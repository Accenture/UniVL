#!/usr/bin/env python3
"""
Sharded FID computation against the 4000-image train_real_4k.jsonl reference.

torchmetrics FrechetInceptionDistance accumulates per-sample feature stats as
SUMS (real_features_sum, real_features_cov_sum, real_features_num_samples),
so we can compute features for shards of the real set in parallel and add the
stats together to recover the exact same FID we'd get from a single pass.

Two modes:

  --mode features  :  process a shard of N reals, save its FID feature stats
                      (one .pt per shard).
  --mode aggregate :  load all shard .pt files, push their sums into a fresh
                      FID, encode the gens, compute final FID.

Usage:
    # 8 parallel feature jobs (run once each, on GPUs 0-3 with 2 jobs/GPU)
    python3 evaluation/fid/compute_fid_shard.py --mode features \\
        --real_list output_inference/train_real_4k.jsonl \\
        --shard_idx 0 --shard_n 8 --gpu 0 \\
        --out output_inference/fid_shards/shard_0.pt

    # 1 aggregate job
    python3 evaluation/fid/compute_fid_shard.py --mode aggregate \\
        --gen_dir output_inference/eval200_Cn50 \\
        --shards_dir output_inference/fid_shards \\
        --gpu 0 --out output_inference/eval200_Cn50/fid_train4k.json
"""
import argparse, glob, json, os
import torch
from PIL import Image
import torchvision.transforms as T

TARGET  = (512, 512)
PANEL_W = 1536


def split_panel(img):
    return (img.crop((0, 0, 512, 512)),
            img.crop((512, 0, 1024, 512)),
            img.crop((1024, 0, 1536, 512)))


def collect_gen(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, "*.jpg")))
    out = []
    for f in files:
        img = Image.open(f).convert("RGB")
        if img.size[0] != PANEL_W:
            continue
        _, g, _ = split_panel(img)
        out.append(g)
    return out


def load_real_paths(real_list_path):
    paths = []
    with open(real_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                paths.append(rec["gt_path"])
            except Exception:
                pass
    return paths


def make_fid(device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    return FrechetInceptionDistance(feature=2048, normalize=True).to(device)


def push_imgs(fid, imgs_iter, device, real, batch_size=16):
    to_tensor = T.ToTensor()
    bb = []
    for img in imgs_iter:
        bb.append(to_tensor(img.resize(TARGET)))
        if len(bb) == batch_size:
            fid.update(torch.stack(bb).to(device), real=real)
            bb = []
    if bb:
        fid.update(torch.stack(bb).to(device), real=real)


def features_mode(args):
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    paths = load_real_paths(args.real_list)
    # contiguous shard split
    shard_size = (len(paths) + args.shard_n - 1) // args.shard_n
    start = args.shard_idx * shard_size
    end = min(len(paths), start + shard_size)
    my_paths = paths[start:end]
    print(f"[shard {args.shard_idx}/{args.shard_n}, gpu={args.gpu}] {len(my_paths)} reals (range {start}:{end})")

    fid = make_fid(device)

    def gen_imgs():
        for p in my_paths:
            try:
                yield Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  skip {p}: {e}")

    push_imgs(fid, gen_imgs(), device, real=True)

    payload = {
        "real_features_sum":         fid.real_features_sum.detach().cpu(),
        "real_features_cov_sum":     fid.real_features_cov_sum.detach().cpu(),
        "real_features_num_samples": fid.real_features_num_samples.detach().cpu(),
        "shard_idx": args.shard_idx,
        "shard_n":   args.shard_n,
        "n_real":    int(fid.real_features_num_samples.item()),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[shard {args.shard_idx}] saved → {args.out}  (n_real={payload['n_real']})")


def aggregate_mode(args):
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    shard_files = sorted(glob.glob(os.path.join(args.shards_dir, "shard_*.pt")))
    print(f"[agg gpu={args.gpu}] loading {len(shard_files)} shards from {args.shards_dir}")
    if not shard_files:
        raise RuntimeError("No shard files found")

    fid = make_fid(device)
    total_n = 0
    for sp in shard_files:
        s = torch.load(sp, map_location="cpu", weights_only=True)
        fid.real_features_sum         = (fid.real_features_sum         + s["real_features_sum"].to(device)).to(fid.real_features_sum.dtype)
        fid.real_features_cov_sum     = (fid.real_features_cov_sum     + s["real_features_cov_sum"].to(device)).to(fid.real_features_cov_sum.dtype)
        fid.real_features_num_samples = (fid.real_features_num_samples + s["real_features_num_samples"].to(device)).to(fid.real_features_num_samples.dtype)
        total_n += int(s["real_features_num_samples"].item())
        print(f"  + {os.path.basename(sp)}: n={int(s['real_features_num_samples'].item())}")
    print(f"[agg] real total n = {total_n}")

    print(f"[agg] encoding gens from {args.gen_dir}")
    gens = collect_gen(args.gen_dir)
    print(f"[agg] gen images = {len(gens)}")
    push_imgs(fid, gens, device, real=False)

    fid_score = fid.compute().item()
    result = {
        "FID": fid_score,
        "n_real": total_n,
        "n_gen": len(gens),
        "gen_dir": args.gen_dir,
        "shards_dir": args.shards_dir,
        "device": device,
    }
    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[agg] saved → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["features", "aggregate"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    # features mode
    ap.add_argument("--real_list", help="JSONL of real GT paths")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shard_n",   type=int, default=8)
    # aggregate mode
    ap.add_argument("--gen_dir",    help="eval dir with 1536x512 panels")
    ap.add_argument("--shards_dir", help="dir containing shard_*.pt files")
    args = ap.parse_args()

    if args.mode == "features":
        assert args.real_list, "--real_list required in features mode"
        features_mode(args)
    else:
        assert args.gen_dir and args.shards_dir, "--gen_dir and --shards_dir required in aggregate mode"
        aggregate_mode(args)


if __name__ == "__main__":
    main()
