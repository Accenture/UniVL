#!/usr/bin/env python3
"""
Compute FID using a pre-selected list of 4000 training GT images as the "real"
distribution and the generated panels of an eval directory as "fake".

The eval dir is expected to contain 1536x512 panels of [cond | gen | gt]; this
script crops out the gen sub-image (cols 512-1024).

Usage:
    python3 evaluation/fid/compute_fid.py \
        --gen_dir output_inference/eval200_Cn50 \
        --real_list output_inference/train_real_4k.jsonl \
        --gpu 0 \
        --out output_inference/eval200_Cn50/fid_train4k.json
"""
import argparse, glob, json, os
import torch
from PIL import Image
import torchvision.transforms as T

TARGET   = (512, 512)
PANEL_W  = 1536


def split_panel(img):
    return (img.crop((0,    0, 512,  512)),
            img.crop((512,  0, 1024, 512)),
            img.crop((1024, 0, 1536, 512)))


def collect_gen(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, "*.jpg")))
    gens = []
    for f in files:
        img = Image.open(f).convert("RGB")
        if img.size[0] != PANEL_W:
            continue
        _, g, _ = split_panel(img)
        gens.append(g)
    return gens


def load_real(real_list_path):
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
                continue
    return paths


def compute_fid(real_paths, gen_imgs, device, batch_size=16):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    to_tensor = T.ToTensor()

    bb = []
    for p in real_paths:
        try:
            img = Image.open(p).convert("RGB").resize(TARGET)
        except Exception:
            continue
        bb.append(to_tensor(img))
        if len(bb) == batch_size:
            fid.update(torch.stack(bb).to(device), real=True)
            bb = []
    if bb:
        fid.update(torch.stack(bb).to(device), real=True)

    bb = []
    for img in gen_imgs:
        bb.append(to_tensor(img.resize(TARGET)))
        if len(bb) == batch_size:
            fid.update(torch.stack(bb).to(device), real=False)
            bb = []
    if bb:
        fid.update(torch.stack(bb).to(device), real=False)

    return fid.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir",   required=True, help="eval dir with 1536x512 panels")
    ap.add_argument("--real_list", required=True, help="JSONL from evaluation/fid/select_reference_images.py")
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--out",       required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    print(f"[gpu={args.gpu}] Loading gen panels from {args.gen_dir} ...")
    gens = collect_gen(args.gen_dir)
    print(f"[gpu={args.gpu}] gen images: {len(gens)}")

    print(f"[gpu={args.gpu}] Loading real list from {args.real_list} ...")
    reals = load_real(args.real_list)
    print(f"[gpu={args.gpu}] real images (pre-load): {len(reals)}")

    fid_score = compute_fid(reals, gens, device)
    result = {
        "FID": fid_score,
        "n_real": len(reals),
        "n_gen": len(gens),
        "gen_dir": args.gen_dir,
        "real_list": args.real_list,
        "device": device,
    }
    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gpu={args.gpu}] Saved → {args.out}")


if __name__ == "__main__":
    main()
