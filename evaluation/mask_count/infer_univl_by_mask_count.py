#!/usr/bin/env python3
"""
Per-bin multibox inference. Loads a JSONL produced by evaluation/mask_count/select_by_mask_count.py
(records in run_multibox_inference's expected format), runs FLUX inference for
each, saves panels to <out_dir>/.

Designed to be launched in parallel (one process per bin × GPU).

Usage (per bin × GPU):
    python -m evaluation.mask_count.infer_univl_by_mask_count \\
        --records output_inference/multibox_by_n/multibox_n2.jsonl \\
        --gpu 0 --bin n2 \\
        --ckpt runs/<run>/ckpt/9000 \\
        --out_dir output_inference/multibox_by_n/eval_n2 \\
        --ab_vae
"""
import argparse, json, os, sys
from evaluation import eval_univl as r200
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--two_condition", action="store_true")
    ap.add_argument("--sigclip_encoder", action="store_true")
    ap.add_argument("--ab_vae", action="store_true")
    ap.add_argument("--use_text_on_mask", action="store_true", default=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # Set runtime flags
    import univl.pipeline.flux_univl as flux_module
    flux_module.TWO_CONDITION = args.two_condition
    if hasattr(flux_module, "SIGCLIP_ENCODER"):
        flux_module.SIGCLIP_ENCODER = args.sigclip_encoder
    if hasattr(flux_module, "AB_VAE"):
        flux_module.AB_VAE = args.ab_vae
    print(f"[bin={args.bin} gpu={args.gpu}] TWO_CONDITION={flux_module.TWO_CONDITION}, "
          f"SIGCLIP_ENCODER={flux_module.SIGCLIP_ENCODER}, AB_VAE={flux_module.AB_VAE}")

    # Load records
    records = []
    with open(args.records) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[bin={args.bin}] {len(records)} records")

    # Load model
    pipe = r200.load_model(
        args.ckpt, device,
        two_condition=args.two_condition,
        sigclip_encoder=args.sigclip_encoder,
        ab_vae=args.ab_vae,
    )

    # Run multibox inference (saves panels to args.out_dir, returns gen/gt/etc.)
    os.makedirs(args.out_dir, exist_ok=True)
    gen, gt, gc, gtc, text, types = r200.run_multibox_inference(
        pipe, records, args.out_dir, device,
        use_text_on_mask=args.use_text_on_mask,
        two_condition=args.two_condition,
    )
    print(f"[bin={args.bin}] inference done, valid pairs: {len(gen)}")


if __name__ == "__main__":
    main()
