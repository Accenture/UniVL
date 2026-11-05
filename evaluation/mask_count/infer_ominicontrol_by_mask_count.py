#!/usr/bin/env python3
"""
Baseline (FLUX + fill_old LoRA) per-bin multibox inference.

Loads a JSONL split (e.g. multibox_n2.jsonl) produced by evaluation/mask_count/select_by_mask_count.py,
runs vanilla fill_old inference with `prompt=""` (empty), saves panels.
"""
import argparse, json, os, sys
from evaluation.baselines import eval_ominicontrol as bl
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_mode", choices=["empty", "class"], default="empty")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    # Force vanilla flags (baseline uses standard FLUX path, no OCR encoder)
    import univl.pipeline.flux_univl as flux_module
    flux_module.TWO_CONDITION = False
    if hasattr(flux_module, "AB_VAE"):          flux_module.AB_VAE = False
    if hasattr(flux_module, "SIGCLIP_ENCODER"): flux_module.SIGCLIP_ENCODER = False
    if hasattr(flux_module, "ATTN_MASKING"):    flux_module.ATTN_MASKING = False
    print(f"[bin={args.bin} gpu={args.gpu}] flags: TWO_COND=False, AB_VAE=False, "
          f"SIGCLIP_ENCODER=False, prompt_mode={args.prompt_mode}")

    # Load records
    records = []
    with open(args.records) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[bin={args.bin}] {len(records)} records")

    # Load baseline pipeline (FLUX + fill_old)
    pipe = bl.load_baseline(device)

    os.makedirs(args.out_dir, exist_ok=True)
    gen, gt, gc, gtc, text, types = bl.run_multibox_baseline(
        pipe, records, args.out_dir, device, args.prompt_mode
    )
    print(f"[bin={args.bin}] inference done, valid pairs: {len(gen)}")


if __name__ == "__main__":
    main()
