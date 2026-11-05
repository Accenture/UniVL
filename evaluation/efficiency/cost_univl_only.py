#!/usr/bin/env python3
"""Cost analysis for OURS only: text encoders are physically removed from memory.

Steps:
  1. Load FluxPipeline (T5 + CLIP-L + VAE + transformer).
  2. Snapshot GPU memory.
  3. del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2;
     gc.collect(); torch.cuda.empty_cache(). Verify they're gone.
  4. Snapshot GPU memory (should drop by ~T5 size, ~9.5 GB at bf16).
  5. Build SAM + DeepSeek CLIP-L + Projector + omini_adapter (OCR encoder).
  6. Snapshot GPU memory.
  7. For each resolution {256, 512, 1024}:
        - OCR encoder forward → FLOPs + runtime + token count.
        - Single transformer step (no-text path) → FLOPs + runtime + token count.
        - Total: 1× encoder + 28× transformer step.
  8. Print param counts (re-counted after deletion) + FLOPs + runtime + memory.

Notes:
- Text encoders/tokenizers are GONE: any code path that touches them will crash.
- We pass `prompt_embeds = zeros(1, 1, 4096)` and `pooled = zeros(1, 768)` directly
  into the transformer (no text encoder forward).
- FLOP counts are MAC×2 from torch.utils.flop_counter.FlopCounterMode.
- Runtime uses cuda.synchronize + perf_counter, 2 warmup + 3 measured runs (mean).
"""
import argparse, gc, json, os, sys, time
from univl.paths import OUTPUT_ROOT
import numpy as np
import torch

DEFAULT_FLUX = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_INFERENCE_STEPS = 28
# OCR encoder input matches the generation resolution (fair comparison: at H×W gen,
# the condition image is also H×W → OCR token count = (H/16)² = baseline VAE-cond count).
# Override with --ocr_input N to fix it across resolutions.
OCR_INPUT_SIZE_DEFAULT = "match"


def count_params(*modules):
    return sum(p.numel() for m in modules if m is not None for p in m.parameters())


def time_block(fn, n_warmup=2, n_runs=3):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def flops_block(fn):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False, depth=0)
    with fc:
        fn()
    return float(fc.get_total_flops()) * 2.0  # MAC → FLOPs


def gpu_mem_gb(device):
    return torch.cuda.memory_allocated(device) / 1024**3


def fmt_t(v): return f"{v/1e12:.3f}T"
def fmt_p(v):
    if v >= 1e9: return f"{v/1e9:.2f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    return f"{v/1e3:.1f}k"
def fmt_ms(v):
    return f"{v:.3f}s" if v >= 1 else f"{v*1000:.1f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resolutions", default="256,512,1024")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--ocr_input", default=OCR_INPUT_SIZE_DEFAULT,
                    help="OCR encoder input size: 'match' (=gen resolution) or an integer (e.g., 1024)")
    ap.add_argument("--out_json",
                    default=OUTPUT_ROOT + "/cost_ours_clean.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # ── Step 1. Load FluxPipeline ──────────────────────────────────────────────
    from diffusers import FluxPipeline
    print("Loading FluxPipeline ...")
    pipe = FluxPipeline.from_pretrained(DEFAULT_FLUX, torch_dtype=DTYPE).to(device)
    pipe.set_progress_bar_config(disable=True)

    # ── Step 2. Snapshot ───────────────────────────────────────────────────────
    torch.cuda.synchronize()
    mem_before = gpu_mem_gb(device)
    text_enc_params = count_params(pipe.text_encoder, pipe.text_encoder_2)
    print(f"  After load:                              {mem_before:6.2f} GB  "
          f"(text_encoders={fmt_p(text_enc_params)})")

    # ── Step 3. Physically remove text encoders + tokenizers ───────────────────
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_del = gpu_mem_gb(device)
    print(f"  After deleting text encoders/tokenizers: {mem_after_del:6.2f} GB  "
          f"(Δ = {mem_before - mem_after_del:+.2f} GB freed)")

    # Sanity check: confirm they're truly gone
    assert pipe.text_encoder is None and pipe.text_encoder_2 is None, "text encoders still present"
    assert pipe.tokenizer is None and pipe.tokenizer_2 is None, "tokenizers still present"

    # ── Step 4. Build OCR encoder (random init — measuring cost only) ──────────
    from univl.deepencoder.sam_vary_sdpa import build_sam_vit_b
    from univl.deepencoder.clip_sdpa import build_clip_l
    from univl.deepencoder.build_linear import MlpProjector
    from addict import Dict as ADict

    target_dim = pipe.vae.config.latent_channels * 4  # 64 for FLUX
    print("Building OCR encoder (SAM + DeepSeek CLIP-L + Projector + adapter) ...")
    sam = build_sam_vit_b().to(device, dtype=DTYPE).eval()
    clip = build_clip_l().to(device, dtype=DTYPE).eval()
    proj_cfg = ADict(projector_type="linear", input_dim=2048, n_embed=1280)
    projector = MlpProjector(proj_cfg).to(device, dtype=DTYPE).eval()
    adapter = torch.nn.Sequential(
        torch.nn.Linear(1024, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, target_dim),
    ).to(device, dtype=DTYPE).eval()

    torch.cuda.synchronize()
    mem_after_ocr = gpu_mem_gb(device)
    ocr_params = count_params(sam, clip, projector, adapter)
    print(f"  After loading OCR encoder:               {mem_after_ocr:6.2f} GB  "
          f"(ocr_encoder={fmt_p(ocr_params)})")

    # ── Step 5. Per-resolution measurement ─────────────────────────────────────
    resolutions = [int(x) for x in args.resolutions.split(",")]
    transformer_params = count_params(pipe.transformer)
    vae_enc_params = count_params(pipe.vae.encoder)

    print(f"\nFinal in-memory module param counts (after text encoder deletion):")
    print(f"  transformer:        {fmt_p(transformer_params):>12s}")
    print(f"  OCR encoder total:  {fmt_p(ocr_params):>12s}")
    print(f"     SAM ViT-B:       {fmt_p(count_params(sam)):>12s}")
    print(f"     DeepSeek CLIP-L: {fmt_p(count_params(clip)):>12s}")
    print(f"     Projector:       {fmt_p(count_params(projector)):>12s}")
    print(f"     adapter (1024→64):{fmt_p(count_params(adapter)):>12s}")
    print(f"  VAE encoder:        {fmt_p(vae_enc_params):>12s}  (still present, used by both)")
    print(f"  TOTAL (in-memory):  {fmt_p(transformer_params + ocr_params + vae_enc_params):>12s}")

    results = []
    for r in resolutions:
        h = w = r
        bs = 1
        in_ch = pipe.transformer.config.in_channels // 4

        # 1. OCR encoder forward (input size = match gen res, or fixed integer)
        ocr_in = r if args.ocr_input == "match" else int(args.ocr_input)
        cond_img = (torch.rand(bs, 3, ocr_in, ocr_in, device=device, dtype=DTYPE) - 0.5) * 0.2

        def encode_ocr():
            with torch.no_grad():
                sam_features = sam(cond_img)
                clip_outputs = clip(cond_img, sam_features)
                features = clip_outputs[:, 1:]   # drop CLS
                features = adapter(features)
            return features

        ocr_features = encode_ocr()
        ocr_tokens_n = ocr_features.shape[1]
        ocr_flops = flops_block(encode_ocr)
        ocr_runtime, _ = time_block(encode_ocr)

        # 2. Single transformer step (no text path; image stream = noise + ocr tokens)
        gen = torch.Generator(device=device).manual_seed(0)
        noise_packed, latent_image_ids = pipe.prepare_latents(bs, in_ch, h, w, DTYPE, device, gen)
        image_tokens = noise_packed.shape[1]
        img_latents = torch.cat([noise_packed, ocr_features], dim=1)
        ocr_ids = torch.zeros(ocr_tokens_n, 3, device=device, dtype=DTYPE)
        img_ids_full = torch.cat([latent_image_ids, ocr_ids], dim=0)

        # FLUX transformer requires non-empty encoder_hidden_states — pass length-1 zeros.
        # (The text encoders themselves have been deleted; we feed the transformer directly.)
        pooled_empty = torch.zeros(bs, 768, device=device, dtype=DTYPE)
        prompt_embeds_empty = torch.zeros(bs, 1, 4096, device=device, dtype=DTYPE)
        text_ids_empty = torch.zeros(1, 3, device=device, dtype=DTYPE)

        timestep = torch.tensor([0.5], device=device, dtype=DTYPE)
        guidance = torch.tensor([3.5], device=device, dtype=DTYPE)

        def transformer_step():
            with torch.no_grad():
                return pipe.transformer(
                    hidden_states=img_latents,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_empty,
                    encoder_hidden_states=prompt_embeds_empty,
                    txt_ids=text_ids_empty,
                    img_ids=img_ids_full,
                    return_dict=False,
                )[0]

        transformer_step()  # warmup
        txer_flops = flops_block(transformer_step)
        txer_runtime, _ = time_block(transformer_step)

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

        n_steps = args.steps
        rec = {
            "resolution": f"{h}x{w}",
            "h": h, "w": w,
            "ocr_input_size": ocr_in,
            "tokens": {
                "text": prompt_embeds_empty.shape[1],   # = 1 (placeholder)
                "image": image_tokens,
                "ocr": ocr_tokens_n,
                "per_step_total": prompt_embeds_empty.shape[1] + image_tokens + ocr_tokens_n,
            },
            "params": {
                "ocr_encoder": ocr_params,
                "vae_encoder": vae_enc_params,
                "transformer": transformer_params,
                "in_memory_total": ocr_params + vae_enc_params + transformer_params,
                "(text_encoder)": "DELETED",
            },
            "flops": {
                "encoder_one_time": ocr_flops,
                "transformer_per_step": txer_flops,
                f"total_x1_plus_x{n_steps}": ocr_flops + n_steps * txer_flops,
            },
            "runtime_s": {
                "encoder_one_time": ocr_runtime,
                "transformer_per_step": txer_runtime,
                f"total_x1_plus_x{n_steps}": ocr_runtime + n_steps * txer_runtime,
            },
            "peak_mem_gb": peak_mem,
        }
        results.append(rec)

        print(f"\n{'='*78}\nResolution {h}×{w}  (steps={n_steps})\n{'='*78}")
        print(f"  Tokens         text={rec['tokens']['text']:>5d}  image={image_tokens:>5d}  "
              f"ocr={ocr_tokens_n:>5d}  total/step={rec['tokens']['per_step_total']:>5d}")
        print(f"  FLOPs (single) encoder={fmt_t(ocr_flops):>10s}  transformer={fmt_t(txer_flops):>10s}")
        print(f"  FLOPs (×{n_steps:>2d})    total = {fmt_t(rec['flops'][f'total_x1_plus_x{n_steps}']):>10s}")
        print(f"  Runtime (1×)   encoder={fmt_ms(ocr_runtime):>10s}  per_step={fmt_ms(txer_runtime):>10s}")
        print(f"  Runtime (×{n_steps:>2d})  total = {fmt_ms(rec['runtime_s'][f'total_x1_plus_x{n_steps}']):>10s}")
        print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # ── Save ───────────────────────────────────────────────────────────────────
    summary = {
        "memory_gb": {
            "after_load": mem_before,
            "after_text_encoder_deletion": mem_after_del,
            "delta_freed": mem_before - mem_after_del,
            "after_ocr_loaded": mem_after_ocr,
        },
        "param_summary": {
            "removed_text_encoders": text_enc_params,
            "ocr_encoder": ocr_params,
            "transformer": transformer_params,
            "vae_encoder": vae_enc_params,
        },
        "per_resolution": results,
        "n_steps_used": args.steps,
        "ocr_input_size": args.ocr_input,
        "method": "ours_text_encoder_deleted",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
