#!/usr/bin/env python3
"""Cost comparison: baseline (text + VAE-cond) vs ours (OCR encoder, no text).

Measures, at each resolution:
  - Tokens per transformer step  (text + image + cond)
  - # parameters in encoder + transformer
  - Encoder TFLOPs + runtime (one-time per generation)
  - Transformer TFLOPs + runtime per step
  - Total (encoder + 28 × transformer)

Baseline (A) — vanilla OminiControl with text:
    encoder = T5 text encoder (encode_prompt) + CLIP-L pooler + VAE encode (cond image @ HxW)
    per-step transformer input tokens = 512 text + image + cond_image_tokens

Ours (B) — OCR encoder, no text:
    encoder = SAM ViT-B + CLIP-L (DeepSeek) + Projector + omini_adapter, run on 1024x1024 image
              (OCR encoder resolution is independent of generation resolution)
    per-step transformer input tokens = 0 text + image + ocr_tokens

FlopCounterMode (PyTorch 2.0+) counts FP MAC ops; reported as TFLOPs (1 MAC = 2 FLOPs).
Wall-clock measured with cuda.synchronize + perf_counter.
"""
import argparse, os, sys, time, json
from univl.paths import OUTPUT_ROOT
import numpy as np
import torch
from PIL import Image

DEFAULT_FLUX = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_INFERENCE_STEPS = 28
OCR_INPUT_SIZE = 1024  # fixed input resolution for SAM+CLIP-L encoder


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
    # FlopCounterMode counts MAC ops; 1 MAC ≈ 2 FLOPs
    return float(fc.get_total_flops()) * 2.0


def build_baseline_pipe(device):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(DEFAULT_FLUX, torch_dtype=DTYPE).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_ours_modules(pipe, device):
    """Build SAM+CLIP-L+Projector OCR encoder + omini_adapter (random init: we measure cost only)."""
    from univl.deepencoder.sam_vary_sdpa import build_sam_vit_b
    from univl.deepencoder.clip_sdpa import build_clip_l
    from univl.deepencoder.build_linear import MlpProjector
    from addict import Dict as ADict

    target_dim = pipe.vae.config.latent_channels * 4  # 64 for FLUX
    sam = build_sam_vit_b().to(device, dtype=DTYPE).eval()
    clip = build_clip_l().to(device, dtype=DTYPE).eval()
    proj_cfg = ADict(projector_type="linear", input_dim=2048, n_embed=1280)
    projector = MlpProjector(proj_cfg).to(device, dtype=DTYPE).eval()
    adapter = torch.nn.Sequential(
        torch.nn.Linear(1024, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, target_dim),
    ).to(device, dtype=DTYPE).eval()
    return sam, clip, projector, adapter


def measure_baseline(pipe, h, w, device, n_inference_steps):
    prompt = "a photo of a cat"

    # 1. Text encoder (T5 + CLIP-L pooler)
    def encode_text():
        with torch.no_grad():
            return pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, device=device,
                num_images_per_prompt=1, max_sequence_length=512,
            )
    prompt_embeds, pooled, text_ids = encode_text()
    text_tokens = prompt_embeds.shape[1]
    text_flops = flops_block(encode_text)
    text_runtime, _ = time_block(encode_text)

    # 2. VAE encode (cond image at H×W)
    cond_img = (torch.rand(1, 3, h, w, device=device, dtype=DTYPE) - 0.5) * 0.2
    def encode_vae():
        with torch.no_grad():
            return pipe.vae.encode(cond_img).latent_dist.sample()
    encode_vae()  # warmup
    vae_flops = flops_block(encode_vae)
    vae_runtime, _ = time_block(encode_vae)

    encoder_flops   = text_flops + vae_flops
    encoder_runtime = text_runtime + vae_runtime
    encoder_params  = count_params(pipe.text_encoder, pipe.text_encoder_2, pipe.vae.encoder)

    # 3. Transformer step  (image stream = noise + VAE-encoded cond, both packed)
    bs = 1
    in_ch = pipe.transformer.config.in_channels // 4  # 16 for FLUX
    gen = torch.Generator(device=device).manual_seed(0)
    dev = torch.device(device) if isinstance(device, str) else device
    noise_packed, latent_image_ids = pipe.prepare_latents(bs, in_ch, h, w, DTYPE, dev, gen)
    cond_packed = noise_packed.clone()
    img_latents = torch.cat([noise_packed, cond_packed], dim=1)
    img_ids = torch.cat([latent_image_ids, latent_image_ids], dim=0)

    timestep = torch.tensor([0.5], device=device, dtype=DTYPE)
    guidance = torch.tensor([3.5], device=device, dtype=DTYPE)

    def transformer_step():
        with torch.no_grad():
            return pipe.transformer(
                hidden_states=img_latents,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
    transformer_step()  # warmup
    txer_flops = flops_block(transformer_step)
    txer_runtime, _ = time_block(transformer_step)

    image_tokens = noise_packed.shape[1]
    cond_tokens  = cond_packed.shape[1]
    transformer_params = count_params(pipe.transformer)

    return {
        "method": "baseline",
        "h": h, "w": w,
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "cond_tokens": cond_tokens,
        "tokens_per_step": text_tokens + image_tokens + cond_tokens,
        "encoder_params": encoder_params,
        "transformer_params": transformer_params,
        "total_params": encoder_params + transformer_params,
        "encoder_flops": encoder_flops,
        "transformer_flops_per_step": txer_flops,
        "total_flops": encoder_flops + n_inference_steps * txer_flops,
        "encoder_runtime_s": encoder_runtime,
        "transformer_runtime_per_step_s": txer_runtime,
        "total_runtime_s": encoder_runtime + n_inference_steps * txer_runtime,
        "_subcost": {
            "text_flops": text_flops, "text_runtime_s": text_runtime,
            "vae_flops": vae_flops, "vae_runtime_s": vae_runtime,
        },
    }


def measure_ours(pipe, sam, clip, projector, adapter, h, w, device, n_inference_steps,
                 ocr_input_size=OCR_INPUT_SIZE):
    """Ours: OCR encoder (SAM+CLIP-L+Projector+adapter) at fixed-size input, no text encoder."""
    bs = 1

    # 1. OCR encoder pass
    cond_img = (torch.rand(bs, 3, ocr_input_size, ocr_input_size, device=device, dtype=DTYPE) - 0.5) * 0.2

    def encode_ocr():
        with torch.no_grad():
            sam_features = sam(cond_img)
            clip_outputs = clip(cond_img, sam_features)
            features = clip_outputs[:, 1:]               # drop CLS
            features = adapter(features)
        return features

    ocr_features = encode_ocr()  # warmup + token-count probe
    ocr_tokens_n = ocr_features.shape[1]
    ocr_flops = flops_block(encode_ocr)
    ocr_runtime, _ = time_block(encode_ocr)
    encoder_params = count_params(sam, clip, projector, adapter)

    # 2. Transformer step (no text path; image stream = noise + ocr tokens)
    in_ch = pipe.transformer.config.in_channels // 4
    gen = torch.Generator(device=device).manual_seed(0)
    dev = torch.device(device) if isinstance(device, str) else device
    noise_packed, latent_image_ids = pipe.prepare_latents(bs, in_ch, h, w, DTYPE, dev, gen)

    img_latents = torch.cat([noise_packed, ocr_features], dim=1)
    ocr_ids = torch.zeros(ocr_tokens_n, 3, device=device, dtype=DTYPE)
    img_ids_full = torch.cat([latent_image_ids, ocr_ids], dim=0)

    # No-text path: minimal length-1 prompt_embeds (FLUX needs a non-empty seq).
    pooled_dim = pipe.text_encoder.config.hidden_size              # CLIP-L = 768
    encoder_dim = pipe.text_encoder_2.config.d_model               # T5 = 4096
    pooled_empty        = torch.zeros(bs, pooled_dim, device=device, dtype=DTYPE)
    prompt_embeds_empty = torch.zeros(bs, 1, encoder_dim, device=device, dtype=DTYPE)
    text_ids_empty      = torch.zeros(1, 3, device=device, dtype=DTYPE)
    text_tokens_n = prompt_embeds_empty.shape[1]

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

    image_tokens = noise_packed.shape[1]
    transformer_params = count_params(pipe.transformer)

    return {
        "method": "ours",
        "h": h, "w": w,
        "text_tokens": text_tokens_n,
        "image_tokens": image_tokens,
        "cond_tokens": ocr_tokens_n,
        "tokens_per_step": text_tokens_n + image_tokens + ocr_tokens_n,
        "encoder_params": encoder_params,
        "transformer_params": transformer_params,
        "total_params": encoder_params + transformer_params,
        "encoder_flops": ocr_flops,
        "transformer_flops_per_step": txer_flops,
        "total_flops": ocr_flops + n_inference_steps * txer_flops,
        "encoder_runtime_s": ocr_runtime,
        "transformer_runtime_per_step_s": txer_runtime,
        "total_runtime_s": ocr_runtime + n_inference_steps * txer_runtime,
        "_meta": {"ocr_input_size": ocr_input_size},
    }


def fmt(v, kind):
    if v is None: return "—"
    if kind == "tokens": return f"{int(v):,}"
    if kind == "params":
        if v >= 1e9: return f"{v/1e9:.2f}B"
        if v >= 1e6: return f"{v/1e6:.1f}M"
        return f"{v/1e3:.1f}k"
    if kind == "flops":
        return f"{v/1e12:.3f}T"
    if kind == "time":
        if v >= 1: return f"{v:.3f}s"
        return f"{v*1000:.1f}ms"
    return str(v)


def print_table(results, n_steps):
    by_res = {}
    for r in results:
        by_res.setdefault(f"{r['h']}×{r['w']}", {})[r['method']] = r

    rows = [
        ("text_tokens",        "text tokens",            "tokens"),
        ("image_tokens",       "image tokens",           "tokens"),
        ("cond_tokens",        "cond/OCR tokens",        "tokens"),
        ("tokens_per_step",    "tokens/step (total)",    "tokens"),
        (None, None, None),
        ("encoder_params",     "encoder params",         "params"),
        ("transformer_params", "transformer params",     "params"),
        ("total_params",       "total params",           "params"),
        (None, None, None),
        ("encoder_flops",      "encoder FLOPs (×1)",     "flops"),
        ("transformer_flops_per_step", "transformer FLOPs (per step)", "flops"),
        ("total_flops",        f"total FLOPs (×1 + ×{n_steps})", "flops"),
        (None, None, None),
        ("encoder_runtime_s",  "encoder runtime (×1)",   "time"),
        ("transformer_runtime_per_step_s", "transformer runtime (per step)", "time"),
        ("total_runtime_s",    f"total runtime (×1 + ×{n_steps})", "time"),
    ]

    for key in by_res:
        b = by_res[key].get("baseline"); o = by_res[key].get("ours")
        print(f"\n{'='*92}\nResolution {key}\n{'='*92}")
        print(f"{'Metric':<36s} {'Baseline (A)':>16s} {'Ours (B)':>16s} {'Δ% (A→B)':>14s}")
        print('-' * 92)
        for k, label, kind in rows:
            if k is None:
                print()
                continue
            va = b.get(k) if b else None
            vb = o.get(k) if o else None
            if va is not None and vb is not None and va != 0:
                delta = (vb - va) / va * 100
                d = f"{delta:+.1f}%"
            else:
                d = "—"
            print(f"{label:<36s} {fmt(va,kind):>16s} {fmt(vb,kind):>16s} {d:>14s}")


def _impr_pct(a, b):
    """Improvement % from A→B (positive means B is smaller/better for cost metrics)."""
    if a is None or b is None or a == 0:
        return None
    return (a - b) / a * 100.0


def _fmt_count(v):
    if v is None: return "—"
    return f"{int(v):,}"


def _fmt_params(v):
    if v is None: return "—"
    if v >= 1e9: return f"{v/1e9:.2f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    return f"{v/1e3:.1f}k"


def _fmt_tflops(v):
    if v is None: return "—"
    return f"{v/1e12:.1f}"


def _fmt_seconds(v):
    if v is None: return "—"
    return f"{v:.3f}"


def _fmt_ms(v):
    if v is None: return "—"
    return f"{v*1000:.1f}"


def _fmt_pct(v):
    if v is None: return "—"
    return f"{v:.1f}%"


def print_table2_style(results, n_steps):
    """Reproduce Table 2 layout: A | B | Impr. for each resolution, side by side."""
    by_res = {}
    for r in results:
        by_res.setdefault((r["h"], r["w"]), {})[r["method"]] = r
    keys_sorted = sorted(by_res.keys(), reverse=True)  # 1024, 512, 256

    def cells(getter, fmt_fn, impr_kind="lower_better"):
        """Return one row of cells: A,B,Impr × 3 resolutions."""
        out = []
        for k in keys_sorted:
            a = by_res[k].get("baseline"); b = by_res[k].get("ours")
            va = getter(a) if a else None
            vb = getter(b) if b else None
            if impr_kind == "lower_better":
                imp = _impr_pct(va, vb)
            elif impr_kind == "higher_better":
                imp = _impr_pct(vb, va) if va is not None and vb is not None else None
            else:
                imp = None
            out += [fmt_fn(va), fmt_fn(vb), _fmt_pct(imp) if imp is not None else "—"]
        return out

    # Build rows
    table_rows = []
    table_rows.append(("Tokens/step",
                       cells(lambda r: r["tokens_per_step"], _fmt_count)))
    table_rows.append(("  └ image tokens",
                       cells(lambda r: r["image_tokens"], _fmt_count)))
    table_rows.append(("  └ cond/OCR tokens",
                       cells(lambda r: r["cond_tokens"], _fmt_count)))
    table_rows.append(("  └ text tokens",
                       cells(lambda r: r["text_tokens"], _fmt_count)))

    def input_size(r):
        # tokens × hidden dim:  text·4096 + image·64 + cond·64 + (pooled 1·768 if baseline)
        if r is None: return None
        sz = r["text_tokens"] * 4096 + r["image_tokens"] * 64 + r["cond_tokens"] * 64
        if r["method"] == "baseline":
            sz += 1 * 768  # CLIP-L pooled
        return sz
    table_rows.append(("Input size (count×dim)",
                       cells(input_size, _fmt_count)))

    # Params (encoder + transformer)
    table_rows.append(("Params (encoder+transformer)",
                       cells(lambda r: r["total_params"], _fmt_params)))
    table_rows.append(("  └ encoder params",
                       cells(lambda r: r["encoder_params"], _fmt_params)))
    table_rows.append(("  └ transformer params",
                       cells(lambda r: r["transformer_params"], _fmt_params)))

    # TFLOPs (all steps)
    table_rows.append((f"TFLOPs (×1 + ×{n_steps})",
                       cells(lambda r: r["total_flops"], _fmt_tflops)))
    table_rows.append(("  └ encoder FLOPs (×1)",
                       cells(lambda r: r["encoder_flops"], _fmt_tflops)))
    table_rows.append((f"  └ transformer FLOPs (×{n_steps})",
                       cells(lambda r: n_steps * r["transformer_flops_per_step"], _fmt_tflops)))

    # Runtime
    table_rows.append((f"Runtime (s, ×1 + ×{n_steps})",
                       cells(lambda r: r["total_runtime_s"], _fmt_seconds)))
    table_rows.append(("Per-step (ms, transformer)",
                       cells(lambda r: r["transformer_runtime_per_step_s"], _fmt_ms)))

    # Print
    res_labels = [f"{h}×{w}" for (h, w) in keys_sorted]
    sep = "  "
    col_w = 9
    header_top = " " * 38
    for rl in res_labels:
        header_top += f"{rl:^{col_w*3 + 2*len(sep)}}{sep*0}"
    print("=" * (38 + (col_w*3 + 2*len(sep) + 0) * len(keys_sorted)))
    print("Cost comparison — Table-2 style")
    print("=" * (38 + (col_w*3 + 2*len(sep) + 0) * len(keys_sorted)))
    print(header_top)
    sub_header = f"{'Metric':<38s}"
    for _ in res_labels:
        sub_header += f"{'A':>{col_w}s}{sep}{'B':>{col_w}s}{sep}{'Impr.':>{col_w}s}"
    print(sub_header)
    print("-" * (38 + (col_w*3 + 2*len(sep)) * len(keys_sorted)))
    for label, c in table_rows:
        line = f"{label:<38s}"
        for i in range(len(keys_sorted)):
            a, b, impr = c[i*3], c[i*3+1], c[i*3+2]
            line += f"{a:>{col_w}s}{sep}{b:>{col_w}s}{sep}{impr:>{col_w}s}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resolutions", default="256,512,1024", help="comma-separated")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--ocr_input_size", type=str, default="match",
                    help="OCR encoder input resolution: 'match' (=gen res) or int (e.g., 1024)")
    ap.add_argument("--out_json",
                    default=OUTPUT_ROOT + "/cost_compare.json")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    resolutions = [int(x) for x in args.resolutions.split(",")]

    print("Loading FluxPipeline (baseline backbone) ...")
    pipe = build_baseline_pipe(device)
    print("Building OCR encoder modules (random init) ...")
    sam, clip, projector, adapter = build_ours_modules(pipe, device)

    results = []
    for r in resolutions:
        print(f"\n>>> Measuring at {r}×{r} ({args.steps} denoising steps)")
        try:
            results.append(measure_baseline(pipe, r, r, device, args.steps))
        except Exception as e:
            import traceback
            print(f"baseline {r} FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
        try:
            ocr_in = r if args.ocr_input_size == "match" else int(args.ocr_input_size)
            results.append(measure_ours(pipe, sam, clip, projector, adapter, r, r, device,
                                        args.steps, ocr_input_size=ocr_in))
        except Exception as e:
            import traceback
            print(f"ours {r} FAIL {type(e).__name__}: {e}")
            traceback.print_exc()

    print_table(results, args.steps)
    print()
    print_table2_style(results, args.steps)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
