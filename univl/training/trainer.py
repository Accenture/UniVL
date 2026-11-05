import lightning as L
from ..paths import DATA_ROOT, OUTPUT_ROOT
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader
import time
from safetensors.torch import load_file, save_file
from typing import List
import math

import prodigyopt
from ..pipeline.flux_univl import transformer_forward, encode_images, encode_images_cond, encode_images_two_cond, get_dropout_prob, ADD_MASK_CHANNEL, GATED_MASK, GatedFusion, TWO_CONDITION, ATTN_MASKING, AB_VAE, SIGCLIP_ENCODER, _build_mask_1d
from ..deepencoder.sam_vary_sdpa import build_sam_vit_b
from ..deepencoder.clip_sdpa import build_clip_l
from ..deepencoder.build_linear import MlpProjector
from addict import Dict
import random
import torchvision
from transformers import CLIPVisionModel
BG_LAMBDA = 0 #0.005

# ── Ablation flag ──────────────────────────────────────────────────────────
# AB_CLIP = True  → load *original* OpenAI CLIP ViT-L/14 weights into the
#                    DeepSeek VitModel (same architecture, different weights).
#                    SAM encoder is still loaded normally; only vision_model
#                    weights are replaced.
# AB_CLIP = False → default behaviour (load OCR-finetuned DeepSeek weights).
AB_CLIP = False


def _clip_to_deepseek_state(clip_vision_state: dict) -> dict:
    """Convert a HuggingFace CLIPVisionModel state_dict into DeepSeek VitModel keys.

    Handles two differences:
      1. Path rename:  encoder.layers.{i}  →  transformer.layers.{i}
      2. QKV fusion:   separate q/k/v_proj →  fused qkv_proj (cat along dim 0)
    Returns a dict loadable by VitModel via load_state_dict(strict=False).
    """
    out: dict = {}
    # Strip the HF "vision_model." prefix if present
    stripped = {}
    for k, v in clip_vision_state.items():
        k2 = k.replace("vision_model.", "") if k.startswith("vision_model.") else k
        stripped[k2] = v

    # Collect separate q/k/v per layer so we can fuse them
    qkv_parts: dict = {}  # layer_i -> {"q"/"k"/"v" -> {"weight": T, "bias": T}}

    for k, v in stripped.items():
        # Rename encoder → transformer
        new_k = k.replace("encoder.layers.", "transformer.layers.")

        # Detect q/k/v_proj keys
        is_qkv = False
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            if f"self_attn.{proj_name}" in new_k:
                parts = new_k.split(".")
                layer_i = parts[2]
                suffix = parts[-1]  # "weight" or "bias"
                letter = proj_name[0]  # q / k / v
                qkv_parts.setdefault(layer_i, {}).setdefault(letter, {})[suffix] = v
                is_qkv = True
                break

        if not is_qkv:
            # Skip post_layernorm (DeepSeek doesn't have it)
            if "post_layernorm" in new_k:
                continue
            out[new_k] = v

    # Fuse q/k/v into qkv_proj for each layer
    for layer_i, qkv in sorted(qkv_parts.items(), key=lambda x: int(x[0])):
        for suffix in ("weight", "bias"):
            q = qkv["q"][suffix]
            k = qkv["k"][suffix]
            v = qkv["v"][suffix]
            fused = torch.cat([q, k, v], dim=0)
            out[f"transformer.layers.{layer_i}.self_attn.qkv_proj.{suffix}"] = fused

    return out


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("UNIVL_CONFIG")
    assert config_path is not None, "Please set the UNIVL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class UniVLModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.target_dim = self.flux_pipe.vae.config.latent_channels * 4
        # --- CUSTOM STEP: Attach OCR Models ---
        train_projector = model_config.get("train_ocr_projector", False)
        self.load_ocr_models_to_pipeline(train_projector=train_projector)
        self.load_deepseek_weights(model_config.get("deepseek_w_path"))#"path/to/deepseek_vl_v2.safetensors")
        self.l2_lambda = model_config.get("l2_loss_lambda", 0.0)
        self.clip_align_lambda = model_config.get("clip_align_lambda", 0.0)
        self.clip_text_lambda = model_config.get("clip_text_lambda", 0.0)
        self.training_phase = model_config.get("training_phase", 2)
        # ATTN_MASKING is controlled only in flux_univl.py
        import univl.pipeline.flux_univl as _flux_mod
        print(f"ATTN_MASKING={_flux_mod.ATTN_MASKING} (from flux_univl.py)")
        if self.clip_align_lambda > 0.0 or self.clip_text_lambda > 0.0:
            self._load_clip_orig(device, dtype)
        # ---------------------------------------------------------
        # 3. Inject LoRA into the Vision Model
        # MUST happen after base weights are loaded to prevent key mismatch
        # ---------------------------------------------------------
        self.apply_vision_lora()
        # Phase 1: enable gradient checkpointing on CLIP vision model to avoid
        # storing all transformer-layer activations for vision_lora backprop.
        # PEFT requires enable_input_require_grads() before GC can work.
        if self.training_phase == 1 and hasattr(self.flux_pipe, "vision_model"):
            try:
                self.flux_pipe.vision_model.enable_input_require_grads()
                self.flux_pipe.vision_model.gradient_checkpointing_enable()
                print("Vision model gradient checkpointing enabled (Phase 1).")
            except Exception as e:
                print(f"Could not enable vision model gradient checkpointing: {e}")
        # Load saved vision LoRA weights if a checkpoint was provided
        if hasattr(self, "_pending_vision_lora_state"):
            set_peft_model_state_dict(self.flux_pipe.vision_model, self._pending_vision_lora_state)
            #del self._pending_vision_lora_state
            print(" - Vision LoRA weights loaded for continued training.")

        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])
        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def _load_clip_orig(self, device, dtype):
        """Load the original (non-OCR-finetuned) CLIP ViT-L/14 as a frozen semantic anchor.
        Used to compute a CLIP alignment loss that recovers the object-semantic grounding
        degraded by OCR fine-tuning (e.g. CLIP no longer knows what 'toy' looks like).
        The 1024-dim patch features are projected to target_dim via a trainable clip_proj.
        Also loads the full CLIP model for text-based semantic loss (clip_text_lambda).
        """
        from transformers import CLIPVisionModel, CLIPModel, CLIPProcessor
        clip_name = "openai/clip-vit-large-patch14"
        print(f"Loading original CLIP ViT-L/14 for semantic alignment loss...")
        self.flux_pipe.clip_orig = CLIPVisionModel.from_pretrained(
            clip_name
        ).to(device=device, dtype=dtype)
        self.flux_pipe.clip_orig.requires_grad_(False).eval()
        # Trainable projection: CLIP patch dim (1024) → conditioning dim (64)
        self.flux_pipe.clip_proj = torch.nn.Linear(
            1024, self.target_dim
        ).to(device=device, dtype=dtype)
        print(f"Original CLIP loaded. clip_proj: 1024 → {self.target_dim}")

        # Full CLIP model for text features + processor
        if self.clip_text_lambda > 0.0:
            print(f"Loading full CLIP model for text-semantic loss...")
            self.clip_text_model = CLIPModel.from_pretrained(
                clip_name, torch_dtype=dtype
            ).to(device)
            self.clip_text_model.requires_grad_(False).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
            # Trainable projection: pooled condition (target_dim) → CLIP text space (768)
            clip_text_hidden = 256
            self.flux_pipe.clip_text_proj = torch.nn.Sequential(
                torch.nn.Linear(self.target_dim, clip_text_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(clip_text_hidden, 768),
            ).to(device=device, dtype=dtype)
            print(f"CLIP text model loaded. clip_text_proj: {self.target_dim} → 256 → 768")

        # Load deferred clip_proj / clip_text_proj weights from trained_adapter_path
        if hasattr(self, "_pending_clip_proj_state"):
            self.flux_pipe.clip_proj.load_state_dict(self._pending_clip_proj_state, strict=True)
            del self._pending_clip_proj_state
            print(" - clip_proj weights loaded (deferred).")
        if hasattr(self, "_pending_clip_text_proj_state") and hasattr(self.flux_pipe, "clip_text_proj"):
            self.flux_pipe.clip_text_proj.load_state_dict(self._pending_clip_text_proj_state, strict=True)
            del self._pending_clip_text_proj_state
            print(" - clip_text_proj weights loaded (deferred).")

    # CLIP normalization constants (ImageNet-derived, used by OpenAI CLIP)
    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

    def apply_vision_lora(self):
        """
        Applies LoRA (rank 4) to the CLIP / SigLIP Vision Model.
        """
        if SIGCLIP_ENCODER:
            # HuggingFace SigLIP uses split q/k/v projections.
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            print("Applying LoRA (r=4) to SigLIP Vision Model...")
        else:
            # DeepSeek VitModel uses a fused qkv_proj.
            target_modules = ["qkv_proj", "out_proj", "fc1", "fc2"]
            print("Applying LoRA (r=4) to CLIP Vision Model...")
        vision_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        # Wrap the vision model with PEFT
        self.flux_pipe.vision_model = get_peft_model(self.flux_pipe.vision_model, vision_lora_config)
        print("****")
        self.flux_pipe.vision_model.print_trainable_parameters()
        
    def load_deepseek_weights(self, checkpoint_path):
        """
        Custom weight loader that implements the user's specific mapping logic:
        if 'sam_model'/'vision_model'/'projector' in name: remove 'model.' prefix.
        """
        print(f"Loading weights from {checkpoint_path}...")
        
        # Load state dict (supports safetensors or bin)
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Containers for the specific sub-modules we attached to pipeline
        sam_state = {}
        vision_state = {}
        projector_state = {}
        
        # Iterate and Apply Mapping Logic
        for name, tensor in state_dict.items():

            # Map the cleaned names to the specific pipeline attributes
            if name.startswith("sam_model."):
                key = name.replace("sam_model.", "")
                sam_state[key] = tensor
            elif name.startswith("vision_model."):
                key = name.replace("vision_model.", "")
                vision_state[key] = tensor
            elif name.startswith("projector."):
                key = name.replace("projector.", "")
                projector_state[key] = tensor
                
        # Load into modules
        # strict=False allows missing keys if your checkpoint has extra stuff,
        # but for the vision parts, we generally expect matches.
        if self.flux_pipe.sam_model:
            self.flux_pipe.sam_model.load_state_dict(sam_state, strict=False)

        if SIGCLIP_ENCODER:
            # SigLIP weights came from from_pretrained; the deepseek checkpoint's vision_model.* keys
            # are architecturally incompatible (different module names, no SAM-conditioned embeddings).
            print("[SIGCLIP_ENCODER] Skipping deepseek vision_model weight load (using SigLIP pretrained).")
        elif AB_CLIP:
            # ── Ablation: replace OCR-finetuned CLIP with original OpenAI CLIP ──
            print("[AB_CLIP] Loading original CLIP ViT-L/14 weights into vision_model...")
            clip_hf = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            vision_state = _clip_to_deepseek_state(clip_hf.state_dict())
            del clip_hf
            info = self.flux_pipe.vision_model.load_state_dict(vision_state, strict=False)
            print(f"[AB_CLIP] Loaded. missing={info.missing_keys}, unexpected={info.unexpected_keys}")
        elif self.flux_pipe.vision_model:
            self.flux_pipe.vision_model.load_state_dict(vision_state, strict=False)

        if self.flux_pipe.projector:
            self.flux_pipe.projector.load_state_dict(projector_state, strict=True)

        print("Deepseek-OCR weights loaded successfully.")

        adapter_path = self.model_config.get("trained_adapter_path")
        if adapter_path:
            state_dict = load_file(adapter_path)
            adapter_state, adapter2_state, conv1_state, conv2_state, vision_lora_state = {}, {}, {}, {}, {}
            clip_proj_state, clip_text_proj_state = {}, {}

            for new_name, tensor in state_dict.items():
                if new_name.startswith("omini_adapter2."):
                    adapter2_state[new_name.replace("omini_adapter2.", "")] = tensor
                elif new_name.startswith("omini_adapter."):
                    adapter_state[new_name.replace("omini_adapter.", "")] = tensor
                elif new_name.startswith("conv1."):
                    conv1_state[new_name.replace("conv1.", "")] = tensor
                elif new_name.startswith("conv2."):
                    conv2_state[new_name.replace("conv2.", "")] = tensor
                elif new_name.startswith("clip_text_proj."):
                    clip_text_proj_state[new_name.replace("clip_text_proj.", "")] = tensor
                elif new_name.startswith("clip_proj."):
                    clip_proj_state[new_name.replace("clip_proj.", "")] = tensor
                elif new_name.startswith("vision_lora."):
                    vision_lora_state[new_name.replace("vision_lora.", "")] = tensor

            if adapter_state and hasattr(self.flux_pipe, "omini_adapter"):
                if AB_CLIP:
                    print("[AB_CLIP] Skipping omini_adapter checkpoint (trained on OCR features).")
                else:
                    self.flux_pipe.omini_adapter.load_state_dict(adapter_state, strict=True)
                    print(" - UniVL adapter weights loaded.")

            if adapter2_state and hasattr(self.flux_pipe, "omini_adapter2"):
                if AB_CLIP:
                    print("[AB_CLIP] Skipping omini_adapter2 checkpoint (trained on OCR features).")
                else:
                    self.flux_pipe.omini_adapter2.load_state_dict(adapter2_state, strict=False)
                    print(" - UniVL fusion adapter weights loaded.")
            
            if conv1_state and hasattr(self.flux_pipe, "conv1"):
                self.flux_pipe.conv1.load_state_dict(conv1_state, strict=True)
                print(" - Conv1 weights loaded.")
                
            if conv2_state and hasattr(self.flux_pipe, "conv2"):
                self.flux_pipe.conv2.load_state_dict(conv2_state, strict=True)
                print(" - Conv2 weights loaded.")
            # Defer vision LoRA loading until after apply_vision_lora() wraps the model
            if vision_lora_state:
                if AB_CLIP:
                    print("[AB_CLIP] Skipping vision_lora checkpoint (incompatible with original CLIP).")
                elif SIGCLIP_ENCODER:
                    print("[SIGCLIP_ENCODER] Skipping vision_lora checkpoint (DeepSeek module names don't map to SigLIP).")
                else:
                    self._pending_vision_lora_state = vision_lora_state

            # Defer clip_proj / clip_text_proj loading: _load_clip_orig() recreates
            # these modules AFTER this method runs, so loading here would be overwritten.
            if clip_proj_state:
                if AB_CLIP:
                    print("[AB_CLIP] Skipping clip_proj checkpoint (trained on OCR features).")
                else:
                    self._pending_clip_proj_state = clip_proj_state
            if clip_text_proj_state:
                self._pending_clip_text_proj_state = clip_text_proj_state

    def load_ocr_models_to_pipeline(self, train_projector=False):
        """
        Initializes Deepseek-OCR models and attaches them to the FluxPipeline.
        """
        device = self.flux_pipe.device
        dtype = self.flux_pipe.dtype

        # 1. Load SAM (Vision Encoder Part 1)
        self.flux_pipe.sam_model = build_sam_vit_b().to(device=device, dtype=dtype)
        self.flux_pipe.sam_model.requires_grad_(False) # ALWAYS FREEZE
        
        # 2. Load CLIP (Vision Encoder Part 2) — or SigLIP when SIGCLIP_ENCODER=True
        if SIGCLIP_ENCODER:
            from transformers import SiglipVisionModel
            siglip_name = "google/siglip-large-patch16-256"
            print(f"[SIGCLIP_ENCODER] Loading {siglip_name} (hidden=1024) in place of DeepSeek CLIP-L.")
            self.flux_pipe.vision_model = SiglipVisionModel.from_pretrained(
                siglip_name
            ).to(device=device, dtype=dtype)
        else:
            self.flux_pipe.vision_model = build_clip_l().to(device=device, dtype=dtype)
        self.flux_pipe.vision_model.requires_grad_(False) # ALWAYS FREEZE

        # 3. Load Projector (Fusion -> 1280 dim)
        proj_cfg = Dict(projector_type="linear", input_dim=2048, n_embed=1280)
        self.flux_pipe.projector = MlpProjector(proj_cfg).to(device=device, dtype=dtype)
        
        # Option to train projector or not
        self.flux_pipe.projector.requires_grad_(train_projector)

        # Adapter is ALWAYS TRAINABLE (it's new, random weights)
        
        #self.flux_pipe.omini_adapter = torch.nn.Linear(1280, self.target_dim).to(device=device, dtype=dtype)
        hidden_dim =  256 #256*2 
        self.flux_pipe.omini_adapter = torch.nn.Sequential(
            torch.nn.Linear(1024, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.target_dim)
        ).to(device=device, dtype=dtype)
        # adapter2 fuses VAE (64) + CLIP (64) features per token
        if GATED_MASK:
            self.flux_pipe.omini_adapter2 = GatedFusion(feat_dim=self.target_dim).to(device=device, dtype=dtype)
        else:
            adapter2_in = 128 + (1 if ADD_MASK_CHANNEL else 0)
            self.flux_pipe.omini_adapter2 = torch.nn.Linear(adapter2_in, self.target_dim).to(device=device, dtype=dtype)
        self.flux_pipe.omini_adapter2.requires_grad_(True)

        self.flux_pipe.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1,  bias=False,).to(device=device, dtype=dtype)
        self.flux_pipe.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3,padding=1,   bias=False,).to(device=device, dtype=dtype)
        
        #torch.nn.Linear(2048+64, self.target_dim).to(device=device, dtype=dtype)
        self.flux_pipe.omini_adapter.requires_grad_(True)

        print(f"Deepseek-OCR modules initialized. UniVL adapter Dim: 1280 -> {self.target_dim}")
        
    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        lora_path = self.model_config.get("trained_lora_path")
        if lora_path:
            print(f"Loading Flux Transformer LoRA weights from {lora_path} ...")
            for adapter_name in self.adapter_set:
                self.flux_pipe.load_lora_weights(
                    lora_path, 
                    adapter_name=adapter_name
                )
                
        # 2. Otherwise, if starting fresh, initialize the PEFT adapter structure
        elif lora_config:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            
        # 3. Diffusers' `load_lora_weights` freezes the layers by default.
        # We must re-enable requires_grad on the LoRA parameters to continue fine-tuning.
        for name, param in self.transformer.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad_(True)
                
        lora_layers = filter(
            lambda p: p.requires_grad, self.transformer.parameters()
        )

        """if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        """
        
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def save_adapter(self, path: str):
        """
        Saves the UniVL adapter and optionally the projector weights.
        """
        
        os.makedirs(path, exist_ok=True)
        
        # Save Projector (only if it was trainable)
        # Check if any param requires grad (implies it was trained)
        if any(p.requires_grad for p in self.flux_pipe.projector.parameters()):
            save_file(
                self.flux_pipe.projector.state_dict(),
                os.path.join(path, "ocr_projector.safetensors")
            )
            print(f"Saved trainable ocr_projector to {path}")

        # 1. Create a dictionary to hold all combined weights
        combined_state_dict = {}

        # Add UniVL adapter weights (with prefix)
        if hasattr(self.flux_pipe, "omini_adapter"):
            for key, tensor in self.flux_pipe.omini_adapter.state_dict().items():
                combined_state_dict[f"omini_adapter.{key}"] = tensor

        if hasattr(self.flux_pipe, "omini_adapter2"):
            for key, tensor in self.flux_pipe.omini_adapter2.state_dict().items():
                combined_state_dict[f"omini_adapter2.{key}"] = tensor

        # Add conv1 weights (with prefix)
        if hasattr(self.flux_pipe, "conv1"):
            for key, tensor in self.flux_pipe.conv1.state_dict().items():
                combined_state_dict[f"conv1.{key}"] = tensor

        # Add conv2 weights (with prefix)
        if hasattr(self.flux_pipe, "conv2"):
            for key, tensor in self.flux_pipe.conv2.state_dict().items():
                combined_state_dict[f"conv2.{key}"] = tensor

        # 3. Gather Vision Model LoRA weights
        if hasattr(self.flux_pipe, "vision_model") and hasattr(self.flux_pipe.vision_model, "peft_config"):
            vision_lora_dict = get_peft_model_state_dict(self.flux_pipe.vision_model)
            for key, tensor in vision_lora_dict.items():
                combined_state_dict[f"vision_lora.{key}"] = tensor

        # Add clip_proj weights if present
        if hasattr(self.flux_pipe, "clip_proj"):
            for key, tensor in self.flux_pipe.clip_proj.state_dict().items():
                combined_state_dict[f"clip_proj.{key}"] = tensor

        # Add clip_text_proj weights if present
        if hasattr(self.flux_pipe, "clip_text_proj"):
            for key, tensor in self.flux_pipe.clip_text_proj.state_dict().items():
                combined_state_dict[f"clip_text_proj.{key}"] = tensor

        # Save the combined dictionary to a single file
        if combined_state_dict:
            save_file(
                combined_state_dict,
                os.path.join(path, "omini_adapter.safetensors")
            )
            print(f"Saved omini_adapter, conv1, conv2, clip_proj combined weights to {path}/omini_adapter.safetensors")

    def configure_optimizers(self):
        # Freeze the transformer (this freezes everything including LoRA initially)
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # 1. Gather LoRA Parameters
        self.trainable_params = self.lora_layers
        
        # 2. Gather OCR Trainable Parameters (Adapter + Projector)
        ocr_params = []
        
        # Check Adapter
        if hasattr(self.flux_pipe, "omini_adapter"):
            ocr_params.extend([p for p in self.flux_pipe.omini_adapter.parameters() if p.requires_grad])
        if hasattr(self.flux_pipe, "omini_adapter2"):
            ocr_params.extend([p for p in self.flux_pipe.omini_adapter2.parameters() if p.requires_grad])
            
        # Check Projector
        if hasattr(self.flux_pipe, "projector"):
            ocr_params.extend([p for p in self.flux_pipe.projector.parameters() if p.requires_grad])
        
        if hasattr(self.flux_pipe, "conv1"):
            ocr_params.extend([p for p in self.flux_pipe.conv1.parameters() if p.requires_grad])
        if hasattr(self.flux_pipe, "conv2"):
            ocr_params.extend([p for p in self.flux_pipe.conv2.parameters() if p.requires_grad])
        # clip_proj is trainable (maps frozen CLIP patch features → conditioning dim)
        if hasattr(self.flux_pipe, "clip_proj"):
            ocr_params.extend([p for p in self.flux_pipe.clip_proj.parameters() if p.requires_grad])
        # clip_text_proj: maps pooled condition latents → CLIP text space (768-dim)
        if hasattr(self.flux_pipe, "clip_text_proj"):
            ocr_params.extend([p for p in self.flux_pipe.clip_text_proj.parameters() if p.requires_grad])
        # 3. Gather Vision Model LoRA Parameters
        # Included in Phase 1 too: CLIP alignment loss flows gradients back through
        # vision_lora, allowing the OCR-finetuned CLIP backbone to recover semantic
        # grounding. Excluded from the DiT-LoRA freeze but still part of conditioning.
        vision_lora_params = []
        if hasattr(self.flux_pipe, "vision_model"):
            vision_lora_params.extend([p for p in self.flux_pipe.vision_model.parameters() if p.requires_grad])

        # Combine lists: Phase 1 trains only OCR adapter (no DiT LoRA)
        if self.training_phase == 1:
            self.trainable_params = ocr_params
            print("Phase 1 (feature alignment): training OCR adapter only, DiT LoRA frozen.")
        else:
            self.trainable_params = self.trainable_params + ocr_params  # LoRA + OCR adapter
            print("Phase 2 (full diffusion): training DiT LoRA + OCR adapter.")

        # Unfreeze all trainable parameters explicitly
        for p in self.trainable_params:
            p.requires_grad_(True)
            
        print(f"Optimizer initialized with {len(self.trainable_params)} trainable parameter tensors.")
        # 3. Create Parameter Groups with custom learning rates
        optim_kwargs = opt_config["params"].copy()
        
        # Extract the base learning rate (defaulting to 1e-4 if not explicitly set in config)
        base_lr = float(optim_kwargs.get("lr", 1e-1))
        
        param_groups = [
            # Group 1: Regular parameters (uses the global base_lr)
            {"params": self.trainable_params}, 
            # Group 2: Vision LoRA parameters (0.1x of the base_lr)
            {"params": vision_lora_params, "lr": base_lr *0.2} #*0.2 }# 1e-4}
        ]
        
        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                param_groups,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(param_groups, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def _training_step_phase1(self, batch, batch_idx):
        """Phase 1: Feature alignment only — no DiT forward pass.
        Trains OCR adapter + clip_proj using L2 VAE alignment and CLIP semantic
        alignment losses. This is cheap (no transformer) and teaches the adapter
        to produce target-consistent conditioning features before Phase 2.
        """
        imgs = batch["image"]
        image_latent_mask = batch.get("image_latent_mask", None)
        pts_raw = batch.get("norm_bbox", None)

        # Convert multibox format (same as Phase 2)
        pts = None
        if pts_raw is not None:
            if isinstance(pts_raw, torch.Tensor):
                pts = pts_raw
            elif isinstance(pts_raw, list) and len(pts_raw) > 0:
                if isinstance(pts_raw[0], list) and len(pts_raw[0]) > 0 and isinstance(pts_raw[0][0], list):
                    B = len(pts_raw)
                    max_boxes = max(len(item_bboxes) for item_bboxes in pts_raw)
                    all_bboxes = []
                    for box_idx in range(max_boxes):
                        pt10 = torch.zeros(B)
                        pt11 = torch.zeros(B)
                        pt20 = torch.zeros(B)
                        pt21 = torch.zeros(B)
                        for b in range(B):
                            if box_idx < len(pts_raw[b]):
                                bb = pts_raw[b][box_idx]
                                pt10[b], pt11[b], pt20[b], pt21[b] = bb
                        all_bboxes.append([pt10, pt11, pt20, pt21])
                    pts = all_bboxes
                else:
                    pts = pts_raw

        conditions = []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])

        with torch.no_grad():
            x_0, _ = encode_images(self.flux_pipe, imgs)

        # Encode conditions with drop_prob=0.0 — OCR always active in Phase 1
        condition_latents, mask_1d = [], None
        for cond in conditions:
            if AB_VAE:
                # AB_VAE ablation: skip OCR encoder, use plain VAE encoding.
                c_latents, _ = encode_images(self.flux_pipe, cond)
                c_mask_1d = _build_mask_1d(pts, c_latents.shape[0], c_latents.device, c_latents.dtype)
            else:
                c_latents, _, c_mask_1d = encode_images_cond(
                    self.flux_pipe, cond, pts, isTrain=True, drop_prob=0.0
                )
            condition_latents.append(c_latents)
            mask_1d = c_mask_1d

        step_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # Component losses stored for wandb logging in the callback
        self.last_l2_text_loss = 0.0
        self.last_l2_bg_loss = 0.0
        self.last_clip_align_loss = 0.0
        self.last_clip_text_loss = 0.0
        # Cosine similarity at text positions (higher = better alignment)
        self.last_cosine_sim = 0.0

        # L2 VAE alignment loss
        if self.l2_lambda > 0.0 and len(condition_latents) > 0 and image_latent_mask is None:
            x0 = x_0.detach()  # local ref; allows del x_0 below
            diff = condition_latents[0] - x0
            C = x0.shape[-1]
            n_mask = mask_1d.sum().clamp(min=1.0)
            text_l2 = (diff * mask_1d).pow(2).sum() / (n_mask * C)
            n_bg = (1.0 - mask_1d).sum().clamp(min=1.0)
            bg_l2 = (diff * (1.0 - mask_1d)).pow(2).sum() / (n_bg * C)
            step_loss = step_loss + self.l2_lambda * (text_l2 + BG_LAMBDA * bg_l2)
            self.last_l2_text_loss = text_l2.item()
            self.last_l2_bg_loss = bg_l2.item()
            # Cosine similarity at text positions: target metric for Phase 1 convergence.
            with torch.no_grad():
                a = (condition_latents[0].detach() * mask_1d).reshape(-1, C)
                b = (x0 * mask_1d).reshape(-1, C)
                norms = (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True)).clamp(min=1e-8)
                self.last_cosine_sim = (a * b).sum(dim=-1, keepdim=True).div(norms).mean().item()
            del x0, diff  # free VAE latents — not needed for CLIP branch

        del x_0  # free original (no_grad) VAE latents regardless of l2_lambda

        # CLIP semantic alignment loss
        if self.clip_align_lambda > 0.0 and len(condition_latents) > 0 and image_latent_mask is None and hasattr(self.flux_pipe, "clip_orig"):
            with torch.no_grad():
                imgs_224 = torch.nn.functional.interpolate(
                    imgs.float(), size=(224, 224), mode="bilinear", align_corners=False
                )
                clip_mean = torch.tensor(self._CLIP_MEAN, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
                clip_std  = torch.tensor(self._CLIP_STD,  device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
                imgs_clip = (imgs_224 - clip_mean) / clip_std
                clip_out = self.flux_pipe.clip_orig(
                    pixel_values=imgs_clip.to(self.flux_pipe.clip_orig.dtype)
                )
                clip_patches = clip_out.last_hidden_state[:, 1:]  # [B, 256, 1024]

            from ..pipeline.flux_univl import _get_spatial_params
            _spatial_dim, _n_tokens = _get_spatial_params()
            clip_proj_feats = self.flux_pipe.clip_proj(clip_patches.to(self.dtype))
            B = clip_proj_feats.shape[0]
            clip_spatial = clip_proj_feats.reshape(B, 16, 16, self.target_dim).permute(0, 3, 1, 2)
            clip_spatial_up = torch.nn.functional.interpolate(
                clip_spatial.float(), size=(_spatial_dim, _spatial_dim), mode="bilinear", align_corners=False
            ).to(self.dtype)
            clip_1d = clip_spatial_up.permute(0, 2, 3, 1).reshape(B, _n_tokens, self.target_dim)
            del clip_proj_feats, clip_spatial, clip_spatial_up  # free upsampled CLIP features
            n_mask_clip = mask_1d.sum().clamp(min=1.0)
            ocr_cond_for_clip_ = condition_latents[1] if TWO_CONDITION else condition_latents[0]
            diff_clip = ocr_cond_for_clip_ - clip_1d.detach()
            del clip_1d
            clip_align_loss = (diff_clip * mask_1d).pow(2).sum() / (n_mask_clip * self.target_dim)
            del diff_clip
            step_loss = step_loss + self.clip_align_lambda * clip_align_loss
            self.last_clip_align_loss = clip_align_loss.item()

        ## CLIP text loss: pooled masked condition → CLIP text embedding of class name
        if self.clip_text_lambda > 0.0 and len(condition_latents) > 0 and hasattr(self, "clip_text_model"):
            objn_list = batch.get("objn", None)
            if objn_list is not None:
                # Multibox: objn="cat|dog" → "cat, dog" for CLIP text
                if isinstance(objn_list, (list, tuple)):
                    objn_list = [o.replace("|", ", ") if isinstance(o, str) else o for o in objn_list]
                with torch.no_grad():
                    text_inputs = self.clip_processor(
                        text=objn_list, return_tensors="pt", padding=True, truncation=True,
                    ).to(self.device)
                    clip_text_embeds = self.clip_text_model.get_text_features(**text_inputs)
                    clip_text_embeds = torch.nn.functional.normalize(clip_text_embeds, dim=-1)

                # Pool masked condition latents (OCR condition in TWO_CONDITION mode)
                ocr_for_text_ = condition_latents[1] if TWO_CONDITION else condition_latents[0]
                masked_cond = ocr_for_text_ * mask_1d
                mask_sum = mask_1d.sum(dim=1).clamp(min=1.0)
                pooled_cond = masked_cond.sum(dim=1) / mask_sum  # (B, target_dim)

                # Project to CLIP text space and normalize
                proj_cond = self.flux_pipe.clip_text_proj(pooled_cond)  # (B, 768)
                proj_cond = torch.nn.functional.normalize(proj_cond, dim=-1)

                clip_text_loss = (1.0 - (proj_cond * clip_text_embeds).sum(dim=-1)).mean()
                step_loss = step_loss + self.clip_text_lambda * clip_text_loss
                self.last_clip_text_loss = clip_text_loss.item()
        ## end CLIP text loss

        self.last_t = 0.5  # dummy for logging compatibility
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def training_step(self, batch, batch_idx):
        if self.training_phase == 1:
            return self._training_step_phase1(batch, batch_idx)
        imgs, prompts = batch["image"], batch["description"] # imgshspae, (B, 3, 512, 512)
        image_latent_mask = batch.get("image_latent_mask", None)


        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))
        #conditions[0].shape torch.Size([16, 3, 512, 512])
        pts_raw = batch.get("norm_bbox", None)

        # Convert multibox format: per-batch-item list-of-bboxes → per-bbox batched tensors
        # Input:  [[bbox_list_for_item0], [bbox_list_for_item1], ...]
        # Output: [[pt10_tensor, pt11_tensor, pt20_tensor, pt21_tensor], ...] (one per unique bbox slot)
        pts = None
        if pts_raw is not None:
            if isinstance(pts_raw, torch.Tensor):
                # Standard single-box tensor (B, 4) — wrap as single bbox
                pts = pts_raw
            elif isinstance(pts_raw, list) and len(pts_raw) > 0:
                if isinstance(pts_raw[0], list) and len(pts_raw[0]) > 0 and isinstance(pts_raw[0][0], list):
                    # Multibox: [[bboxes_item0], [bboxes_item1], ...]
                    # Find max number of bboxes across batch
                    B = len(pts_raw)
                    max_boxes = max(len(item_bboxes) for item_bboxes in pts_raw)
                    # Build per-bbox batched tensors; pad missing boxes with zeros
                    all_bboxes = []
                    for box_idx in range(max_boxes):
                        pt10 = torch.zeros(B)
                        pt11 = torch.zeros(B)
                        pt20 = torch.zeros(B)
                        pt21 = torch.zeros(B)
                        for b in range(B):
                            if box_idx < len(pts_raw[b]):
                                bb = pts_raw[b][box_idx]
                                pt10[b], pt11[b], pt20[b], pt21[b] = bb
                        all_bboxes.append([pt10, pt11, pt20, pt21])
                    pts = all_bboxes  # list of [pt10, pt11, pt20, pt21] per bbox slot
                else:
                    # Single-box list format from collate
                    pts = pts_raw

        """if True:
            # Check the first two images in the batch
            batch_size = conditions[0].shape[0]
            num_to_check = min(2, batch_size)
            
            for b in range(num_to_check):
                img = conditions[0][b] # Shape: [3, 512, 512]
                _, h, w = img.shape
                
                # De-normalize image from [-1, 1] back to [0, 1] for saving
                img_disp = (img + 1.0) / 2.0 
                img_disp = torch.clamp(img_disp, 0.0, 1.0)
                
                # Scale normalized coordinates to absolute pixels
                x_min = int(pt10[b].item() * w)
                y_min = int(pt11[b].item() * h)
                x_max = int(pt20[b].item() * w)
                y_max = int(pt21[b].item() * h)
                
                # Clamp coordinates to valid image dimensions to prevent indexing errors
                x_min, x_max = max(0, x_min), min(w, x_max)
                y_min, y_max = max(0, y_min), min(h, y_max)
                
                # Crop the bounding box region
                if x_max > x_min and y_max > y_min:
                    cropped = img_disp[:, y_min:y_max, x_min:x_max]
                    torchvision.utils.save_image(
                        cropped, 
                        f"debug_batch{batch_idx}_img{b}_cropped.png"
                    )
                else:
                    print(f"Warning: Invalid crop dims for image {b}: x({x_min}-{x_max}), y({y_min}-{y_max})")
                
                # Save the full condition image
                torchvision.utils.save_image(
                    img_disp, 
                    f"debug_batch{batch_idx}_img{b}_full.png"
                )"""

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs) #target image
            
            #print(x_0.shape, img_ids) #torch.Size([2, 1024, 64])
            '''(
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=['blue-sports-car'],
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 2),
                lora_scale=None,
            )'''
            
            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )
            #st()
            #prompt: prompts ['image mask area contains flowers', 'image mask area contains baked-goods']
            #(Pdb) prompt_embeds.shape torch.Size([2, 512, 4096]) (Pdb) pooled_prompt_embeds.shape torch.Size([2, 768])
            # spatial aware t5 embedding?

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]
            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )
        # Prepare conditions
        # ---------------------------------------------------------
        # 2. GRADIENT BLOCK: Trainable Encoders (OCR -> UniVL adapter)
        # ---------------------------------------------------------
        # Moved outside `torch.no_grad()` so gradients flow to the adapter
        condition_latents, condition_ids = [], []

        if TWO_CONDITION:
            # ── TWO_CONDITION: split into VAE + OCR conditions ──
            cond = conditions[0]
            p_delta = position_deltas[0]
            p_scale = position_scales[0]
            vae_tokens, ocr_tokens, shared_ids, mask_1d = encode_images_two_cond(
                self.flux_pipe, cond, pts, isTrain=True,
                drop_prob=get_dropout_prob(self.global_step))

            for tokens in [vae_tokens, ocr_tokens]:
                c_ids = shared_ids.clone()
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                condition_latents.append(tokens)
                condition_ids.append(c_ids)
        else:
            # ── Original single-condition path ──
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                if AB_VAE:
                    # AB_VAE ablation: skip OCR encoder, use plain VAE encoding.
                    c_latents, c_ids = encode_images(self.flux_pipe, cond)
                    mask_1d = _build_mask_1d(pts, c_latents.shape[0], c_latents.device, c_latents.dtype)
                else:
                    c_latents, c_ids, mask_1d = encode_images_cond(self.flux_pipe, cond, pts, isTrain=True, drop_prob=get_dropout_prob(self.global_step))
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)



        n_cond = len(condition_latents)  # 2 in TWO_CONDITION, 1 otherwise
        branch_n = 2 + n_cond
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * n_cond))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Build adapters list: [text=None, noise=None, cond1="default", ...]
        adapters = [None, None] + ["default"] * n_cond

        # Build spatial_mask_info for ATTN_MASKING
        from ..pipeline.flux_univl import ATTN_MASKING
        extra_kwargs = {}
        if ATTN_MASKING and mask_1d is not None:
            h2_n = 1  # text branch count
            extra_kwargs["spatial_mask_info"] = {
                "mask_1d": mask_1d,      # (B, 1024, 1)
                "noise_idx": h2_n + 0,   # noise is first in image_features
                "vae_idx": h2_n + 1,     # VAE/single-cond is second
            }

        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            timesteps=[t, t] + [torch.zeros_like(t)] * n_cond,
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            adapters=adapters,
            return_dict=False,
            group_mask=group_mask,
            **extra_kwargs,
        )
        pred = transformer_out[0]

        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        ## l2 VAE alignment loss
        latent_l2_loss = 0.0

        if self.l2_lambda > 0.0 and len(condition_latents) > 0 and image_latent_mask is None:
            C = x_0.shape[-1]
            n_mask = mask_1d.sum().clamp(min=1.0)
            n_bg = (1.0 - mask_1d).sum().clamp(min=1.0)

            if TWO_CONDITION:
                # condition_latents[0] = VAE (background), condition_latents[1] = OCR (mask)
                vae_cond = condition_latents[0]  # has (1-mask) applied
                ocr_cond = condition_latents[1]  # has mask applied

                # VAE L2: background region should match GT
                vae_diff = vae_cond - (x_0 * (1.0 - mask_1d)).detach()
                bg_l2 = (vae_diff * (1.0 - mask_1d)).pow(2).sum() / (n_bg * C)

                # OCR L2: mask region should match GT
                ocr_diff = ocr_cond - (x_0 * mask_1d).detach()
                text_l2 = (ocr_diff * mask_1d).pow(2).sum() / (n_mask * C)

                latent_l2_loss = text_l2 + BG_LAMBDA * bg_l2
            else:
                diff = condition_latents[0] - x_0.detach()  # [B, 1024, C]

                # Text region: normalize by actual number of mask tokens, not total tokens.
                text_l2 = (diff * mask_1d).pow(2).sum() / (n_mask * C)

                # Background region
                bg_l2 = (diff * (1.0 - mask_1d)).pow(2).sum() / (n_bg * C)

                latent_l2_loss = text_l2 + BG_LAMBDA * bg_l2

            step_loss = step_loss + self.l2_lambda * latent_l2_loss
        ## end l2 VAE alignment loss

        ## CLIP semantic alignment loss
        # Aligns OCR conditioning features at mask positions to frozen original CLIP features,
        # preventing semantic collapse (e.g. "toy" → ball) caused by OCR fine-tuning degrading
        # object-semantic grounding in the OCR-finetuned CLIP backbone.
        if self.clip_align_lambda > 0.0 and len(condition_latents) > 0 and image_latent_mask is None and hasattr(self.flux_pipe, "clip_orig"):
            with torch.no_grad():
                # imgs is [B, 3, H, W] in [0,1]; resize to 224x224 for CLIP
                imgs_224 = torch.nn.functional.interpolate(
                    imgs.float(), size=(224, 224), mode="bilinear", align_corners=False
                )
                clip_mean = torch.tensor(self._CLIP_MEAN, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
                clip_std  = torch.tensor(self._CLIP_STD,  device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
                imgs_clip = (imgs_224 - clip_mean) / clip_std
                clip_out = self.flux_pipe.clip_orig(
                    pixel_values=imgs_clip.to(self.flux_pipe.clip_orig.dtype)
                )
                # Drop CLS token → [B, 256, 1024] patch features on a 16×16 grid
                clip_patches = clip_out.last_hidden_state[:, 1:]

            # Project 1024 → target_dim and upsample 16×16 → spatial_dim to match OCR spatial resolution
            from ..pipeline.flux_univl import _get_spatial_params
            _spatial_dim, _n_tokens = _get_spatial_params()
            clip_proj_feats = self.flux_pipe.clip_proj(clip_patches.to(self.dtype))  # [B, 256, target_dim]
            B = clip_proj_feats.shape[0]
            clip_spatial = clip_proj_feats.reshape(B, 16, 16, self.target_dim).permute(0, 3, 1, 2)
            clip_spatial_up = torch.nn.functional.interpolate(
                clip_spatial.float(), size=(_spatial_dim, _spatial_dim), mode="bilinear", align_corners=False
            ).to(self.dtype)
            clip_1d = clip_spatial_up.permute(0, 2, 3, 1).reshape(B, _n_tokens, self.target_dim)

            # Masked L2 at text region only: encourage OCR conditioning to match CLIP semantics
            n_mask_clip = mask_1d.sum().clamp(min=1.0)
            # In TWO_CONDITION: [0]=VAE, [1]=OCR; otherwise [0]=mixed
            ocr_cond_for_clip = condition_latents[1] if TWO_CONDITION else condition_latents[0]
            diff_clip = ocr_cond_for_clip - clip_1d.detach()
            clip_align_loss = (diff_clip * mask_1d).pow(2).sum() / (n_mask_clip * self.target_dim)
            step_loss = step_loss + self.clip_align_lambda * clip_align_loss
        ## end CLIP alignment loss

        ## CLIP text loss: pooled masked condition → CLIP text embedding of class name
        if self.clip_text_lambda > 0.0 and len(condition_latents) > 0 and hasattr(self, "clip_text_model"):
            objn_list = batch.get("objn", None)
            if objn_list is not None:
                # Multibox: objn="cat|dog" → "cat, dog" for CLIP text
                if isinstance(objn_list, (list, tuple)):
                    objn_list = [o.replace("|", ", ") if isinstance(o, str) else o for o in objn_list]
                with torch.no_grad():
                    text_inputs = self.clip_processor(
                        text=objn_list, return_tensors="pt", padding=True, truncation=True,
                    ).to(self.device)
                    clip_text_embeds = self.clip_text_model.get_text_features(**text_inputs)
                    clip_text_embeds = torch.nn.functional.normalize(clip_text_embeds, dim=-1)

                # In TWO_CONDITION: [1]=OCR (already masked); otherwise [0]=mixed
                ocr_for_text2 = condition_latents[1] if TWO_CONDITION else condition_latents[0]
                masked_cond = ocr_for_text2 * mask_1d
                mask_sum = mask_1d.sum(dim=1).clamp(min=1.0)
                pooled_cond = masked_cond.sum(dim=1) / mask_sum  # (B, target_dim)

                proj_cond = self.flux_pipe.clip_text_proj(pooled_cond)  # (B, 768)
                proj_cond = torch.nn.functional.normalize(proj_cond, dim=-1)

                clip_text_loss = (1.0 - (proj_cond * clip_text_embeds).sum(dim=-1)).mean()
                step_loss = step_loss + self.clip_text_lambda * clip_text_loss
        ## end CLIP text loss

        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function
        self._eval_records = None  # lazy-loaded eval set
        self._clip_model = None    # lazy-loaded CLIP for metrics

    # ── Eval: 40-sample metrics at save_interval ──────────────────────────
    def _build_eval_records(self):
        """Load fixed eval records from pre-built JSONL (consistent across runs)."""
        import json as _json
        EVAL_JSONL = OUTPUT_ROOT + "/eval_sample_set_200.jsonl"

        if os.path.exists(EVAL_JSONL):
            # Use pre-built eval set — take first N per type (deterministic, no randomness)
            N_PER_TYPE = 5
            all_records = []
            with open(EVAL_JSONL) as f:
                for line in f:
                    all_records.append(_json.loads(line.strip()))
            by_type = {}
            for r in all_records:
                by_type.setdefault(r["type"], []).append(r)
            sampled = []
            for t in ["square", "add", "replace", "extract"]:
                items = by_type.get(t, [])
                sampled.extend(items[:N_PER_TYPE])  # first N, no random
            return sampled

        # Fallback: build from JSONs if JSONL not available
        CACHE_DIR = DATA_ROOT
        TRAIN_JSONS = {
            "add":     DATA_ROOT + "/textmask_add_laion/metadata_coords2.json",
            "replace": DATA_ROOT + "/textmask_replace_laion/metadata_coords2.json",
            "extract": DATA_ROOT + "/textmask_extract_ref/metadata_coords2.json",
        }
        from .train import _extract_objn
        rng = random.Random(42)
        all_records = []
        for json_type, json_path in TRAIN_JSONS.items():
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = _json.load(f)
            records = []
            for k, v in data.items():
                bbox = v.get("bbox")
                res = v.get("resolution", {})
                if bbox is None or "width" not in res:
                    continue
                cond_path = k
                base, ext = os.path.splitext(k)
                gt_path = v.get("gt_path", f"{base}_r{ext}")
                objn = _extract_objn(os.path.basename(k)).replace("-", " ")
                records.append({"cond_path": cond_path, "gt_path": gt_path,
                                "bbox": bbox, "resolution": res, "objn": objn, "type": json_type})
            sampled = rng.sample(records, min(3, len(records)))
            all_records.extend(sampled)
        return all_records

    @torch.no_grad()
    def _run_eval(self, pl_module, device):
        """Generate images on 40 fixed eval samples and compute metrics."""
        from PIL import Image
        from ..pipeline.flux_univl import Condition, generate, TWO_CONDITION, RES256
        from .train import _render_text_in_bbox, USE_TEXT_ON_MASK
        from .multibox_dataset import _render_single_text_on_mask
        import numpy as np
        import torch.nn.functional as F
        import torchvision.transforms as T

        from ..pipeline.flux_univl import RES1024
        if RES256:
            eval_res = 256
        elif RES1024:
            eval_res = 1024
        else:
            eval_res = 512

        if self._eval_records is None:
            self._eval_records = self._build_eval_records()
            print(f"[Eval] Built {len(self._eval_records)} eval records")

        if not self._eval_records:
            return {}

        pipe = pl_module.flux_pipe
        gen_images, gt_images, gen_crops, gt_crops = [], [], [], []
        text_labels, record_types = [], []
        to_tensor = T.ToTensor()

        total_eval = len(self._eval_records)
        for rec_idx, rec in enumerate(self._eval_records):
            cond_path = rec["cond_path"]
            if not os.path.exists(cond_path):
                continue
            print(f"  [Eval gen {rec_idx+1}/{total_eval}] {rec['objn']}", flush=True)
            # Support both "bbox": [x1,y1,x2,y2] and "coords": [[x1,y1],[x2,y2]]
            if "bbox" in rec:
                bbox = rec["bbox"]
            elif "coords" in rec:
                pt1, pt2 = rec["coords"]
                bbox = [pt1[0], pt1[1], pt2[0], pt2[1]]
            else:
                continue
            res = rec.get("resolution")
            if res is None:
                # Infer from condition image
                _tmp = Image.open(cond_path)
                res = {"width": _tmp.size[0], "height": _tmp.size[1]}
            w, h = res["width"], res["height"]
            pts = [max(0, min(1, bbox[0]/w)), max(0, min(1, bbox[1]/h)),
                   max(0, min(1, bbox[2]/w)), max(0, min(1, bbox[3]/h))]
            objn = rec["objn"]

            # Render text on mask region
            cond_img = Image.open(cond_path).convert("RGB")
            ow, oh = cond_img.size
            cond_img = cond_img.resize((eval_res, eval_res))
            if bbox:
                x1, y1 = int(bbox[0]*eval_res/ow), int(bbox[1]*eval_res/oh)
                x2, y2 = int(bbox[2]*eval_res/ow), int(bbox[3]*eval_res/oh)
                if USE_TEXT_ON_MASK:
                    cond_img = _render_text_in_bbox(cond_img, x1, y1, x2, y2, objn)
                else:
                    cond_img = _render_single_text_on_mask(cond_img, x1, y1, x2, y2, objn)

            condition = Condition(cond_img, "default", [0, 0], 1.0)
            generator = torch.Generator(device=device).manual_seed(42)
            result = generate(pipe, prompt="", conditions=[condition],
                height=eval_res, width=eval_res, num_inference_steps=28, guidance_scale=3.5,
                generator=generator, model_config={"independent_condition": False},
                kv_cache=False, pts=pts)

            gen_img = result.images[0]
            gen_images.append(gen_img)
            text_labels.append(objn)
            record_types.append(rec["type"])

            # Load GT
            gt_path = rec["gt_path"]
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert("RGB").resize((eval_res, eval_res))
                gt_images.append(gt_img)
            else:
                gt_images.append(None)

            # Crop bbox region
            gw, gh = gen_img.size
            cx1, cy1 = int(pts[0]*gw), int(pts[1]*gh)
            cx2, cy2 = int(pts[2]*gw), int(pts[3]*gh)
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(gw, cx2), min(gh, cy2)
            if cx2 > cx1 and cy2 > cy1:
                gen_crops.append(gen_img.crop((cx1, cy1, cx2, cy2)))
                if gt_images[-1] is not None:
                    gt_crops.append(gt_images[-1].crop((cx1, cy1, cx2, cy2)))
                else:
                    gt_crops.append(None)
            else:
                gen_crops.append(gen_img)
                gt_crops.append(gt_images[-1])

        print(f"  [Eval] Generation done ({len(gen_images)} images), computing metrics...", flush=True)
        # ── Compute metrics ──
        valid = [(g, gt) for g, gt in zip(gen_images, gt_images) if gt is not None]
        if not valid:
            return {}
        gen_list = [p[0] for p in valid]
        gt_list = [p[1] for p in valid]

        results = {}
        # SSIM + PSNR
        print("  [Eval] Computing SSIM/PSNR...", flush=True)
        from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_vals, psnr_vals = [], []
        for g, gt in zip(gen_list, gt_list):
            gt_t = to_tensor(gt).unsqueeze(0).to(device)
            g_t = to_tensor(g.resize(gt.size)).unsqueeze(0).to(device)
            ssim_vals.append(ssim_m(g_t, gt_t).item())
            psnr_vals.append(psnr_m(g_t, gt_t).item())
        results["eval/SSIM"] = np.mean(ssim_vals)
        results["eval/PSNR"] = np.mean(psnr_vals)
        del ssim_m, psnr_m

        # Skip FID — not statistically meaningful with <50 samples
        results["eval/FID"] = float("nan")

        # MUSIQ (no-reference image quality)
        print("  [Eval] Computing MUSIQ...", flush=True)
        try:
            if not hasattr(self, '_musiq_model'):
                import pyiqa
                self._musiq_model = pyiqa.create_metric("musiq", device=device)
            musiq_vals = [self._musiq_model(to_tensor(g.resize((eval_res, eval_res))).unsqueeze(0).to(device)).item() for g in gen_list]
            results["eval/MUSIQ"] = np.mean(musiq_vals)
        except Exception as e:
            print(f"[Eval] MUSIQ failed: {e}")
            results["eval/MUSIQ"] = float("nan")

        # CLIP metrics
        from transformers import CLIPModel, CLIPProcessor
        clip_name = "openai/clip-vit-large-patch14"
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(clip_name, torch_dtype=torch.float32).to(device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(clip_name)
        cm, cp = self._clip_model, self._clip_processor

        print("  [Eval] Computing CLIP metrics...", flush=True)
        # CLIP-Text
        clip_text_vals = []
        valid_texts = [t for t, gt in zip(text_labels, gt_images) if gt is not None]
        for g, text in zip(gen_list, valid_texts):
            inputs = cp(text=[text], images=g, return_tensors="pt", padding=True).to(device)
            out = cm(**inputs)
            clip_text_vals.append((out.logits_per_image / 100.0).item())
        results["eval/CLIP-Text"] = np.mean(clip_text_vals)

        # CLIP-Image
        clip_img_vals = []
        for g, gt in zip(gen_list, gt_list):
            g_f = F.normalize(cm.get_image_features(**cp(images=g, return_tensors="pt").to(device)), dim=-1)
            gt_f = F.normalize(cm.get_image_features(**cp(images=gt, return_tensors="pt").to(device)), dim=-1)
            clip_img_vals.append((g_f * gt_f).sum(dim=-1).item())
        results["eval/CLIP-Image"] = np.mean(clip_img_vals)

        # Region CLIP-T and CLIP-I
        valid_gen_crops = [c for c, gt in zip(gen_crops, gt_images) if gt is not None]
        valid_gt_crops = [c for c, gt in zip(gt_crops, gt_images) if gt is not None and c is not None]
        if valid_gen_crops:
            region_t_vals = []
            for gc, text in zip(valid_gen_crops, valid_texts):
                inputs = cp(text=[text], images=gc.resize((224, 224)), return_tensors="pt", padding=True).to(device)
                out = cm(**inputs)
                region_t_vals.append((out.logits_per_image / 100.0).item())
            results["eval/Region CLIP-T"] = np.mean(region_t_vals)

            region_i_vals = []
            for gc, gtc in zip(valid_gen_crops, valid_gt_crops):
                if gtc is None:
                    continue
                g_f = F.normalize(cm.get_image_features(**cp(images=gc.resize((224, 224)), return_tensors="pt").to(device)), dim=-1)
                gt_f = F.normalize(cm.get_image_features(**cp(images=gtc.resize((224, 224)), return_tensors="pt").to(device)), dim=-1)
                region_i_vals.append((g_f * gt_f).sum(dim=-1).item())
            results["eval/Region CLIP-I"] = np.mean(region_i_vals) if region_i_vals else float("nan")

        torch.cuda.empty_cache()

        # Print summary
        print(f"\n[Eval @ step {self.total_steps}] {len(valid)} samples:")
        for k in ["eval/FID", "eval/SSIM", "eval/PSNR", "eval/MUSIQ",
                   "eval/CLIP-Text", "eval/CLIP-Image", "eval/Region CLIP-T", "eval/Region CLIP-I"]:
            v = results.get(k, float("nan"))
            print(f"  {k:22s}: {v:.4f}")

        return results

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            # Lightning 2.x: training_step returning a plain tensor means outputs IS
            # that tensor, not a dict. Handle both cases to avoid TypeError.
            if isinstance(outputs, dict) and "loss" in outputs:
                raw_loss = outputs["loss"].item()
            elif isinstance(outputs, torch.Tensor):
                raw_loss = outputs.item()
            else:
                raw_loss = float(outputs)
            report_dict = {
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
                "loss": raw_loss * trainer.accumulate_grad_batches,
                "t": pl_module.last_t,
            }
            # Phase 1: log component losses for convergence monitoring
            if getattr(pl_module, "training_phase", 2) == 1:
                report_dict["p1/l2_text"] = getattr(pl_module, "last_l2_text_loss", 0.0)
                report_dict["p1/l2_bg"] = getattr(pl_module, "last_l2_bg_loss", 0.0)
                report_dict["p1/clip_align"] = getattr(pl_module, "last_clip_align_loss", 0.0)
                report_dict["p1/cosine_sim"] = getattr(pl_module, "last_cosine_sim", 0.0)
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            save_dir = f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            # Skip saving DiT LoRA in Phase 1 — it isn't trained there
            if getattr(pl_module, "training_phase", 2) != 1:
                pl_module.save_lora(save_dir)
            pl_module.save_adapter(save_dir)

            # Run eval metrics (Phase 2 only, rank 0 only)
            if getattr(pl_module, "training_phase", 2) != 1:
                if get_rank() == 0:
                    try:
                        pl_module.eval()
                        with torch.no_grad():
                            eval_results = self._run_eval(pl_module, pl_module.device)
                        pl_module.train()
                        torch.cuda.empty_cache()
                        if self.use_wandb and eval_results:
                            eval_results["steps"] = self.total_steps
                            wandb.log(eval_results)
                    except Exception as e:
                        import traceback
                        print(f"[Eval] Failed at step {self.total_steps}: {e}")
                        traceback.print_exc()
                        pl_module.train()
                        torch.cuda.empty_cache()

        # Generate and save a sample image at specified intervals (Phase 2 only)
        if self.total_steps % self.sample_interval == 0 and self.test_function and getattr(pl_module, "training_phase", 2) != 1:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            images = self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()
            if self.use_wandb and images:
                wandb.log({
                    "validation": [wandb.Image(img) for img in images],
                    "steps": self.total_steps,
                })


    def on_train_end(self, trainer, pl_module):
        """Save final adapter weights at end of training.
        For Phase 1, prints the path to use as trained_adapter_path in Phase 2 config.
        """
        save_dir = f"{self.save_path}/{self.run_name}/ckpt/final"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nTraining ended — saving final weights to {save_dir}")
        training_phase = getattr(pl_module, "training_phase", 2)
        if training_phase == 1:
            pl_module.save_adapter(save_dir)
            print(
                f"\n=== Phase 1 complete ===\n"
                f"To start Phase 2, update your YAML:\n"
                f"  model:\n"
                f"    training_phase: 2\n"
                f"    trained_adapter_path: {save_dir}/omini_adapter.safetensors\n"
            )
        else:
            pl_module.save_lora(save_dir)
            pl_module.save_adapter(save_dir)


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    # Use custom collate if dataset has multibox support (variable-length norm_bbox)
    collate_fn = None
    try:
        from .train import multibox_collate_fn
        if hasattr(dataset, 'multibox') and dataset.multibox is not None:
            collate_fn = multibox_collate_fn
            print("Using multibox collate function")
    except ImportError:
        pass
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        collate_fn=collate_fn,
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
