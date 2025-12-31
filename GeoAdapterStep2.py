#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoAdapterStep2.py

Step 2 — Injection into SDXL (baseline concat, minimal pipeline integration)

Goal:
  Use Step-1 packaged conditioning E' as encoder_hidden_states in SDXL UNet cross-attention.
  No attention rewrite yet, just baseline concatenation.
  Train ONLY the geometry mapper f_theta with diffusion loss.

This script:
  - loads an SDXL pipeline (local path or HF id)
  - freezes UNet, VAE, and text encoders
  - computes prompt embeddings
  - computes geometry descriptors G (+ confidence c) from reference images via Step 1
  - maps G -> geometry tokens, concatenates to text hidden states -> E'
  - calls UNet with encoder_hidden_states=E'
  - optimizes mapper weights only

Notes:
  - For a quick end-to-end check, use the default tiny SDXL pipeline:
      --model hf-internal-testing/tiny-stable-diffusion-xl-pipe

FFHQ full-dataset training (the new default workflow):
  1) Ensure you have FFHQ baseline CSVs:
       baselines/<run_id>/baseline_index.csv
       baselines/<run_id>/baseline_geometry.csv
     (This repo already includes an FFHQ baseline folder under `baselines/`.)

  2) Produce a Step1 mapper checkpoint (and optionally tokens) over FFHQ:
       python GeoAdapterStep1_batch.py --data-source ffhq --device auto --save-tokens
     Output: geo_step1_batch_out/geo_mapper_step1.pt

  3) Train Step2 over the FULL FFHQ dataset:
       python GeoAdapterStep2.py --mode ffhq --device cpu --epochs 1 --batch-size 2
     For real SDXL training you likely want:
       --model stabilityai/stable-diffusion-xl-base-1.0 --resolution 1024

Tip:
  - If you want a fast smoke test without requiring Step1 ckpt compatibility, use:
       --model hf-internal-testing/tiny-stable-diffusion-xl-pipe --init-mapper fresh
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Step 1 utilities live in the same repo as a single-file module.
import GeoAdapterStep1 as geo1


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: torch (required for Step 2).") from e
    return torch


def _require_diffusers():
    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
        from diffusers import DDPMScheduler  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: diffusers (pip install diffusers).") from e
    return StableDiffusionXLPipeline, DDPMScheduler


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pandas (required for FFHQ dataset mode).") from e
    return pd


def _pil_load_rgb(path: Path):
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _image_to_tensor_0_1(im, *, size: int):
    """
    Convert PIL RGB to float tensor in [0,1] with shape (3,H,W) resized to (size,size).
    """
    torch = _require_torch()
    im = im.resize((size, size))
    arr = np.asarray(im).astype(np.float32) / 255.0  # (H,W,3)
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x


def _make_time_ids(batch: int, height: int, width: int, device, dtype):
    """
    SDXL conditioning uses `time_ids` with:
      [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    Common baseline: crop=(0,0) and target=(height,width).
    """
    torch = _require_torch()
    base = torch.tensor([height, width, 0, 0, height, width], device=device, dtype=dtype)
    return base.view(1, 6).repeat(batch, 1)


@dataclass
class TrainConfig:
    model: str
    resolution: int = 256  # Default to 256x256 for training
    batch_size: int = 1
    steps: int = 1
    lr: float = 1e-4
    n_geo_tokens: int = 4
    token_norm: str = "layernorm"
    lambda_geo: float = 1.0
    prompt: str = "a photo of a person"
    device: str = "cuda"
    use_fp16: bool = True  # Default to FP16 for GPU to save memory
    out_dir: str = "geo_step2_out"
    seed: int = 123

    # Dataset training (FFHQ baseline)
    mode: str = "smoke"  # "smoke" or "ffhq"
    baseline_dir: str = ""  # folder containing baseline_index.csv + baseline_geometry.csv
    step1_out_dir: str = "geo_step1_batch_out"  # folder containing geo_mapper_step1.pt
    init_mapper: str = "step1"  # "step1" (load Step1 ckpt) or "fresh" (random init with matching C)
    mapper_hidden_dim: int = 256
    mapper_n_layers: int = 3
    max_samples: int = 0  # 0 = all
    num_epochs: int = 1
    max_train_steps: int = 0  # 0 = no cap
    log_every: int = 50
    save_every: int = 500
    num_workers: int = 0
    use_landmark_conf: bool = False  # if true, uses baseline_index.csv's landmark_conf as c


def _resolve_baseline_files(baseline_dir: Path) -> Tuple[Path, Path]:
    """
    Baseline directory must contain:
      - baseline_index.csv (image_id -> path)
      - baseline_geometry.csv (image_id -> 8-D descriptor)
    """
    idx = baseline_dir / "baseline_index.csv"
    geom = baseline_dir / "baseline_geometry.csv"
    if not idx.exists():
        raise FileNotFoundError(f"baseline_index.csv not found: {idx}")
    if not geom.exists():
        raise FileNotFoundError(f"baseline_geometry.csv not found: {geom}")
    return idx, geom


def _load_ffhq_table(
    *,
    baseline_dir: Path,
    max_samples: int,
    use_landmark_conf: bool,
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows:
      { "image_id", "path", "G": np.ndarray shape (8,), "c": float }
    """
    pd = _require_pandas()

    idx_csv, geom_csv = _resolve_baseline_files(baseline_dir)
    df_idx = pd.read_csv(idx_csv)
    df_geom = pd.read_csv(geom_csv)

    # Inner join on image_id to ensure consistent ordering
    df = df_geom.merge(df_idx[["image_id", "path", "landmark_conf"]], on="image_id", how="inner")
    if len(df) == 0:
        raise RuntimeError(f"Join produced 0 rows. Check image_id consistency between {geom_csv} and {idx_csv}.")

    feat_cols = list(geo1.GEOMETRY_FEATURE_NAMES)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"baseline_geometry.csv is missing expected feature columns: {missing}")

    if max_samples and max_samples > 0:
        df = df.iloc[: int(max_samples)].copy()

    rows: List[Dict[str, Any]] = []
    for _i, r in df.iterrows():
        G = r[feat_cols].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(G)):
            continue
        p = Path(str(r["path"])).expanduser()
        c = float(r["landmark_conf"]) if (use_landmark_conf and "landmark_conf" in r) else 1.0
        rows.append({"image_id": str(r["image_id"]), "path": p, "G": G, "c": float(c)})
    if len(rows) == 0:
        raise RuntimeError("No valid FFHQ rows after filtering non-finite geometry descriptors.")
    return rows


def load_sdxl_pipeline(cfg: TrainConfig):
    torch = _require_torch()
    StableDiffusionXLPipeline, DDPMScheduler = _require_diffusers()

    # Determine actual device to use
    gpu_memory_gb = 0.0
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested but not available. PyTorch version: {torch.__version__}")
            print(f"[WARNING] PyTorch has CUDA support compiled, but CUDA runtime is not available.")
            print(f"[WARNING] This usually means NVIDIA drivers need to be installed/updated.")
            print(f"[WARNING] Falling back to CPU. Install NVIDIA drivers and restart to enable GPU.")
            actual_device = "cpu"
        else:
            actual_device = cfg.device
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
            # Clear GPU cache before loading to avoid OOM
            torch.cuda.empty_cache()
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU memory: {gpu_memory_gb:.2f} GB")
            # For GPUs with < 8GB, force FP16 to save memory
            if gpu_memory_gb < 8.0 and not cfg.use_fp16:
                print(f"[WARNING] GPU has < 8GB memory. Enabling FP16 to save memory (use --fp16 explicitly).")
                cfg.use_fp16 = True
            # Warn if memory is very tight
            if gpu_memory_gb < 4.5:
                print(f"[WARNING] GPU memory is very limited ({gpu_memory_gb:.2f} GB).")
                print(f"[WARNING] Estimated requirement: ~3.95 GB - this is a tight fit!")
                print(f"[WARNING] Training may fail if memory fragmentation occurs.")
    else:
        actual_device = cfg.device
    
    dtype = torch.float16 if (cfg.use_fp16 and actual_device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    print(f"[INFO] Loading model with dtype: {dtype}")
    
    # For low-memory GPUs, enable memory-efficient attention if available
    enable_attention_slicing = False
    if actual_device.startswith("cuda") and gpu_memory_gb < 6.0:
        enable_attention_slicing = True
        print(f"[INFO] GPU has < 6GB, will enable attention slicing to save memory")
    
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model, torch_dtype=dtype, variant=None)
        
        # Enable memory optimizations for low-memory GPUs
        if enable_attention_slicing:
            try:
                pipe.enable_attention_slicing()
                print(f"[INFO] Attention slicing enabled")
            except Exception:
                pass  # Not all models support this
        
        print(f"[INFO] Moving pipeline to device: {actual_device}")
        # Move components one by one to avoid OOM during transfer
        if actual_device.startswith("cuda"):
            pipe.unet.to(actual_device)
            torch.cuda.empty_cache()
            pipe.vae.to(actual_device)
            torch.cuda.empty_cache()
            pipe.text_encoder.to(actual_device)
            torch.cuda.empty_cache()
            pipe.text_encoder_2.to(actual_device)
            torch.cuda.empty_cache()
        else:
            pipe.to(actual_device)
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"[ERROR] CUDA out of memory! GPU has {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"[ERROR] Estimated requirement: ~3.95 GB (very tight fit)")
        print(f"[ERROR] Try:")
        print(f"  1. Ensure --fp16 is enabled (auto-enabled for GPU)")
        print(f"  2. Reduce batch size: --batch-size 1")
        print(f"  3. Use smaller model: --model hf-internal-testing/tiny-stable-diffusion-xl-pipe --init-mapper fresh")
        print(f"  4. Use CPU: --device cpu (slower but will work)")
        raise

    # Use a training-friendly scheduler (DDPM) if present; otherwise keep pipeline's.
    try:
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    return pipe


def freeze_module(m) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def maybe_cache_geometry(cache_dir: Path, ref_path: Path, *, allow_rotation: bool = True) -> Tuple[np.ndarray, float]:
    """
    Cache geometry descriptor per reference image to avoid repeated MediaPipe calls.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = ref_path.as_posix().replace("/", "_").replace(":", "").replace("\\", "_")
    cache_path = cache_dir / f"{key}.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["G"].astype(np.float32), float(z["c"])

    G, c = geo1.geometry_from_reference_image(ref_path, allow_rotation=allow_rotation)
    np.savez_compressed(cache_path, G=G.astype(np.float32), c=float(c))
    return G, float(c)


class _FFHQBaselineDataset:
    def __init__(self, rows: List[Dict[str, Any]], *, resolution: int):
        self.rows = rows
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[int(idx)]
        img_path: Path = r["path"]
        if not img_path.exists():
            raise FileNotFoundError(str(img_path))

        im = _pil_load_rgb(img_path)
        x01 = _image_to_tensor_0_1(im, size=self.resolution)  # (3,H,W) float32
        return {
            "image_id": r["image_id"],
            "pixel_0_1": x01,
            "G": r["G"],  # np.ndarray (8,)
            "c": float(r["c"]),
            "path": str(img_path),
        }


def _collate_ffhq(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    torch = _require_torch()
    good: List[Dict[str, Any]] = []
    for b in batch:
        # Dataset __getitem__ may raise; DataLoader will handle, but be defensive for manual calls.
        if b is None:
            continue
        good.append(b)
    if len(good) == 0:
        return None

    px = torch.stack([x["pixel_0_1"] for x in good], dim=0).contiguous()  # (B,3,H,W)
    G = torch.from_numpy(np.stack([x["G"] for x in good], axis=0).astype(np.float32))  # (B,8)
    c = torch.tensor([float(x["c"]) for x in good], dtype=torch.float32)  # (B,)
    return {
        "image_id": [x["image_id"] for x in good],
        "pixel_0_1": px,
        "G": G,
        "c": c,
        "path": [x["path"] for x in good],
    }


def train_one_or_few_steps(cfg: TrainConfig, *, image_path: Path, ref_image_path: Optional[Path] = None) -> Dict[str, Any]:
    torch = _require_torch()

    # Verify GPU availability if CUDA is requested
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but PyTorch CUDA is not available. "
                f"Current PyTorch version: {torch.__version__}. "
                f"Please install PyTorch with CUDA support: "
                f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
        print(f"[INFO] Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Training resolution: {cfg.resolution}x{cfg.resolution}")
    else:
        print(f"[INFO] Training on CPU (device: {cfg.device})")
        print(f"[INFO] Training resolution: {cfg.resolution}x{cfg.resolution}")

    # Repro
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    pipe = load_sdxl_pipeline(cfg)
    device = pipe.device
    dtype = pipe.unet.dtype

    # Freeze everything except mapper
    freeze_module(pipe.unet)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.text_encoder_2)

    # Read cross-attention dim C from UNet config (do not hardcode)
    C = int(getattr(pipe.unet.config, "cross_attention_dim"))
    D = len(geo1.GEOMETRY_FEATURE_NAMES)
    mapper = geo1.GeometryTokenMapper(
        geom_dim=D,
        cross_attention_dim=C,
        n_geo_tokens=int(cfg.n_geo_tokens),
        hidden_dim=256,
        n_layers=3,
        token_norm=str(cfg.token_norm),
    ).to(device)

    mapper.train()

    opt = torch.optim.AdamW(mapper.parameters(), lr=float(cfg.lr))

    # Prepare a single sample repeatedly (baseline smoke)
    img_pil = _pil_load_rgb(image_path)
    x01 = _image_to_tensor_0_1(img_pil, size=int(cfg.resolution)).unsqueeze(0).to(device=device, dtype=torch.float32)
    pixel_values = x01 * 2.0 - 1.0  # [-1,1]

    ref_p = ref_image_path if ref_image_path is not None else image_path
    cache_dir = Path(cfg.out_dir) / "geom_cache"
    G_np, c = maybe_cache_geometry(cache_dir, ref_p, allow_rotation=True)

    G = torch.from_numpy(G_np[None, :]).to(device=device, dtype=torch.float32)
    conf = torch.tensor([float(c)], device=device, dtype=torch.float32)

    # Encode prompt -> E_text
    with torch.no_grad():
        enc = pipe.encode_prompt(
            prompt=cfg.prompt,
            prompt_2=cfg.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        # Diffusers SDXL commonly returns:
        #   (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        # even when CFG is disabled.
        if isinstance(enc, (list, tuple)):
            if len(enc) >= 3:
                prompt_embeds = enc[0]
                pooled_prompt_embeds = enc[2]
            elif len(enc) == 2:
                prompt_embeds, pooled_prompt_embeds = enc
            else:
                raise ValueError(f"Unexpected encode_prompt() return length: {len(enc)}")
        else:
            raise TypeError(f"Unexpected encode_prompt() return type: {type(enc)}")

        # prompt_embeds: (B, L, C)
        # pooled_prompt_embeds: (B, C2) for SDXL added conditioning

    # Build E' = [E_text; T_geo_scaled]
    T_geo_scaled = geo1.make_geo_tokens(mapper, G, conf=conf, lambda_geo=float(cfg.lambda_geo), lambda_vec=None)
    E_prime = geo1.concat_conditioning(prompt_embeds, T_geo_scaled, drop_if_zero=True)

    # Add SDXL added_cond_kwargs
    time_ids = _make_time_ids(batch=1, height=int(cfg.resolution), width=int(cfg.resolution), device=device, dtype=pooled_prompt_embeds.dtype)
    added = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

    # Encode image -> latents
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values.to(dtype=pipe.vae.dtype)).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # Training loop
    losses: List[float] = []
    for step in range(int(cfg.steps)):
        opt.zero_grad(set_to_none=True)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, int(pipe.scheduler.config.num_train_timesteps), (latents.shape[0],), device=device, dtype=torch.long
        )
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # UNet forward uses encoder_hidden_states for cross-attn
        model_pred = pipe.unet(
            noisy_latents.to(dtype=dtype),
            timesteps,
            encoder_hidden_states=E_prime.to(dtype=dtype),
            added_cond_kwargs=added,
            return_dict=False,
        )[0]

        # Diffusion target
        pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = torch.mean((model_pred.float() - target.float()) ** 2)
        loss.backward()
        opt.step()

        losses.append(float(loss.detach().cpu().item()))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "geo_mapper_step2.pt"
    geo1.save_geo_adapter(mapper, ckpt)

    # Sanity: load and reproduce token shape
    mapper2 = geo1.load_geo_adapter(ckpt, map_location="cpu")
    assert mapper2.cross_attention_dim == mapper.cross_attention_dim
    assert mapper2.n_geo_tokens == mapper.n_geo_tokens

    return {
        "model": cfg.model,
        "cross_attention_dim": C,
        "geom_dim": D,
        "n_geo_tokens": int(cfg.n_geo_tokens),
        "lambda_geo": float(cfg.lambda_geo),
        "c": float(c),
        "E_text_shape": tuple(prompt_embeds.shape),
        "E_prime_shape": tuple(E_prime.shape),
        "losses": losses,
        "ckpt_path": str(ckpt),
        "ref_image": str(ref_p),
        "image": str(image_path),
    }


def train_ffhq_dataset(cfg: TrainConfig) -> Dict[str, Any]:
    """
    Full-dataset training loop over FFHQ baseline (geometry from baseline CSVs, images from baseline_index.csv).

    This mode trains ONLY the geometry mapper f_theta with diffusion loss.
    """
    torch = _require_torch()
    from torch.utils.data import DataLoader  # type: ignore

    # Verify GPU availability if CUDA is requested
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but PyTorch CUDA is not available. "
                f"Current PyTorch version: {torch.__version__}. "
                f"Please install PyTorch with CUDA support: "
                f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
        print(f"[INFO] Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Training resolution: {cfg.resolution}x{cfg.resolution}")
    else:
        print(f"[INFO] Training on CPU (device: {cfg.device})")
        print(f"[INFO] Training resolution: {cfg.resolution}x{cfg.resolution}")

    # Repro
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading FFHQ baseline data from: {cfg.baseline_dir}")
    baseline_dir = Path(cfg.baseline_dir).expanduser().resolve()
    rows = _load_ffhq_table(
        baseline_dir=baseline_dir,
        max_samples=int(cfg.max_samples),
        use_landmark_conf=bool(cfg.use_landmark_conf),
    )
    print(f"[INFO] Loaded {len(rows)} samples from FFHQ baseline")

    print(f"[INFO] Loading SDXL pipeline: {cfg.model}")
    print(f"[INFO] This may take several minutes if downloading from HuggingFace (~7GB)...")
    pipe = load_sdxl_pipeline(cfg)
    print(f"[INFO] Pipeline loaded successfully")
    device = pipe.device
    dtype = pipe.unet.dtype

    # Freeze everything except mapper
    freeze_module(pipe.unet)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.text_encoder_2)

    C = int(getattr(pipe.unet.config, "cross_attention_dim"))
    D = len(geo1.GEOMETRY_FEATURE_NAMES)

    # Init mapper
    print(f"[INFO] Initializing geometry mapper...")
    step1_ckpt: Optional[Path] = None
    if str(cfg.init_mapper).lower() == "fresh":
        print(f"[INFO] Creating fresh mapper (random initialization)")
        mapper = geo1.GeometryTokenMapper(
            geom_dim=int(D),
            cross_attention_dim=int(C),
            n_geo_tokens=int(cfg.n_geo_tokens),
            hidden_dim=int(cfg.mapper_hidden_dim),
            n_layers=int(cfg.mapper_n_layers),
            token_norm=str(cfg.token_norm),
        ).to(device)
    else:
        # Load from Step1 batch output (recommended for real training)
        step1_ckpt = Path(cfg.step1_out_dir).expanduser().resolve() / "geo_mapper_step1.pt"
        if not step1_ckpt.exists():
            raise FileNotFoundError(f"Step1 mapper checkpoint not found: {step1_ckpt}")
        print(f"[INFO] Loading Step1 mapper from: {step1_ckpt}")
        mapper = geo1.load_geo_adapter(step1_ckpt, map_location="cpu").to(device)
        print(f"[INFO] Step1 mapper loaded successfully")
    mapper.train()
    print(f"[INFO] Mapper initialized on device: {device}")

    if int(mapper.cross_attention_dim) != int(C):
        raise ValueError(
            f"Mapper cross_attention_dim ({mapper.cross_attention_dim}) != UNet cross_attention_dim ({C}). "
            f"Use an SDXL model with matching C, set --init-mapper fresh for smoke tests, "
            f"or regenerate Step1 batch outputs for this model."
        )
    if int(mapper.n_geo_tokens) != int(cfg.n_geo_tokens):
        # Allow overriding cfg.n_geo_tokens from mapper (mapper is the source of truth)
        cfg.n_geo_tokens = int(mapper.n_geo_tokens)

    opt = torch.optim.AdamW(mapper.parameters(), lr=float(cfg.lr))
    print(f"[INFO] Optimizer initialized (lr={cfg.lr})")

    print(f"[INFO] Creating dataset with {len(rows)} samples at {cfg.resolution}x{cfg.resolution} resolution")
    ds = _FFHQBaselineDataset(rows, resolution=int(cfg.resolution))
    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate_ffhq,
    )
    print(f"[INFO] DataLoader ready (batch_size={cfg.batch_size}, epochs={cfg.num_epochs})")

    # Encode prompt once (fixed prompt)
    print(f"[INFO] Encoding prompt: '{cfg.prompt}'")
    with torch.no_grad():
        enc = pipe.encode_prompt(
            prompt=cfg.prompt,
            prompt_2=cfg.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        if isinstance(enc, (list, tuple)):
            if len(enc) >= 3:
                prompt_embeds_1 = enc[0]  # (1, L, C)
                pooled_prompt_embeds_1 = enc[2]  # (1, C2)
            elif len(enc) == 2:
                prompt_embeds_1, pooled_prompt_embeds_1 = enc
            else:
                raise ValueError(f"Unexpected encode_prompt() return length: {len(enc)}")
        else:
            raise TypeError(f"Unexpected encode_prompt() return type: {type(enc)}")

    global_step = 0
    losses: List[float] = []
    running: List[float] = []

    print(f"[INFO] Starting training loop...")
    print(f"[INFO] Total epochs: {cfg.num_epochs}, Batches per epoch: {len(dl)}")
    for epoch in range(int(cfg.num_epochs)):
        for batch in dl:
            if batch is None:
                continue
            if int(cfg.max_train_steps) > 0 and global_step >= int(cfg.max_train_steps):
                break

            px01 = batch["pixel_0_1"].to(device=device, dtype=torch.float32)  # (B,3,H,W)
            pixel_values = px01 * 2.0 - 1.0  # [-1,1]
            B = int(pixel_values.shape[0])

            G = batch["G"].to(device=device, dtype=torch.float32)  # (B,8)
            conf = batch["c"].to(device=device, dtype=torch.float32)  # (B,)

            # Prepare conditioning
            prompt_embeds = prompt_embeds_1.repeat(B, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds_1.repeat(B, 1)
            time_ids = _make_time_ids(batch=B, height=int(cfg.resolution), width=int(cfg.resolution), device=device, dtype=pooled_prompt_embeds.dtype)
            added = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

            # VAE encode image -> latents (no gradients)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values.to(dtype=pipe.vae.dtype)).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            opt.zero_grad(set_to_none=True)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, int(pipe.scheduler.config.num_train_timesteps), (B,), device=device, dtype=torch.long
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Build E' = [E_text; T_geo_scaled]
            T_geo_scaled = geo1.make_geo_tokens(mapper, G, conf=conf, lambda_geo=float(cfg.lambda_geo), lambda_vec=None)
            E_prime = geo1.concat_conditioning(prompt_embeds, T_geo_scaled, drop_if_zero=True)

            model_pred = pipe.unet(
                noisy_latents.to(dtype=dtype),
                timesteps,
                encoder_hidden_states=E_prime.to(dtype=dtype),
                added_cond_kwargs=added,
                return_dict=False,
            )[0]

            pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
            if pred_type == "epsilon":
                target = noise
            elif pred_type == "v_prediction":
                target = pipe.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_type: {pred_type}")

            loss = torch.mean((model_pred.float() - target.float()) ** 2)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            running.append(loss_val)
            if len(running) > 200:
                running.pop(0)

            global_step += 1

            if int(cfg.log_every) > 0 and (global_step % int(cfg.log_every) == 0):
                mean_recent = float(np.mean(running)) if running else loss_val
                print(f"[TRAIN] step={global_step} epoch={epoch+1}/{cfg.num_epochs} loss={loss_val:.6f} mean200={mean_recent:.6f}")

            if int(cfg.save_every) > 0 and (global_step % int(cfg.save_every) == 0):
                ckpt = out_dir / "geo_mapper_step2.pt"
                geo1.save_geo_adapter(mapper, ckpt)
                (out_dir / "step2_train_report.json").write_text(
                    json.dumps(
                        {
                            "mode": "ffhq",
                            "model": cfg.model,
                            "baseline_dir": str(baseline_dir),
                            "init_mapper": str(cfg.init_mapper),
                            "step1_ckpt": str(step1_ckpt) if step1_ckpt is not None else "",
                            "cross_attention_dim": int(C),
                            "geom_dim": int(D),
                            "n_geo_tokens": int(mapper.n_geo_tokens),
                            "lambda_geo": float(cfg.lambda_geo),
                            "resolution": int(cfg.resolution),
                            "batch_size": int(cfg.batch_size),
                            "lr": float(cfg.lr),
                            "num_epochs": int(cfg.num_epochs),
                            "max_train_steps": int(cfg.max_train_steps),
                            "max_samples": int(cfg.max_samples),
                            "global_step": int(global_step),
                            "last_loss": float(loss_val),
                            "mean_loss_200": float(np.mean(running)) if running else float(loss_val),
                            "ckpt_path": str(ckpt),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        if int(cfg.max_train_steps) > 0 and global_step >= int(cfg.max_train_steps):
            break

    ckpt = out_dir / "geo_mapper_step2.pt"
    geo1.save_geo_adapter(mapper, ckpt)

    return {
        "mode": "ffhq",
        "model": cfg.model,
        "baseline_dir": str(baseline_dir),
        "init_mapper": str(cfg.init_mapper),
        "step1_ckpt": str(step1_ckpt) if step1_ckpt is not None else "",
        "cross_attention_dim": int(C),
        "geom_dim": int(D),
        "n_geo_tokens": int(mapper.n_geo_tokens),
        "lambda_geo": float(cfg.lambda_geo),
        "resolution": int(cfg.resolution),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "num_epochs": int(cfg.num_epochs),
        "max_train_steps": int(cfg.max_train_steps),
        "max_samples": int(cfg.max_samples),
        "global_step": int(global_step),
        "losses_head": losses[:10],
        "losses_tail": losses[-10:],
        "ckpt_path": str(ckpt),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2: inject E' into SDXL UNet and train mapper only.")
    ap.add_argument("--mode", type=str, default="smoke", choices=["smoke", "ffhq"], help="smoke: single-image check; ffhq: full baseline dataset training loop.")
    ap.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Local SDXL path or HF id (default: real SDXL base model).")
    ap.add_argument("--image", type=str, default="face.jpg", help="Training target image path (single-sample smoke by default).")
    ap.add_argument("--ref-image", type=str, default="", help="Reference image path for geometry (defaults to --image).")
    ap.add_argument("--prompt", type=str, default="a photo of a person", help="Text prompt for SDXL.")
    ap.add_argument("--steps", type=int, default=1, help="Number of optimization steps (smoke test default=1).")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size (smoke test uses 1).")
    ap.add_argument("--resolution", type=int, default=256, help="Train resolution (use 256 for tiny model smoke; 1024 for real SDXL).")
    ap.add_argument("--n-geo-tokens", type=int, default=4, help="Number of geometry tokens N_geo.")
    ap.add_argument("--lambda-geo", type=float, default=1.0, help="Scalar geometry strength λ_geo.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 when possible (auto-enabled for GPU to save memory).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable fp16 (not recommended for GPU).")
    ap.set_defaults(fp16=None)  # None means auto-detect
    ap.add_argument("--out-dir", type=str, default="geo_step2_out", help="Output directory for checkpoints and cache.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")

    # FFHQ dataset mode args
    ap.add_argument(
        "--baseline-dir",
        type=str,
        default="baselines/ffhq_mediapipe468_20251225T102206Z_cdc39a62e7c1",
        help="FFHQ baseline folder containing baseline_index.csv + baseline_geometry.csv.",
    )
    ap.add_argument(
        "--step1-out-dir",
        type=str,
        default="geo_step1_batch_out",
        help="Output folder from GeoAdapterStep1_batch.py (must contain geo_mapper_step1.pt).",
    )
    ap.add_argument(
        "--init-mapper",
        type=str,
        default="step1",
        choices=["step1", "fresh"],
        help="ffhq mode: mapper init. step1 loads geo_mapper_step1.pt; fresh random-inits a mapper matching the model C (useful for tiny-model smoke tests).",
    )
    ap.add_argument("--mapper-hidden-dim", type=int, default=256, help="Only used with --init-mapper fresh.")
    ap.add_argument("--mapper-n-layers", type=int, default=3, help="Only used with --init-mapper fresh.")
    ap.add_argument("--max-samples", type=int, default=0, help="Limit dataset to first N samples (0 = all).")
    ap.add_argument("--epochs", type=int, default=1, help="Number of epochs over the dataset.")
    ap.add_argument("--max-train-steps", type=int, default=0, help="Stop after N optimizer steps (0 = no cap).")
    ap.add_argument("--log-every", type=int, default=50, help="Log every N steps.")
    ap.add_argument("--save-every", type=int, default=500, help="Save checkpoint/report every N steps.")
    ap.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader workers (0 is safest on Windows).")
    ap.add_argument("--use-landmark-conf", action="store_true", help="Use baseline_index.csv landmark_conf as c (otherwise c=1).")
    args = ap.parse_args()

    # Default to FP16 for GPU to save memory (especially important for 4GB GPUs)
    # args.fp16 is None by default, meaning auto-detect based on device
    if args.fp16 is None:
        use_fp16 = args.device.startswith("cuda")  # Auto-enable for GPU
    else:
        use_fp16 = bool(args.fp16)
    
    cfg = TrainConfig(
        model=str(args.model),
        resolution=int(args.resolution),
        batch_size=int(args.batch_size),
        steps=int(args.steps),
        lr=float(args.lr),
        n_geo_tokens=int(args.n_geo_tokens),
        lambda_geo=float(args.lambda_geo),
        prompt=str(args.prompt),
        device=str(args.device),
        use_fp16=use_fp16,
        out_dir=str(args.out_dir),
        seed=int(args.seed),
        mode=str(args.mode),
        baseline_dir=str(args.baseline_dir),
        step1_out_dir=str(args.step1_out_dir),
        init_mapper=str(args.init_mapper),
        mapper_hidden_dim=int(args.mapper_hidden_dim),
        mapper_n_layers=int(args.mapper_n_layers),
        max_samples=int(args.max_samples),
        num_epochs=int(args.epochs),
        max_train_steps=int(args.max_train_steps),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        num_workers=int(args.num_workers),
        use_landmark_conf=bool(args.use_landmark_conf),
    )

    if cfg.mode == "ffhq":
        report = train_ffhq_dataset(cfg)
    else:
        image_path = Path(args.image).resolve()
        ref_image_path = Path(args.ref_image).resolve() if str(args.ref_image).strip() else None
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if ref_image_path is not None and not ref_image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
        report = train_one_or_few_steps(cfg, image_path=image_path, ref_image_path=ref_image_path)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if cfg.mode == "ffhq":
        print("[OK] Step2 FFHQ train completed.")
        print("  steps:", report["global_step"])
        print("  ckpt:", report["ckpt_path"])
    else:
        print("[OK] Step2 smoke train completed.")
        print("  loss:", report["losses"])
        print("  E_text:", report["E_text_shape"])
        print("  E_prime:", report["E_prime_shape"])
        print("  ckpt:", report["ckpt_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


