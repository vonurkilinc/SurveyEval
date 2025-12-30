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
    resolution: int = 1024
    batch_size: int = 1
    steps: int = 1
    lr: float = 1e-4
    n_geo_tokens: int = 4
    token_norm: str = "layernorm"
    lambda_geo: float = 1.0
    prompt: str = "a photo of a person"
    device: str = "cuda"
    use_fp16: bool = True
    out_dir: str = "geo_step2_out"
    seed: int = 123


def load_sdxl_pipeline(cfg: TrainConfig):
    torch = _require_torch()
    StableDiffusionXLPipeline, DDPMScheduler = _require_diffusers()

    dtype = torch.float16 if (cfg.use_fp16 and torch.cuda.is_available() and cfg.device.startswith("cuda")) else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model, torch_dtype=dtype, variant=None)
    pipe.to(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

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


def train_one_or_few_steps(cfg: TrainConfig, *, image_path: Path, ref_image_path: Optional[Path] = None) -> Dict[str, Any]:
    torch = _require_torch()

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

    opt = torch.optim.AdamW(mapper.net.parameters(), lr=float(cfg.lr))

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2: inject E' into SDXL UNet and train mapper only.")
    ap.add_argument("--model", type=str, default="hf-internal-testing/tiny-stable-diffusion-xl-pipe", help="Local SDXL path or HF id.")
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
    ap.add_argument("--fp16", action="store_true", help="Use fp16 when possible.")
    ap.add_argument("--out-dir", type=str, default="geo_step2_out", help="Output directory for checkpoints and cache.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = ap.parse_args()

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
        use_fp16=bool(args.fp16),
        out_dir=str(args.out_dir),
        seed=int(args.seed),
    )

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

    print("[OK] Step2 smoke train completed.")
    print("  loss:", report["losses"])
    print("  E_text:", report["E_text_shape"])
    print("  E_prime:", report["E_prime_shape"])
    print("  ckpt:", report["ckpt_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


