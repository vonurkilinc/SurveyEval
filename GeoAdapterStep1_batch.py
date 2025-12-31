#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoAdapterStep1_batch.py

Batch processing script that uses GeoAdapterStep1 to process all geometry data
and generate geometry tokens for training.

This script:
  1. Loads existing geometry descriptors (from features/geometry_descriptors.npz or FFHQ baseline)
  2. Creates/loads a geometry token mapper
  3. Generates tokens for all images
  4. Saves tokens and mapper checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import GeoAdapterStep1 as geo1


def load_caricature_geometry(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load geometry descriptors from features/geometry_descriptors.npz
    
    Note: step3.py saves dense landmarks (468*2=936 or similar), not 8-D CQS descriptors.
    We need to extract landmarks and compute CQS descriptors from them.
    """
    z = np.load(npz_path)
    keys = z["keys"]
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    G_dense = z["G"].astype(np.float32)
    
    # Check if this is dense landmarks (shape ends with ~936 or 956) or already 8-D
    if G_dense.shape[1] == 8:
        # Already CQS format
        return G_dense, keys
    elif G_dense.shape[1] >= 900:
        # Dense landmarks - need to extract from image_records.jsonl and recompute
        print(f"[WARN] NPZ contains dense landmarks ({G_dense.shape[1]}-D), not 8-D CQS descriptors.")
        print(f"[INFO] Will extract landmarks from images and compute CQS descriptors...")
        return None, keys  # Signal to recompute from images
    else:
        raise ValueError(f"Unexpected geometry descriptor shape: {G_dense.shape}")


def load_ffhq_geometry(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load geometry descriptors from FFHQ baseline CSV"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    feature_cols = [
        "face_width_over_height",
        "left_eye_width_over_interocular",
        "right_eye_width_over_interocular",
        "interocular_over_face_width",
        "mouth_width_over_face_width",
        "nose_wing_width_over_face_width",
        "nose_length_over_face_height",
        "jaw_opening_angle",
    ]
    
    G = df[feature_cols].to_numpy(dtype=np.float32)
    keys = df["image_id"].tolist()
    return G, keys


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch process geometry data with GeoAdapterStep1")
    ap.add_argument("--data-source", type=str, default="caricature", choices=["caricature", "ffhq"],
                    help="Data source: caricature (features/geometry_descriptors.npz) or ffhq (baseline CSV)")
    ap.add_argument("--caricature-npz", type=str, default="features/geometry_descriptors.npz",
                    help="Path to caricature geometry NPZ file")
    ap.add_argument("--ffhq-csv", type=str,
                    default="baselines/ffhq_mediapipe468_20251225T102206Z_cdc39a62e7c1/baseline_geometry.csv",
                    help="Path to FFHQ baseline geometry CSV")
    ap.add_argument("--sdxl-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="SDXL model path/ID to read cross_attention_dim from")
    ap.add_argument("--cross-attention-dim", type=int, default=0,
                    help="Override cross-attention dim (0 = read from SDXL model)")
    ap.add_argument("--n-geo-tokens", type=int, default=4, help="Number of geometry tokens")
    ap.add_argument("--hidden-dim", type=int, default=256, help="Mapper hidden dimension")
    ap.add_argument("--n-layers", type=int, default=3, help="Mapper layers")
    ap.add_argument("--lambda-geo", type=float, default=1.0, help="Geometry strength")
    ap.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu). 'auto' detects CUDA availability.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for token generation")
    ap.add_argument("--out-dir", type=str, default="geo_step1_batch_out", help="Output directory")
    ap.add_argument("--save-tokens", action="store_true", help="Save generated tokens to disk")
    ap.add_argument("--max-samples", type=int, default=0, help="Limit to N samples (0 = all)")
    
    args = ap.parse_args()
    
    torch, _ = geo1._require_torch()
    
    # Auto-detect device if requested
    device = str(args.device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[INFO] CUDA available, using GPU")
        else:
            device = "cpu"
            print(f"[INFO] CUDA not available, using CPU (PyTorch: {torch.__version__})")
    else:
        device = device.lower()
        if device == "cuda" and not torch.cuda.is_available():
            print(f"[WARN] CUDA requested but not available (PyTorch: {torch.__version__}). Falling back to CPU.")
            device = "cpu"
    
    # Load geometry data
    print(f"[INFO] Loading geometry data from: {args.data_source}")
    if args.data_source == "caricature":
        npz_path = Path(args.caricature_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"Caricature geometry NPZ not found: {npz_path}")
        result = load_caricature_geometry(npz_path)
        if result[0] is None:
            # Need to recompute from images
            print(f"[INFO] Recomputing CQS descriptors from images...")
            records_path = Path("features/image_records.jsonl")
            if not records_path.exists():
                raise FileNotFoundError(f"image_records.jsonl not found: {records_path}")
            
            keys_all = result[1]
            G_all = []
            valid_keys = []
            
            with open(records_path, "r", encoding="utf-8") as f:
                records = {json.loads(line)["key"]: json.loads(line) for line in f if line.strip()}
            
            for key in keys_all:
                if key not in records:
                    continue
                rec = records[key]
                img_path = Path(rec.get("resolved_path") or rec.get("image_abs", ""))
                if not img_path.exists():
                    continue
                try:
                    G, c = geo1.geometry_from_reference_image(img_path, allow_rotation=True)
                    G_all.append(G)
                    valid_keys.append(key)
                    if len(valid_keys) % 10 == 0:
                        print(f"  Processed {len(valid_keys)}/{len(keys_all)} images...")
                except Exception as e:
                    print(f"  [WARN] Failed for {key}: {e}")
                    continue
            
            G_all = np.stack(G_all, axis=0).astype(np.float32)
            keys_all = valid_keys
            print(f"[INFO] Computed {len(keys_all)} CQS geometry descriptors from images")
        else:
            G_all, keys_all = result
            print(f"[INFO] Loaded {len(keys_all)} caricature geometry descriptors")
    else:  # ffhq
        csv_path = Path(args.ffhq_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"FFHQ geometry CSV not found: {csv_path}")
        G_all, keys_all = load_ffhq_geometry(csv_path)
        print(f"[INFO] Loaded {len(keys_all)} FFHQ geometry descriptors")
    
    # Limit samples if requested
    if args.max_samples > 0 and args.max_samples < len(keys_all):
        G_all = G_all[:args.max_samples]
        keys_all = keys_all[:args.max_samples]
        print(f"[INFO] Limited to {len(keys_all)} samples")
    
    # Determine cross-attention dimension
    if args.cross_attention_dim > 0:
        C = args.cross_attention_dim
        print(f"[INFO] Using cross-attention dim: {C} (from --cross-attention-dim)")
    else:
        try:
            # Try to read from SDXL model
            model_path = Path(args.sdxl_model)
            if model_path.exists() and (model_path / "unet" / "config.json").exists():
                C = geo1.read_sdxl_cross_attention_dim(model_path)
                print(f"[INFO] Read cross-attention dim from local model: {C}")
            else:
                # Try loading from HuggingFace (will cache)
                from diffusers import StableDiffusionXLPipeline
                print(f"[INFO] Loading SDXL model to read cross_attention_dim...")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    args.sdxl_model, torch_dtype=torch.float32
                )
                C = int(getattr(pipe.unet.config, "cross_attention_dim"))
                print(f"[INFO] Read cross-attention dim from HF model: {C}")
                del pipe
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[WARN] Could not read cross_attention_dim: {e}")
            print(f"[INFO] Using default SDXL cross_attention_dim: 2048")
            C = 2048
    
    D = len(geo1.GEOMETRY_FEATURE_NAMES)
    assert D == 8, f"Expected 8-D geometry descriptor, got D={D}"
    
    # Create mapper
    print(f"[INFO] Creating geometry token mapper:")
    print(f"  geom_dim: {D}")
    print(f"  cross_attention_dim: {C}")
    print(f"  n_geo_tokens: {args.n_geo_tokens}")
    print(f"  hidden_dim: {args.hidden_dim}")
    print(f"  n_layers: {args.n_layers}")
    
    mapper = geo1.GeometryTokenMapper(
        geom_dim=D,
        cross_attention_dim=C,
        n_geo_tokens=args.n_geo_tokens,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        token_norm="layernorm",
    ).to(device)
    mapper.eval()
    
    # Generate tokens in batches
    print(f"[INFO] Generating geometry tokens for {len(keys_all)} images...")
    batch_size = args.batch_size
    all_tokens = []
    
    with torch.no_grad():
        for i in range(0, len(keys_all), batch_size):
            batch_end = min(i + batch_size, len(keys_all))
            G_batch = G_all[i:batch_end]
            
            G_tensor = torch.from_numpy(G_batch).to(device=device, dtype=torch.float32)
            T_batch = mapper.forward(G_tensor, lambda_geo=args.lambda_geo, conf=1.0)
            
            all_tokens.append(T_batch.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {batch_end}/{len(keys_all)} images...")
    
    tokens_array = np.concatenate(all_tokens, axis=0)  # (N, n_geo_tokens, C)
    print(f"[INFO] Generated tokens shape: {tokens_array.shape}")
    
    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mapper checkpoint
    ckpt_path = out_dir / "geo_mapper_step1.pt"
    geo1.save_geo_adapter(mapper, ckpt_path)
    print(f"[INFO] Saved mapper checkpoint: {ckpt_path}")
    
    # Save tokens if requested
    if args.save_tokens:
        tokens_path = out_dir / "geometry_tokens.npz"
        np.savez_compressed(
            tokens_path,
            keys=np.array(keys_all),
            tokens=tokens_array.astype(np.float32),
        )
        print(f"[INFO] Saved tokens: {tokens_path}")
    
    # Save report
    report = {
        "data_source": args.data_source,
        "n_samples": len(keys_all),
        "geom_dim": D,
        "cross_attention_dim": C,
        "n_geo_tokens": args.n_geo_tokens,
        "lambda_geo": args.lambda_geo,
        "tokens_shape": list(tokens_array.shape),
        "mapper_config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "token_norm": "layernorm",
        },
        "ckpt_path": str(ckpt_path),
    }
    
    report_path = out_dir / "step1_batch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Saved report: {report_path}")
    
    print(f"\n[OK] Batch processing complete!")
    print(f"  Processed: {len(keys_all)} images")
    print(f"  Tokens shape: {tokens_array.shape}")
    print(f"  Mapper checkpoint: {ckpt_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

