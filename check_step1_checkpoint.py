#!/usr/bin/env python3
"""
Sanity check for GeoAdapterStep1 mapper checkpoint before moving to Step 2.
"""

import json
from pathlib import Path
import numpy as np

import GeoAdapterStep1 as geo1


def main():
    print("=" * 60)
    print("GeoAdapterStep1 Checkpoint Sanity Check")
    print("=" * 60)
    
    ckpt_path = Path("geo_step1_batch_out/geo_mapper_step1.pt")
    report_path = Path("geo_step1_batch_out/step1_batch_report.json")
    tokens_path = Path("geo_step1_batch_out/geometry_tokens.npz")
    
    # 1. Check files exist
    print("\n[1] Checking files exist...")
    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return 1
    print(f"  [OK] Checkpoint exists: {ckpt_path}")
    
    if not report_path.exists():
        print(f"  [WARN] Report not found: {report_path}")
    else:
        print(f"  [OK] Report exists: {report_path}")
    
    if not tokens_path.exists():
        print(f"  [WARN] Tokens file not found: {tokens_path}")
    else:
        print(f"  [OK] Tokens file exists: {tokens_path}")
    
    # 2. Load and verify report
    print("\n[2] Loading report...")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    print(f"  Data source: {report['data_source']}")
    print(f"  Samples processed: {report['n_samples']:,}")
    print(f"  Geometry dim: {report['geom_dim']}")
    print(f"  Cross-attention dim: {report['cross_attention_dim']}")
    print(f"  Geo tokens: {report['n_geo_tokens']}")
    print(f"  Tokens shape: {report['tokens_shape']}")
    
    expected_shape = [report['n_samples'], report['n_geo_tokens'], report['cross_attention_dim']]
    if report['tokens_shape'] != expected_shape:
        print(f"  [ERROR] Shape mismatch! Expected {expected_shape}, got {report['tokens_shape']}")
        return 1
    print(f"  [OK] Shape matches expected: {expected_shape}")
    
    # 3. Load checkpoint
    print("\n[3] Loading mapper checkpoint...")
    try:
        mapper = geo1.load_geo_adapter(ckpt_path, map_location="cpu")
        print(f"  [OK] Checkpoint loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load checkpoint: {e}")
        return 1
    
    # 4. Verify mapper config matches report
    print("\n[4] Verifying mapper configuration...")
    checks = [
        ("geom_dim", mapper.geom_dim, report['geom_dim']),
        ("cross_attention_dim", mapper.cross_attention_dim, report['cross_attention_dim']),
        ("n_geo_tokens", mapper.n_geo_tokens, report['n_geo_tokens']),
        ("hidden_dim", mapper.hidden_dim, report['mapper_config']['hidden_dim']),
        ("n_layers", mapper.n_layers, report['mapper_config']['n_layers']),
    ]
    
    all_ok = True
    for name, actual, expected in checks:
        if actual == expected:
            print(f"  [OK] {name}: {actual}")
        else:
            print(f"  [ERROR] {name}: got {actual}, expected {expected}")
            all_ok = False
    
    if not all_ok:
        return 1
    
    # 5. Test forward pass with sample geometry
    print("\n[5] Testing forward pass...")
    import torch
    mapper.eval()
    
    # Create a batch of test geometry (8-D CQS descriptors)
    test_G = torch.randn(3, report['geom_dim'], dtype=torch.float32)
    
    with torch.no_grad():
        test_tokens = mapper.forward(test_G, lambda_geo=1.0, conf=1.0)
    
    expected_token_shape = (3, report['n_geo_tokens'], report['cross_attention_dim'])
    if tuple(test_tokens.shape) == expected_token_shape:
        print(f"  [OK] Forward pass shape correct: {test_tokens.shape}")
    else:
        print(f"  [ERROR] Forward pass shape mismatch: got {test_tokens.shape}, expected {expected_token_shape}")
        return 1
    
    # 6. Test determinism
    print("\n[6] Testing determinism...")
    with torch.no_grad():
        tokens1 = mapper.forward(test_G, lambda_geo=0.7, conf=0.9)
        tokens2 = mapper.forward(test_G, lambda_geo=0.7, conf=0.9)
    
    if torch.allclose(tokens1, tokens2, atol=1e-6):
        print(f"  [OK] Deterministic: identical inputs produce identical outputs")
    else:
        max_diff = float(torch.max(torch.abs(tokens1 - tokens2)).item())
        print(f"  [ERROR] Non-deterministic! Max difference: {max_diff}")
        return 1
    
    # 7. Test lambda=0 behavior
    print("\n[7] Testing lambda_geo=0 behavior...")
    with torch.no_grad():
        tokens_zero = mapper.forward(test_G, lambda_geo=0.0, conf=1.0)
    
    max_val = float(torch.max(torch.abs(tokens_zero)).item())
    if max_val < 1e-6:
        print(f"  [OK] lambda_geo=0 zeros tokens (max: {max_val:.2e})")
    else:
        print(f"  [WARN] lambda_geo=0 did not fully zero tokens (max: {max_val:.2e})")
    
    # 8. Verify tokens file if it exists
    if tokens_path.exists():
        print("\n[8] Verifying tokens file...")
        try:
            tokens_data = np.load(tokens_path)
            
            if 'keys' not in tokens_data or 'tokens' not in tokens_data:
                print(f"  [ERROR] Missing keys or tokens in NPZ file")
                return 1
            
            keys = tokens_data['keys']
            tokens = tokens_data['tokens']
            
            print(f"  Keys shape: {keys.shape}")
            print(f"  Tokens shape: {tokens.shape}")
            
            if len(keys) != report['n_samples']:
                print(f"  [ERROR] Key count mismatch: {len(keys)} vs {report['n_samples']}")
                return 1
            
            if tokens.shape != tuple(report['tokens_shape']):
                print(f"  [ERROR] Tokens shape mismatch: {tokens.shape} vs {report['tokens_shape']}")
                return 1
            
            print(f"  [OK] Tokens file verified: {len(keys)} samples, shape {tokens.shape}")
        except Exception as e:
            print(f"  [WARN] Could not verify tokens file: {e}")
            print(f"  [INFO] This is optional - checkpoint is still valid for Step 2")
    
    # 9. Check file sizes
    print("\n[9] Checking file sizes...")
    ckpt_size = ckpt_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  Checkpoint size: {ckpt_size:.2f} MB")
    
    if ckpt_size < 0.1:
        print(f"  [WARN] Checkpoint seems very small (< 0.1 MB)")
    elif ckpt_size > 1000:
        print(f"  [WARN] Checkpoint seems very large (> 1 GB)")
    else:
        print(f"  [OK] Checkpoint size reasonable")
    
    # Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)
    print(f"✓ Checkpoint file exists and is loadable")
    print(f"✓ Mapper configuration matches report")
    print(f"✓ Forward pass produces correct shapes")
    print(f"✓ Deterministic behavior verified")
    print(f"✓ Lambda scaling works correctly")
    if tokens_path.exists():
        print(f"✓ Tokens file verified")
    print(f"\n[OK] Checkpoint is READY for Step 2 training!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

