#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step3_sanity_check.py

Sanity check script for Step 3 (feature extraction) outputs.
Validates:
- File existence and integrity
- Data consistency (shapes, counts, key matching)
- Data quality (ranges, norms, NaN/Inf checks)
- Error analysis from image_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def check_file_exists(path: Path, name: str) -> tuple[bool, str]:
    """Check if file exists."""
    if not path.exists():
        return False, f"{name} not found: {path}"
    if path.stat().st_size == 0:
        return False, f"{name} is empty: {path}"
    return True, f"{name} exists ({path.stat().st_size:,} bytes)"


def check_npz_file(path: Path, name: str, required_keys: List[str]) -> tuple[bool, Dict[str, Any], str]:
    """Check NPZ file structure and return data."""
    exists, msg = check_file_exists(path, name)
    if not exists:
        return False, {}, msg

    try:
        data = np.load(path, allow_pickle=False)
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            return False, {}, f"{name} missing required keys: {missing_keys}"

        info = {}
        for key in required_keys:
            arr = data[key]
            info[key] = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "size": arr.size,
            }
            if arr.size > 0:
                # Only compute numeric stats for numeric arrays
                if np.issubdtype(arr.dtype, np.number):
                    info[key]["min"] = float(np.min(arr))
                    info[key]["max"] = float(np.max(arr))
                    info[key]["mean"] = float(np.mean(arr))
                    info[key]["has_nan"] = bool(np.isnan(arr).any())
                    info[key]["has_inf"] = bool(np.isinf(arr).any())
                else:
                    # For string arrays (like keys), just note the type
                    info[key]["is_string_array"] = True
                    info[key]["sample_values"] = arr[:5].tolist() if len(arr) > 0 else []

        return True, info, f"{name} OK"
    except Exception as e:
        return False, {}, f"{name} load error: {e}"


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze image records for errors and warnings."""
    stats = {
        "total": len(records),
        "ok": 0,
        "failed": 0,
        "warnings": 0,
        "error_types": defaultdict(int),
        "failed_images": [],
        "warned_images": [],
    }

    for rec in records:
        status = rec.get("status", {})
        if status.get("ok", False):
            stats["ok"] += 1
        else:
            stats["failed"] += 1
            errors = status.get("errors", [])
            for err in errors:
                stats["error_types"][err] += 1
            stats["failed_images"].append({
                "key": rec.get("key", "unknown"),
                "image_rel": rec.get("image_rel", "unknown"),
                "errors": errors,
            })

        warnings = status.get("warnings", [])
        if warnings:
            stats["warnings"] += len(warnings)
            stats["warned_images"].append({
                "key": rec.get("key", "unknown"),
                "image_rel": rec.get("image_rel", "unknown"),
                "warnings": warnings,
            })

    return stats


def check_key_consistency(
    geom_keys: np.ndarray,
    clip_keys: np.ndarray,
    id_keys: np.ndarray,
    record_keys: List[str],
) -> Dict[str, Any]:
    """Check key consistency across all files."""
    record_key_set = set(record_keys)
    geom_key_set = set(geom_keys)
    clip_key_set = set(clip_keys)
    id_key_set = set(id_keys)

    # Expected: all records should have CLIP (since it's always enabled)
    # Geometry and identity may have fewer due to failures

    issues = []
    consistency = {
        "record_keys": len(record_key_set),
        "geom_keys": len(geom_key_set),
        "clip_keys": len(clip_key_set),
        "id_keys": len(id_key_set),
        "issues": issues,
    }

    # CLIP should match all records (since it's always enabled)
    if clip_key_set != record_key_set:
        missing_clip = record_key_set - clip_key_set
        extra_clip = clip_key_set - record_key_set
        if missing_clip:
            issues.append(f"CLIP missing keys: {len(missing_clip)} (expected all records)")
        if extra_clip:
            issues.append(f"CLIP extra keys: {len(extra_clip)}")

    # Geometry should be subset of records (only successful landmarks)
    if not geom_key_set.issubset(record_key_set):
        extra_geom = geom_key_set - record_key_set
        issues.append(f"Geometry has keys not in records: {len(extra_geom)}")

    # Identity should be subset of records (only successful detections)
    if not id_key_set.issubset(record_key_set):
        extra_id = id_key_set - record_key_set
        issues.append(f"Identity has keys not in records: {len(extra_id)}")

    # Check for missing geometry/identity in successful records
    successful_records = record_key_set  # All records (CLIP should work for all)
    missing_geom = successful_records - geom_key_set
    missing_id = successful_records - id_key_set

    consistency["missing_geom_count"] = len(missing_geom)
    consistency["missing_id_count"] = len(missing_id)

    if missing_geom:
        consistency["missing_geom_samples"] = list(missing_geom)[:5]

    if missing_id:
        consistency["missing_id_samples"] = list(missing_id)[:5]

    return consistency


def check_data_quality(geom_data: Dict, clip_data: Dict, id_data: Dict) -> Dict[str, Any]:
    """Check data quality metrics."""
    quality = {
        "geometry": {},
        "clip": {},
        "identity": {},
    }

    # Geometry checks
    if "G" in geom_data:
        G = geom_data["G"]
        if G["size"] > 0:
            quality["geometry"] = {
                "shape": G["shape"],
                "expected_shape_2d": f"(N, 936)" if G["shape"][1] == 936 else f"Unexpected: {G['shape']}",
                "has_nan": G["has_nan"],
                "has_inf": G["has_inf"],
                "value_range": (G["min"], G["max"]),
                "mean": G["mean"],
            }

    # CLIP checks
    if "clip" in clip_data:
        clip = clip_data["clip"]
        if clip["size"] > 0:
            # Load actual data to check norms
            clip_npz = np.load("features/clip_embeddings.npz")
            clip_arr = clip_npz["clip"]
            norms = np.linalg.norm(clip_arr, axis=1)
            quality["clip"] = {
                "shape": clip["shape"],
                "expected_shape_2d": f"(N, 512)" if clip["shape"][1] == 512 else f"Unexpected: {clip['shape']}",
                "has_nan": clip["has_nan"],
                "has_inf": clip["has_inf"],
                "norm_range": (float(norms.min()), float(norms.max())),
                "norm_mean": float(norms.mean()),
                "should_be_normalized": "Yes (norms should be ~1.0)",
            }

    # Identity checks
    if "id" in id_data:
        id_arr_data = id_data["id"]
        if id_arr_data["size"] > 0:
            # Load actual data to check norms
            id_npz = np.load("features/identity_embeddings.npz")
            id_arr = id_npz["id"]
            norms = np.linalg.norm(id_arr, axis=1)
            quality["identity"] = {
                "shape": id_arr_data["shape"],
                "expected_shape_2d": f"(N, 512)" if id_arr_data["shape"][1] == 512 else f"Unexpected: {id_arr_data['shape']}",
                "has_nan": id_arr_data["has_nan"],
                "has_inf": id_arr_data["has_inf"],
                "norm_range": (float(norms.min()), float(norms.max())),
                "norm_mean": float(norms.mean()),
                "should_be_normalized": "Yes (norms should be ~1.0)",
            }

    return quality


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check for Step 3 feature extraction outputs")
    ap.add_argument("--features-dir", type=str, default="features", help="Features directory")
    ap.add_argument("--report-out", type=str, default="step3_sanity_report.json", help="Output report file")
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    report_path = Path(args.report_out)

    print("=" * 70)
    print("Step 3 Sanity Check")
    print("=" * 70)
    print(f"Features directory: {features_dir}")
    print()

    report = {
        "features_dir": str(features_dir),
        "checks": {},
        "summary": {},
        "errors": [],
        "warnings": [],
    }

    # 1. Check file existence
    print("1. Checking file existence...")
    files_to_check = {
        "image_records.jsonl": ["image_records.jsonl"],
        "geometry_descriptors.npz": ["geometry_descriptors.npz"],
        "clip_embeddings.npz": ["clip_embeddings.npz"],
        "identity_embeddings.npz": ["identity_embeddings.npz"],
        "stepD_report.json": ["stepD_report.json"],
    }

    file_checks = {}
    for name, paths in files_to_check.items():
        path = features_dir / paths[0]
        exists, msg = check_file_exists(path, name)
        file_checks[name] = {"exists": exists, "message": msg}
        print(f"   {name}: {msg}")
        if not exists:
            report["errors"].append(msg)

    report["checks"]["files"] = file_checks
    print()

    # 2. Load and check NPZ files
    print("2. Checking NPZ file structure...")
    geom_ok, geom_info, geom_msg = check_npz_file(
        features_dir / "geometry_descriptors.npz", "geometry_descriptors.npz", ["keys", "G"]
    )
    print(f"   Geometry: {geom_msg}")
    if not geom_ok:
        report["errors"].append(geom_msg)

    clip_ok, clip_info, clip_msg = check_npz_file(
        features_dir / "clip_embeddings.npz", "clip_embeddings.npz", ["keys", "clip"]
    )
    print(f"   CLIP: {clip_msg}")
    if not clip_ok:
        report["errors"].append(clip_msg)

    id_ok, id_info, id_msg = check_npz_file(
        features_dir / "identity_embeddings.npz", "identity_embeddings.npz", ["keys", "id"]
    )
    print(f"   Identity: {id_msg}")
    if not id_ok:
        report["errors"].append(id_msg)

    report["checks"]["npz_files"] = {
        "geometry": geom_info,
        "clip": clip_info,
        "identity": id_info,
    }
    print()

    # 3. Load image records and analyze
    print("3. Analyzing image records...")
    records_path = features_dir / "image_records.jsonl"
    if records_path.exists():
        records = load_jsonl(records_path)
        record_stats = analyze_records(records)
        print(f"   Total records: {record_stats['total']}")
        print(f"   Successful: {record_stats['ok']}")
        print(f"   Failed: {record_stats['failed']}")
        print(f"   Warnings: {record_stats['warnings']}")
        print(f"   Error types: {dict(record_stats['error_types'])}")

        if record_stats["failed_images"]:
            print(f"\n   Failed images (first 5):")
            for img in record_stats["failed_images"][:5]:
                print(f"     - {img['image_rel']}: {img['errors']}")

        report["checks"]["records"] = record_stats
    else:
        report["errors"].append("image_records.jsonl not found")
    print()

    # 4. Check key consistency
    print("4. Checking key consistency...")
    if geom_ok and clip_ok and id_ok and records_path.exists():
        geom_keys = np.load(features_dir / "geometry_descriptors.npz")["keys"]
        clip_keys = np.load(features_dir / "clip_embeddings.npz")["keys"]
        id_keys = np.load(features_dir / "identity_embeddings.npz")["keys"]
        record_keys = [r["key"] for r in records]

        consistency = check_key_consistency(geom_keys, clip_keys, id_keys, record_keys)
        print(f"   Record keys: {consistency['record_keys']}")
        print(f"   Geometry keys: {consistency['geom_keys']}")
        print(f"   CLIP keys: {consistency['clip_keys']}")
        print(f"   Identity keys: {consistency['id_keys']}")
        print(f"   Missing geometry: {consistency['missing_geom_count']}")
        print(f"   Missing identity: {consistency['missing_id_count']}")

        if consistency["issues"]:
            print(f"   Issues: {consistency['issues']}")
            report["warnings"].extend(consistency["issues"])

        report["checks"]["key_consistency"] = consistency
    print()

    # 5. Check data quality
    print("5. Checking data quality...")
    if geom_ok and clip_ok and id_ok:
        quality = check_data_quality(geom_info, clip_info, id_info)

        if quality.get("geometry"):
            gq = quality["geometry"]
            print(f"   Geometry:")
            print(f"     Shape: {gq['shape']}")
            print(f"     Expected: {gq['expected_shape_2d']}")
            print(f"     Has NaN: {gq['has_nan']}")
            print(f"     Has Inf: {gq['has_inf']}")
            if gq["has_nan"] or gq["has_inf"]:
                report["errors"].append(f"Geometry has NaN/Inf values")

        if quality.get("clip"):
            cq = quality["clip"]
            print(f"   CLIP:")
            print(f"     Shape: {cq['shape']}")
            print(f"     Expected: {cq['expected_shape_2d']}")
            print(f"     Norm range: {cq['norm_range'][0]:.4f} - {cq['norm_range'][1]:.4f}")
            print(f"     Norm mean: {cq['norm_mean']:.4f}")
            if cq["has_nan"] or cq["has_inf"]:
                report["errors"].append(f"CLIP has NaN/Inf values")
            if cq["norm_range"][0] < 0.9 or cq["norm_range"][1] > 1.1:
                report["warnings"].append(f"CLIP norms not normalized (range: {cq['norm_range']})")

        if quality.get("identity"):
            iq = quality["identity"]
            print(f"   Identity:")
            print(f"     Shape: {iq['shape']}")
            print(f"     Expected: {iq['expected_shape_2d']}")
            print(f"     Norm range: {iq['norm_range'][0]:.4f} - {iq['norm_range'][1]:.4f}")
            print(f"     Norm mean: {iq['norm_mean']:.4f}")
            if iq["has_nan"] or iq["has_inf"]:
                report["errors"].append(f"Identity has NaN/Inf values")
            if iq["norm_range"][0] < 0.9 or iq["norm_range"][1] > 1.1:
                report["warnings"].append(f"Identity norms not normalized (range: {iq['norm_range']})")

        report["checks"]["data_quality"] = quality
    print()

    # 6. Load and check stepD_report.json
    print("6. Checking stepD_report.json...")
    report_json_path = features_dir / "stepD_report.json"
    if report_json_path.exists():
        with report_json_path.open("r", encoding="utf-8") as f:
            stepd_report = json.load(f)
        print(f"   Unique images: {stepd_report.get('unique_images', 'N/A')}")
        print(f"   Geometry saved: {stepd_report.get('geometry_saved', 'N/A')}")
        print(f"   CLIP saved: {stepd_report.get('clip_saved', 'N/A')}")
        print(f"   Identity saved: {stepd_report.get('identity_saved', 'N/A')}")
        print(f"   Failures: {stepd_report.get('failures', {})}")
        report["checks"]["stepd_report"] = stepd_report
    else:
        report["errors"].append("stepD_report.json not found")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    num_errors = len(report["errors"])
    num_warnings = len(report["warnings"])

    report["summary"] = {
        "status": "PASS" if num_errors == 0 else "FAIL",
        "errors": num_errors,
        "warnings": num_warnings,
    }

    if num_errors == 0:
        print("✓ All checks passed!")
    else:
        print(f"✗ Found {num_errors} error(s):")
        for err in report["errors"]:
            print(f"  - {err}")

    if num_warnings > 0:
        print(f"\n⚠ Found {num_warnings} warning(s):")
        for warn in report["warnings"]:
            print(f"  - {warn}")

    # Save report
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed report saved to: {report_path}")

    sys.exit(0 if num_errors == 0 else 1)


if __name__ == "__main__":
    main()

