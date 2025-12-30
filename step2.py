#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_index_experiment_images.py

Builds a canonical evaluation index for CQS scoring.

Expected layout (example):
img/
  p000/
    reference.jpg
    methodA/
      0001.jpg
      0002.png
    methodB/
      sample_01.jpg
      nested/
        sample_02.jpg
  p001/
    reference.jpg
    methodA/...

Outputs:
- CSV (default): rows of (person_id, method, sample_id, reference_path, caricature_path, ...)
- Optional JSON with the same rows

Usage examples:
  python 20_index_experiment_images.py --root img --out index.csv
  python 20_index_experiment_images.py --root img --out index.csv --json index.json
  python 20_index_experiment_images.py --root img --out index.csv --recursive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(p: Path, exts: set[str]) -> bool:
    return p.is_file() and p.suffix.lower() in exts


def stable_id_from_path(p: Path, length: int = 12) -> str:
    # Stable ID based on relative path string (portable across machines if rel paths preserved)
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()
    return h[:length]


def find_reference(person_dir: Path, reference_name: str) -> Optional[Path]:
    ref = person_dir / reference_name
    if ref.exists() and ref.is_file():
        return ref
    # Fallback: search shallow (some users place reference in a subfolder)
    for cand in person_dir.glob(f"**/{reference_name}"):
        if cand.is_file():
            return cand
    return None


def list_method_dirs(person_dir: Path, reference_path: Path) -> List[Path]:
    """
    Method directories are immediate subfolders of person_dir, excluding the folder containing reference.jpg
    if it is not the person_dir itself.
    """
    method_dirs: List[Path] = []
    for child in sorted(person_dir.iterdir()):
        if not child.is_dir():
            continue
        # Exclude hidden/system-ish folders
        if child.name.startswith("."):
            continue
        method_dirs.append(child)

    # If reference is inside a subfolder, that subfolder might be mistaken as a method folder.
    # Exclude it explicitly.
    ref_parent = reference_path.parent.resolve()
    method_dirs = [d for d in method_dirs if d.resolve() != ref_parent]

    return method_dirs


def collect_images(method_dir: Path, exts: set[str], recursive: bool) -> List[Path]:
    if recursive:
        cands = method_dir.rglob("*")
    else:
        cands = method_dir.glob("*")

    images = [p for p in cands if is_image(p, exts)]
    images.sort(key=lambda p: str(p).lower())
    return images


def build_index(
    root: Path,
    reference_name: str,
    exts: set[str],
    recursive: bool,
    strict_reference: bool,
) -> Tuple[List[Dict], List[str]]:
    """
    Returns: (rows, warnings)
    """
    warnings: List[str] = []
    rows: List[Dict] = []

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Person dirs: p000, p001, ... but we also allow any folder under root
    person_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not person_dirs:
        warnings.append(f"No subdirectories found under root: {root}")

    for person_dir in person_dirs:
        person_id = person_dir.name

        ref_path = find_reference(person_dir, reference_name)
        if ref_path is None:
            msg = f"[WARN] Missing reference for {person_id}: expected '{reference_name}' under {person_dir}"
            if strict_reference:
                warnings.append(msg)
                continue
            else:
                warnings.append(msg + " (continuing; reference_path will be blank)")
        else:
            ref_path = ref_path.resolve()

        # method subfolders under person_dir
        method_dirs = list_method_dirs(person_dir, ref_path if ref_path else person_dir)
        if not method_dirs:
            warnings.append(f"[WARN] No method subfolders found for {person_id} under {person_dir}")
            continue

        for method_dir in method_dirs:
            method = method_dir.name
            images = collect_images(method_dir, exts, recursive)

            # Exclude any accidental reference copies
            if ref_path is not None:
                images = [p for p in images if p.resolve() != ref_path]

            if not images:
                warnings.append(f"[WARN] No images found in method folder: {method_dir}")
                continue

            for img_path in images:
                img_path = img_path.resolve()

                # Relative paths for portability
                rel_ref = str(ref_path.relative_to(root)) if ref_path is not None else ""
                rel_img = str(img_path.relative_to(root))

                # Stable sample ID
                sample_id = stable_id_from_path(Path(rel_img))

                rows.append(
                    {
                        "person_id": person_id,
                        "method": method,
                        "sample_id": sample_id,
                        "reference_path": str(ref_path) if ref_path is not None else "",
                        "caricature_path": str(img_path),
                        "reference_rel": rel_ref,
                        "caricature_rel": rel_img,
                    }
                )

    # Deterministic order
    rows.sort(key=lambda r: (r["person_id"], r["method"], r["caricature_rel"].lower()))
    return rows, warnings


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "person_id",
        "method",
        "sample_id",
        "reference_path",
        "caricature_path",
        "reference_rel",
        "caricature_rel",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(rows: List[Dict], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_warnings(warnings: List[str], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for w in warnings:
            f.write(w.rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="img", help="Root folder containing person folders (e.g., img/)")
    ap.add_argument("--reference-name", type=str, default="reference.jpg", help="Reference image filename")
    ap.add_argument("--out", type=str, default="evaluation_index.csv", help="Output CSV path")
    ap.add_argument("--json", type=str, default="", help="Optional output JSON path")
    ap.add_argument("--warnings", type=str, default="evaluation_index_warnings.txt", help="Warnings output text file")
    ap.add_argument("--recursive", action="store_true", help="Recursively search images inside method folders")
    ap.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(IMAGE_EXTS_DEFAULT)),
        help="Comma-separated allowed image extensions (e.g., .jpg,.png,.webp)",
    )
    ap.add_argument(
        "--strict-reference",
        action="store_true",
        help="If set, skip a person_id entirely when reference is missing",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out).resolve()
    out_json = Path(args.json).resolve() if args.json else None
    out_warnings = Path(args.warnings).resolve()

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    # Normalize: ensure leading dot
    exts = {e if e.startswith(".") else "." + e for e in exts}

    rows, warnings = build_index(
        root=root,
        reference_name=args.reference_name,
        exts=exts,
        recursive=args.recursive,
        strict_reference=args.strict_reference,
    )

    write_csv(rows, out_csv)
    if out_json is not None:
        write_json(rows, out_json)
    write_warnings(warnings, out_warnings)

    print(f"[OK] Root: {root}")
    print(f"[OK] Rows: {len(rows)}")
    print(f"[OK] CSV:  {out_csv}")
    if out_json is not None:
        print(f"[OK] JSON: {out_json}")
    print(f"[OK] Warnings: {out_warnings}")
    if warnings:
        print(f"[WARN] {len(warnings)} warning(s). See: {out_warnings}")


if __name__ == "__main__":
    main()
