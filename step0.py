#!/usr/bin/env python3
"""
step0_setup.py
--------------
Combines:
  - 00_download_models.py
  - 01_verify_runtime.py

What it does:
  1) Creates a local project structure (models/, cache/, reports/, etc.)
  2) Downloads or verifies presence of required model assets:
       - Landmark model (dlib 68)  [default enabled]
       - Identity model (InsightFace ArcFace ONNX) [default enabled]
       - Optional CLIP (OpenCLIP via pip; weights handled by package)
  3) Runs a sanity verification on a single image:
       - Face detection + landmarks (dlib)
       - Identity embedding extraction (InsightFace ONNX)
       - Optional CLIP embedding (open_clip)
  4) Writes a manifest.json with versions, paths, and checksums
  5) Writes a verification report (reports/step0_report.json)

Notes:
  - This script is designed to be practical and robust, not minimal.
  - It prefers *pinned, reproducible* downloads when possible.
  - You can disable components via CLI flags.

Install dependencies (example):
  pip install -U numpy opencv-python tqdm requests onnxruntime insightface pillow
  pip install -U dlib   # may require build tools; on Windows consider prebuilt wheels
  pip install -U open_clip_torch torch torchvision  # optional if --enable-clip

Usage:
  python step0_setup.py --help
  python step0_setup.py --test-image path/to/face.jpg
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional imports will be performed lazily inside verification functions.


# -----------------------------
# Utilities
# -----------------------------

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> Tuple[int, str]:
    """Run a command and capture stdout+stderr."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, (e.output or "").strip()
    except Exception as e:
        return 999, str(e)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def http_download(url: str, dst: Path, expected_sha256: Optional[str] = None) -> None:
    """
    Streaming download with optional SHA256 verification.

    Requires: requests
    """
    import requests
    from tqdm import tqdm

    safe_mkdir(dst.parent)

    # If file exists and matches hash, skip.
    if dst.exists() and expected_sha256:
        got = sha256_file(dst)
        if got.lower() == expected_sha256.lower():
            return

    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.replace(dst)

    if expected_sha256:
        got = sha256_file(dst)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {dst.name}\nExpected: {expected_sha256}\nGot:      {got}"
            )


def pip_show(pkg: str) -> Optional[str]:
    code, out = run_cmd([sys.executable, "-m", "pip", "show", pkg])
    return out if code == 0 and out else None


def get_python_env_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


# -----------------------------
# Model download configuration
# -----------------------------

@dataclass(frozen=True)
class RemoteAsset:
    name: str
    url: str
    filename: str
    sha256: Optional[str] = None  # strongly recommended for reproducibility


def default_assets() -> Dict[str, RemoteAsset]:
    """
    Provide sensible defaults.

    Landmark model:
      - dlib shape_predictor_68_face_landmarks.dat.bz2 is commonly used,
        but the canonical URL can vary. We include a robust fallback path.
      - You may prefer to place the .dat manually if your environment blocks downloads.

    Identity model:
      - InsightFace provides model packs; easiest is to let insightface download into ~/.insightface.
        For reproducibility, we also support placing a specific ONNX in models/identity/.
    """
    # NOTE: The dlib model's canonical hosting is often via dlib.net; some environments block it.
    # You can override via CLI: --dlib-url.
    dlib_url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    return {
        "dlib_68_bz2": RemoteAsset(
            name="dlib_68_landmarks_bz2",
            url=dlib_url,
            filename="shape_predictor_68_face_landmarks.dat.bz2",
            sha256=None,  # If you have the sha, put it here.
        ),
    }


# -----------------------------
# Setup: folders, downloads, manifest
# -----------------------------

def create_project_structure(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "models": root / "models",
        "models_landmarks": root / "models" / "landmarks",
        "models_identity": root / "models" / "identity",
        "cache": root / "cache",
        "reports": root / "reports",
        "manifests": root / "manifests",
        "tmp": root / "cache" / "tmp",
    }
    for p in paths.values():
        safe_mkdir(p)
    return paths


def ensure_dlib_landmark_model(
    paths: Dict[str, Path],
    *,
    dlib_url: str,
    force: bool = False,
) -> Path:
    """
    Downloads and extracts dlib's shape predictor 68 landmarks model.

    Produces:
      models/landmarks/shape_predictor_68_face_landmarks.dat
    """
    import bz2

    bz2_dst = paths["models_landmarks"] / "shape_predictor_68_face_landmarks.dat.bz2"
    dat_dst = paths["models_landmarks"] / "shape_predictor_68_face_landmarks.dat"

    if dat_dst.exists() and not force:
        return dat_dst

    # Download .bz2
    http_download(dlib_url, bz2_dst, expected_sha256=None)

    # Extract
    with bz2.open(bz2_dst, "rb") as f_in, dat_dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return dat_dst


def build_manifest(
    paths: Dict[str, Path],
    *,
    dlib_dat: Optional[Path],
    identity_mode: str,
    enable_clip: bool,
) -> Dict[str, Any]:
    """
    Create a reproducibility manifest: environment + model files + versions.
    """
    manifest: Dict[str, Any] = {
        "created_utc": now_utc_iso(),
        "project_root": str(paths["root"].resolve()),
        "environment": get_python_env_info(),
        "packages": {},
        "models": {},
        "settings": {
            "identity_mode": identity_mode,
            "enable_clip": enable_clip,
        },
    }

    # Record key package versions (if present)
    for pkg in [
        "numpy",
        "opencv-python",
        "onnxruntime",
        "insightface",
        "dlib",
        "open_clip_torch",
        "torch",
        "torchvision",
        "Pillow",
        "requests",
    ]:
        info = pip_show(pkg)
        if info:
            # Extract Version line if possible
            version = None
            for line in info.splitlines():
                if line.lower().startswith("version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            manifest["packages"][pkg] = {"version": version, "pip_show": info}

    # Record model files
    if dlib_dat and dlib_dat.exists():
        manifest["models"]["dlib_shape_predictor_68"] = {
            "path": str(dlib_dat.resolve()),
            "sha256": sha256_file(dlib_dat),
            "bytes": dlib_dat.stat().st_size,
        }

    # Identity model recording:
    # - If using "insightface_pack", it downloads to ~/.insightface; we record that fact.
    # - If using "onnx_path", we record the onnx file path (provided by user elsewhere).
    manifest["models"]["identity"] = {"mode": identity_mode}

    # CLIP is package-managed; record package versions already.
    return manifest


# -----------------------------
# Verification (runtime sanity)
# -----------------------------

def _load_image_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path)

    # Handle uncommon modes robustly
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # Keep grayscale as 8-bit L; dlib accepts it
        pass
    else:
        # RGB already
        pass

    arr = np.asarray(img)

    # Force uint8 (handles 16-bit images or odd conversions)
    if arr.dtype != np.uint8:
        # If it's 16-bit, scale down; otherwise clip
        if arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Ensure shape is either HxW (grayscale) or HxWx3 (RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def verify_landmarks_dlib(test_image: Path, dlib_dat: Path) -> Dict[str, Any]:
    """
    Returns:
      - face_detected: bool
      - num_faces
      - landmarks_shape: (K,2) if face found else None
      - confidence proxy: num_faces>0 (dlib doesn't provide a calibrated confidence)
    """
    import dlib

    img = _load_image_rgb(test_image)
    
    # Initialize dlib face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(dlib_dat))
    
    # Convert to grayscale if needed (dlib works better with grayscale)
    # Use numpy to convert RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
    if img.ndim == 3:
        gray = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114).astype(np.uint8)
    else:
        gray = img
    
    # Detect faces
    faces = detector(gray)
    
    out: Dict[str, Any] = {
        "component": "landmarks_dlib_68",
        "test_image": str(test_image.resolve()),
        "face_detected": len(faces) > 0,
        "num_faces": int(len(faces)),
    }
    
    if not faces:
        return out
    
    # Use largest face
    faces = sorted(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()), reverse=True)
    face = faces[0]
    
    # Get landmarks
    landmarks = predictor(gray, face)
    landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    out["landmarks_shape"] = list(landmarks_np.shape)
    out["landmarks_example_first5"] = landmarks_np[:5].tolist()
    
    return out


def verify_identity_insightface(test_image: Path) -> Dict[str, Any]:
    """
    Uses InsightFace FaceAnalysis to get an embedding.

    This will download models into ~/.insightface automatically if missing.
    Returns:
      - face_detected
      - embedding_dim
      - embedding_norm
    """
    from insightface.app import FaceAnalysis

    img = _load_image_rgb(test_image)

    # Use a conservative provider selection: CUDA if available, else CPU.
    # InsightFace uses onnxruntime providers under the hood.
    app = FaceAnalysis(name="buffalo_l")  # common pack
    app.prepare(ctx_id=0)  # 0 = GPU if available; if not, it will fall back in many setups

    faces = app.get(img)
    out: Dict[str, Any] = {
        "component": "identity_insightface_arcface",
        "test_image": str(test_image.resolve()),
        "face_detected": len(faces) > 0,
        "num_faces": int(len(faces)),
    }
    if not faces:
        return out

    # Use largest face
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].embedding.astype(np.float32)
    out["embedding_dim"] = int(emb.shape[0])
    out["embedding_norm"] = float(np.linalg.norm(emb))
    out["embedding_example_first5"] = emb[:5].round(5).tolist()
    return out


def verify_clip_openclip(test_image: Path) -> Dict[str, Any]:
    """
    Optional CLIP verification using open_clip.
    """
    import torch
    import open_clip
    from PIL import Image

    model_name = "ViT-B-32"
    pretrained = "openai"

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    img = Image.open(test_image).convert("RGB")
    image_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(image_t).float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

    feat_np = feat.cpu().numpy().squeeze(0)
    return {
        "component": "clip_openclip",
        "test_image": str(test_image.resolve()),
        "model": {"name": model_name, "pretrained": pretrained},
        "device": device,
        "embedding_dim": int(feat_np.shape[0]),
        "embedding_example_first5": feat_np[:5].round(6).tolist(),
    }


def verify_onnxruntime_providers() -> Dict[str, Any]:
    with contextlib.suppress(Exception):
        import onnxruntime as ort
        return {"onnxruntime_providers": ort.get_available_providers()}
    return {"onnxruntime_providers": None}


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step0 setup: download/verify models and runtime for CQS.")
    p.add_argument("--root", type=str, default=".", help="Project root folder (default: current directory).")

    # Toggle components
    p.add_argument("--disable-dlib-landmarks", action="store_true", help="Skip dlib landmark model download/verification.")
    p.add_argument("--disable-identity", action="store_true", help="Skip identity model verification (InsightFace).")
    p.add_argument("--enable-clip", action="store_true", help="Enable CLIP verification via open_clip (optional).")

    # Dlib download control
    p.add_argument("--dlib-url", type=str, default=default_assets()["dlib_68_bz2"].url,
                   help="URL for dlib 68 landmarks model .bz2")
    p.add_argument("--force-redownload", action="store_true", help="Force re-download/re-extract where applicable.")

    # Test image
    p.add_argument("--test-image", type=str, required=True,
                   help="Path to a test image containing a clear face (required).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    test_image = Path(args.test_image).expanduser().resolve()

    if not test_image.exists():
        raise FileNotFoundError(f"Test image not found: {test_image}")

    paths = create_project_structure(root)

    report: Dict[str, Any] = {
        "created_utc": now_utc_iso(),
        "root": str(root),
        "test_image": str(test_image),
        "system": get_python_env_info(),
        "providers": verify_onnxruntime_providers(),
        "downloads": {},
        "verifications": [],
        "status": "ok",
        "warnings": [],
    }

    # 1) Download / verify dlib landmark model
    dlib_dat: Optional[Path] = None
    if not args.disable_dlib_landmarks:
        try:
            dlib_dat = ensure_dlib_landmark_model(
                paths,
                dlib_url=args.dlib_url,
                force=args.force_redownload,
            )
            report["downloads"]["dlib_shape_predictor_68"] = {
                "path": str(dlib_dat),
                "sha256": sha256_file(dlib_dat),
                "bytes": dlib_dat.stat().st_size,
            }
        except Exception as e:
            report["status"] = "failed"
            report["warnings"].append(f"Failed to download/extract dlib landmark model: {e}")

    # 2) Build manifest
    identity_mode = "insightface_pack" if not args.disable_identity else "disabled"
    manifest = build_manifest(
        paths,
        dlib_dat=dlib_dat,
        identity_mode=identity_mode,
        enable_clip=bool(args.enable_clip),
    )
    manifest_path = paths["manifests"] / "manifest_step0.json"
    write_json(manifest_path, manifest)
    report["manifest_path"] = str(manifest_path)

    # 3) Verification runs
    # 3.1 Landmarks
    if not args.disable_dlib_landmarks and dlib_dat and dlib_dat.exists():
        try:
            v = verify_landmarks_dlib(test_image, dlib_dat) or {}
            report["verifications"].append(v)

            if not isinstance(v, dict):
                raise RuntimeError(f"verify_landmarks_dlib must return dict, got {type(v)}")

            if not v.get("face_detected", False):
                report["warnings"].append("Dlib landmarks: no face detected on test image.")
                report["warnings"].append("Dlib landmarks: no face detected on test image.")
        except Exception as e:
            report["status"] = "failed"
            report["warnings"].append(f"Dlib landmarks verification failed: {e}")

    # 3.2 Identity
    if not args.disable_identity:
        try:
            v = verify_identity_insightface(test_image)
            report["verifications"].append(v)
            if not v.get("face_detected", False):
                report["warnings"].append("Identity (InsightFace): no face detected on test image.")
        except Exception as e:
            report["status"] = "failed"
            report["warnings"].append(f"Identity verification failed: {e}")

    # 3.3 Optional CLIP
    if args.enable_clip:
        try:
            v = verify_clip_openclip(test_image)
            report["verifications"].append(v)
        except Exception as e:
            report["warnings"].append(f"CLIP verification failed (optional): {e}")

    # 4) Write report
    report_path = paths["reports"] / "step0_report.json"
    write_json(report_path, report)

    # 5) Console summary (concise)
    print(f"[step0] Root: {root}")
    print(f"[step0] Manifest: {manifest_path}")
    print(f"[step0] Report:   {report_path}")
    print(f"[step0] Status:   {report['status']}")
    if report["warnings"]:
        print("[step0] Warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")

    # Exit code
    if report["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
