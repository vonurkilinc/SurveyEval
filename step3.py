#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30_stepD_extract_features.py

Combined Step D script:
- Reads eval index (CSV or JSON) produced by Step C
- Resolves image paths (absolute if present, otherwise uses --img-root + relative)
- Extracts (per unique image):
  1) Landmarks + confidence (default: MediaPipe FaceMesh)
  2) Geometry descriptor G (default: dense landmarks after similarity alignment + normalization)
  3) Identity embedding (optional; default backend: insightface if installed)
  4) CLIP image embedding (ENABLED; default: open_clip ViT-B-32)

Outputs (in --out-dir):
- image_records.jsonl                 # one line per unique image (paths + metadata + status)
- landmarks_cache/<key>.npz           # per-image landmarks + conf + alignment params (debuggable)
- geometry_descriptors.npz            # stacked G vectors + keys
- clip_embeddings.npz                 # stacked CLIP embeddings + keys
- identity_embeddings.npz             # stacked identity embeddings + keys (if enabled and available)
- stepD_report.json                   # summary counts, failures

Typical usage:
  python 30_stepD_extract_features.py \
    --index manifests/eval_index.csv \
    --img-root img \
    --out-dir features \
    --clip-model ViT-B-32 --clip-pretrained openai \
    --landmarks-backend mediapipe \
    --identity-backend insightface \
    --geometry-mode dense_landmarks

Notes:
- This script is intentionally "single-file" and robust to partial failures:
  if a model is missing/unavailable, it will skip that feature and record warnings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def stable_key_from_rel(rel_path: str, length: int = 12) -> str:
    import hashlib

    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    return h[:length]


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if index_path.suffix.lower() == ".json":
        with index_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("JSON index must be a list of records.")
        return rows

    # CSV fallback
    import csv

    rows: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def resolve_path(abs_path: str, rel_path: str, img_root: Path) -> Optional[Path]:
    """
    Prefer absolute path if it exists. Otherwise, join img_root with rel_path.
    rel_path in Windows CSVs may contain backslashes; normalize.
    """
    if abs_path:
        p = Path(abs_path)
        if p.exists():
            return p.resolve()

    if rel_path:
        rel_norm = rel_path.replace("\\", "/")
        p2 = (img_root / rel_norm)
        if p2.exists():
            return p2.resolve()

    return None


def pil_load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# -----------------------------
# Landmarks backends
# -----------------------------

@dataclass
class LandmarkResult:
    ok: bool
    conf: float
    landmarks_xy: Optional[np.ndarray]  # shape: (K, 2), pixel coords
    img_w: int
    img_h: int
    err: str = ""


class MediaPipeFaceMesh:
    """
    Uses mediapipe FaceMesh with refine_landmarks=True.
    Produces 468 landmarks in normalized coords -> converted to pixel coords.

    Confidence: uses FaceDetection score if available; otherwise 1.0 when landmarks present.
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mediapipe not installed. Install with: pip install mediapipe"
            ) from e

        self.mp = mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Optional face detection for confidence
        self.face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def __call__(self, img_rgb: np.ndarray) -> LandmarkResult:
        h, w = img_rgb.shape[:2]
        conf = 0.0
        try:
            # Confidence via detector
            det = self.face_det.process(img_rgb)
            if det and det.detections:
                conf = float(det.detections[0].score[0])
            else:
                conf = 0.0

            res = self.face_mesh.process(img_rgb)
            if not res.multi_face_landmarks:
                return LandmarkResult(ok=False, conf=conf, landmarks_xy=None, img_w=w, img_h=h, err="No face landmarks")

            lm = res.multi_face_landmarks[0].landmark  # 468
            pts = np.zeros((len(lm), 2), dtype=np.float32)
            for i, p in enumerate(lm):
                pts[i, 0] = p.x * w
                pts[i, 1] = p.y * h

            # If no detector confidence but landmarks exist, set a minimal positive conf
            if conf <= 0.0:
                conf = 0.5

            return LandmarkResult(ok=True, conf=conf, landmarks_xy=pts, img_w=w, img_h=h, err="")
        except Exception as e:
            return LandmarkResult(ok=False, conf=conf, landmarks_xy=None, img_w=w, img_h=h, err=str(e))


# -----------------------------
# Geometry descriptor
# -----------------------------

def similarity_align_dense_landmarks(pts_xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple similarity alignment using two points: approximate left/right eye centers.

    For MediaPipe 468: we avoid brittle named indices by using a robust heuristic:
    - compute centroid of landmarks
    - approximate eye centers using top-half points split by x around centroid:
      take the mean of the 3% smallest y among left-half / right-half points.

    This is not "semantic perfect" but works as a stable canonicalization for dense landmark vectors.

    Returns:
      aligned_pts: (K,2) in a canonical frame
      params: dict with scale and rotation used (debuggable)
    """
    pts = pts_xy.astype(np.float32)
    c = pts.mean(axis=0)  # centroid
    left = pts[pts[:, 0] < c[0]]
    right = pts[pts[:, 0] >= c[0]]

    def eye_center(side: np.ndarray) -> np.ndarray:
        if side.shape[0] < 10:
            return c.copy()
        # take top 3% by y (smallest y), then average those points
        k = max(5, int(0.03 * side.shape[0]))
        top = side[np.argsort(side[:, 1])[:k]]
        return top.mean(axis=0)

    eL = eye_center(left)
    eR = eye_center(right)

    v = eR - eL
    dist = float(np.linalg.norm(v) + 1e-8)
    angle = float(np.arctan2(v[1], v[0]))  # radians
    # rotate so eyes are horizontal
    ca, sa = float(np.cos(-angle)), float(np.sin(-angle))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    # translate so centroid at origin, rotate, then scale so eye distance = 1.0
    pts0 = pts - c
    pts1 = pts0 @ R.T
    # recompute aligned eye points
    eL1 = (eL - c) @ R.T
    eR1 = (eR - c) @ R.T
    eye_dist1 = float(np.linalg.norm(eR1 - eL1) + 1e-8)
    scale = 1.0 / eye_dist1
    pts2 = pts1 * scale

    params = {"eye_dist_px": dist, "eye_dist_aligned": eye_dist1, "scale": scale, "rot_rad": -angle}
    return pts2, params


def geometry_dense_landmarks(pts_xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Geometry descriptor G for dense landmarks:
      - Align with similarity transform
      - Flatten normalized (x,y) coordinates -> vector length 2K

    Returns:
      G: shape (2K,)
      align_params: debug info
    """
    aligned, params = similarity_align_dense_landmarks(pts_xy)
    G = aligned.reshape(-1).astype(np.float32)
    return G, params


# -----------------------------
# CLIP embeddings (OpenCLIP)
# -----------------------------

class OpenCLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu") -> None:
        try:
            import torch
            import open_clip  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "open_clip_torch not installed. Install with: pip install open_clip_torch torch torchvision"
            ) from e

        self.torch = torch
        self.open_clip = open_clip
        # Check if CUDA is available, fall back to CPU if not
        if device.lower() == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print(f"[WARN] CUDA requested but not available, using CPU instead", file=sys.stderr)
        else:
            self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        model.to(self.device)
        self.model = model
        self.preprocess = preprocess

    def encode_image(self, img: Image.Image) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# -----------------------------
# Identity embeddings (InsightFace)
# -----------------------------

class InsightFaceEncoder:
    """
    Uses insightface FaceAnalysis to detect and produce ArcFace embedding.
    This is optional; if not installed or fails, identity is skipped.
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            import insightface  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "insightface not installed. Install with: pip install insightface onnxruntime-gpu (or onnxruntime)"
            ) from e

        self.insightface = insightface
        providers = None
        if device.lower() == "cuda":
            # Prefer GPU if available
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.app = insightface.app.FaceAnalysis(providers=providers)
        # ctx_id: 0 for GPU, -1 for CPU
        ctx_id = 0 if device.lower() == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id)

    def encode(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        # Use the most confident / largest face (first is usually best)
        emb = faces[0].embedding
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)


# -----------------------------
# Main pipeline
# -----------------------------

def gather_unique_images(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a unique-image list with metadata, so we compute features once per image.

    Each returned record includes:
      key, abs_path, rel_path, person_id, method, role (reference/caricature)
    """
    uniq: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        person_id = str(r.get("person_id", ""))
        method = str(r.get("method", ""))

        ref_abs = str(r.get("reference_path", ""))
        ref_rel = str(r.get("reference_rel", ""))

        car_abs = str(r.get("caricature_path", ""))
        car_rel = str(r.get("caricature_rel", ""))

        # reference image record
        if ref_rel:
            k = stable_key_from_rel(ref_rel.replace("\\", "/"))
            if k not in uniq:
                uniq[k] = {
                    "key": k,
                    "role": "reference",
                    "person_id": person_id,
                    "method": "",  # reference is method-agnostic
                    "reference_abs": ref_abs,
                    "reference_rel": ref_rel,
                    "image_abs": ref_abs,
                    "image_rel": ref_rel,
                }

        # caricature image record
        if car_rel:
            k2 = stable_key_from_rel(car_rel.replace("\\", "/"))
            if k2 not in uniq:
                uniq[k2] = {
                    "key": k2,
                    "role": "caricature",
                    "person_id": person_id,
                    "method": method,
                    "reference_abs": ref_abs,
                    "reference_rel": ref_rel,
                    "image_abs": car_abs,
                    "image_rel": car_rel,
                }

    # deterministic order
    recs = list(uniq.values())
    recs.sort(key=lambda x: (x["person_id"], x["role"], x["method"], x["image_rel"].lower()))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True, help="Step C output index CSV/JSON (eval_index.csv/json)")
    ap.add_argument("--img-root", type=str, default="img", help="Root folder for relative paths (e.g., img/)")
    ap.add_argument("--out-dir", type=str, default="features", help="Output directory for Step D artifacts")

    # Landmarks
    ap.add_argument("--landmarks-backend", type=str, default="mediapipe", choices=["mediapipe"],
                    help="Landmarks backend (currently supported: mediapipe)")
    ap.add_argument("--min-landmark-conf", type=float, default=0.25,
                    help="Minimum landmark confidence to accept (lower allows more stylized faces)")

    # Geometry
    ap.add_argument("--geometry-mode", type=str, default="dense_landmarks", choices=["dense_landmarks"],
                    help="Geometry descriptor mode (currently supported: dense_landmarks)")

    # CLIP (enabled)
    ap.add_argument("--clip-enable", action="store_true", help="Enable CLIP embeddings (recommended)")
    ap.add_argument("--clip-model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    ap.add_argument("--clip-pretrained", type=str, default="openai", help="OpenCLIP pretrained tag")
    ap.add_argument("--clip-batch", type=int, default=1, help="Batch size (kept simple for robustness)")

    # Identity (optional)
    ap.add_argument("--identity-enable", action="store_true", help="Enable identity embeddings")
    ap.add_argument("--identity-backend", type=str, default="insightface", choices=["insightface"],
                    help="Identity backend (currently supported: insightface)")

    # Device
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")

    # Caching / overwrite
    ap.add_argument("--overwrite", action="store_true", default=True, help="Recompute and overwrite existing outputs (default: True)")

    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    img_root = Path(args.img_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    # Output subdirs
    landmarks_dir = out_dir / "landmarks_cache"
    ensure_dir(landmarks_dir)

    # Load Step C index
    rows = load_index(index_path)
    unique_images = gather_unique_images(rows)

    # Initialize landmark extractor
    lm_extractor = None
    if args.landmarks_backend == "mediapipe":
        lm_extractor = MediaPipeFaceMesh(static_image_mode=True, max_num_faces=1)

    # Initialize CLIP encoder (enabled by request; also controllable)
    clip_encoder = None
    if args.clip_enable or True:
        # default behavior: enabled unless user explicitly doesn't want it
        try:
            clip_encoder = OpenCLIPEncoder(
                model_name=args.clip_model,
                pretrained=args.clip_pretrained,
                device=args.device,
            )
        except Exception as e:
            print(f"[WARN] CLIP init failed; CLIP embeddings will be skipped. Reason: {e}", file=sys.stderr)
            clip_encoder = None

    # Initialize identity encoder (optional)
    id_encoder = None
    if args.identity_enable:
        try:
            if args.identity_backend == "insightface":
                id_encoder = InsightFaceEncoder(device=args.device)
        except Exception as e:
            print(f"[WARN] Identity init failed; identity embeddings will be skipped. Reason: {e}", file=sys.stderr)
            id_encoder = None

    # Storage
    records_path = out_dir / "image_records.jsonl"
    geom_out = out_dir / "geometry_descriptors.npz"
    clip_out = out_dir / "clip_embeddings.npz"
    id_out = out_dir / "identity_embeddings.npz"
    report_out = out_dir / "stepD_report.json"

    # Handle overwrite mode - always overwrite by default to avoid duplicate records
    if args.overwrite:
        if records_path.exists():
            records_path.unlink()
        if geom_out.exists():
            geom_out.unlink()
        if clip_out.exists():
            clip_out.unlink()
        if id_out.exists():
            id_out.unlink()
        if report_out.exists():
            report_out.unlink()
        # Also clear landmarks cache if overwriting
        import shutil
        if landmarks_dir.exists():
            shutil.rmtree(landmarks_dir)
            ensure_dir(landmarks_dir)
    elif records_path.exists():
        print(f"[WARN] {records_path} exists and --overwrite not set. Appending may create duplicate records.", file=sys.stderr)

    # Per-image results
    keys: List[str] = []
    geom_list: List[np.ndarray] = []
    clip_list: List[np.ndarray] = []
    id_list: List[np.ndarray] = []
    id_keys: List[str] = []
    clip_keys: List[str] = []
    geom_keys: List[str] = []

    failures = {
        "missing_file": 0,
        "landmarks_fail": 0,
        "low_landmark_conf": 0,
        "geom_fail": 0,
        "clip_fail": 0,
        "id_fail": 0,
    }

    start = time.time()

    # Use write mode if overwriting, append mode otherwise
    mode = "w" if args.overwrite else "a"
    with records_path.open(mode, encoding="utf-8") as rec_f:
        for rec in tqdm(unique_images, desc="Step D: extracting features"):
            key = rec["key"]
            abs_path = str(rec.get("image_abs", ""))
            rel_path = str(rec.get("image_rel", ""))

            img_path = resolve_path(abs_path, rel_path, img_root)
            status = {"ok": True, "warnings": [], "errors": []}

            if img_path is None or not img_path.exists():
                failures["missing_file"] += 1
                status["ok"] = False
                status["errors"].append("Image file not found via absolute or relative path.")
                out_line = {**rec, "resolved_path": "", "status": status, "timestamp": now_iso()}
                rec_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                continue

            # --- Load image
            try:
                img_pil = pil_load_rgb(img_path)
                img_np = np.array(img_pil)  # RGB, HWC
            except Exception as e:
                failures["missing_file"] += 1
                status["ok"] = False
                status["errors"].append(f"Image load failed: {e}")
                out_line = {**rec, "resolved_path": str(img_path), "status": status, "timestamp": now_iso()}
                rec_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                continue

            # --- Landmarks
            lm_ok = False
            lm_conf = 0.0
            pts_xy = None
            align_params: Dict[str, float] = {}

            try:
                lm_res = lm_extractor(img_np)  # type: ignore
                lm_ok = lm_res.ok
                lm_conf = float(lm_res.conf)
                pts_xy = lm_res.landmarks_xy
                if not lm_ok or pts_xy is None:
                    failures["landmarks_fail"] += 1
                    status["ok"] = False
                    status["errors"].append(f"Landmarks failed: {lm_res.err}")
                elif lm_conf < args.min_landmark_conf:
                    failures["low_landmark_conf"] += 1
                    status["ok"] = False
                    status["errors"].append(f"Landmark confidence too low: {lm_conf:.3f} < {args.min_landmark_conf:.3f}")
            except Exception as e:
                failures["landmarks_fail"] += 1
                status["ok"] = False
                status["errors"].append(f"Landmarks exception: {e}")

            # --- Geometry descriptor
            if status["ok"]:
                try:
                    if args.geometry_mode == "dense_landmarks":
                        G, align_params = geometry_dense_landmarks(pts_xy)  # type: ignore
                        geom_list.append(G)
                        geom_keys.append(key)
                    else:
                        raise ValueError(f"Unsupported geometry-mode: {args.geometry_mode}")
                except Exception as e:
                    failures["geom_fail"] += 1
                    status["ok"] = False
                    status["errors"].append(f"Geometry failed: {e}")

            # Cache landmarks for debugging / downstream reuse
            try:
                lm_cache_path = landmarks_dir / f"{key}.npz"
                if args.overwrite or (not lm_cache_path.exists()):
                    np.savez_compressed(
                        lm_cache_path,
                        key=key,
                        image_rel=rel_path.replace("\\", "/"),
                        conf=np.float32(lm_conf),
                        landmarks_xy=(pts_xy.astype(np.float32) if pts_xy is not None else np.zeros((0, 2), np.float32)),
                        align_params=np.array([align_params.get("scale", 0.0), align_params.get("rot_rad", 0.0)], np.float32),
                    )
            except Exception as e:
                status["warnings"].append(f"Landmarks cache write failed: {e}")

            # --- CLIP embedding (enabled)
            if clip_encoder is not None:
                try:
                    clip_vec = clip_encoder.encode_image(img_pil)
                    clip_list.append(clip_vec)
                    clip_keys.append(key)
                except Exception as e:
                    failures["clip_fail"] += 1
                    status["warnings"].append(f"CLIP failed: {e}")
            else:
                status["warnings"].append("CLIP encoder unavailable; skipped.")

            # --- Identity embedding (optional)
            if id_encoder is not None:
                try:
                    # InsightFace expects BGR uint8
                    img_bgr = img_np[:, :, ::-1].copy()
                    emb = id_encoder.encode(img_bgr)
                    if emb is None:
                        failures["id_fail"] += 1
                        status["warnings"].append("Identity: no face detected by insightface.")
                    else:
                        id_list.append(emb)
                        id_keys.append(key)
                except Exception as e:
                    failures["id_fail"] += 1
                    status["warnings"].append(f"Identity failed: {e}")

            # Write record
            out_line = {**rec, "resolved_path": str(img_path), "status": status, "timestamp": now_iso()}
            rec_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    # Save stacked outputs (overwrite mode - always write fresh)
    if geom_list:
        Gmat = np.stack(geom_list, axis=0).astype(np.float32)
        np.savez_compressed(geom_out, keys=np.array(geom_keys), G=Gmat)
    if clip_list:
        Cmat = np.stack(clip_list, axis=0).astype(np.float32)
        np.savez_compressed(clip_out, keys=np.array(clip_keys), clip=Cmat)
    if id_list:
        Imat = np.stack(id_list, axis=0).astype(np.float32)
        np.savez_compressed(id_out, keys=np.array(id_keys), id=Imat)

    elapsed = time.time() - start
    report = {
        "timestamp": now_iso(),
        "index_path": str(index_path),
        "img_root": str(img_root),
        "out_dir": str(out_dir),
        "unique_images": len(unique_images),
        "geometry_saved": int(len(geom_list)),
        "clip_saved": int(len(clip_list)),
        "identity_saved": int(len(id_list)),
        "failures": failures,
        "elapsed_sec": elapsed,
        "notes": [
            "Geometry mode: dense_landmarks (flattened aligned landmarks).",
            "CLIP uses OpenCLIP; identity uses InsightFace if enabled.",
            "Review image_records.jsonl for per-image warnings/errors.",
        ],
    }
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[OK] Step D complete")
    print(f" - Records:   {records_path}")
    print(f" - Geometry:  {geom_out if geom_out.exists() else '(none)'}")
    print(f" - CLIP:      {clip_out if clip_out.exists() else '(none)'}")
    print(f" - Identity:  {id_out if id_out.exists() else '(none)'}")
    print(f" - Report:    {report_out}")
    print(f" - Elapsed:   {elapsed:.1f}s")


if __name__ == "__main__":
    main()
