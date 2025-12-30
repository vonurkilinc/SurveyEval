#!/usr/bin/env python3
"""
build_cqs_baseline_dataset.py

Unified baseline builder for Caricature Quality Score (CQS).

This script:
1) Scans a neutral face dataset folder for images
2) Runs face + landmark detection (default: MediaPipe FaceMesh)
3) Filters bad samples (no face, too many faces, too small face, low confidence)
4) Aligns landmarks (scale/rotate/translate) to a canonical frame
5) Builds geometry descriptor vectors G (ratios + angles)
6) Saves:
   - baseline_index.csv
   - baseline_rejects.csv
   - baseline_geometry.parquet (or baseline_geometry.csv fallback)
   - geometry_stats.json
   - geometry_bounds.json
   - symmetry_pairs.json
   - baseline_report.json
   - run_config.json

Dependencies (minimum):
  - numpy
  - pandas
  - opencv-python
  - mediapipe  (default landmark backend)

Optional:
  - pyarrow  (for parquet output)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- Optional imports ---
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python. Install with: pip install opencv-python") from e

# MediaPipe is the default backend. If not installed, we fail with a helpful message.
try:
    import mediapipe as mp  # type: ignore
    # Try to import new Tasks API (MediaPipe 0.10+)
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        MP_TASKS_AVAILABLE = True
    except ImportError:
        MP_TASKS_AVAILABLE = False
except Exception as e:
    mp = None
    MP_TASKS_AVAILABLE = False


# -----------------------------
# Configuration and templates
# -----------------------------

SUPPORTED_EXTS_DEFAULT = ("jpg", "jpeg", "png", "webp", "bmp")


@dataclasses.dataclass
class SymmetryTemplate:
    landmark_topology: str
    pairs: List[Tuple[int, int]]
    midline: List[int]  # indices used to estimate midline direction; for saving only


def default_symmetry_template_mediapipe468() -> SymmetryTemplate:
    """
    A small, safe symmetry-pair set for MediaPipe FaceMesh (468 landmarks).

    Note: FaceMesh has many possible symmetry pairs; we include a conservative subset
    commonly used in face analysis. You can replace/extend via --symmetry-schema.
    """
    pairs = [
        # Eyes (outer/inner corners)
        (33, 263),
        (133, 362),
        # Mouth corners
        (61, 291),
        # Cheeks (approx)
        (234, 454),
        # Eyebrow-ish anchors (approx)
        (70, 300),
        (105, 334),
        # Nose wings-ish (approx)
        (94, 331),
    ]
    # Midline: nose tip + chin (common indices)
    midline = [1, 152]
    return SymmetryTemplate(landmark_topology="mediapipe468", pairs=pairs, midline=midline)


@dataclasses.dataclass
class GeometryFeature:
    name: str
    kind: str  # "ratio" or "angle"
    # For ratio: a,b define numerator distance; denom defines denominator ("interocular", "face_height", "face_width")
    a: int
    b: int
    denom: str = "interocular"
    # For angle: uses points (a,b,c) with angle at b. We'll store a,b,c using a,b plus c in extra dict.
    c: Optional[int] = None


def default_geometry_schema_mediapipe468() -> List[GeometryFeature]:
    """
    Default geometry descriptor schema for MediaPipe FaceMesh (468).
    Chosen to be:
      - stable and widely referenced indices
      - mostly pose/scale invariant after alignment
      - interpretable for caricature exaggeration

    Denominators:
      - interocular: dist(left_eye_center, right_eye_center) after alignment (will be 1.0 by construction)
      - face_height: dist(forehead(10), chin(152)) after alignment
      - face_width: dist(left_cheek(234), right_cheek(454)) after alignment
    """
    feats: List[GeometryFeature] = [
        # Face proportions
        GeometryFeature("face_width_over_height", "ratio", 234, 454, denom="face_height"),
        # Eyes
        GeometryFeature("left_eye_width_over_interocular", "ratio", 33, 133, denom="interocular"),
        GeometryFeature("right_eye_width_over_interocular", "ratio", 263, 362, denom="interocular"),
        GeometryFeature("interocular_over_face_width", "ratio", 33, 263, denom="face_width"),
        # Mouth
        GeometryFeature("mouth_width_over_face_width", "ratio", 61, 291, denom="face_width"),
        # Nose (approx)
        GeometryFeature("nose_wing_width_over_face_width", "ratio", 94, 331, denom="face_width"),
        GeometryFeature("nose_length_over_face_height", "ratio", 168, 1, denom="face_height"),  # bridge->tip
        # Jaw-ish angle (approx): angle at chin between cheeks
        GeometryFeature("jaw_opening_angle", "angle", 234, 152, c=454),
    ]
    return feats


# -----------------------------
# Utility functions
# -----------------------------

def now_utc_compact() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def sha1_short(s: str, n: int = 12) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:n]


def list_images(data_root: Path, exts: Sequence[str]) -> List[Path]:
    exts_set = {("." + e.lower().lstrip(".")) for e in exts}
    out: List[Path] = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_set:
            out.append(p)
    out.sort()
    return out


def read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imread returned None (unsupported/corrupt image?)")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def l2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


def dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> float:
    """
    Returns angle at b in radians, between vectors (a-b) and (c-b).
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2 + eps))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.acos(cosv))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def try_save_parquet(df: pd.DataFrame, path: Path) -> Tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, "parquet"
    except Exception as e:
        return False, f"parquet_failed: {e}"


# -----------------------------
# Landmark backend: MediaPipe
# -----------------------------

@dataclasses.dataclass
class LandmarksResult:
    ok: bool
    num_faces: int
    confidence: float
    landmarks_xy: Optional[np.ndarray]  # shape (K,2) in pixel coords
    img_w: int
    img_h: int
    reject_reason: Optional[str] = None


class MediaPipeFaceMeshBackend:
    """
    MediaPipe FaceMesh backend.

    Notes:
    - It does not expose a single face score; we compute a proxy confidence from landmark visibility/presence.
    - For datasets, this is typically sufficient for filtering.
    - Supports both old API (mp.solutions.face_mesh) and new Tasks API (MediaPipe 0.10+).
    """
    def __init__(self, max_num_faces: int = 2):
        if mp is None:
            raise RuntimeError(
                "MediaPipe not installed. Install with: pip install mediapipe\n"
                "If you insist on dlib/insightface, adapt this backend accordingly."
            )
        self.max_num_faces = max_num_faces
        self._use_tasks_api = MP_TASKS_AVAILABLE
        
        if self._use_tasks_api:
            # Use new Tasks API (MediaPipe 0.10+)
            self._init_tasks_api()
        else:
            # Use old solutions API (MediaPipe < 0.10)
            try:
                self._mp_face_mesh = mp.solutions.face_mesh
                self._mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except AttributeError:
                raise RuntimeError(
                    "MediaPipe installed but incompatible version. "
                    "MediaPipe 0.10+ requires Tasks API. Please install mediapipe==0.10.31 or update the code."
                )

    def _init_tasks_api(self) -> None:
        """Initialize MediaPipe 0.10+ Tasks API with automatic model download."""
        import urllib.request
        from pathlib import Path
        
        # Model URL for face landmarker
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_dir = Path.home() / ".mediapipe_models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        
        # Download model if not exists
        if not model_path.exists():
            print(f"[INFO] Downloading MediaPipe face landmarker model to {model_path}...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"[INFO] Model downloaded successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe face landmarker model: {e}\n"
                    f"Please download manually from: {model_url}\n"
                    f"and place it at: {model_path}"
                ) from e
        
        # Create FaceLandmarker
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_num_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        try:
            if self._use_tasks_api:
                self._landmarker.close()
            else:
                self._mesh.close()
        except Exception:
            pass

    def extract(self, img_rgb: np.ndarray) -> LandmarksResult:
        h, w = img_rgb.shape[:2]
        
        if self._use_tasks_api:
            # New Tasks API (MediaPipe 0.10+)
            from mediapipe import Image as MPImage
            
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = self._landmarker.detect(mp_image)
            
            if result.face_landmarks is None or len(result.face_landmarks) == 0:
                return LandmarksResult(
                    ok=False, num_faces=0, confidence=0.0, landmarks_xy=None, img_w=w, img_h=h,
                    reject_reason="no_face_detected"
                )
            
            num_faces = len(result.face_landmarks)
            # Use first face
            landmarks = result.face_landmarks[0]
            K = len(landmarks)
            xy = np.zeros((K, 2), dtype=np.float32)
            
            # Extract landmarks (Tasks API uses normalized coordinates)
            for i, landmark in enumerate(landmarks):
                xy[i, 0] = landmark.x * w
                xy[i, 1] = landmark.y * h
            
            # Confidence: use face detection score if available, otherwise 1.0
            if result.face_blendshapes is not None and len(result.face_blendshapes) > 0:
                # Try to get a confidence from blendshapes or use default
                conf = 1.0
            else:
                conf = 1.0
            
        else:
            # Old solutions API (MediaPipe < 0.10)
            res = self._mesh.process(img_rgb)
            faces = res.multi_face_landmarks or []
            num_faces = len(faces)
            if num_faces == 0:
                return LandmarksResult(
                    ok=False, num_faces=0, confidence=0.0, landmarks_xy=None, img_w=w, img_h=h,
                    reject_reason="no_face_detected"
                )
            # pick first (or largest would require bbox; FaceMesh order is usually stable)
            face = faces[0]
            K = len(face.landmark)
            xy = np.zeros((K, 2), dtype=np.float32)

            # confidence proxy: mean visibility/presence when available
            vis_list = []
            for i, lm in enumerate(face.landmark):
                xy[i, 0] = lm.x * w
                xy[i, 1] = lm.y * h
                # MediaPipe landmarks may include visibility/presence
                v = getattr(lm, "visibility", None)
                p = getattr(lm, "presence", None)
                # Some builds set these to 0; we treat missing as 1.0 (only face found)
                if v is not None:
                    vis_list.append(float(v))
                elif p is not None:
                    vis_list.append(float(p))

            if len(vis_list) == 0:
                conf = 1.0
            else:
                conf = float(np.clip(np.mean(vis_list), 0.0, 1.0))

        return LandmarksResult(
            ok=True, num_faces=num_faces, confidence=conf, landmarks_xy=xy, img_w=w, img_h=h,
            reject_reason=None
        )


# -----------------------------
# Alignment + geometry extraction
# -----------------------------

@dataclasses.dataclass
class AlignResult:
    ok: bool
    aligned_xy: Optional[np.ndarray]  # (K,2) float32 in canonical coords
    interocular: float
    face_height: float
    face_width: float
    reject_reason: Optional[str] = None


def compute_eye_centers_mediapipe(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eye centers based on stable corner indices.
    Left eye corners: 33 (outer), 133 (inner)
    Right eye corners: 263 (outer), 362 (inner)
    """
    left = 0.5 * (xy[33] + xy[133])
    right = 0.5 * (xy[263] + xy[362])
    return left, right


def align_landmarks_similarity(xy: np.ndarray, eps: float = 1e-9) -> AlignResult:
    """
    Canonical alignment:
    - translate so eye midpoint is origin
    - rotate so eye line is horizontal
    - scale so interocular distance = 1

    Returns aligned_xy (K,2), and also face_height/face_width computed in aligned coordinates.
    """
    left_eye, right_eye = compute_eye_centers_mediapipe(xy)
    eye_mid = 0.5 * (left_eye + right_eye)
    v = right_eye - left_eye
    inter = float(np.linalg.norm(v))
    if inter < 10.0:  # in pixels; too small / invalid
        return AlignResult(False, None, interocular=inter, face_height=0.0, face_width=0.0,
                          reject_reason="interocular_too_small")

    # translate
    xy0 = xy.astype(np.float32) - eye_mid.astype(np.float32)

    # rotate: angle to x-axis
    theta = math.atan2(float(v[1]), float(v[0]))
    c = math.cos(-theta)
    s = math.sin(-theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    xy1 = (xy0 @ R.T).astype(np.float32)

    # scale to interocular = 1
    xy2 = xy1 / (inter + eps)

    # compute face measures in aligned space
    # forehead(10)-chin(152), cheeks(234)-(454)
    face_h = dist(xy2[10], xy2[152])
    face_w = dist(xy2[234], xy2[454])
    if not np.isfinite(face_h) or face_h < 0.2:
        return AlignResult(False, None, interocular=1.0, face_height=face_h, face_width=face_w,
                          reject_reason="face_height_invalid")

    return AlignResult(True, xy2, interocular=1.0, face_height=face_h, face_width=face_w)


def compute_face_bbox_size(xy_px: np.ndarray) -> Tuple[float, float]:
    x_min = float(np.min(xy_px[:, 0]))
    x_max = float(np.max(xy_px[:, 0]))
    y_min = float(np.min(xy_px[:, 1]))
    y_max = float(np.max(xy_px[:, 1]))
    return (x_max - x_min), (y_max - y_min)


def geometry_vector(
    aligned_xy: np.ndarray,
    face_height: float,
    face_width: float,
    schema: List[GeometryFeature],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Build named geometry features dict.
    """
    out: Dict[str, float] = {}
    for feat in schema:
        if feat.kind == "ratio":
            num = dist(aligned_xy[feat.a], aligned_xy[feat.b])
            if feat.denom == "interocular":
                den = 1.0
            elif feat.denom == "face_height":
                den = face_height
            elif feat.denom == "face_width":
                den = face_width
            else:
                raise ValueError(f"Unknown denom '{feat.denom}' for feature '{feat.name}'")
            out[feat.name] = float(num / (den + eps))
        elif feat.kind == "angle":
            if feat.c is None:
                raise ValueError(f"Angle feature '{feat.name}' missing point c")
            ang = angle_abc(aligned_xy[feat.a], aligned_xy[feat.b], aligned_xy[feat.c])
            out[feat.name] = float(ang)
        else:
            raise ValueError(f"Unknown feature kind '{feat.kind}' for '{feat.name}'")
    return out


# -----------------------------
# Baseline stats/bounds
# -----------------------------

def compute_stats(df_geom: pd.DataFrame, feature_cols: List[str], mode: str) -> Dict[str, Any]:
    X = df_geom[feature_cols].to_numpy(dtype=np.float64)
    if mode == "meanstd":
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        return {"mode": "meanstd", "mean": mu.tolist(), "std": sd.tolist()}
    elif mode == "robust":
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med[None, :]), axis=0)
        # robust std estimate: 1.4826 * MAD
        rsd = (1.4826 * mad).astype(np.float64)
        return {"mode": "robust", "median": med.tolist(), "mad": mad.tolist(), "robust_std": rsd.tolist()}
    else:
        raise ValueError("stats-mode must be one of: meanstd, robust")


def compute_bounds(
    df_geom: pd.DataFrame,
    feature_cols: List[str],
    bounds_mode: str,
    lower_p: float,
    upper_p: float,
    k: float,
    stats_obj: Dict[str, Any],
) -> Dict[str, Any]:
    X = df_geom[feature_cols].to_numpy(dtype=np.float64)
    if bounds_mode == "percentiles":
        lo = np.nanpercentile(X, lower_p, axis=0)
        hi = np.nanpercentile(X, upper_p, axis=0)
        return {
            "mode": "percentiles",
            "lower_p": lower_p,
            "upper_p": upper_p,
            "lower": lo.tolist(),
            "upper": hi.tolist(),
        }
    elif bounds_mode == "ksigma":
        if stats_obj.get("mode") == "meanstd":
            mu = np.asarray(stats_obj["mean"], dtype=np.float64)
            sd = np.asarray(stats_obj["std"], dtype=np.float64)
        else:
            # robust uses median and robust_std
            mu = np.asarray(stats_obj["median"], dtype=np.float64)
            sd = np.asarray(stats_obj["robust_std"], dtype=np.float64)
        lo = (mu - k * sd).tolist()
        hi = (mu + k * sd).tolist()
        return {"mode": "ksigma", "k": k, "lower": lo, "upper": hi}
    else:
        raise ValueError("bounds-mode must be one of: percentiles, ksigma")


# -----------------------------
# Main pipeline
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CQS baseline dataset artifacts (single unified script).")

    # dataset scan
    p.add_argument("--data-root", type=str, required=True, help="Path to baseline face dataset root.")
    p.add_argument("--baseline-name", type=str, required=True, help="Short name for this baseline (e.g., ffhq).")
    p.add_argument("--extensions", type=str, default=",".join(SUPPORTED_EXTS_DEFAULT),
                   help="Comma-separated image extensions to include.")
    p.add_argument("--max-images", type=int, default=0, help="If >0, limit to first N images (debug).")

    # backend
    p.add_argument("--landmarks", type=str, default="mediapipe468",
                   choices=["mediapipe468"],  # extend if you add dlib/insightface
                   help="Landmark topology/backend.")
    p.add_argument("--max-faces", type=int, default=1, help="Reject images with > max-faces detected.")
    p.add_argument("--min-landmark-conf", type=float, default=0.0,
                   help="Minimum landmark confidence proxy. For MediaPipe, this is visibility/presence average; 0 disables.")
    p.add_argument("--min-face-size", type=int, default=128,
                   help="Minimum face bbox side length in pixels (approx).")

    # geometry
    p.add_argument("--geometry-schema", type=str, default="",
                   help="Optional path to geometry_schema.json. If omitted, uses built-in default for the topology.")
    p.add_argument("--align-mode", type=str, default="similarity", choices=["similarity"], help="Landmark alignment mode.")

    # stats/bounds
    p.add_argument("--stats-mode", type=str, default="meanstd", choices=["meanstd", "robust"],
                   help="How to compute baseline geometry statistics.")
    p.add_argument("--bounds-mode", type=str, default="percentiles", choices=["percentiles", "ksigma"],
                   help="How to compute plausible feature bounds.")
    p.add_argument("--lower-p", type=float, default=1.0, help="Lower percentile (for percentiles mode).")
    p.add_argument("--upper-p", type=float, default=99.0, help="Upper percentile (for percentiles mode).")
    p.add_argument("--k", type=float, default=4.0, help="k for ksigma bounds mode.")

    # symmetry
    p.add_argument("--symmetry-schema", type=str, default="auto",
                   help="auto or path to symmetry_pairs.json")

    # output
    p.add_argument("--out-root", type=str, default="baselines", help="Output root folder.")
    p.add_argument("--seed", type=int, default=123, help="Random seed (for reproducibility where relevant).")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu"], help="Device flag (placeholder).")
    p.add_argument("--save-landmarks-cache", action="store_true",
                   help="If set, saves aligned landmarks per image for debugging (can be large).")

    return p.parse_args()


def load_geometry_schema(path: Path, topology: str) -> List[GeometryFeature]:
    """
    geometry_schema.json expected structure:
    {
      "landmark_topology": "mediapipe468",
      "features": [
        {"name":"...", "kind":"ratio", "a":33, "b":133, "denom":"interocular"},
        {"name":"...", "kind":"angle", "a":234, "b":152, "c":454}
      ]
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"geometry schema file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    topo = obj.get("landmark_topology", "")
    if topo and topo != topology:
        raise ValueError(f"Geometry schema topology '{topo}' != requested '{topology}'")
    feats: List[GeometryFeature] = []
    for f in obj.get("features", []):
        feats.append(
            GeometryFeature(
                name=str(f["name"]),
                kind=str(f["kind"]),
                a=int(f["a"]),
                b=int(f["b"]),
                denom=str(f.get("denom", "interocular")),
                c=int(f["c"]) if "c" in f and f["c"] is not None else None,
            )
        )
    if len(feats) == 0:
        raise ValueError("geometry schema contains no features")
    return feats


def load_symmetry_schema(symmetry_schema_arg: str, topology: str) -> SymmetryTemplate:
    if symmetry_schema_arg == "auto":
        if topology == "mediapipe468":
            return default_symmetry_template_mediapipe468()
        raise ValueError(f"No auto symmetry template for topology: {topology}")
    p = Path(symmetry_schema_arg)
    if not p.exists():
        raise FileNotFoundError(f"symmetry schema not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    topo = obj.get("landmark_topology", "")
    if topo and topo != topology:
        raise ValueError(f"Symmetry schema topology '{topo}' != requested '{topology}'")
    pairs = [(int(a), int(b)) for a, b in obj["pairs"]]
    midline = [int(x) for x in obj.get("midline", [])]
    return SymmetryTemplate(landmark_topology=topology, pairs=pairs, midline=midline)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        print(f"[ERROR] data-root does not exist: {data_root}", file=sys.stderr)
        return 2

    exts = [e.strip() for e in args.extensions.split(",") if e.strip()]
    out_root = Path(args.out_root).expanduser().resolve()

    run_id = f"{args.baseline_name}_{args.landmarks}_{now_utc_compact()}_{sha1_short(str(data_root))}"
    out_dir = out_root / run_id
    ensure_dir(out_dir)

    # Load schema/templates
    if args.geometry_schema:
        geom_schema = load_geometry_schema(Path(args.geometry_schema), args.landmarks)
    else:
        if args.landmarks == "mediapipe468":
            geom_schema = default_geometry_schema_mediapipe468()
        else:
            raise ValueError(f"No default geometry schema for: {args.landmarks}")

    sym_template = load_symmetry_schema(args.symmetry_schema, args.landmarks)

    # Save run config
    run_config = {
        "run_id": run_id,
        "timestamp_utc": now_utc_compact(),
        "data_root": str(data_root),
        "baseline_name": args.baseline_name,
        "landmarks": args.landmarks,
        "extensions": exts,
        "max_images": args.max_images,
        "filters": {
            "max_faces": args.max_faces,
            "min_landmark_conf": args.min_landmark_conf,
            "min_face_size": args.min_face_size,
        },
        "geometry": {
            "align_mode": args.align_mode,
            "num_features": len(geom_schema),
            "feature_names": [f.name for f in geom_schema],
        },
        "stats": {
            "stats_mode": args.stats_mode,
            "bounds_mode": args.bounds_mode,
            "lower_p": args.lower_p,
            "upper_p": args.upper_p,
            "k": args.k,
        },
        "symmetry": dataclasses.asdict(sym_template),
        "save_landmarks_cache": bool(args.save_landmarks_cache),
        "python": sys.version,
        "platform": {
            "cwd": os.getcwd(),
        },
    }
    safe_write_json(out_dir / "run_config.json", run_config)

    # Scan dataset
    print(f"[INFO] Scanning images under: {data_root}")
    paths = list_images(data_root, exts)
    if args.max_images and args.max_images > 0:
        paths = paths[: args.max_images]
    if len(paths) == 0:
        print("[ERROR] No images found. Check --data-root and --extensions.", file=sys.stderr)
        return 2
    print(f"[INFO] Found {len(paths)} candidate images")

    # Initialize backend
    if args.landmarks == "mediapipe468":
        backend = MediaPipeFaceMeshBackend(max_num_faces=max(2, args.max_faces + 1))
    else:
        raise ValueError(f"Unsupported landmarks backend: {args.landmarks}")

    # Prepare outputs
    rejects: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    geom_rows: List[Dict[str, Any]] = []

    lm_cache_dir = out_dir / "baseline_landmarks_cache"
    if args.save_landmarks_cache:
        ensure_dir(lm_cache_dir)

    t0 = dt.datetime.utcnow()

    # Process images
    for i, pth in enumerate(paths, start=1):
        rel = str(pth.relative_to(data_root)) if pth.is_relative_to(data_root) else str(pth)
        image_id = sha1_short(rel, 16)

        try:
            img = read_image_rgb(pth)
        except Exception as e:
            rejects.append({"path": str(pth), "reason": f"image_read_error: {e}"})
            continue

        lr = backend.extract(img)

        # Basic checks
        if not lr.ok or lr.landmarks_xy is None:
            rejects.append({"path": str(pth), "reason": lr.reject_reason or "no_face_detected"})
            continue

        if lr.num_faces > args.max_faces:
            rejects.append({"path": str(pth), "reason": f"too_many_faces: {lr.num_faces}"})
            continue

        if args.min_landmark_conf > 0 and lr.confidence < args.min_landmark_conf:
            rejects.append({"path": str(pth), "reason": f"low_landmark_conf: {lr.confidence:.4f}"})
            continue

        # Face size filter
        fw, fh = compute_face_bbox_size(lr.landmarks_xy)
        if min(fw, fh) < args.min_face_size:
            rejects.append({"path": str(pth), "reason": f"face_too_small: bbox=({fw:.1f},{fh:.1f})"})
            continue

        # Alignment
        if args.align_mode == "similarity":
            ar = align_landmarks_similarity(lr.landmarks_xy)
        else:
            rejects.append({"path": str(pth), "reason": f"unknown_align_mode: {args.align_mode}"})
            continue

        if not ar.ok or ar.aligned_xy is None:
            rejects.append({"path": str(pth), "reason": ar.reject_reason or "align_failed"})
            continue

        # Geometry features
        try:
            g = geometry_vector(ar.aligned_xy, ar.face_height, ar.face_width, geom_schema)
        except Exception as e:
            rejects.append({"path": str(pth), "reason": f"geometry_failed: {e}"})
            continue

        # NaN/inf check
        gvals = np.array(list(g.values()), dtype=np.float64)
        if not np.all(np.isfinite(gvals)):
            rejects.append({"path": str(pth), "reason": "geometry_nonfinite"})
            continue

        # Save rows
        index_rows.append({
            "image_id": image_id,
            "path": str(pth),
            "rel_path": rel,
            "face_detected": True,
            "num_faces": lr.num_faces,
            "landmark_conf": float(lr.confidence),
            "bbox_w": float(fw),
            "bbox_h": float(fh),
        })

        geom_row = {"image_id": image_id}
        geom_row.update(g)
        geom_rows.append(geom_row)

        if args.save_landmarks_cache:
            # Save aligned landmarks as npy (compact)
            np.save(lm_cache_dir / f"{image_id}_aligned.npy", ar.aligned_xy.astype(np.float32))

        if (i % 250) == 0:
            print(f"[INFO] Processed {i}/{len(paths)} images... accepted={len(index_rows)} rejected={len(rejects)}")

    backend.close()
    t1 = dt.datetime.utcnow()
    elapsed_s = (t1 - t0).total_seconds()

    # Build DataFrames
    df_index = pd.DataFrame(index_rows)
    df_rejects = pd.DataFrame(rejects)
    df_geom = pd.DataFrame(geom_rows)

    # Save index/rejects
    df_index.to_csv(out_dir / "baseline_index.csv", index=False)
    df_rejects.to_csv(out_dir / "baseline_rejects.csv", index=False)

    # Save geometry (parquet preferred)
    geom_path_parquet = out_dir / "baseline_geometry.parquet"
    ok_parquet, msg = try_save_parquet(df_geom, geom_path_parquet)
    if not ok_parquet:
        df_geom.to_csv(out_dir / "baseline_geometry.csv", index=False)
        geom_storage = msg
    else:
        geom_storage = "parquet"

    # Sanity: need enough samples
    n_acc = len(df_geom)
    if n_acc < 100:
        print(f"[WARN] Only {n_acc} accepted samples. Baseline stats may be unstable.", file=sys.stderr)

    feature_cols = [f.name for f in geom_schema]

    # Compute stats/bounds
    stats_obj = compute_stats(df_geom, feature_cols, mode=args.stats_mode)
    bounds_obj = compute_bounds(
        df_geom,
        feature_cols,
        bounds_mode=args.bounds_mode,
        lower_p=args.lower_p,
        upper_p=args.upper_p,
        k=args.k,
        stats_obj=stats_obj,
    )

    # Write stats/bounds
    safe_write_json(out_dir / "geometry_stats.json", {
        "landmark_topology": args.landmarks,
        "feature_names": feature_cols,
        **stats_obj,
    })
    safe_write_json(out_dir / "geometry_bounds.json", {
        "landmark_topology": args.landmarks,
        "feature_names": feature_cols,
        **bounds_obj,
    })

    # Write symmetry pairs
    safe_write_json(out_dir / "symmetry_pairs.json", dataclasses.asdict(sym_template))

    # Report
    report = {
        "run_id": run_id,
        "data_root": str(data_root),
        "candidates": len(paths),
        "accepted": int(len(df_index)),
        "rejected": int(len(df_rejects)),
        "accept_rate": float(len(df_index) / max(1, len(paths))),
        "elapsed_seconds": float(elapsed_s),
        "geometry_storage": geom_storage,
        "stats_mode": args.stats_mode,
        "bounds_mode": args.bounds_mode,
        "created_files": [
            "run_config.json",
            "baseline_index.csv",
            "baseline_rejects.csv",
            "baseline_geometry.parquet" if ok_parquet else "baseline_geometry.csv",
            "geometry_stats.json",
            "geometry_bounds.json",
            "symmetry_pairs.json",
            "baseline_report.json",
        ],
    }
    safe_write_json(out_dir / "baseline_report.json", report)

    print(f"[DONE] Baseline built at: {out_dir}")
    print(f"       candidates={report['candidates']} accepted={report['accepted']} rejected={report['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
