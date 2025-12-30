#!/usr/bin/env python3
"""
Step 4 (refactored): CQS component computation (per pair) + summary.

Key refactors vs original:
1) Step-3 OK flags are treated as the inclusion filter (geometry_ok/identity_ok[/clip_ok]).
2) Identity is NOT re-gated by tau_id unless you explicitly request strict mode.
3) If identity embeddings are missing, we compute a no-identity fallback CQS (weights renormalized).
4) Better path normalization for Windows-style paths; improved face selection strategy.

Original file behavior that caused surprises:
- Re-gating by tau_id after filtering identity_ok.
- Hard-zeroing if reference identity embedding missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# -------------------------
# Optional dependencies
# -------------------------
_HAS_MEDIAPIPE = False
_HAS_INSIGHTFACE = False
_HAS_OPENCLIP = False

try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

try:
    from insightface.app import FaceAnalysis  # type: ignore
    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False

try:
    import torch  # type: ignore
    import open_clip  # type: ignore
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


# -------------------------
# Utility
# -------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm_path_str(p: str) -> str:
    """
    Normalize paths that may be saved with Windows backslashes into the current OS format.
    Keeps them usable on Windows, and makes them usable if you ever run on Linux.
    """
    s = str(p).strip().strip('"').strip("'")
    if os.sep == "/" and "\\" in s:
        s = s.replace("\\", "/")
    return s


def _imread_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y", "ok", "pass")
    return False


# -------------------------
# Geometry (MediaPipe FaceMesh)
# -------------------------
@dataclass
class GeoConfig:
    eps: float = 1e-8
    sigma_sym: float = 0.06
    sigma_range: float = 0.35


class FaceMeshGeometry:
    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not available. Install mediapipe.")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.idx = {
            "le_outer": 33,
            "le_inner": 133,
            "re_outer": 263,
            "re_inner": 362,
            "nose_tip": 1,
            "nose_bridge": 168,
            "chin": 152,
            "mouth_l": 61,
            "mouth_r": 291,
            "cheek_l": 234,
            "cheek_r": 454,
            "nose_l": 94,
            "nose_r": 331,
            "forehead": 10,
            "jaw_l": 172,
            "jaw_r": 397,
        }

    def detect_landmarks(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        h, w = rgb.shape[:2]
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, 0.0

        face_lm = res.multi_face_landmarks[0]
        num_landmarks = len(face_lm.landmark)
        pts = np.zeros((num_landmarks, 2), dtype=np.float32)
        for i, p in enumerate(face_lm.landmark):
            pts[i, 0] = p.x * w
            pts[i, 1] = p.y * h
        return pts, 1.0

    def _eye_centers(self, lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        le = 0.5 * (lm[self.idx["le_outer"]] + lm[self.idx["le_inner"]])
        re = 0.5 * (lm[self.idx["re_outer"]] + lm[self.idx["re_inner"]])
        return le, re

    def align(self, lm: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
        le, re = self._eye_centers(lm)
        center = 0.5 * (le + re)
        v = re - le
        iod = float(np.linalg.norm(v) + 1e-8)

        angle = math.atan2(v[1], v[0])
        ca = math.cos(-angle)
        sa = math.sin(-angle)
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

        lm0 = lm - center[None, :]
        lm_rot = (lm0 @ R.T)
        lm_aligned = lm_rot / iod
        return lm_aligned, iod, angle, center

    def build_descriptor(self, lm_aligned: np.ndarray, cfg: GeoConfig) -> np.ndarray:
        I = self.idx
        eps = cfg.eps

        def d(a, b) -> float:
            return float(np.linalg.norm(lm_aligned[a] - lm_aligned[b]) + eps)

        def angle(a, b, c) -> float:
            ba = lm_aligned[a] - lm_aligned[b]
            bc = lm_aligned[c] - lm_aligned[b]
            num = float(np.dot(ba, bc))
            den = float(np.linalg.norm(ba) * np.linalg.norm(bc) + eps)
            x = max(-1.0, min(1.0, num / den))
            return float(math.acos(x))

        face_w = d(I["cheek_l"], I["cheek_r"])
        face_h = d(I["forehead"], I["chin"])
        jaw_w = d(I["jaw_l"], I["jaw_r"])

        eye_w_l = d(I["le_outer"], I["le_inner"])
        eye_w_r = d(I["re_outer"], I["re_inner"])
        eye_w = 0.5 * (eye_w_l + eye_w_r)

        le, re = self._eye_centers(lm_aligned)
        iod = float(np.linalg.norm(re - le) + eps)

        nose_w = d(I["nose_l"], I["nose_r"])
        nose_len = d(I["nose_bridge"], I["nose_tip"])

        mouth_w = d(I["mouth_l"], I["mouth_r"])
        mouth_to_chin = d(I["mouth_l"], I["chin"])

        jaw_angle = angle(I["jaw_l"], I["chin"], I["jaw_r"])
        nose_angle = angle(I["nose_bridge"], I["nose_tip"], I["chin"])

        G = np.array(
            [
                face_w,
                face_h,
                jaw_w,
                eye_w / (face_w + eps),
                iod / (face_w + eps),
                nose_w / (face_w + eps),
                nose_len / (face_h + eps),
                mouth_w / (face_w + eps),
                mouth_to_chin / (face_h + eps),
                jaw_angle / math.pi,
                nose_angle / math.pi,
            ],
            dtype=np.float32,
        )
        return G

    def symmetry_score(self, lm_aligned: np.ndarray, cfg: GeoConfig) -> float:
        I = self.idx
        eps = cfg.eps
        sigma = cfg.sigma_sym
        x_mid = float(lm_aligned[I["nose_bridge"], 0])

        pairs = [
            (I["le_outer"], I["re_outer"]),
            (I["le_inner"], I["re_inner"]),
            (I["cheek_l"], I["cheek_r"]),
            (I["jaw_l"], I["jaw_r"]),
            (I["mouth_l"], I["mouth_r"]),
            (I["nose_l"], I["nose_r"]),
        ]

        errs = []
        for a, b in pairs:
            pa = lm_aligned[a]
            pb = lm_aligned[b]
            pb_m = np.array([2 * x_mid - pb[0], pb[1]], dtype=np.float32)
            errs.append(float(np.linalg.norm(pa - pb_m) + eps))

        e = float(np.mean(errs))
        return float(math.exp(-(e * e) / (2 * sigma * sigma)))


# -------------------------
# Identity (InsightFace)
# -------------------------
class IdentityEmbedder:
    def __init__(
        self,
        det_size: int = 640,
        ctx_id: int = 0,
        model_name: str = "",
        face_select: str = "score_area",  # "score", "area", "score_area"
    ):
        if not _HAS_INSIGHTFACE:
            raise RuntimeError("insightface is not available. Install insightface + onnxruntime.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id == 0 else ["CPUExecutionProvider"]

        # Allow explicit model selection if desired (keeps default behavior if blank).
        kwargs = {"providers": providers}
        if model_name:
            kwargs["name"] = model_name
        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        if face_select not in ("score", "area", "score_area"):
            raise ValueError("face_select must be one of: score, area, score_area")
        self.face_select = face_select

    def _pick_best(self, faces) -> Optional[object]:
        best = None
        best_val = -1e18
        for f in faces:
            box = f.bbox.astype(np.float32)
            area = float(max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1])))
            score = float(getattr(f, "det_score", 1.0))
            if self.face_select == "area":
                val = area
            elif self.face_select == "score":
                val = score
            else:
                val = score * (area + 1.0)
            if val > best_val:
                best_val = val
                best = f
        return best

    def embed(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        # InsightFace expects BGR
        bgr = rgb[:, :, ::-1].copy()
        faces = self.app.get(bgr)
        if not faces:
            return None, 0.0

        best = self._pick_best(faces)
        if best is None or getattr(best, "normed_embedding", None) is None:
            return None, 0.0

        emb = best.normed_embedding.astype(np.float32)
        det_score = float(getattr(best, "det_score", 1.0))
        return emb, max(0.0, min(1.0, det_score))


# -------------------------
# CLIP (optional)
# -------------------------
class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        if not _HAS_OPENCLIP:
            raise RuntimeError("open_clip_torch is not available. Install torch + open_clip_torch.")
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.model.to(self.device)

    def embed(self, img_path: Path) -> np.ndarray:
        im = Image.open(img_path).convert("RGB")
        x = self.preprocess(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.model.encode_image(x)
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z.squeeze(0).detach().cpu().numpy().astype(np.float32)


# -------------------------
# CQS config
# -------------------------
@dataclass
class CQSWeights:
    alpha_id: float = 0.35
    beta_sal: float = 0.30
    gamma_plaus: float = 0.25
    delta_coh: float = 0.10
    eta_mag: float = 0.0
    # New: reward exaggeration aligned with distinctive reference features
    zeta_exag: float = 0.0


@dataclass
class CQSConfig:
    tau_id: float = 0.60
    use_distinctiveness: bool = True
    use_clip: bool = False
    use_mag_window: bool = False
    mu_mag: float = 0.0
    sigma_mag: float = 1.0
    # New: scale for distinctive-exaggeration saturation (larger -> slower saturation)
    exag_tau: float = 0.25

    # New:
    cqs_mode: str = "noid_fallback"  # "manifest", "strict", "noid_fallback"
    geo: GeoConfig = GeoConfig()
    w: CQSWeights = CQSWeights()


# -------------------------
# Salience & plausibility helpers
# -------------------------
def salience_concentration(deltaG: np.ndarray, eps: float = 1e-8) -> float:
    dg = np.maximum(deltaG.astype(np.float32), 0.0)
    s = float(np.sum(dg))
    if s <= eps:
        return 0.0
    p = dg / (s + eps)
    H = float(-np.sum(p * np.log(p + eps)))
    Hmax = float(np.log(len(p) + eps))
    return float(max(0.0, min(1.0, 1.0 - (H / (Hmax + eps)))))


def salience_distinctiveness(deltaG: np.ndarray, D: np.ndarray, eps: float = 1e-8) -> float:
    dg = deltaG.astype(np.float32)
    D = D.astype(np.float32)
    if np.linalg.norm(dg) < eps or np.linalg.norm(D) < eps:
        return 0.0
    c = _cosine(dg, D, eps=eps)
    return float(max(0.0, min(1.0, c)))


def range_score(Gc: np.ndarray, lower: np.ndarray, upper: np.ndarray, cfg: GeoConfig) -> float:
    eps = cfg.eps
    sigma = cfg.sigma_range

    Gc = Gc.astype(np.float32)
    lower = lower.astype(np.float32)
    upper = upper.astype(np.float32)

    width = np.maximum(upper - lower, eps)
    r = np.zeros_like(Gc, dtype=np.float32)

    below = Gc < lower
    above = Gc > upper

    r[below] = (lower[below] - Gc[below]) / width[below]
    r[above] = (Gc[above] - upper[above]) / width[above]

    E = float(np.mean(r * r))
    return float(math.exp(-E / (2 * sigma * sigma)))


def load_ok_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".json", ".jsonl"):
        if path.suffix.lower() == ".jsonl":
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            df = pd.DataFrame(items)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported manifest extension: {path.suffix}")

    for c in ("geometry_ok", "clip_ok", "identity_ok"):
        df[c] = df[c].apply(_to_bool) if c in df.columns else True

    for req in ("person_id", "method", "image_path"):
        if req not in df.columns:
            raise ValueError(f"Manifest missing required column: {req}")

    # Normalize path strings
    df["image_path"] = df["image_path"].astype(str).apply(_norm_path_str)
    if "reference_path" in df.columns:
        df["reference_path"] = df["reference_path"].astype(str).apply(_norm_path_str)

    # Infer reference_path if missing
    if "reference_path" not in df.columns:
        refs = []
        for p in df["image_path"].astype(str).tolist():
            pp = Path(p)
            person_dir = pp.parent.parent
            ref = person_dir / "reference.jpg"
            if not ref.exists():
                ref = person_dir / "reference.png"
            refs.append(str(ref))
        df["reference_path"] = refs

    return df


def load_geometry_stats(stats_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if stats_path is None or (not stats_path.exists()):
        return None, None
    if stats_path.suffix.lower() == ".json":
        with open(stats_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mean = np.array(d["mean"], dtype=np.float32)
        std = np.array(d.get("std", np.ones_like(mean)), dtype=np.float32)
        return mean, std
    if stats_path.suffix.lower() == ".npz":
        z = np.load(stats_path)
        mean = z["mean"].astype(np.float32)
        std = z.get("std", np.ones_like(mean)).astype(np.float32)
        return mean, std
    return None, None


def load_geometry_bounds(bounds_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if bounds_path is None or (not bounds_path.exists()):
        return None, None
    if bounds_path.suffix.lower() == ".json":
        with open(bounds_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        lower = np.array(d["lower"], dtype=np.float32)
        upper = np.array(d["upper"], dtype=np.float32)
        return lower, upper
    if bounds_path.suffix.lower() == ".npz":
        z = np.load(bounds_path)
        lower = z["lower"].astype(np.float32)
        upper = z["upper"].astype(np.float32)
        return lower, upper
    return None, None


def _cqs_noid_fallback(cfg: CQSConfig, Ssal: float, Splaus: float, Scoh: float, Smag: float, Sexag: float) -> float:
    """
    If identity cannot be computed, drop alpha_id*Sid and renormalize remaining weights.
    """
    w = cfg.w
    rest = np.array([w.beta_sal, w.gamma_plaus, w.delta_coh, w.eta_mag, w.zeta_exag], dtype=np.float32)
    denom = float(np.sum(rest))
    if denom <= 1e-12:
        return 0.0
    val = (
        w.beta_sal * Ssal +
        w.gamma_plaus * Splaus +
        w.delta_coh * Scoh +
        w.eta_mag * Smag +
        w.zeta_exag * Sexag
    ) / denom
    return float(max(0.0, min(1.0, val)))


def compute_cqs_for_pairs(
    pairs_ok: pd.DataFrame,
    out_dir: Path,
    cfg: CQSConfig,
    stats_mean: Optional[np.ndarray],
    stats_std: Optional[np.ndarray],
    bounds_lower: Optional[np.ndarray],
    bounds_upper: Optional[np.ndarray],
    device: str,
    insightface_model: str,
    face_select: str,
    det_size: int,
) -> pd.DataFrame:
    _safe_mkdir(out_dir)

    if not _HAS_MEDIAPIPE:
        raise RuntimeError("MediaPipe required for geometry. Install mediapipe.")
    if not _HAS_INSIGHTFACE:
        raise RuntimeError("InsightFace required for identity. Install insightface + onnxruntime.")

    geom = FaceMeshGeometry(static_image_mode=True, max_num_faces=1)

    ctx_id = 0 if device.startswith("cuda") else -1
    ident = IdentityEmbedder(
        det_size=det_size,
        ctx_id=ctx_id,
        model_name=insightface_model,
        face_select=face_select,
    )

    clip = None
    if cfg.use_clip:
        if not _HAS_OPENCLIP:
            raise RuntimeError("CLIP requested but open_clip_torch is not installed.")
        clip = ClipEmbedder(model_name="ViT-B-32", pretrained="openai", device=device)

    # Cache reference computations
    ref_cache: Dict[str, Dict[str, object]] = {}
    rows_out: List[Dict[str, object]] = []

    # If distinctiveness is requested but external stats are missing or incompatible with our descriptor,
    # compute mean/std from the available reference descriptors in this manifest (small-dataset fallback).
    if cfg.use_distinctiveness:
        refs_df = (
            pairs_ok[["person_id", "reference_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, rrow in refs_df.iterrows():
            person_id = str(rrow["person_id"])
            ref_path = Path(str(rrow["reference_path"]))
            if person_id in ref_cache:
                continue
            if not ref_path.exists():
                ref_cache[person_id] = dict(valid=False)
                continue

            rgb_r = _imread_rgb(ref_path)
            lm_r, qlm_r = geom.detect_landmarks(rgb_r)
            if lm_r is None:
                ref_cache[person_id] = dict(valid=False)
                continue

            lm_r_al, _, _, _ = geom.align(lm_r)
            G_r = geom.build_descriptor(lm_r_al, cfg.geo)
            emb_r, qid_r = ident.embed(rgb_r)
            z_r = clip.embed(ref_path) if clip is not None else None
            ref_cache[person_id] = dict(
                valid=True,
                G_r=G_r,
                emb_r=emb_r,          # may be None
                qlm_r=float(qlm_r),
                qid_r=float(qid_r),
                z_r=z_r,
            )

        G_list = [
            entry.get("G_r")  # type: ignore
            for entry in ref_cache.values()
            if entry.get("valid", False) and (entry.get("G_r", None) is not None)
        ]
        if len(G_list) > 0:
            G_stack = np.stack(G_list, axis=0).astype(np.float32)
            internal_mean = np.mean(G_stack, axis=0)
            internal_std = np.std(G_stack, axis=0)

            # If external stats missing or wrong shape, replace with internal stats.
            if (stats_mean is None) or (stats_std is None) or (np.shape(stats_mean) != np.shape(internal_mean)):
                stats_mean = internal_mean
                stats_std = internal_std
        else:
            # Cannot compute distinctiveness without any valid references.
            stats_mean, stats_std = None, None

    for _, row in pairs_ok.iterrows():
        person_id = str(row["person_id"])
        method = str(row["method"])
        img_path = Path(str(row["image_path"]))
        ref_path = Path(str(row["reference_path"]))

        # Manifest gates (these are *your* Step 3 OK decisions)
        gate_manifest_geometry = int(_to_bool(row.get("geometry_ok", True)))
        gate_manifest_identity = int(_to_bool(row.get("identity_ok", True)))
        gate_manifest_clip = int(_to_bool(row.get("clip_ok", True)))

        if cfg.use_clip and gate_manifest_clip == 0:
            continue
        if gate_manifest_geometry == 0 or gate_manifest_identity == 0:
            continue

        if not img_path.exists() or not ref_path.exists():
            continue

        # Reference cache
        if person_id not in ref_cache:
            rgb_r = _imread_rgb(ref_path)
            lm_r, qlm_r = geom.detect_landmarks(rgb_r)
            if lm_r is None:
                ref_cache[person_id] = dict(valid=False)
            else:
                lm_r_al, _, _, _ = geom.align(lm_r)
                G_r = geom.build_descriptor(lm_r_al, cfg.geo)

                emb_r, qid_r = ident.embed(rgb_r)
                z_r = clip.embed(ref_path) if clip is not None else None

                ref_cache[person_id] = dict(
                    valid=True,
                    G_r=G_r,
                    emb_r=emb_r,          # may be None
                    qlm_r=float(qlm_r),
                    qid_r=float(qid_r),
                    z_r=z_r,
                )

        ref_entry = ref_cache[person_id]
        if not ref_entry.get("valid", False):
            continue

        G_r = ref_entry["G_r"]  # type: ignore
        emb_r = ref_entry.get("emb_r", None)
        z_r = ref_entry.get("z_r", None)

        # Caricature
        rgb_c = _imread_rgb(img_path)

        lm_c, qlm_c = geom.detect_landmarks(rgb_c)
        if lm_c is None:
            # geometry_ok said OK, but backend could still fail; record it explicitly
            rows_out.append(
                dict(
                    person_id=person_id,
                    method=method,
                    image_path=str(img_path),
                    reference_path=str(ref_path),
                    gate_face=0,
                    gate_id_manifest=gate_manifest_identity,
                    gate_id_tau=0,
                    Sid=np.nan,
                    sid_raw=np.nan,
                    sid_status="no_landmarks",
                    mag=np.nan,
                    Sconc=0.0,
                    Sdist=0.0,
                    Ssal=0.0,
                    Sconf=0.0,
                    Ssym=0.0,
                    Srange=0.0,
                    Splaus=0.0,
                    Scoh=0.0,
                    Smag=0.0,
                    CQS=0.0,
                )
            )
            continue

        lm_c_al, _, _, _ = geom.align(lm_c)
        G_c = geom.build_descriptor(lm_c_al, cfg.geo)

        # Identity embeddings
        emb_c, _ = ident.embed(rgb_c)

        Sid = np.nan
        sid_raw = np.nan
        sid_status = "ok"
        if (emb_r is None) or (emb_c is None):
            sid_status = "missing_ref_emb" if emb_r is None else "missing_car_emb"
        else:
            sid_raw = _cosine(emb_r, emb_c)
            Sid = (sid_raw + 1.0) / 2.0

        # Geometry deviation
        deltaG = np.abs(G_c - G_r).astype(np.float32)
        mag = float(np.linalg.norm(deltaG))

        # Salience
        Sconc = salience_concentration(deltaG, eps=cfg.geo.eps)
        D = None
        if cfg.use_distinctiveness and (stats_mean is not None) and (stats_std is not None):
            D = np.abs(G_r - stats_mean) / (stats_std + cfg.geo.eps)
            Sdist = salience_distinctiveness(deltaG, D, eps=cfg.geo.eps)
        else:
            Sdist = 1.0
        Ssal = float(max(0.0, min(1.0, Sconc * Sdist)))

        # Plausibility
        Sconf = float(qlm_c)
        Ssym = float(geom.symmetry_score(lm_c_al, cfg.geo))
        if (
            (bounds_lower is not None)
            and (bounds_upper is not None)
            and (np.shape(bounds_lower) == np.shape(G_c))
            and (np.shape(bounds_upper) == np.shape(G_c))
        ):
            Srange = range_score(G_c, bounds_lower, bounds_upper, cfg.geo)
        else:
            Srange = 1.0
        Splaus = float(max(0.0, min(1.0, Sconf * Ssym * Srange)))

        # Coherence
        Scoh = 1.0
        if clip is not None and z_r is not None:
            z_c = clip.embed(img_path)
            Scoh = ( _cosine(z_r, z_c) + 1.0 ) / 2.0

        # Magnitude window (optional)
        Smag = 1.0
        if cfg.use_mag_window:
            Smag = float(math.exp(-((mag - cfg.mu_mag) ** 2) / (2 * (cfg.sigma_mag ** 2) + cfg.geo.eps)))

        # Distinctive exaggeration (new, optional)
        # Reward changes that align with what is distinctive in the reference (D).
        # This is monotonic increasing w.r.t. projection of deltaG onto D, with saturation via exag_tau.
        Sexag = 0.0
        exag_proj = 0.0
        if D is not None:
            Dp = np.maximum(D.astype(np.float32), 0.0)
            Dn = Dp / (float(np.linalg.norm(Dp)) + cfg.geo.eps)
            exag_proj = float(np.dot(deltaG, Dn))
            Sexag = float(1.0 - math.exp(-exag_proj / (cfg.exag_tau + cfg.geo.eps)))
            Sexag = float(max(0.0, min(1.0, Sexag)))

        # Gates
        gate_face = 1
        gate_id_manifest = gate_manifest_identity  # after filtering: should be 1
        gate_id_tau = 0
        if not np.isnan(Sid):
            gate_id_tau = 1 if float(Sid) >= cfg.tau_id else 0

        # Compute CQS in the requested mode
        w = cfg.w
        if cfg.cqs_mode == "strict":
            g = gate_face * gate_id_tau
            CQS = float(
                g * (
                    w.alpha_id * (0.0 if np.isnan(Sid) else float(Sid)) +
                    w.beta_sal * Ssal +
                    w.gamma_plaus * Splaus +
                    w.delta_coh * Scoh +
                    w.eta_mag * Smag +
                    w.zeta_exag * Sexag
                )
            )
        elif cfg.cqs_mode == "manifest":
            # Do not re-gate by tau; use Step-3 inclusion as the gate
            g = gate_face * gate_id_manifest
            # If Sid missing, treat identity term as 0 but still keep the row (explicitly marked)
            CQS = float(
                g * (
                    w.alpha_id * (0.0 if np.isnan(Sid) else float(Sid)) +
                    w.beta_sal * Ssal +
                    w.gamma_plaus * Splaus +
                    w.delta_coh * Scoh +
                    w.eta_mag * Smag +
                    w.zeta_exag * Sexag
                )
            )
        elif cfg.cqs_mode == "noid_fallback":
            g = gate_face * gate_id_manifest
            if np.isnan(Sid):
                CQS = float(g * _cqs_noid_fallback(cfg, Ssal, Splaus, Scoh, Smag, Sexag))
            else:
                CQS = float(
                    g * (
                        w.alpha_id * float(Sid) +
                        w.beta_sal * Ssal +
                        w.gamma_plaus * Splaus +
                        w.delta_coh * Scoh +
                        w.eta_mag * Smag +
                        w.zeta_exag * Sexag
                    )
                )
        else:
            raise ValueError("cqs_mode must be one of: strict, manifest, noid_fallback")

        rows_out.append(
            dict(
                person_id=person_id,
                method=method,
                image_path=str(img_path),
                reference_path=str(ref_path),
                gate_face=int(gate_face),
                gate_id_manifest=int(gate_id_manifest),
                gate_id_tau=int(gate_id_tau),
                Sid=float(Sid) if not np.isnan(Sid) else np.nan,
                sid_raw=float(sid_raw) if not np.isnan(sid_raw) else np.nan,
                sid_status=str(sid_status),
                mag=float(mag),
                Sconc=float(Sconc),
                Sdist=float(Sdist),
                Ssal=float(Ssal),
                Sconf=float(Sconf),
                Ssym=float(Ssym),
                Srange=float(Srange),
                Splaus=float(Splaus),
                Scoh=float(Scoh),
                Smag=float(Smag),
                exag_proj=float(exag_proj),
                Sexag=float(Sexag),
                CQS=float(CQS),
            )
        )

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(out_dir / "cqs_pairs.csv", index=False)

    if not df_out.empty:
        summary = (
            df_out.groupby("method")
            .agg(
                n=("CQS", "count"),
                cqs_mean=("CQS", "mean"),
                cqs_median=("CQS", "median"),
                sid_mean=("Sid", "mean"),
                sal_mean=("Ssal", "mean"),
                plaus_mean=("Splaus", "mean"),
                coh_mean=("Scoh", "mean"),
                sexag_mean=("Sexag", "mean"),
                id_missing_rate=("sid_status", lambda s: float(np.mean(s != "ok"))),
                gate_tau_pass_rate=("gate_id_tau", "mean"),
            )
            .reset_index()
            .sort_values("cqs_mean", ascending=False)
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "method","n","cqs_mean","cqs_median","sid_mean","sal_mean","plaus_mean","coh_mean",
                "id_missing_rate","gate_tau_pass_rate"
            ]
        )

    summary.to_csv(out_dir / "cqs_summary_by_method.csv", index=False)

    run_cfg = dict(
        cqs_config=asdict(cfg),
        model_info=dict(
            mediapipe_available=_HAS_MEDIAPIPE,
            insightface_available=_HAS_INSIGHTFACE,
            openclip_available=_HAS_OPENCLIP,
            use_clip=cfg.use_clip,
        ),
        n_input_pairs=int(len(pairs_ok)),
        n_scored_pairs=int(len(df_out)),
    )
    with open(out_dir / "cqs_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    return df_out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("step4_cqs_refactored.py")

    ap.add_argument("--ok_manifest", type=str, required=True, help="CSV/JSON from Step 3 with geometry_ok/clip_ok/identity_ok.")
    ap.add_argument("--out_dir", type=str, default="reports/cqs", help="Output directory for CQS results.")

    ap.add_argument("--use_clip", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--geometry_stats", type=str, default="")
    ap.add_argument("--geometry_bounds", type=str, default="")

    ap.add_argument("--tau_id", type=float, default=0.60)
    ap.add_argument("--use_distinctiveness", action="store_true")

    ap.add_argument("--cqs_mode", type=str, default="noid_fallback", choices=["strict", "manifest", "noid_fallback"])

    # weights (optimized to match human survey preferences: overall > exaggeration > identity)
    # Optimization results: overall r=0.9042, exaggeration r=-0.7213, identity r=0.9608* (significant)
    ap.add_argument("--alpha_id", type=float, default=0.465116, help="Identity weight (optimized)")
    ap.add_argument("--beta_sal", type=float, default=0.023256, help="Salience weight (optimized)")
    ap.add_argument("--gamma_plaus", type=float, default=0.465115, help="Plausibility weight (optimized)")
    ap.add_argument("--delta_coh", type=float, default=0.023256, help="Coherence weight (optimized)")
    ap.add_argument("--eta_mag", type=float, default=0.023256, help="Magnitude weight (optimized)")
    ap.add_argument("--zeta_exag", type=float, default=0.023256, help="Distinctive exaggeration weight (new)")

    # plausibility

    ap.add_argument("--sigma_sym", type=float, default=0.06)

    # optional magnitude window
    ap.add_argument("--use_mag_window", action="store_true")
    ap.add_argument("--mu_mag", type=float, default=0.0)
    ap.add_argument("--sigma_mag", type=float, default=1.0)
    ap.add_argument("--exag_tau", type=float, default=0.25, help="Distinctive exaggeration saturation scale (new)")

    # insightface controls
    ap.add_argument("--insightface_model", type=str, default="", help="Optional InsightFace model name (e.g., buffalo_l).")
    ap.add_argument("--face_select", type=str, default="score_area", choices=["score", "area", "score_area"])
    ap.add_argument("--det_size", type=int, default=640)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    pairs = load_ok_manifest(Path(args.ok_manifest))
    if pairs.empty:
        print(f"[ERROR] OK-manifest loaded but empty: {args.ok_manifest}", file=sys.stderr)
        sys.exit(1)

    # Hard filter to OK-only (your current plan)
    pairs["geometry_ok"] = pairs["geometry_ok"].apply(_to_bool)
    pairs["identity_ok"] = pairs["identity_ok"].apply(_to_bool)
    pairs["clip_ok"] = pairs["clip_ok"].apply(_to_bool)

    if args.use_clip:
        pairs_ok = pairs[(pairs["geometry_ok"]) & (pairs["identity_ok"]) & (pairs["clip_ok"])].copy()
    else:
        pairs_ok = pairs[(pairs["geometry_ok"]) & (pairs["identity_ok"])].copy()

    if pairs_ok.empty:
        print("[ERROR] After filtering to OK-only images, nothing remains. Check your manifest flags.", file=sys.stderr)
        sys.exit(1)

    stats_mean, stats_std = load_geometry_stats(Path(args.geometry_stats) if args.geometry_stats else None)
    bounds_lower, bounds_upper = load_geometry_bounds(Path(args.geometry_bounds) if args.geometry_bounds else None)

    cfg = CQSConfig(
        tau_id=float(args.tau_id),
        use_distinctiveness=bool(args.use_distinctiveness),
        use_clip=bool(args.use_clip),
        use_mag_window=bool(args.use_mag_window),
        mu_mag=float(args.mu_mag),
        sigma_mag=float(args.sigma_mag),
        exag_tau=float(args.exag_tau),
        cqs_mode=str(args.cqs_mode),
        geo=GeoConfig(eps=1e-8, sigma_sym=float(args.sigma_sym), sigma_range=0.35),
        w=CQSWeights(
            alpha_id=float(args.alpha_id),
            beta_sal=float(args.beta_sal),
            gamma_plaus=float(args.gamma_plaus),
            delta_coh=float(args.delta_coh),
            eta_mag=float(args.eta_mag),
            zeta_exag=float(args.zeta_exag),
        ),
    )

    df = compute_cqs_for_pairs(
        pairs_ok=pairs_ok,
        out_dir=out_dir,
        cfg=cfg,
        stats_mean=stats_mean,
        stats_std=stats_std,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        device=str(args.device),
        insightface_model=str(args.insightface_model),
        face_select=str(args.face_select),
        det_size=int(args.det_size),
    )

    print(f"[OK] Scored pairs: {len(df)}")
    print(f"[OK] Wrote: {out_dir / 'cqs_pairs.csv'}")
    print(f"[OK] Wrote: {out_dir / 'cqs_summary_by_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'cqs_run_config.json'}")


if __name__ == "__main__":
    main()
