#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoAdapterStep1.py

Goal (Step 1):
  Prepare extended conditioning tensors for SDXL by converting an explicit,
  landmark-derived facial geometry descriptor G into a small set of cross-attention
  tokens T_geo, then concatenating them to the text encoder hidden states:

    E' = [E_text ; λ_geo * c * T_geo]

This step intentionally DOES NOT modify UNet attention internals.

Important constraints:
  - Geometry descriptor MUST exactly match the CQS descriptor used in this repo:
    same features, ordering, and landmark normalization as `step1.py`'s
    default_geometry_schema_mediapipe468() + align_landmarks_similarity().
  - Optional confidence c ∈ [0,1] may modulate geometry strength; do not add new
    geometry features.
  - Provide clean save/load for mapper weights and deterministic shape checks.

Dependencies:
  - numpy
  - torch (for the mapper)
  - mediapipe (optional; only needed if you call landmark extraction from images)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Geometry descriptor (CQS-compatible) + explicit Step-1 I/O API
# -----------------------------

# This ordering matches:
# - `step1.py` default_geometry_schema_mediapipe468()
# - `baselines/*/geometry_stats.json` "feature_names"
GEOMETRY_FEATURE_NAMES: Tuple[str, ...] = (
    "face_width_over_height",
    "left_eye_width_over_interocular",
    "right_eye_width_over_interocular",
    "interocular_over_face_width",
    "mouth_width_over_face_width",
    "nose_wing_width_over_face_width",
    "nose_length_over_face_height",
    "jaw_opening_angle",
)


LANDMARK_IDXS: Dict[str, int] = {
    # Stable MediaPipe FaceMesh indices used by CQS baseline
    "forehead": 10,
    "chin": 152,
    "cheek_l": 234,
    "cheek_r": 454,
    "le_outer": 33,
    "le_inner": 133,
    "re_outer": 263,
    "re_inner": 362,
    "mouth_l": 61,
    "mouth_r": 291,
    "nose_l": 94,
    "nose_r": 331,
    "nose_bridge": 168,
    "nose_tip": 1,
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> float:
    """
    Angle at b in radians, between vectors (a-b) and (c-b).
    """
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2 + eps))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.acos(cosv))


def compute_eye_centers_mediapipe468(xy_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches `step1.py` compute_eye_centers_mediapipe().
    Eye centers based on stable corner indices:
      left eye corners: 33 (outer), 133 (inner)
      right eye corners: 263 (outer), 362 (inner)
    """
    left = 0.5 * (xy_px[LANDMARK_IDXS["le_outer"]] + xy_px[LANDMARK_IDXS["le_inner"]])
    right = 0.5 * (xy_px[LANDMARK_IDXS["re_outer"]] + xy_px[LANDMARK_IDXS["re_inner"]])
    return left.astype(np.float32), right.astype(np.float32)


@dataclass(frozen=True)
class AlignResult:
    ok: bool
    aligned_xy: Optional[np.ndarray]  # (K,2) float32 in canonical coords
    interocular: float
    face_height: float
    face_width: float
    reject_reason: Optional[str] = None


def extract_landmarks(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Step-1 API: extract_landmarks(image) -> P

    Returns:
      P: (468,2) float32 landmarks in pixel coordinates.

    Notes:
      - If `image` is a path, it will be loaded as RGB first.
      - If `image` is an RGB ndarray, it is used directly.
      - If you also want confidence/meta, use `extract_landmarks_with_meta`.
    """
    P, _meta = extract_landmarks_with_meta(image)
    return P


def extract_landmarks_with_meta(image: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience helper that returns landmarks + meta, including a confidence scalar candidate.
    """
    if isinstance(image, (str, Path)):
        rgb = load_image_rgb(image)
    else:
        rgb = image
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must be a path or an RGB ndarray of shape (H,W,3).")

    fm = MediaPipeFaceMesh468(static_image_mode=True, max_num_faces=1)
    try:
        lr = fm(rgb)
    finally:
        fm.close()
    if (not lr.ok) or (lr.landmarks_xy_px is None):
        raise RuntimeError(f"Landmark extraction failed: {lr.err}")

    meta = {
        "backend": "mediapipe_facemesh_468",
        "conf_raw": float(lr.conf),
        "img_w": int(lr.img_w),
        "img_h": int(lr.img_h),
    }
    return lr.landmarks_xy_px.astype(np.float32), meta


def normalize_landmarks(P: np.ndarray, *, allow_rotation: bool = True) -> Tuple[np.ndarray, AlignResult]:
    """
    Step-1 API: normalize_landmarks(P) -> P_tilde

    Canonical normalization (CQS-compatible): similarity alignment
      - translate so eye midpoint is origin
      - rotate so eyes horizontal (optional)
      - scale so interocular distance = 1
    """
    ar = align_landmarks_similarity(P, allow_rotation=allow_rotation)
    if not ar.ok or ar.aligned_xy is None:
        raise ValueError(f"normalize_landmarks failed: {ar.reject_reason}")
    return ar.aligned_xy, ar


def build_geometry_descriptor(P_tilde: np.ndarray, *, align: AlignResult) -> np.ndarray:
    """
    Step-1 API: build_geometry_descriptor(P_tilde) -> G (D,)

    Hard requirement: feature ordering/normalization matches the repo's CQS descriptor.
    """
    if not align.ok:
        raise ValueError(f"build_geometry_descriptor requires a valid AlignResult; got reject_reason={align.reject_reason}")
    return build_cqs_geometry_descriptor_from_aligned(
        P_tilde,
        face_height=align.face_height,
        face_width=align.face_width,
    )


def geometry_confidence(
    P: np.ndarray,
    P_tilde: Optional[np.ndarray],
    meta: Optional[Dict[str, Any]] = None,
    *,
    align: Optional[AlignResult] = None,
) -> float:
    """
    Step-1 API: geometry_confidence(P, P_tilde, meta) -> c in [0,1]

    Design:
      - c is a scalar modulator only; it MUST NOT introduce any new geometry features.
      - Uses landmark detector confidence when available, gated by alignment validity.
    """
    conf_raw = 1.0
    if meta is not None and "conf_raw" in meta:
        try:
            conf_raw = float(meta["conf_raw"])
        except Exception:
            conf_raw = 1.0

    # Gate by alignment if provided (keeps behavior deterministic and conservative).
    ok_align = True
    if align is not None:
        ok_align = bool(align.ok) and (align.aligned_xy is not None)

    # Additional soft sanity: require we have at least 468 points.
    ok_pts = isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[0] >= 468 and P.shape[1] == 2
    ok = ok_align and ok_pts and (P_tilde is None or (isinstance(P_tilde, np.ndarray) and P_tilde.shape[1] == 2))

    c = conf_raw if ok else 0.0
    c = float(max(0.0, min(1.0, c)))
    return c


def align_landmarks_similarity(
    xy_px: np.ndarray,
    *,
    allow_rotation: bool = True,
    eps: float = 1e-9,
) -> AlignResult:
    """
    Canonical alignment (CQS baseline):
      - translate so eye midpoint is origin
      - (optional) rotate so eye line is horizontal
      - scale so interocular distance = 1

    Returns:
      aligned_xy: (K,2) float32 in canonical coords
      face_height/face_width: computed in aligned coords
    """
    if xy_px.ndim != 2 or xy_px.shape[1] != 2:
        raise ValueError(f"xy_px must have shape (K,2); got {tuple(xy_px.shape)}")
    if xy_px.shape[0] < 468:
        raise ValueError(f"Expected MediaPipe 468 landmarks; got K={xy_px.shape[0]}")

    left_eye, right_eye = compute_eye_centers_mediapipe468(xy_px)
    eye_mid = 0.5 * (left_eye + right_eye)
    v = right_eye - left_eye
    inter = float(np.linalg.norm(v))
    if inter < 10.0:  # matches step1's conservative pixel threshold
        return AlignResult(
            ok=False,
            aligned_xy=None,
            interocular=inter,
            face_height=0.0,
            face_width=0.0,
            reject_reason="interocular_too_small",
        )

    xy0 = xy_px.astype(np.float32) - eye_mid.astype(np.float32)

    if allow_rotation:
        theta = math.atan2(float(v[1]), float(v[0]))
        c = math.cos(-theta)
        s = math.sin(-theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        xy1 = (xy0 @ R.T).astype(np.float32)
    else:
        xy1 = xy0

    xy2 = xy1 / (inter + eps)  # scale so interocular=1

    face_h = _dist(xy2[LANDMARK_IDXS["forehead"]], xy2[LANDMARK_IDXS["chin"]])
    face_w = _dist(xy2[LANDMARK_IDXS["cheek_l"]], xy2[LANDMARK_IDXS["cheek_r"]])

    if (not np.isfinite(face_h)) or face_h < 0.2:
        return AlignResult(
            ok=False,
            aligned_xy=None,
            interocular=1.0,
            face_height=float(face_h),
            face_width=float(face_w),
            reject_reason="face_height_invalid",
        )

    return AlignResult(
        ok=True,
        aligned_xy=xy2.astype(np.float32),
        interocular=1.0,
        face_height=float(face_h),
        face_width=float(face_w),
        reject_reason=None,
    )


def build_cqs_geometry_descriptor_from_aligned(
    aligned_xy: np.ndarray,
    *,
    face_height: float,
    face_width: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Builds the fixed-order CQS geometry descriptor G (8-D) from aligned landmarks.
    Ordering matches GEOMETRY_FEATURE_NAMES.
    """
    I = LANDMARK_IDXS
    # Denominators in CQS baseline:
    interocular = 1.0  # by construction
    fh = float(face_height)
    fw = float(face_width)

    # Numerators
    face_w_over_h = _dist(aligned_xy[I["cheek_l"]], aligned_xy[I["cheek_r"]]) / (fh + eps)
    left_eye_w_over_ioc = _dist(aligned_xy[I["le_outer"]], aligned_xy[I["le_inner"]]) / (interocular + eps)
    right_eye_w_over_ioc = _dist(aligned_xy[I["re_outer"]], aligned_xy[I["re_inner"]]) / (interocular + eps)
    interocular_over_fw = _dist(aligned_xy[I["le_outer"]], aligned_xy[I["re_outer"]]) / (fw + eps)
    mouth_w_over_fw = _dist(aligned_xy[I["mouth_l"]], aligned_xy[I["mouth_r"]]) / (fw + eps)
    nose_wing_over_fw = _dist(aligned_xy[I["nose_l"]], aligned_xy[I["nose_r"]]) / (fw + eps)
    nose_len_over_fh = _dist(aligned_xy[I["nose_bridge"]], aligned_xy[I["nose_tip"]]) / (fh + eps)
    jaw_opening_angle = _angle_abc(aligned_xy[I["cheek_l"]], aligned_xy[I["chin"]], aligned_xy[I["cheek_r"]], eps=eps)

    G = np.array(
        [
            face_w_over_h,
            left_eye_w_over_ioc,
            right_eye_w_over_ioc,
            interocular_over_fw,
            mouth_w_over_fw,
            nose_wing_over_fw,
            nose_len_over_fh,
            jaw_opening_angle,
        ],
        dtype=np.float32,
    )
    return G


def build_cqs_geometry_descriptor(
    landmarks_xy_px: np.ndarray,
    *,
    allow_rotation: bool = True,
) -> Tuple[np.ndarray, AlignResult]:
    """
    Convenience wrapper: align landmarks then compute the 8-D CQS geometry descriptor.
    """
    ar = align_landmarks_similarity(landmarks_xy_px, allow_rotation=allow_rotation)
    if not ar.ok or ar.aligned_xy is None:
        raise ValueError(f"Alignment failed: {ar.reject_reason}")
    G = build_cqs_geometry_descriptor_from_aligned(
        ar.aligned_xy,
        face_height=ar.face_height,
        face_width=ar.face_width,
    )
    if not np.all(np.isfinite(G)):
        raise ValueError("Non-finite values in geometry descriptor G.")
    return G, ar


# -----------------------------
# Optional: landmark extraction (MediaPipe)
# -----------------------------

@dataclass(frozen=True)
class LandmarkResult:
    ok: bool
    conf: float
    landmarks_xy_px: Optional[np.ndarray]  # (468,2) in pixel coords
    img_w: int
    img_h: int
    err: str = ""


class MediaPipeFaceMesh468:
    """
    Minimal FaceMesh wrapper for 468 landmarks.

    Confidence:
      - If MediaPipe FaceDetection is available, uses its score
      - Otherwise falls back to 1.0 when landmarks exist
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe") from e

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Optional: a separate detector for a confidence scalar
        try:
            self._det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            self._det = None

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass
        try:
            if self._det is not None:
                self._det.close()
        except Exception:
            pass

    def __call__(self, img_rgb: np.ndarray) -> LandmarkResult:
        h, w = img_rgb.shape[:2]
        conf = 0.0
        try:
            if self._det is not None:
                det_res = self._det.process(img_rgb)
                if det_res and det_res.detections:
                    conf = float(det_res.detections[0].score[0])

            res = self._mesh.process(img_rgb)
            if not res.multi_face_landmarks:
                return LandmarkResult(ok=False, conf=conf, landmarks_xy_px=None, img_w=w, img_h=h, err="no_face")

            lm = res.multi_face_landmarks[0].landmark
            pts = np.zeros((len(lm), 2), dtype=np.float32)
            for i, p in enumerate(lm):
                pts[i, 0] = p.x * w
                pts[i, 1] = p.y * h

            if pts.shape[0] < 468:
                return LandmarkResult(
                    ok=False,
                    conf=conf,
                    landmarks_xy_px=None,
                    img_w=w,
                    img_h=h,
                    err=f"unexpected_landmark_count:{pts.shape[0]}",
                )

            # If we couldn't get a detector score, treat as moderately confident.
            if conf <= 0.0:
                conf = 0.5

            conf = float(max(0.0, min(1.0, conf)))
            return LandmarkResult(ok=True, conf=conf, landmarks_xy_px=pts[:468].copy(), img_w=w, img_h=h, err="")
        except Exception as e:
            return LandmarkResult(ok=False, conf=conf, landmarks_xy_px=None, img_w=w, img_h=h, err=str(e))


def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python not installed. Install with: pip install opencv-python") from e
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imread failed for: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def geometry_from_reference_image(
    image_path: Union[str, Path],
    *,
    allow_rotation: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Extract landmarks from a reference image and produce:
      - G: (D,) float32 CQS geometry descriptor
      - c: scalar confidence in [0,1]
    """
    rgb = load_image_rgb(image_path)
    fm = MediaPipeFaceMesh468(static_image_mode=True, max_num_faces=1)
    try:
        lr = fm(rgb)
    finally:
        fm.close()
    if not lr.ok or lr.landmarks_xy_px is None:
        raise RuntimeError(f"Landmark extraction failed: {lr.err}")
    P = lr.landmarks_xy_px.astype(np.float32)
    meta = {"conf_raw": float(lr.conf), "img_w": int(lr.img_w), "img_h": int(lr.img_h), "backend": "mediapipe_facemesh_468"}
    P_tilde, ar = normalize_landmarks(P, allow_rotation=allow_rotation)
    G = build_geometry_descriptor(P_tilde, align=ar)
    c = geometry_confidence(P, P_tilde, meta, align=ar)
    return G, c


# -----------------------------
# SDXL cross-attention dim reader
# -----------------------------

def read_sdxl_cross_attention_dim(model_root: Union[str, Path]) -> int:
    """
    Read SDXL UNet cross-attention dimension C from a local diffusers-style folder:
      <model_root>/unet/config.json

    Expected keys (diffusers):
      - "cross_attention_dim" OR "cross_attention_dim": [..] (older variants)
    """
    root = Path(model_root)
    cfg_path = root / "unet" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find UNet config at: {cfg_path}")
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "cross_attention_dim" not in obj:
        raise KeyError(f"'cross_attention_dim' missing in: {cfg_path}")
    val = obj["cross_attention_dim"]
    if isinstance(val, int):
        return int(val)
    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], int):
        # Some configs use lists for multi-attn dims; SDXL commonly uses a single int.
        return int(val[0])
    raise TypeError(f"Unsupported cross_attention_dim type: {type(val)}")


# -----------------------------
# Geometry token mapper (PyTorch)
# -----------------------------

def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: torch. Install with: pip install torch") from e
    return torch, nn


class GeometryTokenMapper:
    """
    Trainable mapping network f_theta: G -> T_geo.

    Shapes:
      - Input:  G ∈ R[B, D]
      - Output: T_geo ∈ R[B, N_geo, C]
    """

    def __init__(
        self,
        *,
        geom_dim: int,
        cross_attention_dim: int,
        n_geo_tokens: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 3,
        token_norm: str = "layernorm",  # "layernorm" or "none"
    ) -> None:
        torch, nn = _require_torch()
        if geom_dim <= 0:
            raise ValueError("geom_dim must be > 0")
        if cross_attention_dim <= 0:
            raise ValueError("cross_attention_dim must be > 0")
        if n_geo_tokens <= 0:
            raise ValueError("n_geo_tokens must be > 0")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.torch = torch
        self.nn = nn
        self.geom_dim = int(geom_dim)
        self.cross_attention_dim = int(cross_attention_dim)
        self.n_geo_tokens = int(n_geo_tokens)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.token_norm = str(token_norm)

        layers = []
        in_dim = self.geom_dim
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.n_geo_tokens * self.cross_attention_dim))
        self.net = nn.Sequential(*layers)

        if self.token_norm == "layernorm":
            self.norm = nn.LayerNorm(self.cross_attention_dim)
        elif self.token_norm == "none":
            self.norm = None
        else:
            raise ValueError("token_norm must be one of: layernorm, none")

    def parameters(self):
        return self.net.parameters()

    def state_dict(self) -> Dict[str, Any]:
        return self.net.state_dict() if self.norm is None else {"net": self.net.state_dict(), "norm": self.norm.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = True) -> None:
        if self.norm is None:
            self.net.load_state_dict(sd, strict=strict)
        else:
            if "net" not in sd:
                raise KeyError("Expected key 'net' in state_dict for mapper with norm.")
            self.net.load_state_dict(sd["net"], strict=strict)
            if "norm" in sd:
                self.norm.load_state_dict(sd["norm"], strict=strict)

    def eval(self) -> None:
        self.net.eval()
        if self.norm is not None:
            self.norm.eval()

    def train(self) -> None:
        self.net.train()
        if self.norm is not None:
            self.norm.train()

    def to(self, device: Union[str, "torch.device"]) -> "GeometryTokenMapper":
        self.net.to(device)
        if self.norm is not None:
            self.norm.to(device)
        return self

    def forward(
        self,
        G: "torch.Tensor",  # (B,D)
        *,
        lambda_geo: Union[float, "torch.Tensor"] = 1.0,
        conf: Optional[Union[float, "torch.Tensor"]] = None,
        token_scale: Optional[Union["torch.Tensor", Sequence[float]]] = None,
    ) -> "torch.Tensor":
        """
        Returns scaled tokens:
          λ_geo * c * T_geo  (and optional per-token scaling)
        """
        torch = self.torch
        if G.ndim != 2 or int(G.shape[1]) != self.geom_dim:
            raise ValueError(f"G must have shape (B,{self.geom_dim}); got {tuple(G.shape)}")
        B = int(G.shape[0])

        out = self.net(G)  # (B, N*C)
        T = out.view(B, self.n_geo_tokens, self.cross_attention_dim)  # (B,N,C)
        if self.norm is not None:
            T = self.norm(T)

        # Scaling controls
        if isinstance(lambda_geo, (float, int)):
            lam = torch.tensor(float(lambda_geo), device=T.device, dtype=T.dtype)
        else:
            lam = lambda_geo.to(device=T.device, dtype=T.dtype)

        if conf is None:
            c = torch.tensor(1.0, device=T.device, dtype=T.dtype)
        elif isinstance(conf, (float, int)):
            c = torch.tensor(float(conf), device=T.device, dtype=T.dtype)
        else:
            c = conf.to(device=T.device, dtype=T.dtype)

        # reshape broadcastables
        if lam.ndim == 0:
            lam = lam.view(1, 1, 1)
        elif lam.ndim == 1:
            lam = lam.view(-1, 1, 1)
        elif lam.ndim == 2:
            lam = lam.view(lam.shape[0], lam.shape[1], 1)

        if c.ndim == 0:
            c = c.view(1, 1, 1)
        elif c.ndim == 1:
            c = c.view(-1, 1, 1)
        elif c.ndim == 2:
            c = c.view(c.shape[0], c.shape[1], 1)

        # Optional per-token scaling (vector-based scaling)
        if token_scale is None:
            s = torch.ones((1, self.n_geo_tokens, 1), device=T.device, dtype=T.dtype)
        else:
            if not isinstance(token_scale, torch.Tensor):
                token_scale = torch.tensor(list(token_scale), device=T.device, dtype=T.dtype)
            else:
                token_scale = token_scale.to(device=T.device, dtype=T.dtype)

            if token_scale.ndim == 1:
                if int(token_scale.shape[0]) != self.n_geo_tokens:
                    raise ValueError(f"token_scale must have length N_geo={self.n_geo_tokens}; got {int(token_scale.shape[0])}")
                s = token_scale.view(1, self.n_geo_tokens, 1)
            elif token_scale.ndim == 2:
                # (B, N)
                if int(token_scale.shape[1]) != self.n_geo_tokens:
                    raise ValueError(f"token_scale second dim must be N_geo={self.n_geo_tokens}; got {int(token_scale.shape[1])}")
                s = token_scale.view(B, self.n_geo_tokens, 1)
            elif token_scale.ndim == 3:
                # (B, N, 1) or (B,N,C) allowed, will broadcast multiply
                s = token_scale
            else:
                raise ValueError(f"token_scale must be 1D/2D/3D; got ndim={token_scale.ndim}")

        T_scaled = T * lam * c * s
        return T_scaled

    # ---------- save/load ----------
    def save(self, path: Union[str, Path]) -> None:
        torch, _ = _require_torch()
        p = Path(path)
        obj = {
            "format": "GeoAdapterStep1.GeometryTokenMapper",
            "version": 1,
            "config": {
                "geom_dim": self.geom_dim,
                "cross_attention_dim": self.cross_attention_dim,
                "n_geo_tokens": self.n_geo_tokens,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "token_norm": self.token_norm,
            },
            "state_dict": self.state_dict(),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, str(p))

    @staticmethod
    def load(path: Union[str, Path], *, map_location: str = "cpu") -> "GeometryTokenMapper":
        torch, _ = _require_torch()
        p = Path(path)
        obj = torch.load(str(p), map_location=map_location)
        if not isinstance(obj, dict) or obj.get("format") != "GeoAdapterStep1.GeometryTokenMapper":
            raise ValueError(f"Unrecognized mapper checkpoint: {p}")
        cfg = obj["config"]
        mapper = GeometryTokenMapper(
            geom_dim=int(cfg["geom_dim"]),
            cross_attention_dim=int(cfg["cross_attention_dim"]),
            n_geo_tokens=int(cfg["n_geo_tokens"]),
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            token_norm=str(cfg.get("token_norm", "layernorm")),
        )
        mapper.load_state_dict(obj["state_dict"], strict=True)
        return mapper


def save_geo_adapter(mapper: GeometryTokenMapper, path: Union[str, Path]) -> None:
    """
    Step-1 API: save_geo_adapter(path)

    Saves ONLY the mapper weights + minimal config needed to restore it.
    """
    mapper.save(path)


def load_geo_adapter(path: Union[str, Path], *, map_location: str = "cpu") -> GeometryTokenMapper:
    """
    Step-1 API: load_geo_adapter(path) -> mapper

    Loads ONLY the mapper weights + config saved by save_geo_adapter().
    """
    return GeometryTokenMapper.load(path, map_location=map_location)


@dataclass
class GeometryTokenEncoder:
    """
    Convenience wrapper around:
      (landmarks ->) G,c  +  f_theta(G) -> T_geo  +  concat onto E_text.

    This keeps Step 1 easy to integrate without touching UNet internals.
    """

    mapper: GeometryTokenMapper

    def tokens_from_descriptor(
        self,
        G: "Any",  # torch.Tensor (B,D)
        *,
        lambda_geo: Union[float, "Any"] = 1.0,
        conf: Optional[Union[float, "Any"]] = None,
        token_scale: Optional[Union["Any", Sequence[float]]] = None,
    ) -> "Any":
        return self.mapper.forward(G, lambda_geo=lambda_geo, conf=conf, token_scale=token_scale)

    def concat_to_text(
        self,
        text_hidden_states: "Any",
        geo_tokens_scaled: "Any",
    ) -> "Any":
        return extend_text_conditioning(text_hidden_states, geo_tokens_scaled)

    def extend_conditioning(
        self,
        text_hidden_states: "Any",
        G: "Any",
        *,
        lambda_geo: Union[float, "Any"] = 1.0,
        conf: Optional[Union[float, "Any"]] = None,
        token_scale: Optional[Union["Any", Sequence[float]]] = None,
    ) -> "Any":
        T_scaled = make_geo_tokens(self.mapper, G, conf=conf, lambda_geo=lambda_geo, lambda_vec=token_scale)
        return concat_conditioning(text_hidden_states, T_scaled, drop_if_zero=True)


def make_geo_tokens(
    mapper: GeometryTokenMapper,
    G: "Any",
    *,
    conf: Optional[Union[float, "Any"]] = None,
    lambda_geo: Union[float, "Any"] = 1.0,
    lambda_vec: Optional[Union["Any", Sequence[float]]] = None,
) -> Optional["Any"]:
    """
    Step-1 API: make_geo_tokens(G, c, lambda_geo/lambda_vec) -> T_geo_scaled

    Returns:
      - T_geo_scaled: (B, N_geo, C) tensor, already scaled by (λ_geo * c) and optional per-token λ_vec.
      - None when geometry is effectively disabled (lambda_geo==0 and/or lambda_vec all zeros).
    """
    torch, _ = _require_torch()

    # Fast path: scalar disable
    if isinstance(lambda_geo, (float, int)) and float(lambda_geo) == 0.0:
        return None

    # Fast path: vector disable
    if lambda_vec is not None and isinstance(lambda_vec, (list, tuple)) and len(lambda_vec) > 0:
        try:
            if max(abs(float(x)) for x in lambda_vec) == 0.0:
                return None
        except Exception:
            pass

    T_scaled = mapper.forward(G, lambda_geo=lambda_geo, conf=conf, token_scale=lambda_vec)

    # If scaled tokens are numerically all-zero, treat as disabled.
    try:
        if float(torch.max(torch.abs(T_scaled)).item()) == 0.0:
            return None
    except Exception:
        pass

    return T_scaled


def concat_conditioning(
    E_text: "Any",
    T_geo_scaled: Optional["Any"],
    *,
    drop_if_zero: bool = True,
) -> "Any":
    """
    Step-1 API: concat_conditioning(E_text, T_geo_scaled) -> E_prime

    If drop_if_zero=True and geometry is disabled, returns E_text unchanged (shape: BxLxC).
    Otherwise concatenates along sequence length (shape: Bx(L+N_geo)xC).
    """
    if T_geo_scaled is None:
        return E_text
    if drop_if_zero:
        torch, _ = _require_torch()
        if isinstance(T_geo_scaled, torch.Tensor) and float(torch.max(torch.abs(T_geo_scaled)).item()) == 0.0:
            return E_text
    return extend_text_conditioning(E_text, T_geo_scaled)


def extend_text_conditioning(
    text_hidden_states: "Any",
    geo_tokens_scaled: "Any",
) -> "Any":
    """
    Concatenate geometry tokens to text encoder hidden states.

    Expected:
      - text_hidden_states: (B, N_text, C)
      - geo_tokens_scaled:  (B, N_geo,  C)
    Returns:
      - extended: (B, N_text + N_geo, C)
    """
    torch, _ = _require_torch()
    if not isinstance(text_hidden_states, torch.Tensor) or not isinstance(geo_tokens_scaled, torch.Tensor):
        raise TypeError("extend_text_conditioning expects torch.Tensor inputs.")
    if text_hidden_states.ndim != 3 or geo_tokens_scaled.ndim != 3:
        raise ValueError("Both inputs must be rank-3: (B,N,C).")
    if text_hidden_states.shape[0] != geo_tokens_scaled.shape[0]:
        raise ValueError("Batch dimension mismatch.")
    if text_hidden_states.shape[2] != geo_tokens_scaled.shape[2]:
        raise ValueError("Cross-attention dim mismatch (C).")
    return torch.cat([text_hidden_states, geo_tokens_scaled], dim=1)


# -----------------------------
# Self-tests (shape + determinism)
# -----------------------------

def run_shape_tests() -> None:
    # Always-run, torch-free checks: descriptor definition is stable and dimensionally correct.
    D = len(GEOMETRY_FEATURE_NAMES)
    assert D == 8, f"Expected CQS geometry descriptor dim D=8; got D={D}"
    assert len(set(GEOMETRY_FEATURE_NAMES)) == D, "Duplicate geometry feature names."
    for k in (
        "forehead",
        "chin",
        "cheek_l",
        "cheek_r",
        "le_outer",
        "le_inner",
        "re_outer",
        "re_inner",
        "mouth_l",
        "mouth_r",
        "nose_l",
        "nose_r",
        "nose_bridge",
        "nose_tip",
    ):
        assert k in LANDMARK_IDXS, f"Missing landmark index mapping: {k}"

    # Torch-dependent checks: G -> T_geo -> E' determinism and shapes.
    try:
        torch, _ = _require_torch()
    except Exception:
        # Keep self-test useful even without torch (common in this repo).
        # Full token-mapper tests will run once torch is installed.
        return

    C = 2048
    N = 4
    B = 2
    N_text = 77

    torch.manual_seed(123)
    mapper = GeometryTokenMapper(
        geom_dim=D,
        cross_attention_dim=C,
        n_geo_tokens=N,
        hidden_dim=128,
        n_layers=3,
        token_norm="layernorm",
    )
    mapper.eval()

    G = torch.randn(B, D, dtype=torch.float32)
    E_text = torch.randn(B, N_text, C, dtype=torch.float32)

    T1 = make_geo_tokens(mapper, G, conf=torch.tensor([1.0, 0.5]), lambda_geo=0.7, lambda_vec=None)
    T2 = make_geo_tokens(mapper, G, conf=torch.tensor([1.0, 0.5]), lambda_geo=0.7, lambda_vec=None)
    assert T1 is not None and T2 is not None, "Expected geometry tokens to be enabled for lambda_geo=0.7."

    assert tuple(T1.shape) == (B, N, C), f"Bad T shape: {tuple(T1.shape)}"
    assert torch.allclose(T1, T2), "Mapper is not deterministic in eval mode for identical inputs."

    E_ext = concat_conditioning(E_text, T1, drop_if_zero=True)
    assert tuple(E_ext.shape) == (B, N_text + N, C), f"Bad E' shape: {tuple(E_ext.shape)}"

    # Scaling sanity: lambda=0 should yield E' == E_text (no geometry effect)
    T0 = make_geo_tokens(mapper, G, conf=1.0, lambda_geo=0.0, lambda_vec=None)
    assert T0 is None, "lambda_geo=0 should disable geometry token output."
    E0 = concat_conditioning(E_text, T0, drop_if_zero=True)
    assert E0 is E_text or torch.allclose(E0, E_text), "lambda_geo=0 did not yield E' == E_text."

    # Bad shapes raise
    try:
        _ = mapper.forward(torch.randn(B, D + 1))
        raise AssertionError("Expected shape error for wrong G dim.")
    except ValueError:
        pass

    # Encoder wrapper path: G -> T -> E'
    enc = GeometryTokenEncoder(mapper=mapper)
    E_ext2 = enc.extend_conditioning(E_text, G, lambda_geo=0.7, conf=torch.tensor([1.0, 0.5]))
    assert torch.allclose(E_ext, E_ext2), "Encoder wrapper produced different E' vs manual path."

    # Save/load roundtrip reproducibility (mapper weights only)
    tmp_path = Path("._tmp_geo_adapter_step1.pt")
    try:
        save_geo_adapter(mapper, tmp_path)
        mapper2 = load_geo_adapter(tmp_path, map_location="cpu")
        mapper2.eval()
        T1b = make_geo_tokens(mapper2, G, conf=torch.tensor([1.0, 0.5]), lambda_geo=0.7, lambda_vec=None)
        assert T1b is not None
        assert torch.allclose(T1, T1b), "Tokens changed after save/load roundtrip."
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # py3.8+
        except TypeError:
            if tmp_path.exists():
                tmp_path.unlink()


# -----------------------------
# CLI
# -----------------------------

def _cmd_self_test(_: argparse.Namespace) -> int:
    run_shape_tests()
    try:
        _require_torch()
        print("[OK] GeoAdapterStep1 shape/determinism tests passed (including torch token-mapper checks).")
    except Exception:
        print("[OK] GeoAdapterStep1 basic shape tests passed (torch not installed; skipped token-mapper checks).")
    return 0


def _cmd_descriptor(args: argparse.Namespace) -> int:
    G, c = geometry_from_reference_image(args.image, allow_rotation=(not args.no_rotation))
    print("G (CQS geometry descriptor, 8-D):")
    for name, val in zip(GEOMETRY_FEATURE_NAMES, G.tolist()):
        print(f"  {name}: {val:.6f}")
    print(f"c (confidence): {c:.3f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GeoAdapterStep1: geometry descriptor -> SDXL cross-attn tokens.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("self-test", help="Run deterministic shape tests (requires torch).")
    s1.set_defaults(func=_cmd_self_test)

    s2 = sub.add_parser("descriptor", help="Extract CQS geometry descriptor G from a reference face image.")
    s2.add_argument("--image", type=str, required=True, help="Path to reference face image.")
    s2.add_argument("--no-rotation", action="store_true", help="Disable rotation in similarity alignment (not CQS-default).")
    s2.set_defaults(func=_cmd_descriptor)

    args = p.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


