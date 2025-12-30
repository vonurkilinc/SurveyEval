#!/usr/bin/env python3
"""
Visualize landmarks and geometry on P001 reference and caricature images.
Uses exact calculations from step3.py and step4.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    raise RuntimeError("mediapipe is not installed. Install with: pip install mediapipe")


# -------------------------
# MediaPipe FaceMesh (from step3.py and step4.py)
# -------------------------

class MediaPipeFaceMesh:
    """
    Uses mediapipe FaceMesh with refine_landmarks=True.
    Produces 468 landmarks in normalized coords -> converted to pixel coords.
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1) -> None:
        self.mp = mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_landmarks(self, img_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        h, w = img_rgb.shape[:2]
        res = self.face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None, 0.0

        lm = res.multi_face_landmarks[0].landmark  # 468
        pts = np.zeros((len(lm), 2), dtype=np.float32)
        for i, p in enumerate(lm):
            pts[i, 0] = p.x * w
            pts[i, 1] = p.y * h

        return pts, 1.0


# -------------------------
# Geometry calculations (from step3.py)
# -------------------------

def similarity_align_dense_landmarks(pts_xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple similarity alignment using two points: approximate left/right eye centers.
    Exact copy from step3.py
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


# -------------------------
# Geometry descriptor (from step4.py)
# -------------------------

class GeoConfig:
    eps: float = 1e-8
    sigma_sym: float = 0.06
    sigma_range: float = 0.35


class FaceMeshGeometry:
    """
    Exact copy from step4.py for geometry calculations
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1):
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


# -------------------------
# Visualization functions
# -------------------------

def draw_landmarks(img: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
                   radius: int = 2) -> np.ndarray:
    """Draw all landmarks as small circles"""
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    color_bgr = (color[2], color[1], color[0])  # RGB to BGR
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_bgr, (x, y), radius, color_bgr, -1)
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_key_landmarks(img: np.ndarray, landmarks: np.ndarray, idx: Dict[str, int], 
                       color: Tuple[int, int, int] = (255, 0, 0), radius: int = 6) -> np.ndarray:
    """Draw key landmarks used in geometry calculations"""
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    color_bgr = (color[2], color[1], color[0])  # RGB to BGR
    for name, i in idx.items():
        pt = landmarks[i]
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_bgr, (x, y), radius, color_bgr, -1)
        # Add label
        cv2.putText(img_bgr, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1)
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_geometry_features(img: np.ndarray, landmarks: np.ndarray, idx: Dict[str, int],
                           geom: FaceMeshGeometry, cfg: GeoConfig) -> np.ndarray:
    """Draw geometry measurements (distances and angles) on image"""
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    
    # Align landmarks for geometry calculations
    lm_aligned, _, _, center = geom.align(landmarks)
    
    # Convert aligned landmarks back to image coordinates for visualization
    # We'll use the original landmarks for drawing, but calculate geometry from aligned
    
    def d(a, b) -> float:
        return float(np.linalg.norm(lm_aligned[idx[a]] - lm_aligned[idx[b]]) + cfg.eps)
    
    # Draw key distances
    def draw_line(name_a: str, name_b: str, color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 4):
        pt_a = landmarks[idx[name_a]]
        pt_b = landmarks[idx[name_b]]
        color_bgr = (color[2], color[1], color[0])  # RGB to BGR
        cv2.line(img_bgr, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])), color_bgr, thickness)
        # Draw distance label at midpoint
        mid_x = int((pt_a[0] + pt_b[0]) / 2)
        mid_y = int((pt_a[1] + pt_b[1]) / 2)
        dist = d(name_a, name_b)
        cv2.putText(img_bgr, f"{dist:.3f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
    
    # Draw face width (cheek to cheek)
    draw_line("cheek_l", "cheek_r", (255, 255, 0), 4)
    
    # Draw face height (forehead to chin)
    draw_line("forehead", "chin", (255, 0, 255), 4)
    
    # Draw jaw width
    draw_line("jaw_l", "jaw_r", (0, 255, 255), 4)
    
    # Draw eye widths
    draw_line("le_outer", "le_inner", (0, 255, 0), 2)
    draw_line("re_outer", "re_inner", (0, 255, 0), 2)
    
    # Draw inter-ocular distance (eye centers)
    le_center = 0.5 * (landmarks[idx["le_outer"]] + landmarks[idx["le_inner"]])
    re_center = 0.5 * (landmarks[idx["re_outer"]] + landmarks[idx["re_inner"]])
    cv2.line(img_bgr, (int(le_center[0]), int(le_center[1])), 
             (int(re_center[0]), int(re_center[1])), (0, 0, 255), 4)  # BGR: red
    # Calculate IOD from aligned landmarks (more accurate)
    le_center_aligned = 0.5 * (lm_aligned[idx["le_outer"]] + lm_aligned[idx["le_inner"]])
    re_center_aligned = 0.5 * (lm_aligned[idx["re_outer"]] + lm_aligned[idx["re_inner"]])
    iod_dist = float(np.linalg.norm(re_center_aligned - le_center_aligned) + cfg.eps)
    mid_iod_x = int((le_center[0] + re_center[0]) / 2)
    mid_iod_y = int((le_center[1] + re_center[1]) / 2)
    cv2.putText(img_bgr, f"IOD: {iod_dist:.3f}", (mid_iod_x, mid_iod_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # BGR: red
    
    # Draw nose width
    draw_line("nose_l", "nose_r", (255, 165, 0), 2)
    
    # Draw nose length
    draw_line("nose_bridge", "nose_tip", (255, 165, 0), 2)
    
    # Draw mouth width
    draw_line("mouth_l", "mouth_r", (0, 0, 255), 2)
    
    # Draw mouth to chin
    draw_line("mouth_l", "chin", (0, 0, 255), 2)
    
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def print_geometry_descriptor(G: np.ndarray):
    """Print geometry descriptor values"""
    feature_names = [
        "face_w",
        "face_h",
        "jaw_w",
        "eye_w / face_w",
        "iod / face_w",
        "nose_w / face_w",
        "nose_len / face_h",
        "mouth_w / face_w",
        "mouth_to_chin / face_h",
        "jaw_angle / π",
        "nose_angle / π",
    ]
    print("\nGeometry Descriptor:")
    print("-" * 50)
    for name, value in zip(feature_names, G):
        print(f"  {name:25s}: {value:.6f}")


def visualize_image(img_path: Path, output_path: Path, title: str = ""):
    """Load image, detect landmarks, calculate geometry, and visualize"""
    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)
    
    # Initialize detectors
    mp_detector = MediaPipeFaceMesh()
    geom = FaceMeshGeometry()
    cfg = GeoConfig()
    
    # Detect landmarks
    landmarks, conf = mp_detector.detect_landmarks(img_rgb)
    if landmarks is None:
        print(f"Warning: No landmarks detected in {img_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    print(f"Landmark confidence: {conf:.3f}")
    print(f"Number of landmarks: {len(landmarks)}")
    
    # Calculate alignment (from step3.py)
    aligned_landmarks, align_params = similarity_align_dense_landmarks(landmarks)
    print(f"\nAlignment parameters:")
    print(f"  Eye distance (pixels): {align_params['eye_dist_px']:.2f}")
    print(f"  Eye distance (aligned): {align_params['eye_dist_aligned']:.2f}")
    print(f"  Scale: {align_params['scale']:.6f}")
    print(f"  Rotation (rad): {align_params['rot_rad']:.6f}")
    
    # Calculate geometry descriptor (from step4.py)
    lm_aligned, _, _, _ = geom.align(landmarks)
    G = geom.build_descriptor(lm_aligned, cfg)
    print_geometry_descriptor(G)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Image with all landmarks
    img_landmarks = draw_landmarks(img_rgb, landmarks, color=(0, 255, 0), radius=4)
    axes[1].imshow(img_landmarks)
    axes[1].set_title("All Landmarks (468 points)", fontsize=12)
    axes[1].axis('off')
    
    # Image with key landmarks and geometry
    img_geometry = draw_geometry_features(img_rgb, landmarks, geom.idx, geom, cfg)
    img_geometry = draw_key_landmarks(img_geometry, landmarks, geom.idx, color=(255, 0, 0), radius=6)
    axes[2].imshow(img_geometry)
    axes[2].set_title("Key Landmarks + Geometry Measurements", fontsize=12)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main():
    """Main function to visualize P001 images"""
    img_root = Path("img/P001")
    
    if not img_root.exists():
        print(f"Error: Directory {img_root} not found")
        return
    
    # Reference image
    ref_path = img_root / "reference.jpg"
    if ref_path.exists():
        visualize_image(ref_path, Path("p001_reference_visualization.png"), 
                       "P001 Reference Image - Landmarks & Geometry")
    
    # Caricature images (one from each method)
    caricature_paths = [
        (img_root / "instantid" / "v1.jpg", "P001 Caricature (InstantID v1)"),
        (img_root / "pulid" / "v1.jpg", "P001 Caricature (PulID v1)"),
        (img_root / "qwen" / "v1.jpg", "P001 Caricature (Qwen v1)"),
    ]
    
    for car_path, title in caricature_paths:
        if car_path.exists():
            output_name = f"p001_{car_path.parent.name}_{car_path.stem}_visualization.png"
            visualize_image(car_path, Path(output_name), title)
        else:
            print(f"Warning: {car_path} not found")
    
    print("\n" + "="*60)
    print("Visualization complete!")


if __name__ == "__main__":
    main()

