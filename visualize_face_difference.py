#!/usr/bin/env python3
"""
Visualize differences between a random FFHQ face and the average face.

This script:
1) Selects a random face from FFHQ dataset
2) Detects landmarks and computes geometry features
3) Compares with average face statistics
4) Visualizes differences with thick lines showing deviations
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    raise RuntimeError("mediapipe is not installed. Install with: pip install mediapipe")


# MediaPipe FaceMesh landmark indices (from step1.py)
LANDMARK_INDICES = {
    "forehead": 10,
    "nose_bridge": 168,
    "nose_tip": 1,
    "nose_l": 94,
    "nose_r": 331,
    "chin": 152,
    "le_outer": 33,
    "le_inner": 133,
    "re_outer": 263,
    "re_inner": 362,
    "mouth_l": 61,
    "mouth_r": 291,
    "cheek_l": 234,
    "cheek_r": 454,
}


def load_geometry_stats(stats_path: Path) -> Dict:
    """Load geometry statistics from JSON file."""
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_geometry_schema() -> List[Dict]:
    """Load default geometry schema from step1.py."""
    # This matches step1.py's default_geometry_schema_mediapipe468()
    return [
        {"name": "face_width_over_height", "kind": "ratio", "a": 234, "b": 454, "denom": "face_height"},
        {"name": "left_eye_width_over_interocular", "kind": "ratio", "a": 33, "b": 133, "denom": "interocular"},
        {"name": "right_eye_width_over_interocular", "kind": "ratio", "a": 263, "b": 362, "denom": "interocular"},
        {"name": "interocular_over_face_width", "kind": "ratio", "a": 33, "b": 263, "denom": "face_width"},
        {"name": "mouth_width_over_face_width", "kind": "ratio", "a": 61, "b": 291, "denom": "face_width"},
        {"name": "nose_wing_width_over_face_width", "kind": "ratio", "a": 94, "b": 331, "denom": "face_width"},
        {"name": "nose_length_over_face_height", "kind": "ratio", "a": 168, "b": 1, "denom": "face_height"},
        {"name": "jaw_opening_angle", "kind": "angle", "a": 234, "b": 152, "c": 454},
    ]


class FaceMeshGeometry:
    """MediaPipe FaceMesh geometry calculator (from step4.py)."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.idx = LANDMARK_INDICES

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


def compute_geometry_features(lm_aligned: np.ndarray, schema: List[Dict], eps: float = 1e-12) -> Dict[str, float]:
    """Compute geometry features from aligned landmarks (from step1.py)."""
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))
    
    def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < eps or n2 < eps:
            return float("nan")
        cosv = float(np.dot(v1, v2) / (n1 * n2 + eps))
        cosv = max(-1.0, min(1.0, cosv))
        return float(math.acos(cosv))
    
    # Compute denominators
    face_height = dist(lm_aligned[LANDMARK_INDICES["forehead"]], lm_aligned[LANDMARK_INDICES["chin"]])
    face_width = dist(lm_aligned[LANDMARK_INDICES["cheek_l"]], lm_aligned[LANDMARK_INDICES["cheek_r"]])
    interocular = 1.0  # By construction in aligned space
    
    features = {}
    for feat in schema:
        name = feat["name"]
        kind = feat["kind"]
        
        if kind == "ratio":
            a_idx = feat["a"]
            b_idx = feat["b"]
            num = dist(lm_aligned[a_idx], lm_aligned[b_idx])
            denom_name = feat.get("denom", "interocular")
            
            if denom_name == "interocular":
                den = interocular
            elif denom_name == "face_height":
                den = face_height
            elif denom_name == "face_width":
                den = face_width
            else:
                den = 1.0
            
            features[name] = float(num / (den + eps))
        
        elif kind == "angle":
            a_idx = feat["a"]
            b_idx = feat["b"]
            c_idx = feat.get("c")
            if c_idx is None:
                features[name] = float("nan")
            else:
                ang = angle_abc(lm_aligned[a_idx], lm_aligned[b_idx], lm_aligned[c_idx])
                features[name] = float(ang)
    
    return features


def visualize_differences(
    img_rgb: np.ndarray,
    landmarks: np.ndarray,
    landmarks_aligned: np.ndarray,
    features: Dict[str, float],
    stats_mean: Dict[str, float],
    stats_std: Dict[str, float],
    output_path: Path,
) -> None:
    """Visualize differences from average face with thick lines."""
    
    # Calculate differences (in standard deviations)
    differences = {}
    for name in features:
        if name in stats_mean and name in stats_std:
            diff_std = (features[name] - stats_mean[name]) / (stats_std[name] + 1e-12)
            differences[name] = diff_std
    
    # Create visualization
    img_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    # Color mapping: red for larger than average, blue for smaller
    def get_color(diff_std: float) -> Tuple[int, int, int]:
        """Get color based on deviation: red for positive, blue for negative."""
        # Make colors more vibrant and visible
        intensity = min(abs(diff_std) * 0.4 + 0.6, 1.0)  # Scale to 0.6-1.0 for better visibility
        if diff_std > 0:
            # Red for larger than average (BGR format)
            return (0, 0, int(255 * intensity))
        else:
            # Blue for smaller than average (BGR format)
            return (int(255 * intensity), 0, 0)
    
    # Draw differences with very thick lines
    line_thickness = 15  # Very thick lines for visibility
    
    # Face width/height
    if "face_width_over_height" in differences:
        diff = differences["face_width_over_height"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["cheek_l"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["cheek_r"]].astype(int)),
                color, line_thickness)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["forehead"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["chin"]].astype(int)),
                color, line_thickness)
    
    # Eyes
    if "left_eye_width_over_interocular" in differences:
        diff = differences["left_eye_width_over_interocular"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["le_outer"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["le_inner"]].astype(int)),
                color, line_thickness)
    
    if "right_eye_width_over_interocular" in differences:
        diff = differences["right_eye_width_over_interocular"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["re_inner"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["re_outer"]].astype(int)),
                color, line_thickness)
    
    if "interocular_over_face_width" in differences:
        diff = differences["interocular_over_face_width"]
        color = get_color(diff)
        le_center = 0.5 * (landmarks[LANDMARK_INDICES["le_outer"]] + landmarks[LANDMARK_INDICES["le_inner"]])
        re_center = 0.5 * (landmarks[LANDMARK_INDICES["re_outer"]] + landmarks[LANDMARK_INDICES["re_inner"]])
        cv2.line(img_bgr,
                tuple(le_center.astype(int)),
                tuple(re_center.astype(int)),
                color, line_thickness)
    
    # Nose
    if "nose_length_over_face_height" in differences:
        diff = differences["nose_length_over_face_height"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["nose_bridge"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["nose_tip"]].astype(int)),
                color, line_thickness)
    
    if "nose_wing_width_over_face_width" in differences:
        diff = differences["nose_wing_width_over_face_width"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["nose_l"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["nose_r"]].astype(int)),
                color, line_thickness)
    
    # Mouth
    if "mouth_width_over_face_width" in differences:
        diff = differences["mouth_width_over_face_width"]
        color = get_color(diff)
        cv2.line(img_bgr,
                tuple(landmarks[LANDMARK_INDICES["mouth_l"]].astype(int)),
                tuple(landmarks[LANDMARK_INDICES["mouth_r"]].astype(int)),
                color, line_thickness)
    
    # Jaw angle (draw triangle)
    if "jaw_opening_angle" in differences:
        diff = differences["jaw_opening_angle"]
        color = get_color(diff)
        pts = np.array([
            landmarks[LANDMARK_INDICES["cheek_l"]].astype(int),
            landmarks[LANDMARK_INDICES["chin"]].astype(int),
            landmarks[LANDMARK_INDICES["cheek_r"]].astype(int),
        ], dtype=np.int32)
        cv2.polylines(img_bgr, [pts], True, color, line_thickness)
    
    # Draw landmarks as large dots
    landmark_radius = 12
    for idx in LANDMARK_INDICES.values():
        pt = landmarks[idx].astype(int)
        cv2.circle(img_bgr, tuple(pt), landmark_radius, (0, 255, 0), -1)
        cv2.circle(img_bgr, tuple(pt), landmark_radius + 2, (0, 0, 0), 2)
    
    # Convert back to RGB
    img_vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create figure with statistics
    fig = plt.figure(figsize=(16, 12))
    
    # Image visualization
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(img_vis)
    ax1.set_title("Face Differences from Average (Red = Larger, Blue = Smaller)", 
                  fontsize=18, fontweight='bold')
    ax1.axis('off')
    
    # Statistics table
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(["Feature", "This Face", "Average", "Difference (σ)", "Status"])
    
    for name in sorted(features.keys()):
        if name in stats_mean:
            this_val = features[name]
            avg_val = stats_mean[name]
            std_val = stats_std.get(name, 1.0)
            diff_std = (this_val - avg_val) / (std_val + 1e-12)
            
            if abs(diff_std) < 0.5:
                status = "Normal"
            elif abs(diff_std) < 1.5:
                status = "Moderate"
            elif abs(diff_std) < 2.5:
                status = "Large"
            else:
                status = "Extreme"
            
            table_data.append([
                name.replace("_", " ").title(),
                f"{this_val:.4f}",
                f"{avg_val:.4f}",
                f"{diff_std:+.2f}σ",
                status
            ])
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='left', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code differences
    for i in range(1, len(table_data)):
        diff_std = float(table_data[i][3].replace('σ', ''))
        if abs(diff_std) > 2.0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#ffcccc')
        elif abs(diff_std) > 1.0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#ffffcc')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize differences between a random FFHQ face and average face"
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to geometry_stats.json from baseline run"
    )
    parser.add_argument(
        "--baseline-index",
        type=str,
        default="",
        help="Path to baseline_index.csv (optional, for selecting from processed images)"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Specific image path (optional, overrides random selection)"
    )
    parser.add_argument(
        "--ffhq-root",
        type=str,
        default="ffhq_1024/images1024x1024",
        help="Root directory for FFHQ images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="face_difference_visualization.png",
        help="Output image path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load statistics
    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        print(f"Error: Statistics file not found: {stats_path}", file=sys.stderr)
        return 1
    
    print(f"Loading statistics from: {stats_path}")
    stats = load_geometry_stats(stats_path)
    stats_mean = dict(zip(stats["feature_names"], stats["mean"]))
    stats_std = dict(zip(stats["feature_names"], stats.get("std", [1.0] * len(stats["mean"]))))
    
    # Select image
    if args.image_path:
        img_path = Path(args.image_path)
    elif args.baseline_index:
        index_path = Path(args.baseline_index)
        if not index_path.exists():
            print(f"Error: Baseline index not found: {index_path}", file=sys.stderr)
            return 1
        df = pd.read_csv(index_path)
        if df.empty:
            print("Error: Baseline index is empty", file=sys.stderr)
            return 1
        random_row = df.sample(n=1).iloc[0]
        img_path = Path(random_row["path"])
        print(f"Selected random image: {img_path}")
    else:
        # Select random image from FFHQ directory
        ffhq_root = Path(args.ffhq_root)
        if not ffhq_root.exists():
            print(f"Error: FFHQ root directory not found: {ffhq_root}", file=sys.stderr)
            return 1
        
        # Find all PNG files
        png_files = list(ffhq_root.rglob("*.png"))
        if not png_files:
            print(f"Error: No PNG files found in {ffhq_root}", file=sys.stderr)
            return 1
        
        img_path = random.choice(png_files)
        print(f"Selected random image: {img_path}")
    
    if not img_path.exists():
        print(f"Error: Image file not found: {img_path}", file=sys.stderr)
        return 1
    
    # Load image
    print(f"Loading image: {img_path}")
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)
    
    # Detect landmarks
    print("Detecting landmarks...")
    geom = FaceMeshGeometry()
    landmarks, conf = geom.detect_landmarks(img_rgb)
    if landmarks is None:
        print("Error: No landmarks detected", file=sys.stderr)
        return 1
    
    print(f"Landmarks detected: {len(landmarks)} points, confidence: {conf:.3f}")
    
    # Align landmarks
    landmarks_aligned, _, _, _ = geom.align(landmarks)
    
    # Compute geometry features
    print("Computing geometry features...")
    schema = load_geometry_schema()
    features = compute_geometry_features(landmarks_aligned, schema)
    
    print("\nFeature comparison:")
    for name in sorted(features.keys()):
        if name in stats_mean:
            this_val = features[name]
            avg_val = stats_mean[name]
            diff_std = (this_val - avg_val) / (stats_std.get(name, 1.0) + 1e-12)
            print(f"  {name:35s}: {this_val:8.4f} (avg: {avg_val:8.4f}, diff: {diff_std:+6.2f}σ)")
    
    # Visualize
    print("\nCreating visualization...")
    visualize_differences(
        img_rgb,
        landmarks,
        landmarks_aligned,
        features,
        stats_mean,
        stats_std,
        Path(args.output)
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

