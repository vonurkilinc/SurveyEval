#!/usr/bin/env python3
"""
Visualize the average human face reconstructed from baseline statistics.

This script:
1) Loads geometry_stats.json from a baseline run
2) Reconstructs key landmark positions using the mean statistics
3) Visualizes the average face geometry
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def reconstruct_average_landmarks(stats: Dict) -> np.ndarray:
    """
    Reconstruct average landmark positions in aligned space from statistics.
    
    In aligned space:
    - Interocular distance = 1.0 (by construction)
    - Eye centers are at (-0.5, 0) and (0.5, 0)
    - Origin is at eye midpoint
    
    We use the mean statistics to determine:
    - face_height and face_width
    - Positions of other key landmarks
    """
    feature_names = stats["feature_names"]
    mean_values = stats["mean"]
    
    # Create feature dictionary
    features = dict(zip(feature_names, mean_values))
    
    # Initialize landmark array (468 landmarks, but we'll only set key ones)
    landmarks = np.full((468, 2), np.nan, dtype=np.float32)
    
    # In aligned space, interocular = 1.0
    # Eye centers are at (-0.5, 0) and (0.5, 0)
    left_eye_center = np.array([-0.5, 0.0], dtype=np.float32)
    right_eye_center = np.array([0.5, 0.0], dtype=np.float32)
    
    # Calculate face_width from interocular_over_face_width ratio
    # interocular_over_face_width = IOD / face_width = 1.0 / face_width
    if "interocular_over_face_width" in features:
        interocular_over_face_width = features["interocular_over_face_width"]
        face_width = 1.0 / interocular_over_face_width
    else:
        # Fallback: estimate from typical human proportions
        face_width = 1.5
    
    # Calculate face_height from face_width_over_height ratio
    # face_width_over_height = face_width / face_height
    if "face_width_over_height" in features:
        face_width_over_height = features["face_width_over_height"]
        face_height = face_width / face_width_over_height
    else:
        # Fallback: estimate from typical human proportions
        face_height = face_width / 0.84
    
    # Place cheek landmarks (face width)
    # Use jaw_opening_angle if available to position cheeks relative to chin
    if "jaw_opening_angle" in features:
        jaw_angle = features["jaw_opening_angle"]  # angle at chin between cheeks (in radians)
        # We'll position cheeks symmetrically around the chin
        # The angle gives us the opening, so half-angle on each side
        half_angle = jaw_angle / 2
        # Distance from chin to cheek (approximate from face width)
        cheek_dist = face_width / 2
        # Position cheeks relative to chin
        # Chin is at (0, chin_y), cheeks are at angles ±half_angle from vertical
        # We need to compute this after chin is placed, so we'll update later
        # For now, place them horizontally aligned with eye level
        landmarks[LANDMARK_INDICES["cheek_l"]] = np.array([-face_width / 2, 0.0], dtype=np.float32)
        landmarks[LANDMARK_INDICES["cheek_r"]] = np.array([face_width / 2, 0.0], dtype=np.float32)
    else:
        landmarks[LANDMARK_INDICES["cheek_l"]] = np.array([-face_width / 2, 0.0], dtype=np.float32)
        landmarks[LANDMARK_INDICES["cheek_r"]] = np.array([face_width / 2, 0.0], dtype=np.float32)
    
    # Place forehead and chin (face height)
    # Typically forehead is above eyes, chin is below
    # From typical proportions: forehead ~0.3*face_height above eye center, chin ~0.7*face_height below
    eye_y = 0.0
    forehead_y = eye_y - 0.3 * face_height
    chin_y = eye_y + 0.7 * face_height
    
    landmarks[LANDMARK_INDICES["forehead"]] = np.array([0.0, forehead_y], dtype=np.float32)
    landmarks[LANDMARK_INDICES["chin"]] = np.array([0.0, chin_y], dtype=np.float32)
    
    # Place eye corners
    # left_eye_width_over_interocular gives us the eye width
    if "left_eye_width_over_interocular" in features:
        left_eye_width = features["left_eye_width_over_interocular"]  # relative to IOD=1.0
        left_eye_half_width = left_eye_width / 2
        landmarks[LANDMARK_INDICES["le_outer"]] = left_eye_center + np.array([-left_eye_half_width, 0.0])
        landmarks[LANDMARK_INDICES["le_inner"]] = left_eye_center + np.array([left_eye_half_width, 0.0])
    else:
        landmarks[LANDMARK_INDICES["le_outer"]] = left_eye_center + np.array([-0.22, 0.0])
        landmarks[LANDMARK_INDICES["le_inner"]] = left_eye_center + np.array([0.22, 0.0])
    
    if "right_eye_width_over_interocular" in features:
        right_eye_width = features["right_eye_width_over_interocular"]
        right_eye_half_width = right_eye_width / 2
        landmarks[LANDMARK_INDICES["re_inner"]] = right_eye_center + np.array([-right_eye_half_width, 0.0])
        landmarks[LANDMARK_INDICES["re_outer"]] = right_eye_center + np.array([right_eye_half_width, 0.0])
    else:
        landmarks[LANDMARK_INDICES["re_inner"]] = right_eye_center + np.array([-0.22, 0.0])
        landmarks[LANDMARK_INDICES["re_outer"]] = right_eye_center + np.array([0.22, 0.0])
    
    # Place nose
    # nose_length_over_face_height gives us nose length
    if "nose_length_over_face_height" in features:
        nose_length = features["nose_length_over_face_height"] * face_height
        # Nose bridge is typically around 0.1*face_height below eye center
        nose_bridge_y = eye_y + 0.1 * face_height
        nose_tip_y = nose_bridge_y + nose_length
        landmarks[LANDMARK_INDICES["nose_bridge"]] = np.array([0.0, nose_bridge_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["nose_tip"]] = np.array([0.0, nose_tip_y], dtype=np.float32)
    else:
        nose_bridge_y = eye_y + 0.1 * face_height
        nose_tip_y = nose_bridge_y + 0.26 * face_height
        landmarks[LANDMARK_INDICES["nose_bridge"]] = np.array([0.0, nose_bridge_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["nose_tip"]] = np.array([0.0, nose_tip_y], dtype=np.float32)
    
    # Nose wing width
    if "nose_wing_width_over_face_width" in features:
        nose_wing_width = features["nose_wing_width_over_face_width"] * face_width
        nose_wing_half_width = nose_wing_width / 2
        # Place nose wings at approximately nose tip level
        nose_wing_y = landmarks[LANDMARK_INDICES["nose_tip"]][1]
        landmarks[LANDMARK_INDICES["nose_l"]] = np.array([-nose_wing_half_width, nose_wing_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["nose_r"]] = np.array([nose_wing_half_width, nose_wing_y], dtype=np.float32)
    else:
        nose_wing_y = landmarks[LANDMARK_INDICES["nose_tip"]][1]
        nose_wing_half_width = 0.08 * face_width
        landmarks[LANDMARK_INDICES["nose_l"]] = np.array([-nose_wing_half_width, nose_wing_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["nose_r"]] = np.array([nose_wing_half_width, nose_wing_y], dtype=np.float32)
    
    # Place mouth
    # mouth_width_over_face_width gives us mouth width
    if "mouth_width_over_face_width" in features:
        mouth_width = features["mouth_width_over_face_width"] * face_width
        mouth_half_width = mouth_width / 2
        # Mouth is typically around 0.4*face_height below eye center
        mouth_y = eye_y + 0.4 * face_height
        landmarks[LANDMARK_INDICES["mouth_l"]] = np.array([-mouth_half_width, mouth_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["mouth_r"]] = np.array([mouth_half_width, mouth_y], dtype=np.float32)
    else:
        mouth_y = eye_y + 0.4 * face_height
        mouth_half_width = 0.2 * face_width
        landmarks[LANDMARK_INDICES["mouth_l"]] = np.array([-mouth_half_width, mouth_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["mouth_r"]] = np.array([mouth_half_width, mouth_y], dtype=np.float32)
    
    # Refine cheek positions using jaw_opening_angle if available
    # The angle is at chin (152) between cheek_l (234) and cheek_r (454)
    # Note: Cheeks are at the sides of the face, typically at mouth level or slightly higher,
    # NOT below the chin. The jaw_opening_angle determines the horizontal spread.
    if "jaw_opening_angle" in features:
        jaw_angle = features["jaw_opening_angle"]  # radians
        chin_pos = landmarks[LANDMARK_INDICES["chin"]]
        mouth_y = landmarks[LANDMARK_INDICES["mouth_l"]][1]
        
        # Cheeks are positioned at approximately mouth level or slightly higher
        # Use the jaw angle to determine horizontal spread from center
        # The angle is measured at the chin, so we can use it to determine
        # how far apart the cheeks are horizontally relative to the chin
        half_angle = jaw_angle / 2
        
        # Distance from chin to cheek horizontally (not vertically)
        # Use the face_width and angle to determine cheek positions
        # The cheeks should be at the sides of the face, so use face_width/2
        # but adjust based on the angle to maintain the geometric relationship
        cheek_half_width = face_width / 2
        
        # Position cheeks at mouth level (or slightly higher) horizontally
        # The jaw angle tells us the opening, so we can compute the horizontal distance
        # from chin to cheek using trigonometry
        # If angle is measured at chin, and cheeks are at mouth level:
        # tan(half_angle) = horizontal_distance / vertical_distance
        vertical_dist_to_mouth = abs(chin_pos[1] - mouth_y)
        if vertical_dist_to_mouth > 0.01:  # Avoid division by zero
            horizontal_offset = vertical_dist_to_mouth * math.tan(half_angle)
            # But cheeks should be at face_width/2 from center, not from chin
            # So we use the face_width directly
            left_cheek_x = -cheek_half_width
            right_cheek_x = cheek_half_width
        else:
            left_cheek_x = -cheek_half_width
            right_cheek_x = cheek_half_width
        
        # Position cheeks at mouth level or slightly higher (around 0.05*face_height above mouth)
        cheek_y = mouth_y - 0.05 * face_height
        
        landmarks[LANDMARK_INDICES["cheek_l"]] = np.array([left_cheek_x, cheek_y], dtype=np.float32)
        landmarks[LANDMARK_INDICES["cheek_r"]] = np.array([right_cheek_x, cheek_y], dtype=np.float32)
    
    return landmarks, face_width, face_height


def draw_average_face(landmarks: np.ndarray, face_width: float, face_height: float, 
                      stats: Dict, output_size: Tuple[int, int] = (800, 1000)) -> np.ndarray:
    """
    Draw the average face visualization.
    
    Returns an RGB image array.
    """
    # Create a white canvas
    img = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255
    
    # Scale landmarks to fit the canvas
    # Find bounding box of valid landmarks
    valid_landmarks = landmarks[~np.isnan(landmarks).any(axis=1)]
    if len(valid_landmarks) == 0:
        return img
    
    min_x, min_y = valid_landmarks.min(axis=0)
    max_x, max_y = valid_landmarks.max(axis=0)
    
    # Add padding
    padding = 0.1
    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x -= padding * range_x
    max_x += padding * range_x
    min_y -= padding * range_y
    max_y += padding * range_y
    
    # Scale to fit canvas
    scale_x = (output_size[0] - 40) / (max_x - min_x)
    scale_y = (output_size[1] - 40) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    
    # Center offset
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    offset_x = output_size[0] / 2 - center_x * scale
    offset_y = output_size[1] / 2 - center_y * scale
    
    # Convert to BGR for OpenCV
    img_bgr = img.copy()
    
    # Draw face outline (ellipse approximation)
    face_center_x = int(center_x * scale + offset_x)
    face_center_y = int(center_y * scale + offset_y)
    face_ellipse_w = int(face_width * scale)
    face_ellipse_h = int(face_height * scale)
    cv2.ellipse(img_bgr, (face_center_x, face_center_y), 
                (face_ellipse_w // 2, face_ellipse_h // 2), 
                0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img_bgr, (face_center_x, face_center_y), 
                (face_ellipse_w // 2, face_ellipse_h // 2), 
                0, 0, 360, (200, 200, 200), 2)
    
    # Draw key landmarks and connections
    def draw_landmark(idx: int, name: str, color: Tuple[int, int, int], radius: int = 5):
        if idx < len(landmarks) and not np.isnan(landmarks[idx]).any():
            pt = landmarks[idx]
            x = int(pt[0] * scale + offset_x)
            y = int(pt[1] * scale + offset_y)
            color_bgr = (color[2], color[1], color[0])  # RGB to BGR
            cv2.circle(img_bgr, (x, y), radius, color_bgr, -1)
            cv2.circle(img_bgr, (x, y), radius + 2, (0, 0, 0), 2)
            # Add label with larger font
            cv2.putText(img_bgr, name, (x + radius + 5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    def draw_line(idx1: int, idx2: int, color: Tuple[int, int, int], thickness: int = 3):
        if (idx1 < len(landmarks) and idx2 < len(landmarks) and 
            not np.isnan(landmarks[idx1]).any() and not np.isnan(landmarks[idx2]).any()):
            pt1 = landmarks[idx1]
            pt2 = landmarks[idx2]
            x1 = int(pt1[0] * scale + offset_x)
            y1 = int(pt1[1] * scale + offset_y)
            x2 = int(pt2[0] * scale + offset_x)
            y2 = int(pt2[1] * scale + offset_y)
            color_bgr = (color[2], color[1], color[0])  # RGB to BGR
            cv2.line(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
    
    # Draw connections
    # Face outline
    draw_line(LANDMARK_INDICES["forehead"], LANDMARK_INDICES["cheek_l"], (150, 150, 150), 2)
    draw_line(LANDMARK_INDICES["forehead"], LANDMARK_INDICES["cheek_r"], (150, 150, 150), 2)
    draw_line(LANDMARK_INDICES["cheek_l"], LANDMARK_INDICES["chin"], (150, 150, 150), 2)
    draw_line(LANDMARK_INDICES["cheek_r"], LANDMARK_INDICES["chin"], (150, 150, 150), 2)
    
    # Eyes
    draw_line(LANDMARK_INDICES["le_outer"], LANDMARK_INDICES["le_inner"], (100, 100, 255), 3)
    draw_line(LANDMARK_INDICES["re_inner"], LANDMARK_INDICES["re_outer"], (100, 100, 255), 3)
    draw_line(LANDMARK_INDICES["le_inner"], LANDMARK_INDICES["re_inner"], (255, 100, 100), 3)  # Interocular
    
    # Nose
    draw_line(LANDMARK_INDICES["nose_bridge"], LANDMARK_INDICES["nose_tip"], (100, 200, 100), 3)
    draw_line(LANDMARK_INDICES["nose_l"], LANDMARK_INDICES["nose_r"], (100, 200, 100), 3)
    draw_line(LANDMARK_INDICES["nose_tip"], LANDMARK_INDICES["nose_l"], (100, 200, 100), 2)
    draw_line(LANDMARK_INDICES["nose_tip"], LANDMARK_INDICES["nose_r"], (100, 200, 100), 2)
    
    # Mouth
    draw_line(LANDMARK_INDICES["mouth_l"], LANDMARK_INDICES["mouth_r"], (200, 100, 100), 3)
    
    # Face dimensions
    draw_line(LANDMARK_INDICES["forehead"], LANDMARK_INDICES["chin"], (200, 200, 0), 3)  # Face height
    draw_line(LANDMARK_INDICES["cheek_l"], LANDMARK_INDICES["cheek_r"], (200, 200, 0), 3)  # Face width
    
    # Draw landmarks (with larger dots)
    draw_landmark(LANDMARK_INDICES["forehead"], "forehead", (255, 0, 0), 12)
    draw_landmark(LANDMARK_INDICES["nose_bridge"], "bridge", (0, 255, 0), 10)
    draw_landmark(LANDMARK_INDICES["nose_tip"], "tip", (0, 255, 0), 10)
    draw_landmark(LANDMARK_INDICES["nose_l"], "nose_l", (0, 200, 0), 8)
    draw_landmark(LANDMARK_INDICES["nose_r"], "nose_r", (0, 200, 0), 8)
    draw_landmark(LANDMARK_INDICES["chin"], "chin", (255, 0, 0), 12)
    draw_landmark(LANDMARK_INDICES["le_outer"], "le_o", (0, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["le_inner"], "le_i", (0, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["re_inner"], "re_i", (0, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["re_outer"], "re_o", (0, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["mouth_l"], "mouth_l", (255, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["mouth_r"], "mouth_r", (255, 0, 255), 10)
    draw_landmark(LANDMARK_INDICES["cheek_l"], "cheek_l", (200, 200, 0), 10)
    draw_landmark(LANDMARK_INDICES["cheek_r"], "cheek_r", (200, 200, 0), 10)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def create_statistics_text(stats: Dict) -> str:
    """Create a formatted text string of statistics."""
    lines = ["Average Face Statistics", "=" * 40]
    
    feature_names = stats["feature_names"]
    mean_values = stats["mean"]
    std_values = stats.get("std", [0.0] * len(mean_values))
    
    for name, mean_val, std_val in zip(feature_names, mean_values, std_values):
        lines.append(f"{name:35s}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize average human face from baseline statistics"
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to geometry_stats.json from baseline run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="average_face_visualization.png",
        help="Output image path"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Output image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1000,
        help="Output image height"
    )
    
    args = parser.parse_args()
    
    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        print(f"Error: Statistics file not found: {stats_path}", file=sys.stderr)
        return 1
    
    # Load statistics
    print(f"Loading statistics from: {stats_path}")
    stats = load_geometry_stats(stats_path)
    
    # Print statistics
    print("\n" + create_statistics_text(stats))
    
    # Reconstruct average landmarks
    print("\nReconstructing average face landmarks...")
    landmarks, face_width, face_height = reconstruct_average_landmarks(stats)
    
    print(f"Face width (aligned): {face_width:.4f}")
    print(f"Face height (aligned): {face_height:.4f}")
    print(f"Face aspect ratio: {face_width/face_height:.4f}")
    
    # Draw visualization
    print("\nCreating visualization...")
    img = draw_average_face(landmarks, face_width, face_height, stats, 
                            output_size=(args.width, args.height))
    
    # Create figure with statistics
    fig = plt.figure(figsize=(14, 16))
    
    # Main visualization
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(img)
    ax1.set_title("Average Human Face Geometry (Reconstructed from Statistics)", 
                  fontsize=20, fontweight='bold')
    ax1.axis('off')
    
    # Statistics text
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    stats_text = create_statistics_text(stats)
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=14, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {args.output}")
    plt.close()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

