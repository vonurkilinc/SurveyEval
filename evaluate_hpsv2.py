#!/usr/bin/env python3
"""
Evaluate caricature images using HPSv2.

This script:
1. Maps variants (v1-v4) to the provided prompts
2. Scores all caricature images in the img folder
3. Computes statistics grouped by caricature (person) and by method
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hpsv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# The 4 prompts corresponding to variants v1, v2, v3, v4
PROMPTS = [
    "Stylized caricature portrait, mild exaggeration, slightly larger eyes and smile, clean ink lines, flat pastel colors, soft lighting, realistic and balanced proportions.",
    "Caricature portrait, medium exaggeration, elongated nose, slimmer jaw, big eyes, comic ink style with bold lines and flat color, 3/4 view, clear identity preserved.",
    "Extreme caricature portrait, giant head, tiny body, huge expressive eyes and nose, graffiti cartoon style, vibrant pop colors, dynamic brush strokes, exaggerated yet coherent.",
    "Caricature portrait, medium exaggeration, dynamic pose with head tilted upward and slightly sideways, elongated nose, bold comic style, vibrant flat colors, clear identity.",
]


def get_variant_number(filename: str) -> Optional[int]:
    """Extract variant number from filename (e.g., v1.jpg -> 1, v4.png -> 4)."""
    name = Path(filename).stem.lower()  # Remove extension
    if name.startswith('v') and name[1:].isdigit():
        return int(name[1:])
    return None


def find_caricature_images(img_dir: Path) -> List[Dict[str, str]]:
    """
    Scan img directory and find all caricature images.
    
    Returns list of dicts with keys: person_id, method, variant, image_path
    """
    images = []
    
    # Look for person directories (P000, P001, etc.)
    for person_dir in sorted(img_dir.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith('P'):
            continue
            
        person_id = person_dir.name
        
        # Look for method directories
        for method_dir in sorted(person_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name == 'reference':
                continue
                
            method = method_dir.name
            
            # Find all image files in method directory
            for img_file in sorted(method_dir.iterdir()):
                if not img_file.is_file():
                    continue
                    
                ext = img_file.suffix.lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                
                variant_num = get_variant_number(img_file.name)
                if variant_num is None:
                    continue
                
                images.append({
                    'person_id': person_id,
                    'method': method,
                    'variant': variant_num,
                    'image_path': str(img_file.resolve()),
                })
    
    return images


def score_image(image_path: Path, prompt: str, hps_version: str = "v2.1") -> float:
    """Score a single image with HPSv2."""
    try:
        image = Image.open(image_path).convert("RGB")
        scores = hpsv2.score(image, prompt, hps_version=hps_version)
        # hpsv2.score returns a list, get the first element
        if isinstance(scores, list) and len(scores) > 0:
            return float(scores[0])
        return float(scores) if not isinstance(scores, list) else np.nan
    except Exception as e:
        print(f"Error scoring {image_path}: {e}")
        return np.nan


def evaluate_all_images(
    img_dir: Path,
    prompts: List[str],
    hps_version: str = "v2.1",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate all caricature images and return a DataFrame with scores.
    
    Args:
        img_dir: Directory containing person subdirectories
        prompts: List of prompts (index corresponds to variant-1)
        hps_version: HPS version to use
        output_dir: Optional directory to save intermediate results
    
    Returns:
        DataFrame with columns: person_id, method, variant, prompt, image_path, hps_score
    """
    print(f"Scanning for caricature images in {img_dir}...")
    images = find_caricature_images(img_dir)
    print(f"Found {len(images)} caricature images")
    
    if not images:
        print("No caricature images found!")
        return pd.DataFrame()
    
    results = []
    
    print("Evaluating images with HPSv2...")
    for img_info in tqdm(images, desc="Scoring images"):
        variant = img_info['variant']
        
        # Map variant to prompt (v1 -> prompts[0], v2 -> prompts[1], etc.)
        if variant < 1 or variant > len(prompts):
            print(f"Warning: variant {variant} out of range, skipping {img_info['image_path']}")
            continue
        
        prompt = prompts[variant - 1]
        image_path = Path(img_info['image_path'])
        
        score = score_image(image_path, prompt, hps_version=hps_version)
        
        results.append({
            'person_id': img_info['person_id'],
            'method': img_info['method'],
            'variant': variant,
            'prompt': prompt,
            'image_path': str(image_path),
            'hps_score': score,
        })
    
    df = pd.DataFrame(results)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "hpsv2_scores.csv", index=False)
        print(f"Saved detailed scores to {output_dir / 'hpsv2_scores.csv'}")
    
    return df


def compute_statistics(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute statistics grouped by caricature (person) and by method.
    
    Returns:
        Tuple of (caricature_stats, method_stats) DataFrames
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Remove rows with NaN scores
    df_valid = df.dropna(subset=['hps_score']).copy()
    
    if df_valid.empty:
        print("Warning: No valid scores found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Statistics by caricature (person)
    caricature_stats = (
        df_valid.groupby('person_id')['hps_score']
        .agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
        ])
        .reset_index()
        .round(4)
    )
    caricature_stats.columns = ['person_id', 'n_images', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    
    # Statistics by method
    method_stats = (
        df_valid.groupby('method')['hps_score']
        .agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
        ])
        .reset_index()
        .round(4)
    )
    method_stats.columns = ['method', 'n_images', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    
    # Sort by mean score (descending)
    caricature_stats = caricature_stats.sort_values('mean', ascending=False)
    method_stats = method_stats.sort_values('mean', ascending=False)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        caricature_stats.to_csv(output_dir / "hpsv2_stats_by_caricature.csv", index=False)
        method_stats.to_csv(output_dir / "hpsv2_stats_by_method.csv", index=False)
        print(f"Saved statistics to {output_dir}")
    
    return caricature_stats, method_stats


def print_summary(caricature_stats: pd.DataFrame, method_stats: pd.DataFrame):
    """Print a summary of the statistics."""
    print("\n" + "="*80)
    print("HPSv2 EVALUATION SUMMARY")
    print("="*80)
    
    print("\n--- Statistics by Caricature (Person) ---")
    if not caricature_stats.empty:
        print(caricature_stats.to_string(index=False))
    else:
        print("No data available")
    
    print("\n--- Statistics by Method ---")
    if not method_stats.empty:
        print(method_stats.to_string(index=False))
    else:
        print("No data available")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate caricature images using HPSv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="img",
        help="Directory containing caricature images (default: img)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/hpsv2",
        help="Output directory for results (default: reports/hpsv2)",
    )
    parser.add_argument(
        "--hps-version",
        type=str,
        default="v2.1",
        choices=["v2.0", "v2.1"],
        help="HPS version to use (default: v2.1)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional JSON file with prompts (if not provided, uses default prompts)",
    )
    
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        return 1
    
    # Load prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
            prompts = prompts_data.get('prompts', PROMPTS)
    else:
        prompts = PROMPTS
    
    print(f"Using {len(prompts)} prompts")
    print(f"HPS version: {args.hps_version}")
    
    # Evaluate all images
    df = evaluate_all_images(img_dir, prompts, args.hps_version, output_dir)
    
    if df.empty:
        print("No images were evaluated!")
        return 1
    
    # Compute statistics
    caricature_stats, method_stats = compute_statistics(df, output_dir)
    
    # Print summary
    print_summary(caricature_stats, method_stats)
    
    # Save full results summary as JSON
    summary = {
        'total_images': len(df),
        'valid_scores': len(df.dropna(subset=['hps_score'])),
        'by_caricature': caricature_stats.to_dict('records') if not caricature_stats.empty else [],
        'by_method': method_stats.to_dict('records') if not method_stats.empty else [],
    }
    
    with open(output_dir / "hpsv2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFull summary saved to {output_dir / 'hpsv2_summary.json'}")
    
    return 0


if __name__ == "__main__":
    exit(main())

