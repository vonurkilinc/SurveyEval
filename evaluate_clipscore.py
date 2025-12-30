#!/usr/bin/env python3
"""
Evaluate caricature images using CLIPScore.

This script:
1. Maps variants (v1-v4) to the provided prompts
2. Scores all caricature images in the img folder using CLIPScore
3. Computes statistics grouped by caricature (person) and by method
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    try:
        import open_clip
        HAS_OPEN_CLIP = True
    except ImportError:
        HAS_OPEN_CLIP = False


# The 4 prompts corresponding to variants v1, v2, v3, v4
PROMPTS = [
    "Stylized caricature portrait, mild exaggeration, slightly larger eyes and smile, clean ink lines, flat pastel colors, soft lighting, realistic and balanced proportions.",
    "Caricature portrait, medium exaggeration, elongated nose, slimmer jaw, big eyes, comic ink style with bold lines and flat color, 3/4 view, clear identity preserved.",
    "Extreme caricature portrait, giant head, tiny body, huge expressive eyes and nose, graffiti cartoon style, vibrant pop colors, dynamic brush strokes, exaggerated yet coherent.",
    "Caricature portrait, medium exaggeration, dynamic pose with head tilted upward and slightly sideways, elongated nose, bold comic style, vibrant flat colors, clear identity.",
]


class CLIPScorer:
    """CLIPScore calculator using CLIP embeddings."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to use ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model_name = model_name
        
        if HAS_CLIP:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            self.use_open_clip = False
        elif HAS_OPEN_CLIP:
            # Map CLIP model names to open_clip names
            model_mapping = {
                "ViT-B/32": "ViT-B-32",
                "ViT-B/16": "ViT-B-16",
                "ViT-L/14": "ViT-L-14",
            }
            open_clip_name = model_mapping.get(model_name, model_name)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                open_clip_name, pretrained="openai"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(open_clip_name)
            self.use_open_clip = True
        else:
            raise RuntimeError(
                "Neither 'clip' nor 'open_clip_torch' is installed. "
                "Please install one: pip install clip-by-openai or pip install open-clip-torch"
            )
    
    def encode_image(self, image_path: Path) -> torch.Tensor:
        """Encode an image to CLIP embedding."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_open_clip:
                image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.encode_image(image_tensor)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP embedding."""
        with torch.no_grad():
            if self.use_open_clip:
                text_tokens = self.tokenizer([text]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
            else:
                text_tokens = clip.tokenize([text]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
            
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def score(self, image_path: Path, text: str) -> float:
        """
        Compute CLIPScore between image and text.
        
        CLIPScore is the cosine similarity between image and text embeddings,
        typically scaled by 2.5 and clipped to [0, 1] range.
        
        Args:
            image_path: Path to image
            text: Text prompt
        
        Returns:
            CLIPScore (typically in range [0, 1])
        """
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()
        
        # CLIPScore formula: scale by 2.5 and clip to [0, 1]
        # This is the standard CLIPScore formula from the paper
        clip_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
        return float(clip_score)


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


def score_image(scorer: CLIPScorer, image_path: Path, prompt: str) -> float:
    """Score a single image with CLIPScore."""
    try:
        score = scorer.score(image_path, prompt)
        return float(score)
    except Exception as e:
        print(f"Error scoring {image_path}: {e}")
        return np.nan


def evaluate_all_images(
    img_dir: Path,
    prompts: List[str],
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate all caricature images and return a DataFrame with scores.
    
    Args:
        img_dir: Directory containing person subdirectories
        prompts: List of prompts (index corresponds to variant-1)
        model_name: CLIP model name
        device: Device to use
        output_dir: Optional directory to save intermediate results
    
    Returns:
        DataFrame with columns: person_id, method, variant, prompt, image_path, clip_score
    """
    print(f"Scanning for caricature images in {img_dir}...")
    images = find_caricature_images(img_dir)
    print(f"Found {len(images)} caricature images")
    
    if not images:
        print("No caricature images found!")
        return pd.DataFrame()
    
    print(f"Initializing CLIP model: {model_name}")
    scorer = CLIPScorer(model_name=model_name, device=device)
    
    results = []
    
    print("Evaluating images with CLIPScore...")
    for img_info in tqdm(images, desc="Scoring images"):
        variant = img_info['variant']
        
        # Map variant to prompt (v1 -> prompts[0], v2 -> prompts[1], etc.)
        if variant < 1 or variant > len(prompts):
            print(f"Warning: variant {variant} out of range, skipping {img_info['image_path']}")
            continue
        
        prompt = prompts[variant - 1]
        image_path = Path(img_info['image_path'])
        
        score = score_image(scorer, image_path, prompt)
        
        results.append({
            'person_id': img_info['person_id'],
            'method': img_info['method'],
            'variant': variant,
            'prompt': prompt,
            'image_path': str(image_path),
            'clip_score': score,
        })
    
    df = pd.DataFrame(results)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "clipscore_scores.csv", index=False)
        print(f"Saved detailed scores to {output_dir / 'clipscore_scores.csv'}")
    
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
    df_valid = df.dropna(subset=['clip_score']).copy()
    
    if df_valid.empty:
        print("Warning: No valid scores found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Statistics by caricature (person)
    caricature_stats = (
        df_valid.groupby('person_id')['clip_score']
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
        df_valid.groupby('method')['clip_score']
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
        caricature_stats.to_csv(output_dir / "clipscore_stats_by_caricature.csv", index=False)
        method_stats.to_csv(output_dir / "clipscore_stats_by_method.csv", index=False)
        print(f"Saved statistics to {output_dir}")
    
    return caricature_stats, method_stats


def print_summary(caricature_stats: pd.DataFrame, method_stats: pd.DataFrame):
    """Print a summary of the statistics."""
    print("\n" + "="*80)
    print("CLIPScore EVALUATION SUMMARY")
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
        description="Evaluate caricature images using CLIPScore",
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
        default="reports/clipscore",
        help="Output directory for results (default: reports/clipscore)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use (default: ViT-B/32). Options: ViT-B/32, ViT-B/16, ViT-L/14, etc.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
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
    print(f"CLIP model: {args.model}")
    print(f"Device: {args.device}")
    
    # Evaluate all images
    df = evaluate_all_images(img_dir, prompts, args.model, args.device, output_dir)
    
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
        'valid_scores': len(df.dropna(subset=['clip_score'])),
        'model': args.model,
        'device': args.device,
        'by_caricature': caricature_stats.to_dict('records') if not caricature_stats.empty else [],
        'by_method': method_stats.to_dict('records') if not method_stats.empty else [],
    }
    
    with open(output_dir / "clipscore_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFull summary saved to {output_dir / 'clipscore_summary.json'}")
    
    return 0


if __name__ == "__main__":
    exit(main())

