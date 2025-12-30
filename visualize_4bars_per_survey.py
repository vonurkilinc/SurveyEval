#!/usr/bin/env python3
"""
Create simple 4-bar visualizations for each survey metric:
- Best matching CQS Component
- CQS Overall
- HPSv2
- CLIPScore
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 17,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette for methods
METHOD_COLORS = {
    "instantid": "#2E86AB",      # Blue
    "qwen": "#F18F01",           # Orange
    "pulid": "#A23B72",          # Purple
    "caricaturebooth": "#C73E1D", # Red
}

# Component labels
COMPONENT_LABELS = {
    "Sexag": "Sexag\n(Ref-Tied Exag)",
    "Sexag_free": "Sexag_free\n(Ref-Free Exag)",
    "mag": "mag\n(From Ref)",
    "mag_free": "mag_free\n(From Avg)",
    "exag_proj": "exag_proj\n(Projection)",
    "Sid": "Sid\n(Identity)",
    "Splaus": "Splaus\n(Plausibility)",
}


def load_all_data(
    exaggeration_modes_csv: Path,
    survey_csv: Path,
    method_comparison_csv: Path,
) -> pd.DataFrame:
    """Load and merge all data sources."""
    # Load exaggeration components
    df_exag = pd.read_csv(exaggeration_modes_csv)
    df_exag["method"] = df_exag["method"].str.lower()
    
    # Load survey data (pivot to wide format)
    df_survey_raw = pd.read_csv(survey_csv)
    df_survey = df_survey_raw.pivot(index="method", columns="metric", values="mean").reset_index()
    # Handle column names properly
    new_cols = ["method"]
    for col in df_survey.columns[1:]:
        if col == "alignment":  # Handle alignment specially
            new_cols.append("survey_alignment")
        else:
            new_cols.append(f"survey_{col}")
    df_survey.columns = new_cols
    df_survey["method"] = df_survey["method"].str.lower()
    
    # Load method comparison (has hpsv2, clipscore, cqs)
    df_methods = pd.read_csv(method_comparison_csv)
    df_methods["method"] = df_methods["method"].str.lower()
    
    # Merge all
    df = df_exag.merge(df_survey, on="method", how="inner")
    df = df.merge(df_methods, on="method", how="inner")
    
    return df


def compute_correlation(df: pd.DataFrame, survey_col: str, metric_col: str) -> Tuple[float, float]:
    """Compute Pearson correlation between survey metric and automated metric."""
    if metric_col not in df.columns:
        return np.nan, np.nan
    
    valid_mask = df[[survey_col, metric_col]].notna().all(axis=1)
    if valid_mask.sum() < 2:
        return np.nan, np.nan
    
    try:
        r, p = pearsonr(df.loc[valid_mask, survey_col], df.loc[valid_mask, metric_col])
        return r, p
    except Exception:
        return np.nan, np.nan


def find_best_component(df: pd.DataFrame, survey_col: str) -> Tuple[str, float]:
    """Find the CQS component with highest absolute correlation to survey metric."""
    cqs_components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj", "Sid", "Splaus"]
    
    best_component = None
    best_corr = -np.inf
    
    for comp in cqs_components:
        if comp in df.columns:
            r, _ = compute_correlation(df, survey_col, comp)
            if not np.isnan(r):
                if abs(r) > abs(best_corr):
                    best_corr = r
                    best_component = comp
    
    # If no component found, try with any valid correlation (even negative)
    if best_component is None:
        for comp in cqs_components:
            if comp in df.columns:
                r, _ = compute_correlation(df, survey_col, comp)
                if not np.isnan(r):
                    best_corr = r
                    best_component = comp
                    break
    
    return best_component, best_corr


def create_4bar_visualization(
    df: pd.DataFrame,
    survey_metric: str,
    survey_label: str,
    best_comp: str,
    best_corr: float,
    output_path: Path,
) -> None:
    """Create 4-bar chart: Best CQS Component, CQS Overall, HPSv2, CLIPScore."""
    # Sort by survey metric
    df_sorted = df.sort_values(survey_metric, ascending=False).copy()
    
    # Prepare metrics
    metrics = [
        (best_comp, f"CQS Component\n({best_comp})", COMPONENT_LABELS.get(best_comp, best_comp)),
        ("CQS", "CQS Overall", "CQS Overall"),
        ("hpsv2", "HPSv2", "HPSv2"),
        ("clipscore", "CLIPScore", "CLIPScore"),
    ]
    
    # Filter to available metrics
    available_metrics = [(m, label, full_label) for m, label, full_label in metrics if m in df_sorted.columns]
    
    if len(available_metrics) == 0:
        print(f"[WARNING] No available metrics for {survey_label}, skipping...")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(df_sorted))
    width = 0.2
    n_metrics = len(available_metrics)
    
    # Normalize all metrics to 0-1 for fair comparison
    df_plot = df_sorted.copy()
    for metric, _, _ in available_metrics:
        if metric in df_plot.columns:
            min_val = df_plot[metric].min()
            max_val = df_plot[metric].max()
            if max_val > min_val:
                df_plot[f"{metric}_norm"] = (df_plot[metric] - min_val) / (max_val - min_val)
            else:
                df_plot[f"{metric}_norm"] = 0.5
    
    # Normalize survey metric too
    survey_min = df_plot[survey_metric].min()
    survey_max = df_plot[survey_metric].max()
    if survey_max > survey_min:
        df_plot[f"{survey_metric}_norm"] = (df_plot[survey_metric] - survey_min) / (survey_max - survey_min)
    else:
        df_plot[f"{survey_metric}_norm"] = 0.5
    
    # Plot bars
    bars_list = []
    for idx, (metric, label, _) in enumerate(available_metrics):
        offset = (idx - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            x_pos + offset,
            df_plot[f"{metric}_norm"],
            width,
            label=label,
            alpha=0.85,
            edgecolor="black",
            linewidth=1.5,
        )
        bars_list.append((bars, metric, label))
        
        # Add correlation annotation on bars
        r, p = compute_correlation(df, survey_metric, metric)
        if not np.isnan(r):
            for bar in bars:
                height = bar.get_height()
                sig = "*" if p < 0.05 else ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"r={r:.2f}{sig}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
    
    # Add survey rating as line on secondary axis
    ax_twin = ax.twinx()
    line = ax_twin.plot(
        x_pos,
        df_plot[f"{survey_metric}_norm"],
        "o-",
        color="red",
        linewidth=4,
        markersize=14,
        label=f"Survey {survey_label}",
        zorder=10,
    )
    ax_twin.set_ylabel(f"Survey {survey_label} (Normalized)", fontweight="bold", color="red", fontsize=13)
    ax_twin.tick_params(axis="y", labelcolor="red")
    ax_twin.set_ylim(0, 1)
    
    # Customize main axis
    ax.set_xlabel("Method (sorted by Survey Rating)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Normalized Score (0-1)", fontweight="bold", fontsize=13)
    ax.set_title(
        f"Metrics vs Survey {survey_label}\nBest CQS Component: {best_comp} (r={best_corr:.3f})",
        fontweight="bold",
        pad=15,
        fontsize=15,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted["method"].str.upper(), rotation=0, fontsize=12)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
    ax_twin.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {survey_label} visualization to: {output_path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create 4-bar visualizations per survey metric")
    ap.add_argument("--exaggeration_modes_csv", type=str,
                    default="reports/exaggeration_modes_exag2/exaggeration_modes_by_method.csv",
                    help="Exaggeration modes CSV")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv",
                    help="Survey CSV")
    ap.add_argument("--method_comparison_csv", type=str,
                    default="reports/correlations_exag2/method_scores_comparison.csv",
                    help="Method comparison CSV with hpsv2, clipscore, cqs")
    ap.add_argument("--output_dir", type=str, default="reports/4bars_per_survey",
                    help="Output directory")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATING 4-BAR VISUALIZATIONS PER SURVEY METRIC")
    print("="*80)
    
    # Load data
    print("\n[INFO] Loading data...")
    df = load_all_data(
        Path(args.exaggeration_modes_csv),
        Path(args.survey_csv),
        Path(args.method_comparison_csv),
    )
    print(f"[INFO] Loaded data for {len(df)} methods")
    print(f"  Methods: {', '.join(df['method'].str.upper().tolist())}")
    
    # Get survey metrics (filter out alignment_x/alignment_y duplicates)
    survey_metrics = [col for col in df.columns if col.startswith("survey_")]
    # Keep only the main ones
    main_survey_metrics = []
    seen_base = set()
    for col in survey_metrics:
        base = col.replace("_x", "").replace("_y", "")
        if base not in seen_base:
            if col.endswith("_x"):
                main_survey_metrics.append(col)
                seen_base.add(base)
            elif not col.endswith("_y"):
                main_survey_metrics.append(col)
                seen_base.add(base)
    
    survey_labels = {
        "survey_exaggeration": "Exaggeration",
        "survey_identity": "Identity",
        "survey_overall": "Overall",
        "survey_plausibility": "Plausibility",
        "survey_alignment": "Alignment",
        "survey_alignment_x": "Alignment",
    }
    
    print(f"\n[INFO] Found survey metrics: {main_survey_metrics}")
    
    # Create visualization for each survey metric
    print("\n[INFO] Creating visualizations...")
    best_components_summary = []
    
    for survey_col in main_survey_metrics:
        survey_name = survey_col.replace("survey_", "").replace("_x", "")
        label = survey_labels.get(survey_col, survey_name.title())
        
        best_comp, best_corr = find_best_component(df, survey_col)
        if best_comp:
            best_components_summary.append({
                "survey_metric": label,
                "best_component": best_comp,
                "correlation": best_corr,
            })
            print(f"  Creating visualization for {label}...")
            print(f"    Best component: {best_comp} (r={best_corr:.3f})")
            
            output_path = output_dir / f"4bars_vs_{survey_name}.png"
            create_4bar_visualization(df, survey_col, label, best_comp, best_corr, output_path)
    
    # Save summary
    if best_components_summary:
        df_summary = pd.DataFrame(best_components_summary)
        summary_csv = output_dir / "best_components_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n[OK] Saved best components summary to: {summary_csv}")
        print("\nBest Components Summary:")
        print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"[OK] All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

