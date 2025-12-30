#!/usr/bin/env python3
"""
Visualize CQS components, overall CQS, CLIPScore, and HPSv2 compared to each survey metric.
Creates separate visualizations for each survey dimension (exaggeration, identity, overall, etc.)
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
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
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

# Metric labels and grouping
METRIC_GROUPS = {
    "CQS Components": {
        "Sexag": "Sexag\n(Ref-Tied Exag)",
        "Sexag_free": "Sexag_free\n(Ref-Free Exag)",
        "mag": "mag\n(From Ref)",
        "mag_free": "mag_free\n(From Avg)",
        "exag_proj": "exag_proj\n(Projection)",
        "Sid": "Sid\n(Identity)",
        "Splaus": "Splaus\n(Plausibility)",
    },
    "Overall Scores": {
        "CQS": "CQS\n(Overall)",
        "cqs_step4": "CQS Step4\n(With Sexag)",
        "clipscore": "CLIPScore",
        "hpsv2": "HPSv2",
    }
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
    # Handle column names properly (avoid duplicates)
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


def compute_correlations(df: pd.DataFrame, survey_col: str, metric_cols: List[str]) -> Dict[str, float]:
    """Compute Pearson correlations between survey metric and all automated metrics."""
    correlations = {}
    for metric_col in metric_cols:
        if metric_col in df.columns:
            valid_mask = df[[survey_col, metric_col]].notna().all(axis=1)
            if valid_mask.sum() >= 2:
                try:
                    r, p = pearsonr(df.loc[valid_mask, survey_col], df.loc[valid_mask, metric_col])
                    correlations[metric_col] = r
                except Exception:
                    correlations[metric_col] = np.nan
            else:
                correlations[metric_col] = np.nan
        else:
            correlations[metric_col] = np.nan
    return correlations


def create_survey_dimension_visualization(
    df: pd.DataFrame,
    survey_metric: str,
    survey_label: str,
    output_path: Path,
) -> None:
    """Create comprehensive visualization for one survey dimension."""
    # Get all metrics to compare
    cqs_components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj", "Sid", "Splaus"]
    overall_scores = ["CQS", "cqs_step4", "clipscore", "hpsv2"]
    
    # Filter to available metrics
    available_components = [m for m in cqs_components if m in df.columns]
    available_overall = [m for m in overall_scores if m in df.columns]
    
    # Compute correlations
    all_metrics = available_components + available_overall
    correlations = compute_correlations(df, survey_metric, all_metrics)
    
    # Create figure with two subplots: bar chart and correlation display
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Top left: Bar chart for CQS components
    ax1 = fig.add_subplot(gs[0, 0])
    if available_components:
        df_sorted = df.sort_values(survey_metric, ascending=False)
        x_pos = np.arange(len(df_sorted))
        width = 0.15
        
        for idx, metric in enumerate(available_components):
            offset = (idx - len(available_components) / 2) * width + width / 2
            bars = ax1.bar(
                x_pos + offset,
                df_sorted[metric],
                width,
                label=METRIC_GROUPS["CQS Components"].get(metric, metric),
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            
            # Add correlation annotation
            corr_val = correlations.get(metric, np.nan)
            if not np.isnan(corr_val):
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"r={corr_val:.2f}",
                        ha="center",
                        va="bottom" if corr_val >= 0 else "top",
                        fontsize=8,
                        rotation=90,
                        fontweight="bold",
                    )
        
        ax1.set_xlabel("Method (sorted by Survey Rating)", fontweight="bold")
        ax1.set_ylabel("Component Score", fontweight="bold")
        ax1.set_title(f"CQS Components vs {survey_label}", fontweight="bold", pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_sorted["method"].str.upper(), rotation=0)
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add secondary y-axis for survey rating
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_pos, df_sorted[survey_metric], "o-", color="red", linewidth=3, 
                     markersize=10, label=f"Survey {survey_label}", zorder=10)
        ax1_twin.set_ylabel(f"Survey {survey_label} Rating", fontweight="bold", color="red")
        ax1_twin.tick_params(axis="y", labelcolor="red")
        ax1_twin.legend(loc="upper right", framealpha=0.9)
    
    # Top right: Bar chart for overall scores
    ax2 = fig.add_subplot(gs[0, 1])
    if available_overall:
        df_sorted = df.sort_values(survey_metric, ascending=False)
        x_pos = np.arange(len(df_sorted))
        width = 0.2
        
        for idx, metric in enumerate(available_overall):
            offset = (idx - len(available_overall) / 2) * width + width / 2
            bars = ax2.bar(
                x_pos + offset,
                df_sorted[metric],
                width,
                label=METRIC_GROUPS["Overall Scores"].get(metric, metric),
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            
            # Add correlation annotation
            corr_val = correlations.get(metric, np.nan)
            if not np.isnan(corr_val):
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"r={corr_val:.2f}",
                        ha="center",
                        va="bottom" if corr_val >= 0 else "top",
                        fontsize=8,
                        rotation=90,
                        fontweight="bold",
                    )
        
        ax2.set_xlabel("Method (sorted by Survey Rating)", fontweight="bold")
        ax2.set_ylabel("Score", fontweight="bold")
        ax2.set_title(f"Overall Scores vs {survey_label}", fontweight="bold", pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df_sorted["method"].str.upper(), rotation=0)
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add secondary y-axis for survey rating
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, df_sorted[survey_metric], "o-", color="red", linewidth=3,
                     markersize=10, label=f"Survey {survey_label}", zorder=10)
        ax2_twin.set_ylabel(f"Survey {survey_label} Rating", fontweight="bold", color="red")
        ax2_twin.tick_params(axis="y", labelcolor="red")
        ax2_twin.legend(loc="upper right", framealpha=0.9)
    
    # Bottom left: Scatter plots for CQS components
    if available_components:
        n_components = len(available_components)
        cols = min(4, n_components)
        rows = (n_components + cols - 1) // cols
        
        # Create subplot grid for scatter plots
        gs_sub = gs[1, 0].subgridspec(rows, cols, hspace=0.4, wspace=0.4)
        
        for idx, metric in enumerate(available_components):
            row = idx // cols
            col = idx % cols
            sub_ax = fig.add_subplot(gs_sub[row, col])
            
            for method in df["method"].unique():
                df_method = df[df["method"] == method]
                sub_ax.scatter(
                    df_method[survey_metric],
                    df_method[metric],
                    s=150,
                    color=METHOD_COLORS.get(method, "#808080"),
                    label=method.upper(),
                    edgecolors="black",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=5,
                )
            
            corr_val = correlations.get(metric, np.nan)
            if not np.isnan(corr_val):
                # Add correlation line
                z = np.polyfit(df[survey_metric], df[metric], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df[survey_metric].min(), df[survey_metric].max(), 100)
                sub_ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=1.5)
            
            sub_ax.set_xlabel("Survey Rating", fontsize=9)
            sub_ax.set_ylabel(METRIC_GROUPS["CQS Components"].get(metric, metric), fontsize=9)
            title_text = f"{METRIC_GROUPS['CQS Components'].get(metric, metric)}\nr={corr_val:.3f}" if not np.isnan(corr_val) else METRIC_GROUPS["CQS Components"].get(metric, metric)
            sub_ax.set_title(title_text, fontsize=10, fontweight="bold")
            sub_ax.grid(alpha=0.3, linestyle="--")
            if idx == 0:
                sub_ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    
    # Bottom right: Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    all_metrics_ordered = available_components + available_overall
    corr_values = [correlations.get(m, np.nan) for m in all_metrics_ordered]
    metric_labels = [METRIC_GROUPS["CQS Components"].get(m, METRIC_GROUPS["Overall Scores"].get(m, m)) for m in all_metrics_ordered]
    
    # Create correlation bar chart
    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in corr_values]
    bars = ax4.barh(range(len(all_metrics_ordered)), corr_values, color=colors, alpha=0.7, edgecolor="black")
    ax4.set_yticks(range(len(all_metrics_ordered)))
    ax4.set_yticklabels(metric_labels, fontsize=9)
    ax4.set_xlabel("Pearson Correlation (r)", fontweight="bold")
    ax4.set_title(f"Correlations with Survey {survey_label}", fontweight="bold", pad=10)
    ax4.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlim(-1, 1)
    ax4.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels
    for idx, (bar, val) in enumerate(zip(bars, corr_values)):
        if not np.isnan(val):
            ax4.text(
                val + (0.05 if val >= 0 else -0.05),
                idx,
                f"{val:.3f}",
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    
    plt.suptitle(f"All Metrics vs Survey {survey_label}", fontsize=18, fontweight="bold", y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {survey_label} visualization to: {output_path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize all metrics compared to each survey dimension")
    ap.add_argument("--exaggeration_modes_csv", type=str,
                    default="reports/exaggeration_modes_exag2/exaggeration_modes_by_method.csv",
                    help="Exaggeration modes CSV")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv",
                    help="Survey CSV")
    ap.add_argument("--method_comparison_csv", type=str,
                    default="reports/correlations_exag2/method_scores_comparison.csv",
                    help="Method comparison CSV with hpsv2, clipscore, cqs")
    ap.add_argument("--output_dir", type=str, default="reports/metrics_by_survey_dimension",
                    help="Output directory")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATING METRICS-BY-SURVEY-DIMENSION VISUALIZATIONS")
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
    
    # Get survey metrics
    survey_metrics = [col for col in df.columns if col.startswith("survey_")]
    survey_labels = {
        "survey_exaggeration": "Exaggeration",
        "survey_identity": "Identity",
        "survey_overall": "Overall",
        "survey_plausibility": "Plausibility",
        "survey_alignment": "Alignment",
    }
    
    print(f"\n[INFO] Found survey metrics: {survey_metrics}")
    
    # Create visualization for each survey metric
    print("\n[INFO] Creating visualizations...")
    for survey_col in survey_metrics:
        survey_name = survey_col.replace("survey_", "")
        label = survey_labels.get(survey_col, survey_name.title())
        
        output_path = output_dir / f"metrics_vs_{survey_name}.png"
        print(f"  Creating visualization for {label}...")
        create_survey_dimension_visualization(df, survey_col, label, output_path)
    
    # Save summary CSV
    summary_csv = output_dir / "all_metrics_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\n[OK] Saved summary CSV to: {summary_csv}")
    
    print("\n" + "="*80)
    print(f"[OK] All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

