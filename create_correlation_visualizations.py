#!/usr/bin/env python3
"""
Create scientific report-quality visualizations for correlation analysis.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

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

# Color palette
COLORS = {
    "hpsv2": "#2E86AB",      # Blue
    "clipscore": "#A23B72",  # Purple
    "cqs_optimized": "#F18F01",  # Orange
    "cqs_original": "#C73E1D",   # Red
    "cqs_step4": "#C73E1D",      # Red (step4-computed CQS)
    "sexag": "#2AA198",          # Teal
    "sexag_free": "#6C71C4",     # Indigo
}

METRIC_LABELS = {
    "hpsv2": "HPSv2",
    "clipscore": "CLIPScore",
    "cqs_optimized": "CQS",
    "cqs": "CQS",
    "cqs_step4": "CQS (step4)",
    "sexag": "Sexag (ref-tied)",
    "sexag_free": "Sexag_free (ref-free)",
}


def load_data(correlations_csv: Path, method_comparison_csv: Path, survey_csv: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load correlation and method comparison data."""
    df_corr = pd.read_csv(correlations_csv)
    df_methods = pd.read_csv(method_comparison_csv)
    
    # If survey CSV is provided, merge overall preference data
    if survey_csv and survey_csv.exists():
        df_survey = pd.read_csv(survey_csv)
        df_overall = df_survey[df_survey["metric"] == "overall"][["method", "mean"]].copy()
        df_overall.columns = ["method", "survey_overall"]
        df_overall["method"] = df_overall["method"].str.lower()
        df_methods = df_methods.merge(df_overall, on="method", how="left")
    
    return df_corr, df_methods


def create_correlation_comparison_heatmap(df_corr: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap comparing correlations across metrics and survey dimensions."""
    # Filter for overall preference and remove any original CQS if present
    df_overall = df_corr[df_corr["survey_metric"] == "overall"].copy()
    # Keep all available metrics in the correlations CSV (so new metrics like sexag/sexag_free appear)
    # but preserve a stable ordering for the most common ones.
    preferred = ["hpsv2", "clipscore", "cqs_optimized", "cqs_step4", "sexag", "sexag_free"]
    keep = [m for m in preferred if m in df_overall["metric"].unique()] + [
        m for m in sorted(df_overall["metric"].unique()) if m not in preferred
    ]
    df_overall = df_overall[df_overall["metric"].isin(keep)].copy()
    df_overall["metric"] = pd.Categorical(df_overall["metric"], categories=keep, ordered=True)
    
    # Create pivot table
    pivot_pearson = df_overall.pivot(index="metric", columns="survey_metric", values="pearson_r").sort_index()
    pivot_spearman = df_overall.pivot(index="metric", columns="survey_metric", values="spearman_r").sort_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Pearson correlation heatmap
    sns.heatmap(
        pivot_pearson,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
        linewidths=1,
        linecolor="white",
        ax=ax1,
        cbar_ax=None if ax2 is None else None,
    )
    ax1.set_title("Pearson Correlation", fontweight="bold", pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("Metric", fontweight="bold")
    ax1.set_yticklabels([METRIC_LABELS.get(label.get_text(), label.get_text()) for label in ax1.get_yticklabels()])
    
    # Spearman correlation heatmap
    sns.heatmap(
        pivot_spearman,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman ρ"},
        linewidths=1,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title("Spearman Correlation", fontweight="bold", pad=10)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_yticklabels([METRIC_LABELS.get(label.get_text(), label.get_text()) for label in ax2.get_yticklabels()])
    
    plt.suptitle("Correlation with Human Overall Preferences", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation comparison heatmap to: {output_path}")
    plt.close()


def create_correlation_bar_chart(df_corr: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart comparing correlations for overall preference."""
    df_overall = df_corr[df_corr["survey_metric"] == "overall"].copy()
    # Only keep the three main metrics
    df_overall = df_overall[df_overall["metric"].isin(["hpsv2", "clipscore", "cqs_optimized"])].copy()
    
    # Sort by Pearson correlation
    df_overall = df_overall.sort_values("pearson_r", ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pearson correlations
    colors_pearson = [COLORS.get(metric, "#808080") for metric in df_overall["metric"].values]
    bars1 = ax1.barh(
        [METRIC_LABELS.get(m, m) for m in df_overall["metric"].values],
        df_overall["pearson_r"].values,
        color=colors_pearson,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    
    # Add value labels
    for i, (bar, val, p_val) in enumerate(zip(bars1, df_overall["pearson_r"].values, df_overall["pearson_p"].values)):
        sig = "*" if p_val < 0.05 else ""
        ax1.text(
            val + 0.02 if val >= 0 else val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}{sig}",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    
    ax1.set_xlabel("Pearson Correlation Coefficient (r)", fontweight="bold")
    ax1.set_title("Pearson Correlations", fontweight="bold", pad=10)
    ax1.set_xlim(-0.2, 1.0)
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Spearman correlations
    colors_spearman = [COLORS.get(metric, "#808080") for metric in df_overall["metric"].values]
    bars2 = ax2.barh(
        [METRIC_LABELS.get(m, m) for m in df_overall["metric"].values],
        df_overall["spearman_r"].values,
        color=colors_spearman,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    
    # Add value labels
    for i, (bar, val, p_val) in enumerate(zip(bars2, df_overall["spearman_r"].values, df_overall["spearman_p"].values)):
        sig = "*" if p_val < 0.05 else ""
        ax2.text(
            val + 0.02 if val >= 0 else val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}{sig}",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    
    ax2.set_xlabel("Spearman Correlation Coefficient (ρ)", fontweight="bold")
    ax2.set_title("Spearman Correlations", fontweight="bold", pad=10)
    ax2.set_xlim(-0.2, 1.0)
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    
    plt.suptitle("Automated Metrics vs Human Overall Preferences", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation bar chart to: {output_path}")
    plt.close()


def create_multi_dimension_heatmap(df_corr: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing correlations across all survey dimensions."""
    # Filter to only the three main metrics
    df_filtered = df_corr[df_corr["metric"].isin(["hpsv2", "clipscore", "cqs_optimized"])].copy()
    
    # Create pivot tables
    pivot_pearson = df_filtered.pivot(index="metric", columns="survey_metric", values="pearson_r")
    pivot_spearman = df_filtered.pivot(index="metric", columns="survey_metric", values="spearman_r")
    
    # Reorder columns
    column_order = ["overall", "identity", "alignment", "plausibility", "exaggeration"]
    pivot_pearson = pivot_pearson[[c for c in column_order if c in pivot_pearson.columns]]
    pivot_spearman = pivot_spearman[[c for c in column_order if c in pivot_spearman.columns]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson correlations
    sns.heatmap(
        pivot_pearson,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
        linewidths=1,
        linecolor="white",
        ax=ax1,
        xticklabels=[c.title() for c in pivot_pearson.columns],
        yticklabels=[METRIC_LABELS.get(m, m) for m in pivot_pearson.index],
    )
    ax1.set_title("Pearson Correlations", fontweight="bold", pad=10)
    ax1.set_xlabel("Survey Dimension", fontweight="bold")
    ax1.set_ylabel("Automated Metric", fontweight="bold")
    
    # Spearman correlations
    sns.heatmap(
        pivot_spearman,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman ρ"},
        linewidths=1,
        linecolor="white",
        ax=ax2,
        xticklabels=[c.title() for c in pivot_spearman.columns],
        yticklabels=[METRIC_LABELS.get(m, m) for m in pivot_spearman.index],
    )
    ax2.set_title("Spearman Correlations", fontweight="bold", pad=10)
    ax2.set_xlabel("Survey Dimension", fontweight="bold")
    ax2.set_ylabel("")
    
    plt.suptitle("Correlation Analysis: All Survey Dimensions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved multi-dimension heatmap to: {output_path}")
    plt.close()


def create_scatter_plots(df_methods: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plots showing relationship between metrics and survey preferences."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["hpsv2", "clipscore", "cqs_optimized"]
    # Use survey_overall if available, otherwise use first survey column
    if "survey_overall" in df_methods.columns:
        survey_col = "survey_overall"
    else:
        survey_cols = [c for c in df_methods.columns if c.startswith("survey_")]
        survey_col = survey_cols[0] if survey_cols else None
    
    if survey_col is None:
        print("Warning: No survey column found, skipping scatter plots")
        return
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(
            df_methods[metric].values,
            df_methods[survey_col].values,
            s=200,
            alpha=0.7,
            color=COLORS.get(metric, "#808080"),
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )
        
        # Add method labels
        for _, row in df_methods.iterrows():
            ax.annotate(
                row["method"].title(),
                (row[metric], row[survey_col]),
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        # Add regression line
        x_vals = df_methods[metric].values
        y_vals = df_methods[survey_col].values
        if len(x_vals) >= 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label="Linear fit")
        
        # Labels and title
        metric_label = METRIC_LABELS.get(metric, metric)
        ax.set_xlabel(f"{metric_label} Score", fontweight="bold")
        ax.set_ylabel("Survey Overall Preference", fontweight="bold")
        ax.set_title(metric_label, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)
    
    plt.suptitle("Relationship: Automated Metrics vs Human Overall Preferences", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved scatter plots to: {output_path}")
    plt.close()


def create_metric_comparison(output_path: Path) -> None:
    """Create bar chart comparing all three metrics."""
    # Data for comparison
    metrics = ["HPSv2", "CLIPScore", "CQS"]
    correlations = [0.8235, 0.5040, 0.9052]
    colors = [COLORS["hpsv2"], COLORS["clipscore"], COLORS["cqs_optimized"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        metrics,
        correlations,
        color=colors,
        edgecolor="black",
        linewidth=2,
        alpha=0.8,
    )
    
    # Add value labels
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    
    ax.set_ylabel("Pearson Correlation Coefficient (r)", fontweight="bold", fontsize=12)
    ax.set_title("Automated Metrics vs Human Overall Preferences", fontweight="bold", fontsize=14, pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved metric comparison to: {output_path}")
    plt.close()


def create_radar_chart(df_corr: pd.DataFrame, output_path: Path) -> None:
    """Create radar chart comparing metrics across dimensions."""
    # Prepare data
    dimensions = ["overall", "identity", "alignment", "plausibility", "exaggeration"]
    metrics = ["hpsv2", "clipscore", "cqs_optimized"]
    
    # Filter to only the three main metrics
    df_filtered = df_corr[df_corr["metric"].isin(metrics)].copy()
    
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    
    for metric in metrics:
        values = []
        for dim in dimensions:
            row = df_filtered[(df_filtered["survey_metric"] == dim) & (df_filtered["metric"] == metric)]
            if not row.empty:
                values.append(row["pearson_r"].values[0])
            else:
                values.append(0.0)
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, "o-", linewidth=2, label=METRIC_LABELS.get(metric, metric), 
                color=COLORS.get(metric, "#808080"), markersize=8)
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(metric, "#808080"))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.title() for d in dimensions], fontsize=11, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"], fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title("Correlation Profile Across Survey Dimensions\n(Pearson Correlations)", 
                 fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved radar chart to: {output_path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create scientific report visualizations")
    ap.add_argument("--correlations_csv", type=str, default="reports/correlations_optimized/correlations_optimized_cqs.csv")
    ap.add_argument("--method_comparison_csv", type=str, default="reports/correlations_optimized/method_scores_comparison.csv")
    ap.add_argument("--output_dir", type=str, default="reports/correlations_optimized/figures")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATING SCIENTIFIC REPORT VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\n[INFO] Loading data...")
    survey_csv = Path(args.correlations_csv).parent.parent / "survey_report" / "method_means_ci.csv"
    df_corr, df_methods = load_data(Path(args.correlations_csv), Path(args.method_comparison_csv), survey_csv)
    
    # Create visualizations
    print("\n[INFO] Creating visualizations...")
    
    # 1. Correlation comparison heatmap
    print("  1. Correlation comparison heatmap...")
    create_correlation_comparison_heatmap(df_corr, output_dir / "fig1_correlation_comparison_heatmap.png")
    
    # 2. Correlation bar chart
    print("  2. Correlation bar chart...")
    create_correlation_bar_chart(df_corr, output_dir / "fig2_correlation_bar_chart.png")
    
    # 3. Multi-dimension heatmap
    print("  3. Multi-dimension heatmap...")
    create_multi_dimension_heatmap(df_corr, output_dir / "fig3_multi_dimension_heatmap.png")
    
    # 4. Scatter plots
    print("  4. Scatter plots...")
    create_scatter_plots(df_methods, output_dir / "fig4_scatter_plots.png")
    
    # 5. Metric comparison
    print("  5. Metric comparison...")
    create_metric_comparison(output_dir / "fig5_metric_comparison.png")
    
    # 6. Radar chart
    print("  6. Radar chart...")
    create_radar_chart(df_corr, output_dir / "fig6_radar_chart.png")
    
    print("\n" + "="*80)
    print(f"[OK] All visualizations saved to: {output_dir}")
    print("="*80)
    print("\nGenerated figures:")
    print("  - fig1_correlation_comparison_heatmap.png")
    print("  - fig2_correlation_bar_chart.png")
    print("  - fig3_multi_dimension_heatmap.png")
    print("  - fig4_scatter_plots.png")
    print("  - fig5_metric_comparison.png")
    print("  - fig6_radar_chart.png")


if __name__ == "__main__":
    main()

