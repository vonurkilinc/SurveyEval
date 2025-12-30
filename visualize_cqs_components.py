#!/usr/bin/env python3
"""
Visualize CQS components (exaggeration-related) per method and compare to survey ratings.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
METHOD_COLORS = {
    "instantid": "#2E86AB",      # Blue
    "qwen": "#F18F01",           # Orange
    "pulid": "#A23B72",          # Purple
    "caricaturebooth": "#C73E1D", # Red
}

COMPONENT_LABELS = {
    "Sexag": "Reference-Tied\nDistinctive Exaggeration",
    "Sexag_free": "Reference-Free\nExaggeration",
    "mag": "Magnitude from\nReference",
    "mag_free": "Magnitude from\nAverage Face",
    "exag_proj": "Projection onto\nDistinctive Direction",
    "survey_exaggeration": "Survey\nExaggeration",
}


def load_data(exaggeration_modes_csv: Path, survey_csv: Path) -> pd.DataFrame:
    """Load exaggeration component data and merge with survey ratings."""
    df_comp = pd.read_csv(exaggeration_modes_csv)
    df_comp["method"] = df_comp["method"].str.lower()
    
    # Load survey data
    df_survey = pd.read_csv(survey_csv)
    df_exag = df_survey[df_survey["metric"] == "exaggeration"][["method", "mean"]].copy()
    df_exag.columns = ["method", "survey_exaggeration"]
    df_exag["method"] = df_exag["method"].str.lower()
    
    # Merge
    df = df_comp.merge(df_exag, on="method", how="inner")
    
    # Normalize survey_exaggeration to 0-1 scale for comparison (assuming 1-7 scale)
    df["survey_exaggeration_norm"] = (df["survey_exaggeration"] - 1.0) / 6.0
    
    return df


def create_component_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart comparing all exaggeration components across methods."""
    components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj", "survey_exaggeration_norm"]
    component_labels = [
        "Sexag\n(Ref-Tied)",
        "Sexag_free\n(Ref-Free)",
        "mag\n(From Ref)",
        "mag_free\n(From Avg)",
        "exag_proj\n(Projection)",
        "Survey\nExaggeration",
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (comp, label) in enumerate(zip(components, component_labels)):
        ax = axes[idx]
        df_sorted = df.sort_values(comp, ascending=False)
        
        bars = ax.bar(
            df_sorted["method"].str.upper(),
            df_sorted[comp],
            color=[METHOD_COLORS.get(m, "#808080") for m in df_sorted["method"]],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )
        
        ax.set_title(label, fontweight="bold", pad=10)
        ax.set_ylabel("Score" if idx % 3 == 0 else "")
        ax.set_ylim(0, max(df[comp].max() * 1.1, 0.1))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    
    plt.suptitle("CQS Exaggeration Components Comparison by Method", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved component bar chart to: {output_path}")
    plt.close()


def create_component_scatter_plots(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plots of each component vs survey exaggeration."""
    components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj"]
    component_labels = [
        "Reference-Tied Distinctive Exaggeration (Sexag)",
        "Reference-Free Exaggeration (Sexag_free)",
        "Magnitude from Reference (mag)",
        "Magnitude from Average Face (mag_free)",
        "Projection onto Distinctive Direction (exag_proj)",
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (comp, label) in enumerate(zip(components, component_labels)):
        ax = axes[idx]
        
        for method in df["method"].unique():
            df_method = df[df["method"] == method]
            ax.scatter(
                df_method["survey_exaggeration"],
                df_method[comp],
                s=200,
                color=METHOD_COLORS.get(method, "#808080"),
                label=method.upper(),
                edgecolors="black",
                linewidth=1.5,
                alpha=0.8,
                zorder=5,
            )
        
        # Add correlation line
        if len(df) >= 2:
            z = np.polyfit(df["survey_exaggeration"], df[comp], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["survey_exaggeration"].min(), df["survey_exaggeration"].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label="Linear fit")
            
            # Compute correlation
            corr = np.corrcoef(df["survey_exaggeration"], df[comp])[0, 1]
            ax.text(
                0.05, 0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )
        
        ax.set_xlabel("Survey Exaggeration Rating", fontweight="bold")
        ax.set_ylabel(label, fontweight="bold")
        ax.set_title(label.split("(")[0].strip(), fontweight="bold", pad=10)
        ax.grid(alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(loc="lower right", framealpha=0.9)
    
    # Remove last subplot
    axes[-1].axis("off")
    
    plt.suptitle("CQS Components vs Survey Exaggeration", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved component scatter plots to: {output_path}")
    plt.close()


def create_ranking_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing ranking of each method across components."""
    components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj", "survey_exaggeration"]
    
    # Compute rankings (1 = highest, 4 = lowest)
    rankings = {}
    for comp in components:
        if comp in df.columns:
            df_sorted = df.sort_values(comp, ascending=False)
            rankings[comp] = {method: rank + 1 for rank, method in enumerate(df_sorted["method"])}
    
    # Create DataFrame
    df_rank = pd.DataFrame(rankings).T
    df_rank = df_rank.reindex(components)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        df_rank,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",  # Reversed: green=1 (best), red=4 (worst)
        vmin=1,
        vmax=4,
        cbar_kws={"label": "Rank (1=Best, 4=Worst)"},
        linewidths=1.5,
        linecolor="black",
        ax=ax,
        square=True,
    )
    
    ax.set_title("Method Rankings Across Exaggeration Components", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel("Component", fontweight="bold")
    ax.set_xticklabels([m.upper() for m in df_rank.columns], rotation=0)
    ax.set_yticklabels([COMPONENT_LABELS.get(c, c) for c in df_rank.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved ranking comparison to: {output_path}")
    plt.close()


def create_radar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create radar chart comparing all components for each method."""
    components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj"]
    n_components = len(components)
    
    # Normalize all components to 0-1 scale for radar chart
    df_norm = df.copy()
    for comp in components:
        if comp in df.columns:
            min_val = df[comp].min()
            max_val = df[comp].max()
            if max_val > min_val:
                df_norm[f"{comp}_norm"] = (df[comp] - min_val) / (max_val - min_val)
            else:
                df_norm[f"{comp}_norm"] = 0.5
    
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))
    
    for method in df_norm["method"].unique():
        df_method = df_norm[df_norm["method"] == method]
        values = [df_method[f"{comp}_norm"].iloc[0] for comp in components]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, "o-", linewidth=2, label=method.upper(), 
                color=METHOD_COLORS.get(method, "#808080"), markersize=8)
        ax.fill(angles, values, alpha=0.15, color=METHOD_COLORS.get(method, "#808080"))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([COMPONENT_LABELS.get(c, c) for c in components], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title("Exaggeration Components Profile by Method\n(Normalized to 0-1)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved radar chart to: {output_path}")
    plt.close()


def create_component_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing correlations between components and survey exaggeration."""
    components = ["Sexag", "Sexag_free", "mag", "mag_free", "exag_proj"]
    
    # Compute correlations with survey exaggeration
    correlations = {}
    for comp in components:
        if comp in df.columns:
            corr = np.corrcoef(df["survey_exaggeration"], df[comp])[0, 1]
            correlations[comp] = corr
    
    # Create DataFrame
    df_corr = pd.DataFrame([correlations], index=["Survey Exaggeration"])
    df_corr = df_corr.T
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.heatmap(
        df_corr,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=1.5,
        linecolor="black",
        ax=ax,
        square=False,
    )
    
    ax.set_title("Component Correlations with Survey Exaggeration", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Component", fontweight="bold")
    ax.set_yticklabels([COMPONENT_LABELS.get(c, c) for c in df_corr.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation heatmap to: {output_path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize CQS exaggeration components")
    ap.add_argument("--exaggeration_modes_csv", type=str, 
                    default="reports/exaggeration_modes_exag2/exaggeration_modes_by_method.csv",
                    help="Exaggeration modes CSV")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv",
                    help="Survey CSV")
    ap.add_argument("--output_dir", type=str, default="reports/component_visualizations",
                    help="Output directory")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATING CQS COMPONENT VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\n[INFO] Loading data...")
    df = load_data(Path(args.exaggeration_modes_csv), Path(args.survey_csv))
    print(f"[INFO] Loaded data for {len(df)} methods")
    print(f"  Methods: {', '.join(df['method'].str.upper().tolist())}")
    
    # Create visualizations
    print("\n[INFO] Creating visualizations...")
    
    print("  1. Component bar chart...")
    create_component_bar_chart(df, output_dir / "component_bar_chart.png")
    
    print("  2. Component scatter plots...")
    create_component_scatter_plots(df, output_dir / "component_scatter_plots.png")
    
    print("  3. Ranking comparison heatmap...")
    create_ranking_comparison(df, output_dir / "component_rankings.png")
    
    print("  4. Radar chart...")
    create_radar_chart(df, output_dir / "component_radar.png")
    
    print("  5. Correlation heatmap...")
    create_component_correlation_heatmap(df, output_dir / "component_correlations.png")
    
    # Save summary CSV
    summary_csv = output_dir / "component_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\n[OK] Saved component summary to: {summary_csv}")
    
    print("\n" + "="*80)
    print(f"[OK] All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

