#!/usr/bin/env python3
"""
Compare survey preferences (overall) with automated metrics (HPSv2, CQS, CLIPScore).
Normalizes scores to 0-1 range for fair comparison and creates visualization.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_survey_preferences(survey_csv: Path) -> pd.DataFrame:
    """Load all survey metrics by method."""
    df = pd.read_csv(survey_csv)
    # Pivot to have each metric as a column
    df_pivot = df.pivot(index="method", columns="metric", values="mean").reset_index()
    # Rename columns to have survey_ prefix
    df_pivot.columns = ["method"] + [f"survey_{col}" for col in df_pivot.columns[1:]]
    return df_pivot


def load_hpsv2_scores(hpsv2_csv: Path) -> pd.DataFrame:
    """Load HPSv2 scores by method."""
    df = pd.read_csv(hpsv2_csv)
    df = df[["method", "mean"]].rename(columns={"mean": "hpsv2"})
    # Normalize method names to lowercase for consistency
    df["method"] = df["method"].str.lower()
    return df


def load_cqs_scores(cqs_csv: Path) -> pd.DataFrame:
    """Load CQS scores by method."""
    df = pd.read_csv(cqs_csv)
    df = df[["method", "cqs_mean"]].rename(columns={"cqs_mean": "cqs"})
    # Normalize method names to lowercase for consistency
    df["method"] = df["method"].str.lower()
    return df


def load_clipscore_scores(clipscore_csv: Path) -> pd.DataFrame:
    """Load CLIPScore by method."""
    df = pd.read_csv(clipscore_csv)
    df = df[["method", "mean"]].rename(columns={"mean": "clipscore"})
    # Normalize method names to lowercase for consistency
    df["method"] = df["method"].str.lower()
    return df


def normalize_scores(df: pd.DataFrame, columns: List[str], method: str = "minmax") -> pd.DataFrame:
    """
    Normalize scores to 0-1 range.
    
    Args:
        df: DataFrame with score columns
        columns: List of column names to normalize
        method: "minmax" (0-1) or "zscore" (standardized then minmax)
    
    Returns:
        DataFrame with normalized columns (original columns prefixed with "norm_")
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        if len(values) == 0:
            df_norm[f"norm_{col}"] = np.nan
            continue
        
        if method == "minmax":
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                df_norm[f"norm_{col}"] = 0.5  # All same, set to middle
            else:
                df_norm[f"norm_{col}"] = (df[col] - min_val) / (max_val - min_val)
        elif method == "zscore":
            mean_val = values.mean()
            std_val = values.std()
            if std_val == 0:
                df_norm[f"norm_{col}"] = 0.5
            else:
                z_scores = (df[col] - mean_val) / std_val
                # Convert z-scores to 0-1 range (assuming normal distribution)
                # Using sigmoid-like transformation: 1 / (1 + exp(-z))
                df_norm[f"norm_{col}"] = 1 / (1 + np.exp(-z_scores))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm


def create_comparison_visualization(
    df_combined: pd.DataFrame,
    output_path: Path,
    metrics: List[str] = None,
) -> None:
    """
    Create a grouped bar chart comparing normalized metrics across methods.
    
    Args:
        df_combined: DataFrame with method and normalized metric columns
        output_path: Path to save the figure
        metrics: List of metric column names (without "norm_" prefix)
    """
    if metrics is None:
        metrics = ["survey_overall", "hpsv2", "cqs", "clipscore"]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if f"norm_{m}" in df_combined.columns]
    
    if len(available_metrics) == 0:
        print("Warning: No normalized metrics found for visualization")
        return
    
    # Prepare data for plotting
    methods = df_combined["method"].tolist()
    n_methods = len(methods)
    n_metrics = len(available_metrics)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.5), 6))
    
    # Bar width and positions
    bar_width = 0.8 / n_metrics
    x = np.arange(n_methods)
    
    # Color palette
    colors = sns.color_palette("husl", n_metrics)
    
    # Plot bars for each metric
    bars = []
    labels = []
    for i, metric in enumerate(available_metrics):
        values = df_combined[f"norm_{metric}"].values
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bar = ax.bar(
            x + offset,
            values,
            bar_width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bars.append(bar)
        labels.append(metric.replace("_", " ").title())
    
    # Customize plot
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Score (0-1)", fontsize=12, fontweight="bold")
    ax.set_title("Method Comparison: Survey Preferences vs Automated Metrics\n(Normalized Scores)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved visualization to: {output_path}")
    plt.close()


def create_heatmap_visualization(
    df_combined: pd.DataFrame,
    output_path: Path,
    metrics: List[str] = None,
) -> None:
    """
    Create a heatmap comparing normalized metrics across methods.
    
    Args:
        df_combined: DataFrame with method and normalized metric columns
        output_path: Path to save the figure
        metrics: List of metric column names (without "norm_" prefix)
    """
    if metrics is None:
        metrics = ["survey_overall", "hpsv2", "cqs", "clipscore"]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if f"norm_{m}" in df_combined.columns]
    
    if len(available_metrics) == 0:
        print("Warning: No normalized metrics found for visualization")
        return
    
    # Prepare data for heatmap
    methods = df_combined["method"].tolist()
    heatmap_data = []
    
    for metric in available_metrics:
        values = df_combined[f"norm_{metric}"].values
        heatmap_data.append(values)
    
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), max(6, len(available_metrics) * 0.8)))
    
    metric_labels = [m.replace("_", " ").title() for m in available_metrics]
    
    sns.heatmap(
        heatmap_array,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=methods,
        yticklabels=metric_labels,
        cbar_kws={"label": "Normalized Score (0-1)"},
        linewidths=0.5,
        linecolor="black",
        ax=ax,
    )
    
    ax.set_title("Method Comparison: Survey Preferences vs Automated Metrics\n(Normalized Scores Heatmap)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved heatmap to: {output_path}")
    plt.close()


def compute_correlations(
    df: pd.DataFrame,
    survey_col: str,
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between survey preferences and automated metrics.
    
    Args:
        df: DataFrame with survey and metric columns
        survey_col: Column name for survey preferences
        metric_cols: List of automated metric column names
    
    Returns:
        DataFrame with correlation results
    """
    results = []
    
    for metric_col in metric_cols:
        if metric_col not in df.columns or survey_col not in df.columns:
            continue
        
        # Get valid pairs (non-NaN)
        valid_mask = df[[survey_col, metric_col]].notna().all(axis=1)
        if valid_mask.sum() < 2:
            continue
        
        survey_vals = df.loc[valid_mask, survey_col].values
        metric_vals = df.loc[valid_mask, metric_col].values
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = pearsonr(survey_vals, metric_vals)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = spearmanr(survey_vals, metric_vals)
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan
        
        results.append({
            "metric": metric_col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "pearson_significant": pearson_p < 0.05 if not np.isnan(pearson_p) else False,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "spearman_significant": spearman_p < 0.05 if not np.isnan(spearman_p) else False,
            "n_samples": int(valid_mask.sum()),
        })
    
    return pd.DataFrame(results)


def create_correlation_scatter_plots(
    df: pd.DataFrame,
    survey_col: str,
    metric_cols: List[str],
    output_path: Path,
) -> None:
    """
    Create scatter plots showing correlation between survey preferences and each automated metric.
    
    Args:
        df: DataFrame with survey and metric columns
        survey_col: Column name for survey preferences
        metric_cols: List of automated metric column names
        output_path: Path to save the figure
    """
    available_metrics = [m for m in metric_cols if m in df.columns]
    
    if len(available_metrics) == 0:
        print("Warning: No metrics found for scatter plots")
        return
    
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric_col in enumerate(available_metrics):
        ax = axes[idx]
        
        # Get valid pairs
        valid_mask = df[[survey_col, metric_col]].notna().all(axis=1)
        survey_vals = df.loc[valid_mask, survey_col].values
        metric_vals = df.loc[valid_mask, metric_col].values
        
        # Compute correlation
        try:
            pearson_r, pearson_p = pearsonr(survey_vals, metric_vals)
            spearman_r, spearman_p = spearmanr(survey_vals, metric_vals)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan
        
        # Scatter plot
        ax.scatter(metric_vals, survey_vals, alpha=0.7, s=100, edgecolors="black", linewidth=0.5)
        
        # Add method labels
        if "method" in df.columns:
            methods = df.loc[valid_mask, "method"].values
            for i, method in enumerate(methods):
                ax.annotate(method, (metric_vals[i], survey_vals[i]), 
                           fontsize=8, alpha=0.7, ha="center", va="bottom")
        
        # Add regression line
        if not np.isnan(pearson_r) and len(survey_vals) >= 2:
            z = np.polyfit(metric_vals, survey_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(metric_vals.min(), metric_vals.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label="Fit")
        
        # Labels and title
        metric_label = metric_col.replace("_", " ").title()
        ax.set_xlabel(f"{metric_label} Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Survey Overall Preference", fontsize=11, fontweight="bold")
        
        # Title with correlation info
        sig_pearson = "*" if (not np.isnan(pearson_p) and pearson_p < 0.05) else ""
        sig_spearman = "*" if (not np.isnan(spearman_p) and spearman_p < 0.05) else ""
        title = f"{metric_label}\n"
        title += f"Pearson r={pearson_r:.3f}{sig_pearson} (p={pearson_p:.3f})\n"
        title += f"Spearman ρ={spearman_r:.3f}{sig_spearman} (p={spearman_p:.3f})"
        ax.set_title(title, fontsize=10)
        
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Correlation: Survey Preferences vs Automated Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation scatter plots to: {output_path}")
    plt.close()


def create_correlation_matrix_heatmap(
    df: pd.DataFrame,
    metric_cols: List[str],
    output_path: Path,
) -> None:
    """
    Create a correlation matrix heatmap showing correlations between all metrics.
    
    Args:
        df: DataFrame with metric columns
        metric_cols: List of metric column names
        output_path: Path to save the figure
    """
    available_metrics = [m for m in metric_cols if m in df.columns]
    
    if len(available_metrics) < 2:
        print("Warning: Need at least 2 metrics for correlation matrix")
        return
    
    # Compute correlation matrix
    corr_data = df[available_metrics].corr(method="pearson")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(available_metrics) * 1.2), max(8, len(available_metrics) * 1.2)))
    
    metric_labels = [m.replace("_", " ").title() for m in available_metrics]
    
    sns.heatmap(
        corr_data,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=metric_labels,
        yticklabels=metric_labels,
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=0.5,
        linecolor="black",
        ax=ax,
    )
    
    ax.set_title("Correlation Matrix: Survey Preferences vs Automated Metrics", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation matrix to: {output_path}")
    plt.close()


def create_correlation_summary_heatmap(
    df_correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a heatmap showing correlations for all survey metrics vs automated metrics.
    
    Args:
        df_correlations: DataFrame with columns: survey_metric, metric, pearson_r, spearman_r
        output_path: Path to save the figure
    """
    # Pivot to create matrix: survey_metrics (rows) x automated_metrics (columns)
    pearson_matrix = df_correlations.pivot(index="survey_metric", columns="metric", values="pearson_r")
    
    # Create figure with two subplots (Pearson and Spearman)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pearson correlation heatmap
    sns.heatmap(
        pearson_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        cbar_kws={"label": "Pearson r"},
        linewidths=0.5,
        linecolor="black",
        ax=ax1,
    )
    ax1.set_title("Pearson Correlations", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Automated Metric", fontsize=11)
    ax1.set_ylabel("Survey Metric", fontsize=11)
    
    # Spearman correlation heatmap
    spearman_matrix = df_correlations.pivot(index="survey_metric", columns="metric", values="spearman_r")
    sns.heatmap(
        spearman_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        cbar_kws={"label": "Spearman ρ"},
        linewidths=0.5,
        linecolor="black",
        ax=ax2,
    )
    ax2.set_title("Spearman Correlations", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Automated Metric", fontsize=11)
    ax2.set_ylabel("Survey Metric", fontsize=11)
    
    plt.suptitle("Correlation Summary: All Survey Metrics vs Automated Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved correlation summary heatmap to: {output_path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare survey preferences with automated metrics")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv", help="Survey method means CSV")
    ap.add_argument("--hpsv2_csv", type=str, default="reports/hpsv2/hpsv2_stats_by_method.csv", help="HPSv2 stats CSV")
    ap.add_argument("--cqs_csv", type=str, default="reports/cqs/cqs_summary_by_method.csv", help="CQS summary CSV")
    ap.add_argument("--clipscore_csv", type=str, default="reports/clipscore/clipscore_stats_by_method.csv", help="CLIPScore stats CSV")
    ap.add_argument("--output_dir", type=str, default="reports/method_comparison", help="Output directory")
    ap.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "zscore"], help="Normalization method")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("[INFO] Loading survey preferences (all metrics)...")
    df_survey = load_survey_preferences(Path(args.survey_csv))
    df_survey["method"] = df_survey["method"].str.lower()
    
    # Get list of survey metrics
    survey_metrics = [col for col in df_survey.columns if col.startswith("survey_")]
    print(f"[INFO] Found survey metrics: {survey_metrics}")
    
    print("[INFO] Loading HPSv2 scores...")
    df_hpsv2 = load_hpsv2_scores(Path(args.hpsv2_csv))
    
    print("[INFO] Loading CQS scores...")
    df_cqs = load_cqs_scores(Path(args.cqs_csv))
    
    print("[INFO] Loading CLIPScore...")
    df_clipscore = load_clipscore_scores(Path(args.clipscore_csv))
    
    # Merge all dataframes
    print("[INFO] Merging data...")
    df_combined = df_survey.copy()
    df_combined = df_combined.merge(df_hpsv2, on="method", how="outer")
    df_combined = df_combined.merge(df_cqs, on="method", how="outer")
    df_combined = df_combined.merge(df_clipscore, on="method", how="outer")
    
    # Sort by method name for consistent ordering
    df_combined = df_combined.sort_values("method").reset_index(drop=True)
    
    # Normalize scores (for visualization, normalize all survey metrics and automated metrics)
    print(f"[INFO] Normalizing scores using {args.normalization} method...")
    automated_metrics = ["hpsv2", "cqs", "clipscore"]
    all_score_columns = survey_metrics + automated_metrics
    df_normalized = normalize_scores(df_combined, all_score_columns, method=args.normalization)
    
    # Save normalized data
    output_csv = output_dir / "method_comparison_normalized.csv"
    df_normalized.to_csv(output_csv, index=False)
    print(f"[OK] Saved normalized comparison data to: {output_csv}")
    
    # Print summary (for overall metric only, for backward compatibility)
    print("\n" + "="*80)
    print("NORMALIZED SCORES SUMMARY (0-1 scale) - Overall Metric")
    print("="*80)
    if "survey_overall" in survey_metrics:
        display_cols = ["method"] + [f"norm_survey_overall"] + [f"norm_{c}" for c in automated_metrics if f"norm_{c}" in df_normalized.columns]
        print(df_normalized[[c for c in display_cols if c in df_normalized.columns]].to_string(index=False))
    print("="*80 + "\n")
    
    # Create visualizations (for overall metric, for backward compatibility)
    print("[INFO] Creating comparison visualizations...")
    
    if "survey_overall" in survey_metrics:
        # Grouped bar chart
        bar_chart_path = output_dir / "method_comparison_bar_chart.png"
        create_comparison_visualization(df_normalized, bar_chart_path, metrics=["survey_overall"] + automated_metrics)
        
        # Heatmap
        heatmap_path = output_dir / "method_comparison_heatmap.png"
        create_heatmap_visualization(df_normalized, heatmap_path, metrics=["survey_overall"] + automated_metrics)
    
    # Also save raw scores for reference
    raw_csv = output_dir / "method_comparison_raw.csv"
    df_combined.to_csv(raw_csv, index=False)
    print(f"[OK] Saved raw comparison data to: {raw_csv}")
    
    # CORRELATION ANALYSIS FOR ALL SURVEY METRICS
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: How well do metrics capture human preferences?")
    print("="*80)
    
    all_correlation_results = []
    
    # Compute correlations for each survey metric
    for survey_metric in survey_metrics:
        print(f"\n[INFO] Analyzing correlations for {survey_metric}...")
        correlation_results = compute_correlations(df_combined, survey_metric, automated_metrics)
        
        if not correlation_results.empty:
            # Add survey metric name to results
            correlation_results.insert(0, "survey_metric", survey_metric.replace("survey_", ""))
            all_correlation_results.append(correlation_results)
            
            print(f"  Results for {survey_metric}:")
            print(correlation_results[["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]].to_string(index=False))
    
    # Combine all correlation results
    if all_correlation_results:
        df_all_correlations = pd.concat(all_correlation_results, ignore_index=True)
        correlation_csv = output_dir / "correlation_analysis_all_metrics.csv"
        df_all_correlations.to_csv(correlation_csv, index=False)
        print(f"\n[OK] Saved comprehensive correlation analysis to: {correlation_csv}")
        
        # Summary by survey metric
        print("\n" + "="*80)
        print("CORRELATION SUMMARY BY SURVEY METRIC")
        print("="*80)
        for survey_metric in survey_metrics:
            metric_results = df_all_correlations[df_all_correlations["survey_metric"] == survey_metric.replace("survey_", "")]
            if not metric_results.empty:
                best_pearson = metric_results.loc[metric_results["pearson_r"].abs().idxmax()]
                best_spearman = metric_results.loc[metric_results["spearman_r"].abs().idxmax()]
                print(f"\n{survey_metric.replace('survey_', '').upper()}:")
                print(f"  Best Pearson: {best_pearson['metric']} (r={best_pearson['pearson_r']:.3f}, p={best_pearson['pearson_p']:.3f})")
                print(f"  Best Spearman: {best_spearman['metric']} (ρ={best_spearman['spearman_r']:.3f}, p={best_spearman['spearman_p']:.3f})")
        
        # Create correlation visualizations for each survey metric
        print("\n[INFO] Creating correlation visualizations for each survey metric...")
        
        for survey_metric in survey_metrics:
            metric_name = survey_metric.replace("survey_", "")
            print(f"  Creating visualizations for {metric_name}...")
            
            # Scatter plots
            scatter_path = output_dir / f"correlation_scatter_{metric_name}.png"
            create_correlation_scatter_plots(df_combined, survey_metric, automated_metrics, scatter_path)
            
            # Correlation matrix
            matrix_path = output_dir / f"correlation_matrix_{metric_name}.png"
            all_metrics_for_matrix = [survey_metric] + automated_metrics
            create_correlation_matrix_heatmap(df_combined, all_metrics_for_matrix, matrix_path)
        
        # Create a comprehensive correlation summary heatmap
        print("\n[INFO] Creating comprehensive correlation summary heatmap...")
        summary_heatmap_path = output_dir / "correlation_summary_heatmap.png"
        create_correlation_summary_heatmap(df_all_correlations, summary_heatmap_path)
    else:
        print("Warning: Could not compute correlations (insufficient data)")
    
    print("="*80)
    print(f"\n[OK] All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

