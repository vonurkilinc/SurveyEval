#!/usr/bin/env python3
"""
Create 5 simple graphs (one per survey metric) with 4 bars each:
  1) CQS overall (cqs_optimized correlation)
  2) CQS component that best matches the survey metric (best positive correlation among CQS components)
     - If survey metric is Exaggeration, label that component as "Exaggeration" (for simplicity)
  3) HPSv2
  4) CLIPScore

Bars show Pearson correlation r vs the survey metric (method-level, n=4).

Inputs:
  - reports/correlations_exag2/correlations_optimized_cqs.csv
  - reports/cqs_exag2/cqs_summary_by_method.csv  (for CQS component correlations)
  - survey_report/method_means_ci.csv            (survey means)

Outputs:
  - reports/4bar_corr_per_survey/ (5 pngs + summary csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


SURVEY_METRICS = [
    ("alignment", "Alignment", "survey_alignment"),
    ("exaggeration", "Exaggeration", "survey_exaggeration"),
    ("identity", "Identity", "survey_identity"),
    ("plausibility", "Plausibility", "survey_plausibility"),
    ("overall", "Overall", "survey_overall"),
]

# The 4 metrics we always plot (correlations from correlations_optimized_cqs.csv)
FOUR_METRICS = [
    ("cqs_optimized", "CQS overall"),
    ("hpsv2", "HPSv2"),
    ("clipscore", "CLIPScore"),
]


def _read_corr(corr_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(corr_csv)
    df["survey_metric"] = df["survey_metric"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.lower()
    return df


def _read_survey_means(survey_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(survey_csv)
    # pivot to wide
    wide = df.pivot(index="method", columns="metric", values="mean").reset_index()
    wide["method"] = wide["method"].astype(str).str.lower()
    wide.columns = ["method"] + [f"survey_{c}" for c in wide.columns[1:]]
    return wide


def _read_cqs_summary(cqs_summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cqs_summary_csv)
    df["method"] = df["method"].astype(str).str.lower()
    return df


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def _best_positive_component(
    df_joined: pd.DataFrame,
    survey_col: str,
    component_cols: List[str],
) -> Tuple[Optional[str], float, float]:
    """
    Returns (best_component_col, r, p) where r is the *best positive* Pearson r.
    If no positive correlations exist, returns the max r (could be negative).
    """
    best = None
    best_r = -np.inf
    best_p = np.nan

    for col in component_cols:
        if col not in df_joined.columns:
            continue
        valid = df_joined[[survey_col, col]].notna().all(axis=1)
        r, p = _pearson(df_joined.loc[valid, survey_col].values, df_joined.loc[valid, col].values)
        if np.isnan(r):
            continue
        if r > 0 and r > best_r:
            best = col
            best_r = r
            best_p = p

    if best is not None:
        return best, float(best_r), float(best_p)

    # fallback: choose the max r even if negative
    best = None
    best_r = -np.inf
    best_p = np.nan
    for col in component_cols:
        if col not in df_joined.columns:
            continue
        valid = df_joined[[survey_col, col]].notna().all(axis=1)
        r, p = _pearson(df_joined.loc[valid, survey_col].values, df_joined.loc[valid, col].values)
        if np.isnan(r):
            continue
        if r > best_r:
            best = col
            best_r = r
            best_p = p
    return best, float(best_r), float(best_p)


def _metric_corr(df_corr: pd.DataFrame, survey_metric: str, metric: str) -> Tuple[float, float]:
    sub = df_corr[(df_corr["survey_metric"] == survey_metric) & (df_corr["metric"] == metric)]
    if sub.empty:
        return np.nan, np.nan
    row = sub.iloc[0]
    return float(row["pearson_r"]), float(row["pearson_p"])


def _plot_one(
    out_png: Path,
    title: str,
    bars: List[Tuple[str, float, float]],  # (label, r, p)
) -> None:
    # Academic style settings
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    
    labels = [b[0] for b in bars]
    rs = [b[1] for b in bars]
    ps = [b[2] for b in bars]

    # Academic color scheme (more muted, professional)
    colors = []
    for lab in labels:
        if lab.startswith("CQS component"):
            colors.append("#2C7A7B")  # teal (darker, more academic)
        elif lab == "CQS overall":
            colors.append("#D97706")  # orange (darker)
        elif lab == "HPSv2":
            colors.append("#1E40AF")  # blue (darker)
        elif lab == "CLIPScore":
            colors.append("#7C2D12")  # brown (academic)
        else:
            colors.append("#4B5563")  # gray

    # Larger figure with more space
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.6  # Wider bars for cleaner look
    
    bars_plot = ax.bar(
        x, 
        rs, 
        width=width,
        color=colors, 
        edgecolor="black", 
        linewidth=1.5,
        alpha=0.85
    )
    
    # Zero line
    ax.axhline(0, color="black", linewidth=1.2, zorder=0)
    
    # Y-axis
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Pearson Correlation Coefficient (r)", fontweight="bold", fontsize=12)
    
    # X-axis with better spacing
    ax.set_xticks(x)
    # Shorten labels for better fit
    short_labels = []
    for lab in labels:
        if lab == "CQS overall":
            short_labels.append("CQS\nOverall")
        elif lab.startswith("CQS component: Exaggeration"):
            short_labels.append("CQS Component:\nExaggeration")
        elif lab.startswith("CQS component: Identity"):
            short_labels.append("CQS Component:\nIdentity")
        elif lab.startswith("CQS component: Plausibility"):
            short_labels.append("CQS Component:\nPlausibility")
        elif lab.startswith("CQS component: Salience"):
            short_labels.append("CQS Component:\nSalience")
        elif lab.startswith("CQS component"):
            # Extract component name
            comp_name = lab.replace("CQS component: ", "").split(" (")[0]
            short_labels.append(f"CQS Component:\n{comp_name}")
        else:
            short_labels.append(lab)
    
    ax.set_xticklabels(short_labels, fontsize=11, ha="center")
    
    # Title with more padding
    ax.set_title(title, fontweight="bold", fontsize=14, pad=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Add correlation values on bars with better positioning
    for i, (r, p) in enumerate(zip(rs, ps)):
        if np.isnan(r):
            continue
        sig = "*" if (not np.isnan(p) and p < 0.05) else ""
        # Position text above/below bar with more clearance
        y_offset = 0.08 if r >= 0 else -0.10
        ax.text(
            i,
            r + y_offset,
            f"r = {r:.3f}{sig}",
            ha="center",
            va="bottom" if r >= 0 else "top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        )

    # Add significance note if any
    has_sig = any(not np.isnan(p) and p < 0.05 for p in ps)
    if has_sig:
        ax.text(
            0.02, 0.98,
            "* p < 0.05",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    plt.tight_layout(pad=2.0)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="4-bar correlation plots per survey metric")
    ap.add_argument("--correlations_csv", type=str, default="reports/correlations_exag2/correlations_optimized_cqs.csv")
    ap.add_argument("--cqs_summary_csv", type=str, default="reports/cqs_exag2/cqs_summary_by_method.csv")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv")
    ap.add_argument("--out_dir", type=str, default="reports/4bar_corr_per_survey")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_corr = _read_corr(Path(args.correlations_csv))
    df_survey = _read_survey_means(Path(args.survey_csv))
    df_cqs = _read_cqs_summary(Path(args.cqs_summary_csv))

    # Join survey + CQS components (method-level)
    df_join = df_survey.merge(df_cqs, on="method", how="inner")

    # Candidate component columns from cqs_summary_by_method.csv
    component_cols = [
        "sid_mean",
        "plaus_mean",
        "sal_mean",
        "sexag_mean",
        "sexag_free_mean",
    ]
    # Only keep ones that exist
    component_cols = [c for c in component_cols if c in df_join.columns]

    summary_rows = []

    for survey_key, survey_label, survey_col in SURVEY_METRICS:
        # Find best positive CQS component for THIS survey column
        best_comp, comp_r, comp_p = _best_positive_component(df_join, survey_col, component_cols)
        if best_comp is None:
            comp_label = "CQS component"
        else:
            # Friendly label mapping
            label_map = {
                "sid_mean": "Identity (Sid)",
                "plaus_mean": "Plausibility (Splaus)",
                "sal_mean": "Salience (Ssal)",
                "sexag_mean": "Exaggeration (Sexag)",
                "sexag_free_mean": "Exaggeration-free (Sexag_free)",
            }
            if survey_key == "exaggeration":
                # user asked: name it simply "Exaggeration" for the CQS exaggeration component
                comp_label = "CQS component: Exaggeration"
            else:
                comp_label = f"CQS component: {label_map.get(best_comp, best_comp)}"

        # Pull correlations for the other 3 metrics from correlations CSV
        cqs_r, cqs_p = _metric_corr(df_corr, survey_key, "cqs_optimized")
        hps_r, hps_p = _metric_corr(df_corr, survey_key, "hpsv2")
        clip_r, clip_p = _metric_corr(df_corr, survey_key, "clipscore")

        bars = [
            ("CQS overall", cqs_r, cqs_p),
            (comp_label, comp_r, comp_p),
            ("HPSv2", hps_r, hps_p),
            ("CLIPScore", clip_r, clip_p),
        ]

        out_png = out_dir / f"corr_4bars_{survey_key}.png"
        _plot_one(out_png, f"Correlation vs Survey {survey_label}", bars)

        summary_rows.append(
            dict(
                survey_metric=survey_label,
                cqs_overall_r=cqs_r,
                cqs_component=best_comp,
                cqs_component_label=comp_label,
                cqs_component_r=comp_r,
                hpsv2_r=hps_r,
                clipscore_r=clip_r,
            )
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"[OK] Wrote 5 plots + summary.csv to: {out_dir}")


if __name__ == "__main__":
    main()


