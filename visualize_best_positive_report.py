#!/usr/bin/env python3
"""
Easy-to-read report visuals:

For each survey metric (alignment, exaggeration, identity, overall, plausibility),
compare 4 metrics:
  - CQS Component (Exaggeration)  -> uses `sexag` but labeled "Exaggeration"
  - CQS Overall                  -> `cqs_optimized`
  - HPSv2                        -> `hpsv2`
  - CLIPScore                    -> `clipscore`

Outputs:
  - correlation_matrix_4metrics.png (heatmap)
  - best_positive_by_survey.png (bar chart)
  - tradeoff_bestpos_vs_worstneg.png (bar chart)
  - best_positive_summary.csv
  - report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SURVEY_ORDER = ["exaggeration", "identity", "overall", "plausibility", "alignment"]

# 4 metrics requested by user, with labeling rules:
# - "sexag" is the CQS exaggeration component, labeled simply as "Exaggeration"
METRICS_4 = [
    ("sexag", "Exaggeration (CQS component)"),
    ("cqs_optimized", "CQS (overall)"),
    ("hpsv2", "HPSv2"),
    ("clipscore", "CLIPScore"),
]


def _load_corr(corr_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(corr_csv)
    df["survey_metric"] = df["survey_metric"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.lower()
    return df


def _prep_matrix(df_corr: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _ in METRICS_4]
    df = df_corr[df_corr["metric"].isin(metrics)].copy()
    # Order survey metrics
    df["survey_metric"] = pd.Categorical(df["survey_metric"], categories=SURVEY_ORDER, ordered=True)
    df["metric"] = pd.Categorical(df["metric"], categories=metrics, ordered=True)
    mat = df.pivot(index="survey_metric", columns="metric", values="pearson_r").sort_index()
    return mat


def _best_positive(df_corr: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _ in METRICS_4]
    rows = []
    for s in SURVEY_ORDER:
        sub = df_corr[(df_corr["survey_metric"] == s) & (df_corr["metric"].isin(metrics))].copy()
        if sub.empty:
            continue
        # best positive: max pearson_r (not abs)
        sub_pos = sub[sub["pearson_r"] > 0].copy()
        if sub_pos.empty:
            best = sub.loc[sub["pearson_r"].idxmax()]  # fallback (least bad)
            rows.append(
                dict(
                    survey_metric=s,
                    best_metric=best["metric"],
                    pearson_r=float(best["pearson_r"]),
                    pearson_p=float(best["pearson_p"]),
                    note="no positive correlations; selected max r",
                )
            )
        else:
            best = sub_pos.loc[sub_pos["pearson_r"].idxmax()]
            rows.append(
                dict(
                    survey_metric=s,
                    best_metric=best["metric"],
                    pearson_r=float(best["pearson_r"]),
                    pearson_p=float(best["pearson_p"]),
                    note="best positive r",
                )
            )
    return pd.DataFrame(rows)


def _worst_negative(df_corr: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _ in METRICS_4]
    rows = []
    for s in SURVEY_ORDER:
        sub = df_corr[(df_corr["survey_metric"] == s) & (df_corr["metric"].isin(metrics))].copy()
        if sub.empty:
            continue
        sub_neg = sub[sub["pearson_r"] < 0].copy()
        if sub_neg.empty:
            rows.append(dict(survey_metric=s, worst_metric=None, pearson_r=np.nan, pearson_p=np.nan))
        else:
            worst = sub_neg.loc[sub_neg["pearson_r"].idxmin()]  # most negative
            rows.append(
                dict(
                    survey_metric=s,
                    worst_metric=worst["metric"],
                    pearson_r=float(worst["pearson_r"]),
                    pearson_p=float(worst["pearson_p"]),
                )
            )
    return pd.DataFrame(rows)


def _metric_label(metric: str) -> str:
    m = metric.lower()
    for key, label in METRICS_4:
        if key == m:
            return label
    return metric


def plot_matrix(mat: pd.DataFrame, out_png: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 4.8))
    ax = sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Pearson correlation: Survey metric vs 4 metrics", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([_metric_label(t.get_text()) for t in ax.get_xticklabels()], rotation=20, ha="right")
    ax.set_yticklabels([t.get_text().title() for t in ax.get_yticklabels()], rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_best_positive(df_best: pd.DataFrame, out_png: Path) -> None:
    if df_best.empty:
        return
    df = df_best.copy()
    df["survey_metric"] = pd.Categorical(df["survey_metric"], categories=SURVEY_ORDER, ordered=True)
    df = df.sort_values("survey_metric")
    colors = []
    for m in df["best_metric"].tolist():
        if m == "sexag":
            colors.append("#2AA198")  # teal
        elif m == "cqs_optimized":
            colors.append("#F18F01")  # orange
        elif m == "hpsv2":
            colors.append("#2E86AB")  # blue
        elif m == "clipscore":
            colors.append("#A23B72")  # purple
        else:
            colors.append("#666666")

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()
    ax.bar(df["survey_metric"].str.title(), df["pearson_r"], color=colors, edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Best positive Pearson r", fontweight="bold")
    ax.set_title("Best positive match per survey metric (among 4 metrics)", fontweight="bold", pad=12)
    for i, row in df.iterrows():
        label = _metric_label(row["best_metric"])
        sig = "*" if float(row["pearson_p"]) < 0.05 else ""
        ax.text(
            i,
            float(row["pearson_r"]) + (0.03 if float(row["pearson_r"]) >= 0 else -0.05),
            f"{label}\n(r={row['pearson_r']:.3f}{sig})",
            ha="center",
            va="bottom" if float(row["pearson_r"]) >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_tradeoff(df_best: pd.DataFrame, df_worst: pd.DataFrame, out_png: Path) -> None:
    if df_best.empty:
        return
    df = df_best.merge(df_worst, on="survey_metric", how="left", suffixes=("_best", "_worst"))
    df["survey_metric"] = pd.Categorical(df["survey_metric"], categories=SURVEY_ORDER, ordered=True)
    df = df.sort_values("survey_metric")

    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(10.8, 5.2))
    ax = plt.gca()
    ax.bar(x - width / 2, df["pearson_r_best"], width, label="Best positive", color="#2E7D32", edgecolor="black")
    ax.bar(x + width / 2, df["pearson_r_worst"], width, label="Most negative", color="#C62828", edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in df["survey_metric"].tolist()])
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson r", fontweight="bold")
    ax.set_title("Tradeoff view: best positive vs most negative (among 4 metrics)", fontweight="bold", pad=12)
    ax.legend(framealpha=0.95)

    for i, row in df.iterrows():
        bm = _metric_label(str(row["best_metric"]))
        ax.text(i - width / 2, float(row["pearson_r_best"]) + 0.03, bm, ha="center", va="bottom", fontsize=8, fontweight="bold")
        if pd.notna(row.get("worst_metric")) and pd.notna(row.get("pearson_r_worst")):
            wm = _metric_label(str(row["worst_metric"]))
            ax.text(i + width / 2, float(row["pearson_r_worst"]) - 0.03, wm, ha="center", va="top", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def write_md(df_best: pd.DataFrame, df_worst: pd.DataFrame, out_md: Path) -> None:
    lines = []
    lines.append("## Best-positive correlation report (4 metrics)\n")
    lines.append("Metrics compared:\n")
    for k, label in METRICS_4:
        lines.append(f"- **{label}** (column: `{k}`)\n")
    lines.append("\n### Best positive per survey metric\n")
    if df_best.empty:
        lines.append("_No results._\n")
    else:
        for _, r in df_best.iterrows():
            s = str(r["survey_metric"]).title()
            m = _metric_label(str(r["best_metric"]))
            rr = float(r["pearson_r"])
            pp = float(r["pearson_p"])
            sig = " (p<0.05)" if pp < 0.05 else ""
            lines.append(f"- **{s}**: **{m}** with **r={rr:.3f}**{sig}\n")
    lines.append("\n### Why negative correlation is meaningful here\n")
    lines.append(
        "A negative correlation means methods that score higher on a metric tend to receive lower human ratings on that survey dimension. "
        "In this project that often reflects *competing objectives*: e.g., exaggeration (caricature-ness) can conflict with identity, plausibility, or alignment.\n"
    )
    lines.append("\n### Most negative per survey metric (tradeoff signal)\n")
    if df_worst.empty:
        lines.append("_No results._\n")
    else:
        for _, r in df_worst.iterrows():
            s = str(r["survey_metric"]).title()
            if pd.isna(r["worst_metric"]):
                lines.append(f"- **{s}**: _no negative correlations among the 4 metrics_\n")
            else:
                m = _metric_label(str(r["worst_metric"]))
                rr = float(r["pearson_r"])
                lines.append(f"- **{s}**: **{m}** with **r={rr:.3f}**\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create easy-to-read best-positive correlation visuals")
    ap.add_argument("--correlations_csv", type=str, default="reports/correlations_exag2/correlations_optimized_cqs.csv")
    ap.add_argument("--out_dir", type=str, default="reports/best_positive_report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_corr = _load_corr(Path(args.correlations_csv))

    mat = _prep_matrix(df_corr)
    df_best = _best_positive(df_corr)
    df_worst = _worst_negative(df_corr)

    # Save CSVs
    df_best.to_csv(out_dir / "best_positive_summary.csv", index=False)

    # Plots
    plot_matrix(mat, out_dir / "correlation_matrix_4metrics.png")
    plot_best_positive(df_best, out_dir / "best_positive_by_survey.png")
    plot_tradeoff(df_best, df_worst, out_dir / "tradeoff_bestpos_vs_worstneg.png")

    # Markdown
    write_md(df_best, df_worst, out_dir / "report.md")

    print(f"[OK] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


