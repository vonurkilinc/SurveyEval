#!/usr/bin/env python3
"""
Compare two exaggeration notions:
1) Reference-tied distinctive exaggeration (Sexag)
2) Reference-free exaggeration vs dataset-level average face (Sexag_free)

Inputs: step4.py output cqs_pairs.csv
Outputs: method-level summary CSV + a scatter plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare reference-tied vs reference-free exaggeration metrics")
    ap.add_argument("--cqs_pairs_csv", type=str, required=True, help="Path to cqs_pairs.csv produced by step4.py")
    ap.add_argument("--out_dir", type=str, default="reports/exaggeration_modes", help="Output directory")
    args = ap.parse_args()

    cqs_pairs_csv = Path(args.cqs_pairs_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cqs_pairs_csv)
    df["method"] = df["method"].astype(str).str.lower()

    # Aggregate by method
    cols = [c for c in ["Sexag", "Sexag_free", "exag_proj", "mag", "mag_free", "Sid", "Splaus", "CQS"] if c in df.columns]
    agg = df.groupby("method").agg({c: "mean" for c in cols}).reset_index()
    agg = agg.sort_values(by="Sexag_free" if "Sexag_free" in agg.columns else cols[0], ascending=False)

    out_csv = out_dir / "exaggeration_modes_by_method.csv"
    agg.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")

    # Scatter plot: Sexag_free vs Sexag (if both exist)
    if "Sexag" in agg.columns and "Sexag_free" in agg.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(agg["Sexag_free"], agg["Sexag"])
        for _, row in agg.iterrows():
            ax.annotate(row["method"], (row["Sexag_free"], row["Sexag"]), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("Reference-free exaggeration (Sexag_free)")
        ax.set_ylabel("Reference-tied distinctive exaggeration (Sexag)")
        ax.set_title("Exaggeration modes by method")
        ax.grid(True, alpha=0.3)
        out_png = out_dir / "exaggeration_modes_scatter.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[OK] Wrote: {out_png}")
    else:
        print("[WARN] Missing Sexag and/or Sexag_free columns; scatter plot skipped.")


if __name__ == "__main__":
    main()


