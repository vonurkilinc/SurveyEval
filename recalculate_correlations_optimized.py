#!/usr/bin/env python3
"""
Recalculate correlations for HPSv2, CLIPScore, and CQS (with optimized weights) 
against human survey preferences.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Optimized CQS weights (multi-metric optimization: overall > exaggeration > identity)
OPTIMIZED_WEIGHTS = {
    "alpha_id": 0.465116,
    "beta_sal": 0.023256,
    "gamma_plaus": 0.465115,
    "delta_coh": 0.023256,
    "eta_mag": 0.023256,
    # New: distinctive exaggeration (step4.py writes Sexag)
    "zeta_exag": 0.023256,
}


def load_cqs_components(cqs_pairs_csv: Path) -> pd.DataFrame:
    """Load CQS component scores."""
    df = pd.read_csv(cqs_pairs_csv)
    df["method"] = df["method"].str.lower()
    return df


def compute_cqs_with_weights(
    df_cqs: pd.DataFrame,
    alpha_id: float,
    beta_sal: float,
    gamma_plaus: float,
    delta_coh: float,
    eta_mag: float,
    zeta_exag: float = 0.0,
    cqs_mode: str = "noid_fallback",
) -> pd.DataFrame:
    """
    Compute CQS scores with given weights.
    """
    df = df_cqs.copy()
    
    # Normalize weights to sum to 1.0
    total_weight = alpha_id + beta_sal + gamma_plaus + delta_coh + eta_mag + zeta_exag
    if total_weight > 1e-12:
        alpha_id /= total_weight
        beta_sal /= total_weight
        gamma_plaus /= total_weight
        delta_coh /= total_weight
        eta_mag /= total_weight
        zeta_exag /= total_weight
    
    cqs_scores = []
    
    for _, row in df.iterrows():
        sid = row.get("Sid", np.nan)
        ssal = row.get("Ssal", 0.0)
        splaus = row.get("Splaus", 0.0)
        scoh = row.get("Scoh", 1.0)
        smag = row.get("Smag", 1.0)
        sexag = row.get("Sexag", 0.0)
        gate_face = row.get("gate_face", 1)
        gate_id_manifest = row.get("gate_id_manifest", 1)
        
        if cqs_mode == "noid_fallback":
            g = gate_face * gate_id_manifest
            if pd.isna(sid):
                rest_total = beta_sal + gamma_plaus + delta_coh + eta_mag + zeta_exag
                if rest_total > 1e-12:
                    cqs = g * (
                        (beta_sal / rest_total) * ssal +
                        (gamma_plaus / rest_total) * splaus +
                        (delta_coh / rest_total) * scoh +
                        (eta_mag / rest_total) * smag +
                        (zeta_exag / rest_total) * sexag
                    )
                else:
                    cqs = 0.0
            else:
                cqs = g * (
                    alpha_id * float(sid) +
                    beta_sal * ssal +
                    gamma_plaus * splaus +
                    delta_coh * scoh +
                    eta_mag * smag +
                    zeta_exag * sexag
                )
        elif cqs_mode == "manifest":
            g = gate_face * gate_id_manifest
            sid_val = 0.0 if pd.isna(sid) else float(sid)
            cqs = g * (
                alpha_id * sid_val +
                beta_sal * ssal +
                gamma_plaus * splaus +
                delta_coh * scoh +
                eta_mag * smag +
                zeta_exag * sexag
            )
        else:  # strict
            gate_id_tau = row.get("gate_id_tau", 0)
            g = gate_face * gate_id_tau
            sid_val = 0.0 if pd.isna(sid) else float(sid)
            cqs = g * (
                alpha_id * sid_val +
                beta_sal * ssal +
                gamma_plaus * splaus +
                delta_coh * scoh +
                eta_mag * smag +
                zeta_exag * sexag
            )
        
        cqs_scores.append(max(0.0, min(1.0, cqs)))
    
    df["CQS_optimized"] = cqs_scores
    return df


def load_survey_preferences(survey_csv: Path) -> pd.DataFrame:
    """Load all survey metrics by method."""
    df = pd.read_csv(survey_csv)
    df_pivot = df.pivot(index="method", columns="metric", values="mean").reset_index()
    df_pivot.columns = ["method"] + [f"survey_{col}" for col in df_pivot.columns[1:]]
    df_pivot["method"] = df_pivot["method"].str.lower()
    return df_pivot


def load_hpsv2_scores(hpsv2_csv: Path) -> pd.DataFrame:
    """Load HPSv2 scores by method."""
    df = pd.read_csv(hpsv2_csv)
    df = df[["method", "mean"]].rename(columns={"mean": "hpsv2"})
    df["method"] = df["method"].str.lower()
    return df


def load_clipscore_scores(clipscore_csv: Path) -> pd.DataFrame:
    """Load CLIPScore by method."""
    df = pd.read_csv(clipscore_csv)
    df = df[["method", "mean"]].rename(columns={"mean": "clipscore"})
    df["method"] = df["method"].str.lower()
    return df


def compute_correlations(
    df: pd.DataFrame,
    survey_col: str,
    metric_cols: List[str],
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations."""
    results = []
    
    for metric_col in metric_cols:
        if metric_col not in df.columns or survey_col not in df.columns:
            continue
        
        valid_mask = df[[survey_col, metric_col]].notna().all(axis=1)
        if valid_mask.sum() < 2:
            continue
        
        survey_vals = df.loc[valid_mask, survey_col].values
        metric_vals = df.loc[valid_mask, metric_col].values
        
        try:
            pearson_r, pearson_p = pearsonr(survey_vals, metric_vals)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan
        
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalculate correlations with optimized CQS weights")
    ap.add_argument("--cqs_pairs_csv", type=str, default="reports/cqs/cqs_pairs.csv", help="CQS pairs CSV")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv", help="Survey CSV")
    ap.add_argument("--hpsv2_csv", type=str, default="reports/hpsv2/hpsv2_stats_by_method.csv", help="HPSv2 CSV")
    ap.add_argument("--clipscore_csv", type=str, default="reports/clipscore/clipscore_stats_by_method.csv", help="CLIPScore CSV")
    ap.add_argument("--cqs_mode", type=str, default="noid_fallback", choices=["strict", "manifest", "noid_fallback"])
    ap.add_argument("--output_dir", type=str, default="reports/correlations_optimized", help="Output directory")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RECALCULATING CORRELATIONS WITH OPTIMIZED CQS WEIGHTS")
    print("="*80)
    print(f"\nOptimized CQS weights:")
    for key, value in OPTIMIZED_WEIGHTS.items():
        print(f"  {key}: {value:.6f}")
    
    # Load data
    print("\n[INFO] Loading data...")
    df_cqs = load_cqs_components(Path(args.cqs_pairs_csv))
    df_survey = load_survey_preferences(Path(args.survey_csv))
    df_hpsv2 = load_hpsv2_scores(Path(args.hpsv2_csv))
    df_clipscore = load_clipscore_scores(Path(args.clipscore_csv))
    
    # Compute CQS with optimized weights
    print("[INFO] Computing CQS with optimized weights...")
    df_cqs_opt = compute_cqs_with_weights(
        df_cqs,
        OPTIMIZED_WEIGHTS["alpha_id"],
        OPTIMIZED_WEIGHTS["beta_sal"],
        OPTIMIZED_WEIGHTS["gamma_plaus"],
        OPTIMIZED_WEIGHTS["delta_coh"],
        OPTIMIZED_WEIGHTS["eta_mag"],
        OPTIMIZED_WEIGHTS.get("zeta_exag", 0.0),
        args.cqs_mode,
    )
    
    # Aggregate CQS by method
    method_cqs_opt = df_cqs_opt.groupby("method")["CQS_optimized"].mean().reset_index()
    method_cqs_opt.columns = ["method", "cqs_optimized"]

    # Also aggregate the raw Step-4 CQS column (uses weights passed to step4.py at generation time)
    method_cqs_step4 = None
    if "CQS" in df_cqs.columns:
        method_cqs_step4 = df_cqs.groupby("method")["CQS"].mean().reset_index()
        method_cqs_step4.columns = ["method", "cqs_step4"]
    
    # Aggregate Sexag by method (if present) as a direct exaggeration-alignment metric
    method_sexag = None
    if "Sexag" in df_cqs.columns:
        method_sexag = df_cqs.groupby("method")["Sexag"].mean().reset_index()
        method_sexag.columns = ["method", "sexag"]
    
    # Aggregate Sexag_free by method (if present) as a reference-free exaggeration metric
    method_sexag_free = None
    if "Sexag_free" in df_cqs.columns:
        method_sexag_free = df_cqs.groupby("method")["Sexag_free"].mean().reset_index()
        method_sexag_free.columns = ["method", "sexag_free"]
    
    # Merge all data
    df_combined = df_survey.copy()
    df_combined = df_combined.merge(df_hpsv2, on="method", how="inner")
    df_combined = df_combined.merge(df_clipscore, on="method", how="inner")
    df_combined = df_combined.merge(method_cqs_opt, on="method", how="inner")
    if method_cqs_step4 is not None:
        df_combined = df_combined.merge(method_cqs_step4, on="method", how="inner")
    if method_sexag is not None:
        df_combined = df_combined.merge(method_sexag, on="method", how="inner")
    if method_sexag_free is not None:
        df_combined = df_combined.merge(method_sexag_free, on="method", how="inner")
    
    print(f"\n[INFO] Data merged. Methods: {', '.join(df_combined['method'].tolist())}")
    print(f"  Survey metrics: {[c for c in df_combined.columns if c.startswith('survey_')]}")
    
    # Compute correlations for each survey metric
    survey_metrics = [col for col in df_combined.columns if col.startswith("survey_")]
    automated_metrics = ["hpsv2", "clipscore", "cqs_optimized"]
    if "cqs_step4" in df_combined.columns:
        automated_metrics.append("cqs_step4")
    if "sexag" in df_combined.columns:
        automated_metrics.append("sexag")
    if "sexag_free" in df_combined.columns:
        automated_metrics.append("sexag_free")
    
    all_results = []
    
    print("\n" + "="*80)
    print("CORRELATION RESULTS (with optimized CQS weights)")
    print("="*80)
    
    for survey_metric in survey_metrics:
        metric_name = survey_metric.replace("survey_", "")
        print(f"\n{metric_name.upper()}:")
        print("-" * 80)
        
        correlations = compute_correlations(df_combined, survey_metric, automated_metrics)
        
        if not correlations.empty:
            correlations.insert(0, "survey_metric", metric_name)
            all_results.append(correlations)
            
            # Print results
            for _, row in correlations.iterrows():
                sig_pearson = "*" if row["pearson_significant"] else ""
                sig_spearman = "*" if row["spearman_significant"] else ""
                print(f"  {row['metric']:20s}:")
                print(f"    Pearson r = {row['pearson_r']:7.4f}{sig_pearson} (p = {row['pearson_p']:.4f})")
                # Avoid non-ASCII printing issues on Windows consoles (cp1252)
                print(f"    Spearman rho = {row['spearman_r']:7.4f}{sig_spearman} (p = {row['spearman_p']:.4f})")
            
            # Find best metric
            best_pearson = correlations.loc[correlations["pearson_r"].abs().idxmax()]
            best_spearman = correlations.loc[correlations["spearman_r"].abs().idxmax()]
            print(f"\n  Best Pearson: {best_pearson['metric']} (r={best_pearson['pearson_r']:.4f})")
            print(f"  Best Spearman: {best_spearman['metric']} (rho={best_spearman['spearman_r']:.4f})")
    
    # Combine all results
    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_csv = output_dir / "correlations_optimized_cqs.csv"
        df_all_results.to_csv(output_csv, index=False)
        print(f"\n[OK] Saved correlation results to: {output_csv}")
        
        # Save method comparison
        comparison_csv = output_dir / "method_scores_comparison.csv"
        df_combined[["method"] + [survey_metrics[0]] + automated_metrics].to_csv(comparison_csv, index=False)
        print(f"[OK] Saved method comparison to: {comparison_csv}")
        
        # Summary for overall metric
        if "survey_overall" in survey_metrics:
            overall_results = df_all_results[df_all_results["survey_metric"] == "overall"]
            print("\n" + "="*80)
            print("SUMMARY: Overall Preference Correlations")
            print("="*80)
            print("\nRanking by Pearson correlation:")
            overall_sorted = overall_results.sort_values("pearson_r", ascending=False)
            for idx, (_, row) in enumerate(overall_sorted.iterrows(), 1):
                print(f"{idx}. {row['metric']:20s}: r = {row['pearson_r']:7.4f} (p = {row['pearson_p']:.4f})")
            
            print("\nRanking by Spearman correlation:")
            overall_sorted_rho = overall_results.sort_values("spearman_r", ascending=False)
            for idx, (_, row) in enumerate(overall_sorted_rho.iterrows(), 1):
                print(f"{idx}. {row['metric']:20s}: rho = {row['spearman_r']:7.4f} (p = {row['spearman_p']:.4f})")
    
    print("\n" + "="*80)
    print(f"[OK] All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

