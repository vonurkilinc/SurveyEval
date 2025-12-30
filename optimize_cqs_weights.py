#!/usr/bin/env python3
"""
Optimize CQS weighting coefficients to maximize correlation with human survey overall preferences.

The CQS formula is:
CQS = alpha_id * Sid + beta_sal * Ssal + gamma_plaus * Splaus + delta_coh * Scoh + eta_mag * Smag

We optimize these weights to maximize Pearson correlation with survey overall ratings.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr, spearmanr

# Constraint: weights must sum to 1.0 (or close to it)
# We'll use a soft constraint by normalizing weights


def load_cqs_components(cqs_pairs_csv: Path) -> pd.DataFrame:
    """Load CQS component scores."""
    df = pd.read_csv(cqs_pairs_csv)
    # Normalize method names to lowercase
    df["method"] = df["method"].str.lower()
    return df


def load_survey_ratings(survey_csv: Path, metrics: List[str] = None) -> pd.DataFrame:
    """Load survey ratings aggregated by method for specified metrics."""
    if metrics is None:
        metrics = ["overall"]
    
    df = pd.read_csv(survey_csv)
    df_pivot = df.pivot(index="method", columns="metric", values="mean").reset_index()
    df_pivot.columns = ["method"] + [f"survey_{col}" for col in df_pivot.columns[1:]]
    df_pivot["method"] = df_pivot["method"].str.lower()
    
    # Keep only requested metrics
    cols_to_keep = ["method"] + [f"survey_{m}" for m in metrics if f"survey_{m}" in df_pivot.columns]
    return df_pivot[cols_to_keep]


def compute_cqs_with_weights(
    df_cqs: pd.DataFrame,
    alpha_id: float,
    beta_sal: float,
    gamma_plaus: float,
    delta_coh: float,
    eta_mag: float,
    cqs_mode: str = "noid_fallback",
) -> pd.DataFrame:
    """
    Compute CQS scores with given weights.
    
    Args:
        df_cqs: DataFrame with CQS components (Sid, Ssal, Splaus, Scoh, Smag)
        alpha_id, beta_sal, gamma_plaus, delta_coh, eta_mag: Weight coefficients
        cqs_mode: How to handle missing identity ("noid_fallback", "manifest", "strict")
    
    Returns:
        DataFrame with computed CQS scores
    """
    df = df_cqs.copy()
    
    # Normalize weights to sum to 1.0
    total_weight = alpha_id + beta_sal + gamma_plaus + delta_coh + eta_mag
    if total_weight > 1e-12:
        alpha_id /= total_weight
        beta_sal /= total_weight
        gamma_plaus /= total_weight
        delta_coh /= total_weight
        eta_mag /= total_weight
    
    # Compute CQS for each row
    cqs_scores = []
    
    for _, row in df.iterrows():
        sid = row.get("Sid", np.nan)
        ssal = row.get("Ssal", 0.0)
        splaus = row.get("Splaus", 0.0)
        scoh = row.get("Scoh", 1.0)  # Default to 1.0 if not using CLIP
        smag = row.get("Smag", 1.0)
        gate_face = row.get("gate_face", 1)
        gate_id_manifest = row.get("gate_id_manifest", 1)
        
        if cqs_mode == "noid_fallback":
            g = gate_face * gate_id_manifest
            if pd.isna(sid):
                # Fallback: renormalize remaining weights
                rest_total = beta_sal + gamma_plaus + delta_coh + eta_mag
                if rest_total > 1e-12:
                    cqs = g * (
                        (beta_sal / rest_total) * ssal +
                        (gamma_plaus / rest_total) * splaus +
                        (delta_coh / rest_total) * scoh +
                        (eta_mag / rest_total) * smag
                    )
                else:
                    cqs = 0.0
            else:
                cqs = g * (
                    alpha_id * float(sid) +
                    beta_sal * ssal +
                    gamma_plaus * splaus +
                    delta_coh * scoh +
                    eta_mag * smag
                )
        elif cqs_mode == "manifest":
            g = gate_face * gate_id_manifest
            sid_val = 0.0 if pd.isna(sid) else float(sid)
            cqs = g * (
                alpha_id * sid_val +
                beta_sal * ssal +
                gamma_plaus * splaus +
                delta_coh * scoh +
                eta_mag * smag
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
                eta_mag * smag
            )
        
        cqs_scores.append(max(0.0, min(1.0, cqs)))  # Clamp to [0, 1]
    
    df["CQS_computed"] = cqs_scores
    return df


def objective_function_multi_metric(
    weights: np.ndarray,
    df_cqs: pd.DataFrame,
    df_survey: pd.DataFrame,
    cqs_mode: str,
    metric_weights: Dict[str, float] = None,
) -> float:
    """
    Objective function: weighted negative correlation across multiple metrics (to minimize).
    
    Args:
        weights: [alpha_id, beta_sal, gamma_plaus, delta_coh, eta_mag]
        df_cqs: CQS component scores
        df_survey: Survey ratings by method (with multiple metrics)
        cqs_mode: CQS computation mode
        metric_weights: Dictionary mapping survey metric names to importance weights
    
    Returns:
        Negative weighted correlation (to minimize)
    """
    if metric_weights is None:
        metric_weights = {"overall": 1.0, "exaggeration": 0.5, "identity": 0.3}
    
    alpha_id, beta_sal, gamma_plaus, delta_coh, eta_mag = weights
    
    # Compute CQS with these weights
    df_with_cqs = compute_cqs_with_weights(
        df_cqs, alpha_id, beta_sal, gamma_plaus, delta_coh, eta_mag, cqs_mode
    )
    
    # Aggregate CQS by method
    method_cqs = df_with_cqs.groupby("method")["CQS_computed"].mean().reset_index()
    method_cqs.columns = ["method", "cqs_mean"]
    
    # Merge with survey data
    merged = df_survey.merge(method_cqs, on="method", how="inner")
    
    if len(merged) < 2:
        return 1.0  # Penalty for insufficient data
    
    # Compute weighted correlation across all metrics
    total_weighted_corr = 0.0
    total_weight = 0.0
    
    for metric_name, weight in metric_weights.items():
        survey_col = f"survey_{metric_name}"
        if survey_col not in merged.columns:
            continue
        
        try:
            r, _ = pearsonr(merged[survey_col].values, merged["cqs_mean"].values)
            if not np.isnan(r):
                total_weighted_corr += weight * r
                total_weight += weight
        except Exception:
            pass
    
    if total_weight < 1e-12:
        return 1.0  # Penalty if no valid correlations
    
    # Normalize by total weight
    weighted_corr = total_weighted_corr / total_weight
    
    # Return negative weighted correlation (we want to maximize, so minimize negative)
    return -weighted_corr


def objective_function(
    weights: np.ndarray,
    df_cqs: pd.DataFrame,
    df_survey: pd.DataFrame,
    cqs_mode: str,
) -> float:
    """
    Objective function: negative correlation (to minimize).
    Kept for backward compatibility.
    """
    return objective_function_multi_metric(weights, df_cqs, df_survey, cqs_mode, {"overall": 1.0})


def optimize_weights(
    df_cqs: pd.DataFrame,
    df_survey: pd.DataFrame,
    cqs_mode: str = "noid_fallback",
    method: str = "differential_evolution",
    min_weight: float = 0.05,
    metric_weights: Dict[str, float] = None,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Optimize CQS weights to maximize correlation with survey overall preferences.
    
    Args:
        df_cqs: CQS component scores
        df_survey: Survey overall ratings by method
        cqs_mode: CQS computation mode
        method: Optimization method ("differential_evolution" or "minimize")
        min_weight: Minimum weight for each component (to prevent extreme solutions)
    
    Returns:
        Tuple of (optimized_weights_dict, best_correlation)
    """
    # Bounds: all weights between min_weight and 1.0
    # This ensures no component is completely ignored
    bounds = [(min_weight, 1.0)] * 5
    
    # Initial guess: current default weights
    x0 = np.array([0.15, 0.65, 0.10, 0.10, 0.0])
    
    if metric_weights is None:
        metric_weights = {"overall": 1.0, "exaggeration": 0.5, "identity": 0.3}
    
    if method == "differential_evolution":
        # Differential evolution is good for global optimization
        result = differential_evolution(
            objective_function_multi_metric,
            bounds,
            args=(df_cqs, df_survey, cqs_mode, metric_weights),
            seed=42,
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            atol=1e-6,
        )
        optimal_weights = result.x
        best_corr = -result.fun
    else:
        # Use scipy.optimize.minimize with L-BFGS-B
        result = minimize(
            objective_function_multi_metric,
            x0,
            args=(df_cqs, df_survey, cqs_mode, metric_weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )
        optimal_weights = result.x
        best_corr = -result.fun
    
    # Compute individual correlations for reporting
    df_with_cqs = compute_cqs_with_weights(
        df_cqs,
        optimal_weights[0] / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights[0],
        optimal_weights[1] / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights[1],
        optimal_weights[2] / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights[2],
        optimal_weights[3] / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights[3],
        optimal_weights[4] / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else optimal_weights[4],
        cqs_mode,
    )
    method_cqs = df_with_cqs.groupby("method")["CQS_computed"].mean().reset_index()
    method_cqs.columns = ["method", "cqs_mean"]
    merged = df_survey.merge(method_cqs, on="method", how="inner")
    
    individual_corrs = {}
    for metric_name in metric_weights.keys():
        survey_col = f"survey_{metric_name}"
        if survey_col in merged.columns:
            try:
                r, p = pearsonr(merged[survey_col].values, merged["cqs_mean"].values)
                individual_corrs[metric_name] = {"r": float(r), "p": float(p)}
            except Exception:
                individual_corrs[metric_name] = {"r": np.nan, "p": np.nan}
    
    # Normalize weights to sum to 1.0
    total = np.sum(optimal_weights)
    if total > 1e-12:
        optimal_weights = optimal_weights / total
    
    weights_dict = {
        "alpha_id": float(optimal_weights[0]),
        "beta_sal": float(optimal_weights[1]),
        "gamma_plaus": float(optimal_weights[2]),
        "delta_coh": float(optimal_weights[3]),
        "eta_mag": float(optimal_weights[4]),
    }
    
    return weights_dict, best_corr, individual_corrs


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize CQS weights to match human survey preferences")
    ap.add_argument("--cqs_pairs_csv", type=str, default="reports/cqs/cqs_pairs.csv", help="CQS pairs CSV with component scores")
    ap.add_argument("--survey_csv", type=str, default="survey_report/method_means_ci.csv", help="Survey method means CSV")
    ap.add_argument("--cqs_mode", type=str, default="noid_fallback", choices=["strict", "manifest", "noid_fallback"], help="CQS computation mode")
    ap.add_argument("--optimization_method", type=str, default="differential_evolution", choices=["differential_evolution", "minimize"], help="Optimization method")
    ap.add_argument("--min_weight", type=float, default=0.05, help="Minimum weight for each component (prevents extreme solutions)")
    ap.add_argument("--output_dir", type=str, default="reports/cqs_optimization", help="Output directory")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics and their importance weights
    target_metrics = ["overall", "exaggeration", "identity"]
    metric_weights = {"overall": 1.0, "exaggeration": 0.5, "identity": 0.3}
    
    print("[INFO] Loading CQS component scores...")
    df_cqs = load_cqs_components(Path(args.cqs_pairs_csv))
    print(f"  Loaded {len(df_cqs)} CQS component records")
    
    print(f"[INFO] Loading survey ratings for: {', '.join(target_metrics)}")
    print(f"[INFO] Importance weights: overall={metric_weights['overall']}, exaggeration={metric_weights['exaggeration']}, identity={metric_weights['identity']}")
    df_survey = load_survey_ratings(Path(args.survey_csv), target_metrics)
    print(f"  Loaded survey data for {len(df_survey)} methods")
    print(f"  Methods: {', '.join(df_survey['method'].tolist())}")
    
    # Optimize weights
    print(f"\n[INFO] Optimizing weights using {args.optimization_method}...")
    print(f"  CQS mode: {args.cqs_mode}")
    print(f"  Minimum weight constraint: {args.min_weight}")
    print(f"  Target metrics: {', '.join(target_metrics)}")
    print(f"  Metric importance: {metric_weights}")
    print("  This may take a few minutes...")
    
    optimal_weights, best_correlation, individual_corrs = optimize_weights(
        df_cqs, df_survey, args.cqs_mode, args.optimization_method, args.min_weight, metric_weights
    )
    
    print(f"\n[OK] Optimization complete!")
    print(f"\nOptimal weights:")
    for key, value in optimal_weights.items():
        print(f"  {key} = {value:.6f}")
    print(f"\nWeighted correlation: r = {best_correlation:.4f}")
    print(f"\nIndividual metric correlations:")
    for metric_name, corr_data in individual_corrs.items():
        sig = "*" if corr_data["p"] < 0.05 else ""
        print(f"  {metric_name}: r = {corr_data['r']:.4f}{sig} (p = {corr_data['p']:.4f})")
    
    # Evaluate optimized weights for each metric
    print("\n[INFO] Evaluating optimized weights for each metric...")
    df_optimized = compute_cqs_with_weights(
        df_cqs,
        optimal_weights["alpha_id"],
        optimal_weights["beta_sal"],
        optimal_weights["gamma_plaus"],
        optimal_weights["delta_coh"],
        optimal_weights["eta_mag"],
        args.cqs_mode,
    )
    method_cqs_opt = df_optimized.groupby("method")["CQS_computed"].mean().reset_index()
    method_cqs_opt.columns = ["method", "cqs_mean"]
    merged_opt = df_survey.merge(method_cqs_opt, on="method", how="inner")
    
    if len(merged_opt) >= 2:
        print(f"  Correlations by metric:")
        for metric_name in target_metrics:
            survey_col = f"survey_{metric_name}"
            if survey_col in merged_opt.columns:
                r_opt, p_opt = pearsonr(merged_opt[survey_col].values, merged_opt["cqs_mean"].values)
                rho_opt, p_rho_opt = spearmanr(merged_opt[survey_col].values, merged_opt["cqs_mean"].values)
                sig = "*" if p_opt < 0.05 else ""
                print(f"    {metric_name}:")
                print(f"      Pearson r = {r_opt:.4f}{sig} (p = {p_opt:.4f})")
                print(f"      Spearman ρ = {rho_opt:.4f} (p = {p_rho_opt:.4f})")
    
    # Save results
    results = {
        "optimal_weights": optimal_weights,
        "weighted_correlation": float(best_correlation),
        "individual_correlations": individual_corrs,
        "metric_weights": metric_weights,
        "target_metrics": target_metrics,
        "cqs_mode": args.cqs_mode,
        "optimization_method": args.optimization_method,
    }
    
    import json
    results_json = output_dir / "optimization_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results to: {results_json}")
    
    # Save comparison table
    comparison_data = {"method": merged_opt["method"].tolist()}
    for metric_name in target_metrics:
        survey_col = f"survey_{metric_name}"
        if survey_col in merged_opt.columns:
            comparison_data[f"survey_{metric_name}"] = merged_opt[survey_col].tolist()
    comparison_data["cqs"] = merged_opt["cqs_mean"].tolist()
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = output_dir / "method_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"[OK] Saved comparison table to: {comparison_csv}")
    
    # Print step4.py command line arguments
    print("\n" + "="*80)
    print("RECOMMENDED STEP4.PY ARGUMENTS:")
    print("="*80)
    print(f"--alpha_id {optimal_weights['alpha_id']:.6f}")
    print(f"--beta_sal {optimal_weights['beta_sal']:.6f}")
    print(f"--gamma_plaus {optimal_weights['gamma_plaus']:.6f}")
    print(f"--delta_coh {optimal_weights['delta_coh']:.6f}")
    print(f"--eta_mag {optimal_weights['eta_mag']:.6f}")
    print("="*80)


if __name__ == "__main__":
    main()

