#!/usr/bin/env python3
"""
Survey Analyzer (folder → tables + figures + report)

What it does
------------
Given a folder of participant JSON files (one file per participant; same schema as survey_data_1.json),
this script will:

1) Load and normalize all participant data.
2) Run data-quality checks (attention checks, missingness, extreme RTs, straight-lining).
3) Produce descriptive statistics (overall + by method/identity/variant/demographics).
4) Run inferential comparisons (optional SciPy): Friedman/Wilcoxon for repeated-measures across methods.
5) Compute inter-rater agreement (ICC) when multiple participants rated the same stimulus.
6) Generate visualizations (PNG) and export tidy CSVs + a Markdown report.

Usage
-----
python survey_analyzer.py --input_dir ./data --output_dir ./survey_report

Notes
-----
- The parser prefers `session_data.stimulus_responses` if present; otherwise it falls back to `csv_stimulus`.
- Figures are generated with matplotlib (no seaborn).
- SciPy is optional; if absent, inferential tests are skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Configuration
# ----------------------------

RATING_COLS = ["identity", "exaggeration", "alignment", "plausibility", "overall"]
NUMERIC_COLS = RATING_COLS + ["rt_ms", "duration_ms"]
DEFAULT_RT_MIN_MS = 1500  # conservative "too fast to read" heuristic
DEFAULT_RT_MAX_MS = 10 * 60 * 1000  # 10 minutes
EXPECTED_RATING_MIN = 1
EXPECTED_RATING_MAX = 7


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float]:
    """Basic percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    stats = []
    for _ in range(n_boot):
        samp = rng.choice(values, size=len(values), replace=True)
        stats.append(stat(samp))
    lo = np.percentile(stats, 100 * (alpha / 2))
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ----------------------------
# Data model + loader
# ----------------------------

@dataclass
class ParticipantRecord:
    participant_id: str
    file_path: Path
    summary: Dict[str, Any]
    demographics: Dict[str, Any]
    attention_checks: List[Dict[str, Any]]
    stimulus_df: pd.DataFrame


def _extract_session_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("session_data"), dict):
        return payload["session_data"]
    return payload


def _parse_stimulus_responses(payload: Dict[str, Any]) -> pd.DataFrame:
    session = _extract_session_block(payload)

    if isinstance(session.get("stimulus_responses"), list) and len(session["stimulus_responses"]) > 0:
        return pd.DataFrame(session["stimulus_responses"])

    # fallback: CSV string
    csv_str = payload.get("csv_stimulus") or session.get("csv_stimulus")
    if isinstance(csv_str, str) and "participant_id" in csv_str and "\n" in csv_str:
        return pd.read_csv(StringIO(csv_str))

    return pd.DataFrame(columns=["participant_id"] + RATING_COLS)


def load_participant_json(json_path: Path) -> ParticipantRecord:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    session = _extract_session_block(payload)

    pid = str(payload.get("participant_id") or session.get("summary", {}).get("participant_id") or session.get("participant_id") or json_path.stem)

    summary = payload.get("summary") or session.get("summary") or {}
    demographics = session.get("demographics") or payload.get("demographics") or {}
    attention_checks = session.get("attention_checks") or payload.get("attention_checks") or []

    stim_df = _parse_stimulus_responses(payload).copy()

    # Normalize / type-cast
    if "participant_id" not in stim_df.columns:
        stim_df["participant_id"] = pid
    stim_df["participant_id"] = stim_df["participant_id"].astype(str)

    for col in ["identity_id", "identity_display", "method_true", "method_mask", "variant", "reference_url"]:
        if col not in stim_df.columns:
            stim_df[col] = np.nan

    for col in NUMERIC_COLS:
        if col in stim_df.columns:
            stim_df[col] = pd.to_numeric(stim_df[col], errors="coerce")

    stim_df["variant"] = stim_df["variant"].astype(str)

    for col in RATING_COLS:
        if col in stim_df.columns:
            stim_df.loc[(stim_df[col] < EXPECTED_RATING_MIN) | (stim_df[col] > EXPECTED_RATING_MAX), col] = np.nan

    return ParticipantRecord(
        participant_id=pid,
        file_path=json_path,
        summary=summary,
        demographics=demographics,
        attention_checks=attention_checks,
        stimulus_df=stim_df,
    )


def load_folder(input_dir: Path) -> List[ParticipantRecord]:
    json_files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {input_dir}")

    records: List[ParticipantRecord] = []
    for p in json_files:
        try:
            records.append(load_participant_json(p))
        except Exception as e:
            print(f"[WARN] Failed to parse {p.name}: {e}")

    if not records:
        raise RuntimeError("No participant files could be parsed successfully.")
    return records


# ----------------------------
# Quality control
# ----------------------------

def qc_attention(attention_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not attention_checks:
        return {"attention_any": False, "attention_pass_all": np.nan, "attention_pass_rate": np.nan, "attention_n": 0}
    passes = [bool(ac.get("pass")) for ac in attention_checks]
    return {
        "attention_any": True,
        "attention_pass_all": all(passes),
        "attention_pass_rate": float(np.mean(passes)),
        "attention_n": int(len(passes)),
    }


def qc_rt(stim_df: pd.DataFrame, rt_min_ms: int, rt_max_ms: int) -> Dict[str, Any]:
    if "rt_ms" not in stim_df.columns or stim_df.empty:
        return {"rt_n": 0, "rt_too_fast_n": 0, "rt_too_slow_n": 0, "rt_too_fast_rate": np.nan, "rt_too_slow_rate": np.nan}

    rts = stim_df["rt_ms"].astype(float)
    valid = rts.dropna()
    if valid.empty:
        return {"rt_n": 0, "rt_too_fast_n": 0, "rt_too_slow_n": 0, "rt_too_fast_rate": np.nan, "rt_too_slow_rate": np.nan}

    too_fast = (valid < rt_min_ms).sum()
    too_slow = (valid > rt_max_ms).sum()
    n = len(valid)
    return {
        "rt_n": int(n),
        "rt_too_fast_n": int(too_fast),
        "rt_too_slow_n": int(too_slow),
        "rt_too_fast_rate": float(too_fast / n),
        "rt_too_slow_rate": float(too_slow / n),
        "rt_median_ms": float(np.median(valid)),
        "rt_mean_ms": float(np.mean(valid)),
    }


def qc_straightlining(stim_df: pd.DataFrame) -> Dict[str, Any]:
    if stim_df.empty:
        return {"straightline_any": False}

    out: Dict[str, Any] = {"straightline_any": False}
    for col in RATING_COLS:
        if col not in stim_df.columns:
            continue
        vals = stim_df[col].dropna().astype(float).values
        if len(vals) < 6:
            out[f"{col}_straightline"] = np.nan
            continue
        uniq = len(np.unique(vals))
        out[f"{col}_unique_values"] = int(uniq)
        is_straight = (uniq <= 2)  # heuristic
        out[f"{col}_straightline"] = bool(is_straight)
        out["straightline_any"] = out["straightline_any"] or is_straight
    return out


def build_participant_table(records: List[ParticipantRecord], rt_min_ms: int, rt_max_ms: int) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "participant_id": r.participant_id,
            "file": r.file_path.name,
            "age": r.demographics.get("age"),
            "gender": r.demographics.get("gender"),
            "experience": r.demographics.get("experience"),
            "exit_reason": r.summary.get("exit_reason"),
            "total_duration_minutes": r.summary.get("total_duration_minutes") or (r.summary.get("total_duration_ms", 0) / 60000 if r.summary.get("total_duration_ms") else np.nan),
            "stimulus_trials": (r.summary.get("counts", {}) or {}).get("stimulus_trials"),
        }
        row.update(qc_attention(r.attention_checks))
        row.update(qc_rt(r.stimulus_df, rt_min_ms, rt_max_ms))
        row.update(qc_straightlining(r.stimulus_df))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------
# Stats / agreement
# ----------------------------

def cronbach_alpha(df: pd.DataFrame) -> float:
    X = df.to_numpy(dtype=float)
    X = X[~np.any(np.isnan(X), axis=1)]
    if X.shape[0] < 3 or X.shape[1] < 2:
        return np.nan
    k = X.shape[1]
    variances = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - (variances.sum() / total_var))
    return float(alpha)


def icc_2_1(ratings: pd.DataFrame, target_col: str, rater_col: str, score_col: str) -> float:
    df = ratings[[target_col, rater_col, score_col]].dropna().copy()
    if df.empty:
        return np.nan

    mat = df.pivot_table(index=target_col, columns=rater_col, values=score_col, aggfunc="mean")
    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")

    n, k = mat.shape
    if n < 2 or k < 2:
        return np.nan

    grand_mean = np.nanmean(mat.values)
    mean_target = np.nanmean(mat.values, axis=1)
    mean_rater = np.nanmean(mat.values, axis=0)

    ss_target = k * np.nansum((mean_target - grand_mean) ** 2)
    ss_rater = n * np.nansum((mean_rater - grand_mean) ** 2)
    ss_total = np.nansum((mat.values - grand_mean) ** 2)
    ss_error = ss_total - ss_target - ss_rater

    ms_target = ss_target / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_target - ms_error) / (ms_target + (k - 1) * ms_error)
    return float(icc)


def try_inferential_tests(stim_all: pd.DataFrame, score: str = "overall") -> Dict[str, Any]:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception:
        return {"available": False, "reason": "SciPy not installed"}

    df = stim_all.dropna(subset=[score, "participant_id", "method_true"]).copy()
    if df.empty:
        return {"available": True, "friedman": None, "pairwise": []}

    pm = df.groupby(["participant_id", "method_true"])[score].mean().reset_index()
    methods = sorted(pm["method_true"].unique().tolist())
    pivot = pm.pivot(index="participant_id", columns="method_true", values=score).dropna(axis=0, how="any")
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"available": True, "friedman": None, "pairwise": [], "note": "Not enough complete repeated-measures data."}

    arrays = [pivot[m].values for m in methods]
    stat, p = friedmanchisquare(*arrays)

    pairs = []
    m = len(methods)
    alpha = 0.05
    bonf_alpha = alpha / (m * (m - 1) / 2)

    for i in range(m):
        for j in range(i + 1, m):
            a = pivot[methods[i]].values
            b = pivot[methods[j]].values
            try:
                w_stat, w_p = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
            except Exception:
                w_stat, w_p = np.nan, np.nan
            pairs.append({
                "method_a": methods[i],
                "method_b": methods[j],
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
                "p_value": float(w_p) if not np.isnan(w_p) else np.nan,
                "bonferroni_alpha": bonf_alpha,
                "significant_bonferroni": bool(w_p < bonf_alpha) if (w_p is not None and not np.isnan(w_p)) else False
            })

    return {
        "available": True,
        "friedman": {"statistic": float(stat), "p_value": float(p), "methods": methods, "n_participants_complete": int(pivot.shape[0])},
        "pairwise": pairs
    }


# ----------------------------
# Aggregation + plots
# ----------------------------

def build_stimulus_table(records: List[ParticipantRecord]) -> pd.DataFrame:
    dfs = []
    for r in records:
        df = r.stimulus_df.copy()
        df["participant_id"] = r.participant_id
        dfs.append(df)
    stim_all = pd.concat(dfs, ignore_index=True)
    stim_all["method_true"] = stim_all["method_true"].astype(str)
    return stim_all


def plot_distribution(stim_all: pd.DataFrame, out_dir: Path) -> None:
    for col in RATING_COLS:
        if col not in stim_all.columns:
            continue
        fig = plt.figure()
        vals = stim_all[col].dropna().astype(float).values
        plt.hist(vals, bins=np.arange(EXPECTED_RATING_MIN, EXPECTED_RATING_MAX + 2) - 0.5)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        save_fig(fig, out_dir / f"dist_{col}.png")

    if "rt_ms" in stim_all.columns:
        fig = plt.figure()
        vals = stim_all["rt_ms"].dropna().astype(float).values / 1000.0
        plt.hist(vals, bins=30)
        plt.title("Response time (seconds) distribution")
        plt.xlabel("RT (s)")
        plt.ylabel("count")
        save_fig(fig, out_dir / "dist_rt_seconds.png")


def plot_method_boxplots(stim_all: pd.DataFrame, out_dir: Path) -> None:
    methods = sorted(stim_all["method_true"].dropna().unique().tolist())
    if not methods:
        return

    for col in RATING_COLS:
        if col not in stim_all.columns:
            continue
        fig = plt.figure()
        data = [stim_all.loc[stim_all["method_true"] == m, col].dropna().astype(float).values for m in methods]
        plt.boxplot(data, labels=methods, showfliers=True)
        plt.title(f"{col} by method (boxplot)")
        plt.ylabel(col)
        plt.xticks(rotation=20)
        save_fig(fig, out_dir / f"box_{col}_by_method.png")


def plot_method_means_ci(stim_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    methods = sorted(stim_all["method_true"].dropna().unique().tolist())
    rows = []
    for m in methods:
        sub = stim_all[stim_all["method_true"] == m]
        for col in RATING_COLS:
            vals = sub[col].dropna().astype(float).values
            mean = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci(vals, stat=np.mean) if len(vals) else (np.nan, np.nan)
            rows.append({"method": m, "metric": col, "mean": mean, "ci_low": lo, "ci_high": hi, "n": int(len(vals))})
    df = pd.DataFrame(rows)

    for col in RATING_COLS:
        fig = plt.figure()
        sub = df[df["metric"] == col].copy()
        x = np.arange(len(sub))
        means = sub["mean"].values
        yerr = np.vstack([means - sub["ci_low"].values, sub["ci_high"].values - means])
        plt.errorbar(x, means, yerr=yerr, fmt="o")
        plt.xticks(x, sub["method"].values, rotation=20)
        plt.title(f"{col}: mean with 95% bootstrap CI")
        plt.ylabel(col)
        save_fig(fig, out_dir / f"mean_ci_{col}_by_method.png")

    return df


def plot_identity_method_heatmap(stim_all: pd.DataFrame, out_dir: Path, score_col: str = "overall") -> None:
    if "identity_id" not in stim_all.columns or "method_true" not in stim_all.columns or score_col not in stim_all.columns:
        return
    pivot = stim_all.pivot_table(index="identity_id", columns="method_true", values=score_col, aggfunc="mean")
    if pivot.empty:
        return
    fig = plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.title(f"Mean {score_col} by identity × method")
    plt.xlabel("method")
    plt.ylabel("identity")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=20)
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.colorbar(label=f"mean {score_col}")
    save_fig(fig, out_dir / f"heatmap_{score_col}_identity_by_method.png")


# ----------------------------
# Report generation
# ----------------------------

def df_to_md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    return df2.to_markdown(index=False)


def generate_report(
    out_dir: Path,
    participant_df: pd.DataFrame,
    stim_all: pd.DataFrame,
    method_means_ci: pd.DataFrame,
    inferential: Dict[str, Any],
    icc_value: float,
) -> None:
    n_participants = participant_df["participant_id"].nunique() if not participant_df.empty else 0
    n_trials = len(stim_all)

    alpha = cronbach_alpha(stim_all[["alignment", "plausibility", "overall"]] if set(["alignment","plausibility","overall"]).issubset(stim_all.columns) else pd.DataFrame())

    report = []
    report.append("# Survey Analysis Report\n")
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    report.append("## Dataset overview\n")
    report.append(f"- Participants parsed: **{n_participants}**\n")
    report.append(f"- Stimulus rating trials: **{n_trials}**\n")
    report.append(f"- Methods observed: **{', '.join(sorted(stim_all['method_true'].dropna().unique().tolist())) if 'method_true' in stim_all.columns else ''}**\n")
    report.append(f"- Rating columns: **{', '.join([c for c in RATING_COLS if c in stim_all.columns])}**\n\n")

    report.append("## Data quality (QC)\n")
    qc_cols = [
        "participant_id","attention_n","attention_pass_rate","attention_pass_all",
        "rt_n","rt_too_fast_rate","rt_too_slow_rate","rt_median_ms","straightline_any"
    ]
    qc_view = participant_df[[c for c in qc_cols if c in participant_df.columns]].copy()
    report.append(df_to_md_table(qc_view, max_rows=50))
    report.append("\n\n")

    report.append("## Method-level summary (mean ± 95% CI)\n")
    report.append(df_to_md_table(method_means_ci.sort_values(["metric","method"]), max_rows=200))
    report.append("\n\n")

    report.append("## Reliability / agreement\n")
    report.append(f"- Cronbach's alpha (alignment+plausibility+overall; heuristic composite): **{alpha:.3f}**\n")
    report.append(f"- Inter-rater agreement ICC(2,1) for **overall** (by stimulus): **{icc_value:.3f}**\n\n")

    report.append("## Inferential tests (optional)\n")
    if not inferential.get("available"):
        report.append(f"- SciPy not available; inferential tests skipped. ({inferential.get('reason')})\n")
    else:
        fr = inferential.get("friedman")
        if fr:
            report.append(f"- Friedman test across methods (within participant means): statistic={fr['statistic']:.3f}, p={fr['p_value']:.4g}, n_complete={fr['n_participants_complete']}\n")
        else:
            report.append(f"- Friedman test not run: {inferential.get('note','insufficient data')}\n")
        pw = inferential.get("pairwise", [])
        if pw:
            report.append("\n### Pairwise Wilcoxon (Bonferroni corrected)\n")
            report.append(df_to_md_table(pd.DataFrame(pw), max_rows=200))
    report.append("\n\n")

    report.append("## Figures\n")
    fig_files = sorted([p.name for p in out_dir.glob('*.png')])
    for f in fig_files:
        report.append(f"- {f}\n")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing participant .json files")
    ap.add_argument("--output_dir", type=str, required=True, help="Output folder (will be created)")
    ap.add_argument("--rt_min_ms", type=int, default=DEFAULT_RT_MIN_MS, help="QC threshold: RT too fast (ms)")
    ap.add_argument("--rt_max_ms", type=int, default=DEFAULT_RT_MAX_MS, help="QC threshold: RT too slow (ms)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    records = load_folder(in_dir)

    participant_df = build_participant_table(records, rt_min_ms=args.rt_min_ms, rt_max_ms=args.rt_max_ms)
    stim_all = build_stimulus_table(records)

    participant_df.to_csv(out_dir / "participants_qc.csv", index=False)
    stim_all.to_csv(out_dir / "stimulus_ratings_long.csv", index=False)

    plot_distribution(stim_all, out_dir)
    plot_method_boxplots(stim_all, out_dir)
    method_means_ci = plot_method_means_ci(stim_all, out_dir)
    method_means_ci.to_csv(out_dir / "method_means_ci.csv", index=False)
    plot_identity_method_heatmap(stim_all, out_dir, score_col="overall")

    stim_all["stimulus_key"] = stim_all["identity_id"].astype(str) + "|" + stim_all["method_true"].astype(str) + "|" + stim_all["variant"].astype(str)
    icc_value = icc_2_1(stim_all, target_col="stimulus_key", rater_col="participant_id", score_col="overall")

    inferential = try_inferential_tests(stim_all, score="overall")
    with open(out_dir / "inferential_tests.json", "w", encoding="utf-8") as f:
        json.dump(inferential, f, indent=2)

    generate_report(out_dir, participant_df, stim_all, method_means_ci, inferential, icc_value)

    print(f"[OK] Wrote outputs to: {out_dir}")
    print("Key files: report.md, participants_qc.csv, stimulus_ratings_long.csv, method_means_ci.csv, inferential_tests.json")


if __name__ == "__main__":
    main()
