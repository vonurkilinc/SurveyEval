#!/usr/bin/env python3
"""
step1_sanity_check.py

Sanity check script for step1 baseline dataset generation.

Verifies that step1.py correctly:
1. Produced a clean file list (baseline_index.csv)
2. Computed per-image geometry descriptors G (baseline_geometry.csv/parquet)
3. Computed population statistics Ḡ and σG (geometry_stats.json)
4. Computed plausible ranges [Li,Ui] (geometry_bounds.json)
5. Produced symmetry_pairs.json for the landmark topology

Usage:
    python step1_sanity_check.py --baseline-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class SanityCheckError(Exception):
    """Custom exception for sanity check failures."""
    pass


class SanityChecker:
    """Sanity checker for step1 baseline outputs."""
    
    def __init__(self, baseline_dir: Path):
        self.baseline_dir = Path(baseline_dir).expanduser().resolve()
        if not self.baseline_dir.exists():
            raise FileNotFoundError(f"Baseline directory does not exist: {self.baseline_dir}")
        
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def error(self, msg: str) -> None:
        """Record an error."""
        self.errors.append(msg)
        print(f"[ERROR] {msg}", file=sys.stderr)
    
    def warn(self, msg: str) -> None:
        """Record a warning."""
        self.warnings.append(msg)
        print(f"[WARN] {msg}", file=sys.stderr)
    
    def info_msg(self, msg: str) -> None:
        """Record an info message."""
        self.info.append(msg)
        print(f"[INFO] {msg}")
    
    def check_file_exists(self, filename: str, required: bool = True) -> Optional[Path]:
        """Check if a file exists in the baseline directory."""
        path = self.baseline_dir / filename
        if path.exists():
            return path
        if required:
            self.error(f"Required file missing: {filename}")
        else:
            self.warn(f"Optional file missing: {filename}")
        return None
    
    def check_all_files_exist(self) -> Dict[str, Optional[Path]]:
        """Check that all expected output files exist."""
        self.info_msg("Checking required files exist...")
        
        files = {
            "run_config": self.check_file_exists("run_config.json"),
            "baseline_index": self.check_file_exists("baseline_index.csv"),
            "baseline_rejects": self.check_file_exists("baseline_rejects.csv"),
            "baseline_geometry": None,
            "geometry_stats": self.check_file_exists("geometry_stats.json"),
            "geometry_bounds": self.check_file_exists("geometry_bounds.json"),
            "symmetry_pairs": self.check_file_exists("symmetry_pairs.json"),
            "baseline_report": self.check_file_exists("baseline_report.json"),
        }
        
        # Check for geometry file (parquet or csv)
        geom_parquet = self.baseline_dir / "baseline_geometry.parquet"
        geom_csv = self.baseline_dir / "baseline_geometry.csv"
        
        if geom_parquet.exists():
            files["baseline_geometry"] = geom_parquet
            self.info_msg(f"Found baseline_geometry.parquet")
        elif geom_csv.exists():
            files["baseline_geometry"] = geom_csv
            self.info_msg(f"Found baseline_geometry.csv")
        else:
            self.error("Neither baseline_geometry.parquet nor baseline_geometry.csv found")
        
        return files
    
    def check_run_config(self, config_path: Path) -> Dict[str, Any]:
        """Check run_config.json structure and content."""
        self.info_msg("Checking run_config.json...")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            self.error(f"Failed to read run_config.json: {e}")
            return {}
        
        # Check required fields
        required_fields = [
            "run_id", "baseline_name", "landmarks", "geometry", "stats", "symmetry"
        ]
        for field in required_fields:
            if field not in config:
                self.error(f"run_config.json missing required field: {field}")
        
        # Check geometry schema
        if "geometry" in config:
            geom = config["geometry"]
            if "num_features" not in geom or "feature_names" not in geom:
                self.error("run_config.json geometry section missing num_features or feature_names")
            elif len(geom["feature_names"]) != geom["num_features"]:
                self.error(f"run_config.json geometry.num_features ({geom['num_features']}) != len(feature_names) ({len(geom['feature_names'])})")
        
        return config
    
    def check_baseline_index(self, index_path: Path) -> pd.DataFrame:
        """Check baseline_index.csv structure and content."""
        self.info_msg("Checking baseline_index.csv...")
        
        try:
            df = pd.read_csv(index_path)
        except Exception as e:
            self.error(f"Failed to read baseline_index.csv: {e}")
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ["image_id", "path", "rel_path", "face_detected", "num_faces"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self.error(f"baseline_index.csv missing columns: {missing_cols}")
        
        # Check data quality
        if len(df) == 0:
            self.error("baseline_index.csv is empty")
        else:
            self.info_msg(f"baseline_index.csv contains {len(df)} accepted images")
        
        # Check for duplicates
        dup_ids = df["image_id"].duplicated().sum()
        if dup_ids > 0:
            self.error(f"baseline_index.csv contains {dup_ids} duplicate image_id values")
        
        # Check face_detected is all True
        if "face_detected" in df.columns:
            not_detected = (~df["face_detected"]).sum()
            if not_detected > 0:
                self.warn(f"baseline_index.csv contains {not_detected} images with face_detected=False")
        
        # Check num_faces
        if "num_faces" in df.columns:
            max_faces = df["num_faces"].max()
            if max_faces > 1:
                self.warn(f"baseline_index.csv contains images with {max_faces} faces (expected max 1)")
        
        # Check paths exist (sample check)
        if "path" in df.columns:
            sample_size = min(10, len(df))
            sample_paths = df["path"].head(sample_size)
            missing = sum(1 for p in sample_paths if not Path(p).exists())
            if missing > 0:
                self.warn(f"Sample check: {missing}/{sample_size} image paths do not exist")
        
        return df
    
    def check_baseline_rejects(self, rejects_path: Path) -> pd.DataFrame:
        """Check baseline_rejects.csv structure and content."""
        self.info_msg("Checking baseline_rejects.csv...")
        
        try:
            df = pd.read_csv(rejects_path)
        except Exception as e:
            self.error(f"Failed to read baseline_rejects.csv: {e}")
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ["path", "reason"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self.error(f"baseline_rejects.csv missing columns: {missing_cols}")
        
        if len(df) > 0:
            self.info_msg(f"baseline_rejects.csv contains {len(df)} rejected images")
            
            # Check reject reasons are valid
            valid_reasons = [
                "no_face_detected", "too_many_faces", "low_landmark_conf",
                "face_too_small", "align_failed", "geometry_failed",
                "geometry_nonfinite", "image_read_error"
            ]
            invalid_reasons = df[~df["reason"].str.contains("|".join(valid_reasons), case=False, na=False)]
            if len(invalid_reasons) > 0:
                self.warn(f"Found {len(invalid_reasons)} rejections with unusual reasons")
                unique_reasons = invalid_reasons["reason"].unique()[:5]
                for reason in unique_reasons:
                    self.info_msg(f"  Unusual reason: {reason}")
        else:
            self.info_msg("baseline_rejects.csv is empty (no rejections)")
        
        return df
    
    def check_baseline_geometry(self, geometry_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
        """Check baseline_geometry.csv/parquet structure and content."""
        self.info_msg(f"Checking {geometry_path.name}...")
        
        try:
            if geometry_path.suffix == ".parquet":
                df = pd.read_parquet(geometry_path)
            else:
                df = pd.read_csv(geometry_path)
        except Exception as e:
            self.error(f"Failed to read {geometry_path.name}: {e}")
            return pd.DataFrame()
        
        # Check required columns
        if "image_id" not in df.columns:
            self.error(f"{geometry_path.name} missing image_id column")
        
        # Get expected feature names from config
        expected_features = []
        if "geometry" in config and "feature_names" in config["geometry"]:
            expected_features = config["geometry"]["feature_names"]
        
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                self.error(f"{geometry_path.name} missing feature columns: {missing_features}")
            
            extra_features = [c for c in df.columns if c not in ["image_id"] + expected_features]
            if extra_features:
                self.warn(f"{geometry_path.name} has extra columns: {extra_features}")
        
        # Check data quality
        if len(df) == 0:
            self.error(f"{geometry_path.name} is empty")
        else:
            self.info_msg(f"{geometry_path.name} contains {len(df)} geometry descriptors")
        
        # Check for NaN/inf values
        feature_cols = [c for c in df.columns if c != "image_id"]
        for col in feature_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                self.error(f"{geometry_path.name} column '{col}' contains {nan_count} NaN values")
            
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.error(f"{geometry_path.name} column '{col}' contains {inf_count} inf values")
        
        # Check for duplicates
        dup_ids = df["image_id"].duplicated().sum()
        if dup_ids > 0:
            self.error(f"{geometry_path.name} contains {dup_ids} duplicate image_id values")
        
        return df
    
    def check_geometry_stats(self, stats_path: Path, geometry_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check geometry_stats.json structure and verify against data."""
        self.info_msg("Checking geometry_stats.json...")
        
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except Exception as e:
            self.error(f"Failed to read geometry_stats.json: {e}")
            return {}
        
        # Check required fields
        required_fields = ["landmark_topology", "feature_names", "mode"]
        for field in required_fields:
            if field not in stats:
                self.error(f"geometry_stats.json missing required field: {field}")
        
        # Check mode-specific fields
        mode = stats.get("mode", "")
        if mode == "meanstd":
            if "mean" not in stats or "std" not in stats:
                self.error("geometry_stats.json (meanstd mode) missing mean or std")
        elif mode == "robust":
            if "median" not in stats or "mad" not in stats or "robust_std" not in stats:
                self.error("geometry_stats.json (robust mode) missing median, mad, or robust_std")
        else:
            self.error(f"geometry_stats.json has unknown mode: {mode}")
        
        # Verify stats match data
        if len(geometry_df) > 0 and "feature_names" in stats:
            feature_names = stats["feature_names"]
            missing_cols = [f for f in feature_names if f not in geometry_df.columns]
            if missing_cols:
                self.error(f"geometry_stats.json references features not in geometry data: {missing_cols}")
            else:
                # Recompute stats and compare
                if mode == "meanstd" and "mean" in stats and "std" in stats:
                    computed_mean = geometry_df[feature_names].mean().values
                    computed_std = geometry_df[feature_names].std(ddof=1).values
                    
                    stored_mean = np.array(stats["mean"])
                    stored_std = np.array(stats["std"])
                    
                    mean_diff = np.abs(computed_mean - stored_mean)
                    std_diff = np.abs(computed_std - stored_std)
                    
                    max_mean_diff = np.max(mean_diff)
                    max_std_diff = np.max(std_diff)
                    
                    if max_mean_diff > 1e-6:
                        self.error(f"geometry_stats.json mean values don't match recomputed values (max diff: {max_mean_diff:.2e})")
                    else:
                        self.info_msg(f"Verified mean values match (max diff: {max_mean_diff:.2e})")
                    
                    if max_std_diff > 1e-6:
                        self.error(f"geometry_stats.json std values don't match recomputed values (max diff: {max_std_diff:.2e})")
                    else:
                        self.info_msg(f"Verified std values match (max diff: {max_std_diff:.2e})")
        
        return stats
    
    def check_geometry_bounds(self, bounds_path: Path, geometry_df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check geometry_bounds.json structure and verify against data."""
        self.info_msg("Checking geometry_bounds.json...")
        
        try:
            with open(bounds_path, 'r', encoding='utf-8') as f:
                bounds = json.load(f)
        except Exception as e:
            self.error(f"Failed to read geometry_bounds.json: {e}")
            return {}
        
        # Check required fields
        required_fields = ["landmark_topology", "feature_names", "mode", "lower", "upper"]
        for field in required_fields:
            if field not in bounds:
                self.error(f"geometry_bounds.json missing required field: {field}")
        
        # Check mode-specific fields
        mode = bounds.get("mode", "")
        if mode == "percentiles":
            if "lower_p" not in bounds or "upper_p" not in bounds:
                self.error("geometry_bounds.json (percentiles mode) missing lower_p or upper_p")
        elif mode == "ksigma":
            if "k" not in bounds:
                self.error("geometry_bounds.json (ksigma mode) missing k")
        else:
            self.error(f"geometry_bounds.json has unknown mode: {mode}")
        
        # Verify bounds match data
        if len(geometry_df) > 0 and "feature_names" in bounds and "lower" in bounds and "upper" in bounds:
            feature_names = bounds["feature_names"]
            lower = np.array(bounds["lower"])
            upper = np.array(bounds["upper"])
            
            if len(lower) != len(feature_names) or len(upper) != len(feature_names):
                self.error(f"geometry_bounds.json lower/upper length mismatch with feature_names")
            else:
                # Check that bounds are valid (lower < upper)
                invalid_bounds = np.sum(lower >= upper)
                if invalid_bounds > 0:
                    self.error(f"geometry_bounds.json has {invalid_bounds} features where lower >= upper")
                
                # Verify bounds contain data (sample check)
                if all(f in geometry_df.columns for f in feature_names):
                    sample_size = min(1000, len(geometry_df))
                    sample_data = geometry_df[feature_names].head(sample_size).values
                    
                    out_of_bounds = 0
                    for i, feat in enumerate(feature_names):
                        below_lower = np.sum(sample_data[:, i] < lower[i])
                        above_upper = np.sum(sample_data[:, i] > upper[i])
                        out_of_bounds += below_lower + above_upper
                    
                    if out_of_bounds > 0:
                        self.warn(f"Sample check: {out_of_bounds} values out of bounds (expected for percentiles/ksigma)")
                    
                    # Recompute percentiles if mode is percentiles
                    if mode == "percentiles":
                        lower_p = bounds.get("lower_p", 1.0)
                        upper_p = bounds.get("upper_p", 99.0)
                        computed_lower = np.percentile(geometry_df[feature_names].values, lower_p, axis=0)
                        computed_upper = np.percentile(geometry_df[feature_names].values, upper_p, axis=0)
                        
                        lower_diff = np.abs(computed_lower - lower)
                        upper_diff = np.abs(computed_upper - upper)
                        
                        max_lower_diff = np.max(lower_diff)
                        max_upper_diff = np.max(upper_diff)
                        
                        if max_lower_diff > 1e-6:
                            self.error(f"geometry_bounds.json lower bounds don't match recomputed percentiles (max diff: {max_lower_diff:.2e})")
                        else:
                            self.info_msg(f"Verified lower bounds match (max diff: {max_lower_diff:.2e})")
                        
                        if max_upper_diff > 1e-6:
                            self.error(f"geometry_bounds.json upper bounds don't match recomputed percentiles (max diff: {max_upper_diff:.2e})")
                        else:
                            self.info_msg(f"Verified upper bounds match (max diff: {max_upper_diff:.2e})")
        
        return bounds
    
    def check_symmetry_pairs(self, symmetry_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check symmetry_pairs.json structure and validity."""
        self.info_msg("Checking symmetry_pairs.json...")
        
        try:
            with open(symmetry_path, 'r', encoding='utf-8') as f:
                symmetry = json.load(f)
        except Exception as e:
            self.error(f"Failed to read symmetry_pairs.json: {e}")
            return {}
        
        # Check required fields
        required_fields = ["landmark_topology", "pairs", "midline"]
        for field in required_fields:
            if field not in symmetry:
                self.error(f"symmetry_pairs.json missing required field: {field}")
        
        # Check landmark topology matches config
        topology = symmetry.get("landmark_topology", "")
        if "landmarks" in config and topology != config["landmarks"]:
            self.error(f"symmetry_pairs.json topology '{topology}' != config landmarks '{config['landmarks']}'")
        
        # Check pairs structure
        pairs = symmetry.get("pairs", [])
        if not isinstance(pairs, list):
            self.error("symmetry_pairs.json pairs must be a list")
        elif len(pairs) == 0:
            self.warn("symmetry_pairs.json contains no pairs")
        else:
            self.info_msg(f"symmetry_pairs.json contains {len(pairs)} symmetry pairs")
            
            # Check each pair is valid
            for i, pair in enumerate(pairs):
                if not isinstance(pair, list) or len(pair) != 2:
                    self.error(f"symmetry_pairs.json pair {i} is invalid (expected [int, int])")
                else:
                    a, b = pair
                    if not isinstance(a, int) or not isinstance(b, int):
                        self.error(f"symmetry_pairs.json pair {i} contains non-integer values")
                    elif a < 0 or b < 0:
                        self.error(f"symmetry_pairs.json pair {i} contains negative indices")
                    elif topology == "mediapipe468" and (a >= 468 or b >= 468):
                        self.error(f"symmetry_pairs.json pair {i} contains indices >= 468 (MediaPipe has 468 landmarks)")
        
        # Check midline structure
        midline = symmetry.get("midline", [])
        if not isinstance(midline, list):
            self.error("symmetry_pairs.json midline must be a list")
        elif len(midline) == 0:
            self.warn("symmetry_pairs.json contains no midline indices")
        else:
            self.info_msg(f"symmetry_pairs.json contains {len(midline)} midline indices")
            
            for i, idx in enumerate(midline):
                if not isinstance(idx, int):
                    self.error(f"symmetry_pairs.json midline[{i}] is not an integer")
                elif idx < 0:
                    self.error(f"symmetry_pairs.json midline[{i}] is negative")
                elif topology == "mediapipe468" and idx >= 468:
                    self.error(f"symmetry_pairs.json midline[{i}] >= 468 (MediaPipe has 468 landmarks)")
        
        return symmetry
    
    def check_data_consistency(self, index_df: pd.DataFrame, geometry_df: pd.DataFrame, rejects_df: pd.DataFrame) -> None:
        """Check consistency between index, geometry, and rejects."""
        self.info_msg("Checking data consistency...")
        
        if len(index_df) == 0 or len(geometry_df) == 0:
            return
        
        # Check image_ids match between index and geometry
        index_ids = set(index_df["image_id"].unique())
        geometry_ids = set(geometry_df["image_id"].unique())
        
        missing_in_geometry = index_ids - geometry_ids
        if missing_in_geometry:
            self.error(f"{len(missing_in_geometry)} image_ids in baseline_index.csv missing from geometry data")
        
        extra_in_geometry = geometry_ids - index_ids
        if extra_in_geometry:
            self.error(f"{len(extra_in_geometry)} image_ids in geometry data missing from baseline_index.csv")
        
        if len(index_ids) == len(geometry_ids) and index_ids == geometry_ids:
            self.info_msg("Image IDs match between index and geometry data")
        
        # Check no overlap between index and rejects (by path)
        if len(rejects_df) > 0 and "path" in rejects_df.columns and "path" in index_df.columns:
            index_paths = set(index_df["path"].unique())
            reject_paths = set(rejects_df["path"].unique())
            overlap = index_paths & reject_paths
            if overlap:
                self.error(f"{len(overlap)} paths appear in both index and rejects")
    
    def check_baseline_report(self, report_path: Path) -> Dict[str, Any]:
        """Check baseline_report.json structure."""
        self.info_msg("Checking baseline_report.json...")
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            self.error(f"Failed to read baseline_report.json: {e}")
            return {}
        
        # Check required fields
        required_fields = ["run_id", "candidates", "accepted", "rejected", "accept_rate"]
        for field in required_fields:
            if field not in report:
                self.error(f"baseline_report.json missing required field: {field}")
        
        # Check accept_rate is reasonable
        if "accepted" in report and "candidates" in report:
            accepted = report["accepted"]
            candidates = report["candidates"]
            if candidates > 0:
                computed_rate = accepted / candidates
                stored_rate = report.get("accept_rate", 0.0)
                if abs(computed_rate - stored_rate) > 1e-6:
                    self.error(f"baseline_report.json accept_rate mismatch: stored={stored_rate:.6f}, computed={computed_rate:.6f}")
                else:
                    self.info_msg(f"Accept rate: {stored_rate:.2%} ({accepted}/{candidates})")
        
        return report
    
    def run_all_checks(self) -> int:
        """Run all sanity checks."""
        self.info_msg(f"Starting sanity check for: {self.baseline_dir}")
        self.info_msg("=" * 70)
        
        # Check files exist
        files = self.check_all_files_exist()
        if not all(files.values()):
            self.error("Some required files are missing. Aborting further checks.")
            return 1
        
        # Load and check run_config
        config = self.check_run_config(files["run_config"])
        
        # Load and check data files
        index_df = self.check_baseline_index(files["baseline_index"])
        rejects_df = self.check_baseline_rejects(files["baseline_rejects"])
        geometry_df = self.check_baseline_geometry(files["baseline_geometry"], config)
        
        # Check statistics and bounds
        stats = self.check_geometry_stats(files["geometry_stats"], geometry_df, config)
        bounds = self.check_geometry_bounds(files["geometry_bounds"], geometry_df, stats)
        
        # Check symmetry pairs
        symmetry = self.check_symmetry_pairs(files["symmetry_pairs"], config)
        
        # Check data consistency
        self.check_data_consistency(index_df, geometry_df, rejects_df)
        
        # Check report
        report = self.check_baseline_report(files["baseline_report"])
        
        # Summary
        self.info_msg("=" * 70)
        self.info_msg("Sanity check summary:")
        self.info_msg(f"  Errors: {len(self.errors)}")
        self.info_msg(f"  Warnings: {len(self.warnings)}")
        self.info_msg(f"  Info messages: {len(self.info)}")
        
        if self.errors:
            self.error("Sanity check FAILED - see errors above")
            return 1
        elif self.warnings:
            self.warn("Sanity check PASSED with warnings - see warnings above")
            return 0
        else:
            self.info_msg("Sanity check PASSED - all checks successful!")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check for step1 baseline dataset generation"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=True,
        help="Path to baseline output directory (e.g., baselines/ffhq_mediapipe468_...)",
    )
    
    args = parser.parse_args()
    
    try:
        checker = SanityChecker(args.baseline_dir)
        return checker.run_all_checks()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

