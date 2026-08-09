# -*- coding: utf-8 -*-
"""
Phase 9a: Severity Detection Sweep
====================================
Applies the best detection configuration already identified in Phase 4b
(model, threshold, persistence) for each fault type to every new
severity-level simulation run and computes detection KPIs.

Nothing is re-simulated — the script works entirely from the fdc_runtime_log.csv
files already logged under severity/runs/.

Produces
--------
severity/plots/
    table_severity_detection.csv   — full KPI table (one row per severity level)
    table_severity_detection.tex   — LaTeX tabular for the journal paper

Author : Nima Monghasemi
Date   : March 2026
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths  (all relative to the project root — edit if needed)
# ──────────────────────────────────────────────────────────────────────────────

# Resolve project root as two levels above this script
# severity/analysis/ → severity/ → project_root/
HERE         = Path(__file__).resolve().parent          # severity/analysis/
SEV_DIR      = HERE.parent                              # severity/
PROJECT_ROOT = SEV_DIR.parent                           # project root

MODELS_DIR   = PROJECT_ROOT / "models"
RUNS_DIR     = PROJECT_ROOT / "runs"          # existing Phase 4 detect runs
SEV_RUNS_DIR = SEV_DIR / "runs"               # new severity sweep runs
PLOTS_P4B    = PROJECT_ROOT / "plots"         # Phase 4b best_config JSONs live here
SEV_PLOTS    = SEV_DIR / "plots"

SEV_PLOTS.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TIMESTEP_MINUTES = 10
CP_WATER = 4180.0  # J/(kg·K) — not used here but kept for consistency

# Best-config JSON file for each fault type (from Phase 4b sweep)
BEST_CFG_FILES: Dict[str, str] = {
    "stuck_closed": "fault_analysis_stuckclosed/best_config_stuckclosed.json",
    "stuck_open":   "fault_analysis_stuckopen/best_config_stuckopen.json",
    "supply_curve": "fault_analysis_supplycurve/best_config_supplycurve.json",
}

# Existing Phase-4 detect run folders (used to resolve S3 / baseline)
EXISTING_DETECT_RUNS: Dict[str, str] = {
    "stuck_closed": "run_config_detect_stuckclosed.json",
    "stuck_open":   "run_config_detect_stuckopen.json",
    "supply_curve": "run_config_detect_supplycurve.json",
}

# Severity tags per fault type (NEW levels only; existing S3 appended separately)
SEVERITY_TAGS: Dict[str, List[Tuple[str, float]]] = {
    "stuck_closed": [
        ("sc_s10", 0.10),
        ("sc_s20", 0.20),
        ("sc_s30", 0.30),   # existing — resolved from RUNS_DIR
        ("sc_s50", 0.50),
        ("sc_s70", 0.70),
    ],
    "stuck_open": [
        ("so_b05", 0.5),
        ("so_b10", 1.0),
        ("so_b20", 2.0),    # existing
        ("so_b40", 4.0),
        ("so_b60", 6.0),
    ],
    "supply_curve": [
        ("scu_k02", 2.0),
        ("scu_k05", 5.0),
        ("scu_k08", 8.0),   # existing detect
        ("scu_k12", 12.0),
        ("scu_k15", 15.0),
    ],
}

# Human-readable parameter labels for tables/figures
FAULT_PARAM_LABEL: Dict[str, str] = {
    "stuck_closed": "Flow fraction φ",
    "stuck_open":   "Bias ΔT [°C]",
    "supply_curve": "OAT bias ΔT [K]",
}

FAULT_LABELS: Dict[str, str] = {
    "stuck_closed": "Stuck-Closed",
    "stuck_open":   "Stuck-Open",
    "supply_curve": "Supply-Curve Bias",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers (reused from Phase 4b)
# ──────────────────────────────────────────────────────────────────────────────

def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime column using the month≥10 → 2017 convention."""
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    years = df["month"].apply(lambda m: 2017 if int(m) >= 10 else 2018)
    date_str = (
        years.astype(str) + "-"
        + df["month"].astype(int).astype(str).str.zfill(2) + "-"
        + df["day"].astype(int).astype(str).str.zfill(2)
    )
    base = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    offs = (
        pd.to_timedelta(df["hour"].astype(int), unit="h")
        + pd.to_timedelta(df["minute"].astype(int), unit="m")
    )
    df["datetime"] = base + offs
    return df


def tag_fault_window(df: pd.DataFrame, fw: Dict) -> pd.DataFrame:
    """Add in_fault_window (0/1) column from fault_window dict."""
    df = df.copy()
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]),   int(fw["end_day"])
    sh, eh = int(fw["start_hour"]),  int(fw["end_hour"])
    df["in_fault_window"] = (
        (df["month"] >= sm) & (df["month"] <= em) &
        (df["day"]   >= sd) & (df["day"]   <= ed) &
        (df["hour"]  >= sh) & (df["hour"]  <  eh)
    ).astype(int)
    return df


def compute_gated(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    occ  = df["occupied"]       == 1 if "occupied"       in df.columns else pd.Series(True,  index=df.index)
    htg  = df["heating_active"] == 1 if "heating_active" in df.columns else pd.Series(True,  index=df.index)
    feat = df["feature_ready"]  == 1 if "feature_ready"  in df.columns else pd.Series(True,  index=df.index)
    df["gated"] = (occ & htg & feat).astype(int)
    return df


def reflag(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores < threshold).astype(int)


def compute_session_ids(gated: np.ndarray) -> np.ndarray:
    session = np.zeros(len(gated), dtype=int)
    cur = 0
    prev = 0
    for i, g in enumerate(gated):
        if g == 1 and prev == 0:
            cur += 1
        session[i] = cur if g == 1 else 0
        prev = g
    return session


def apply_persistence(flags: np.ndarray, n: int,
                       sessions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (filtered_retroactive, realtime_causal) flag arrays."""
    filtered = np.zeros_like(flags)
    realtime = np.zeros_like(flags)
    run_len  = 0
    prev_ses = -1
    for i in range(len(flags)):
        if sessions[i] != prev_ses:
            run_len = 0
        prev_ses = sessions[i]
        if flags[i] == 1:
            run_len += 1
        else:
            run_len = 0
        if run_len >= n:
            filtered[max(0, i - n + 1): i + 1] = 1
            realtime[i] = 1
    return filtered, realtime


def compute_metrics(y_true, y_pred, fault_abs_start_idx,
                     gated_indices, realtime_flags=None) -> Dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    latency_steps  = None
    latency_min    = None
    if fault_abs_start_idx is not None and gated_indices is not None:
        flags_lat = realtime_flags if realtime_flags is not None else y_pred
        mask = (y_true == 1) & (flags_lat == 1)
        if mask.any():
            first_pos   = np.where(mask)[0][0]
            first_glob  = gated_indices[first_pos]
            latency_steps = int(first_glob - fault_abs_start_idx)
            latency_min   = latency_steps * TIMESTEP_MINUTES
    return dict(
        precision=prec, recall=recall, f1=f1, fpr=fpr,
        latency_steps=latency_steps, latency_min=latency_min,
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run-folder resolution
# ──────────────────────────────────────────────────────────────────────────────

def find_run_dir(root: Path, config_fname: str) -> Optional[Path]:
    """
    Return the run sub-folder under *root* that contains *config_fname*
    AND an fdc_runtime_log.csv.  Picks newest by CSV mtime.
    """
    matches: List[Tuple[float, Path]] = []
    for p in root.rglob(config_fname):
        rd = p.parent
        csv = rd / "fdc_runtime_log.csv"
        if csv.exists():
            matches.append((csv.stat().st_mtime, rd))
    if not matches:
        return None
    return max(matches, key=lambda x: x[0])[1]


# ──────────────────────────────────────────────────────────────────────────────
# Load models / thresholds
# ──────────────────────────────────────────────────────────────────────────────

def load_thresholds() -> Dict:
    return joblib.load(MODELS_DIR / "thresholds.joblib")


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep logic for one severity run
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_severity_run(
    csv_path:       Path,
    fw:             Dict,
    thresholds:     Dict,
    best_cfg:       Dict,
) -> Dict:
    """
    Apply the best detection config to one severity-level fdc_runtime_log.csv
    and return a dict of KPI values.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = add_datetime(df)
    df = tag_fault_window(df, fw)
    df = compute_gated(df)

    gated_mask    = df["gated"].values == 1
    gated_indices = np.where(gated_mask)[0]
    sessions      = compute_session_ids(df["gated"].values)
    y_true        = df.loc[gated_mask, "in_fault_window"].values

    # Find absolute index of fault-window start
    fw_rows = df[
        (df["month"] == fw["start_month"]) &
        (df["day"]   == fw["start_day"])   &
        (df["hour"]  == fw["start_hour"])
    ]
    fault_abs_start = int(fw_rows.index[0]) if len(fw_rows) > 0 else None

    model     = best_cfg["model"]
    tkey      = best_cfg["threshold_key"]
    n_persist = int(best_cfg["persistence"])

    if model == "majority_vote":
        raw_flags_dict = {}
        for mname in ["ocsvm", "iforest", "lof"]:
            sc = df.loc[gated_mask, f"score_{mname}"].values
            tval = thresholds[mname][tkey]
            raw_flags_dict[mname] = reflag(sc, tval)
        vote_sum = sum(raw_flags_dict.values())
        raw_flags = (vote_sum >= 2).astype(int)
    else:
        sc = df.loc[gated_mask, f"score_{model}"].values
        tval = thresholds[model][tkey]
        raw_flags = reflag(sc, tval)

    if n_persist == 1:
        filtered, realtime = raw_flags, raw_flags
    else:
        filtered, realtime = apply_persistence(
            raw_flags, n_persist, sessions[gated_mask]
        )

    return compute_metrics(
        y_true, filtered, fault_abs_start, gated_indices, realtime
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fault window from run config JSON
# ──────────────────────────────────────────────────────────────────────────────

def get_fault_window(run_dir: Path) -> Optional[Dict]:
    for p in sorted(run_dir.glob("run_config_*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        fw  = cfg.get("fault_window")
        ft  = cfg.get("fault_type", "none")
        if fw and ft != "none":
            return fw
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("Phase 9a — Severity Detection Sweep")
    print("=" * 68)

    thresholds = load_thresholds()
    rows: List[Dict] = []

    for fault_type, sev_list in SEVERITY_TAGS.items():
        # Load best detection config for this fault type
        best_cfg_path = PLOTS_P4B / BEST_CFG_FILES[fault_type]
        if not best_cfg_path.exists():
            print(f"  [WARN] best_config not found for {fault_type}: {best_cfg_path}")
            continue
        best_cfg = json.loads(best_cfg_path.read_text(encoding="utf-8"))
        print(f"\n  Fault: {FAULT_LABELS[fault_type]}")
        print(f"  Best config: model={best_cfg['model']}, "
              f"threshold={best_cfg['threshold_key']}, "
              f"persistence={best_cfg['persistence']}")

        for tag, sev_value in sev_list:
            # Decide where to look for the run folder.
            # Existing (Phase-4) runs live in RUNS_DIR; new severity sweep runs
            # live in SEV_RUNS_DIR.
            #
            # Note on scu_k08 (supply-curve, −8 K):
            #   The original project ran detect at −8 K (exists in RUNS_DIR) but
            #   ran comp at −15 K — a different severity level entirely. For
            #   *detection* Phase 9a only needs the detect run, and the existing
            #   −8 K detect run is physically correct, so is_existing=True here
            #   is safe. Compensation matching (Phase 9b) is handled separately
            #   with independent det_existing / comp_existing flags.
            #
            # scu_k15 is NOT in the existing set: the original comp was at −15 K
            #   but there is no original *detect* run at −15 K, so we need the
            #   new SEV_RUNS_DIR detect run.
            is_existing = tag in ("sc_s30", "so_b20", "scu_k08")

            if is_existing:
                cfg_fname = EXISTING_DETECT_RUNS[fault_type]
                run_dir   = find_run_dir(RUNS_DIR, cfg_fname)
            else:
                cfg_fname = f"run_config_detect_{tag}.json"
                run_dir   = find_run_dir(SEV_RUNS_DIR, cfg_fname)

            if run_dir is None:
                print(f"    [{tag}] run not found — skipping")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **{k: float("nan") for k in
                       ["precision", "recall", "f1", "fpr",
                        "latency_steps", "latency_min"]},
                })
                continue

            csv_path = run_dir / "fdc_runtime_log.csv"
            fw = get_fault_window(run_dir)
            if fw is None:
                # Fall back to existing fault window from best_cfg if available
                fw = best_cfg.get("fault_window")
            if fw is None:
                print(f"    [{tag}] fault_window not found — skipping")
                continue

            print(f"    [{tag}]  sev={sev_value}  ->  {run_dir.name}", end="  ")
            try:
                kpis = evaluate_severity_run(csv_path, fw, thresholds, best_cfg)
                print(f"F1={kpis['f1']:.3f}  Recall={kpis['recall']:.3f}"
                      f"  Latency={kpis['latency_min']} min")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **{k: kpis[k] for k in
                       ["precision", "recall", "f1", "fpr",
                        "latency_steps", "latency_min"]},
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **{k: float("nan") for k in
                       ["precision", "recall", "f1", "fpr",
                        "latency_steps", "latency_min"]},
                })

    if not rows:
        print("\nNo results — check that severity/runs/ is populated.")
        sys.exit(1)

    df_out = pd.DataFrame(rows)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_out = SEV_PLOTS / "table_severity_detection.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"\n  Saved CSV:  {csv_out}")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    tex_cols = {
        "fault_label":    "Fault Type",
        "severity_tag":   "Level",
        "severity_value": "Severity",
        "precision":      "Precision",
        "recall":         "Recall",
        "f1":             "F1",
        "fpr":            "FPR",
        "latency_min":    "Latency [min]",
    }
    df_tex = df_out[list(tex_cols.keys())].rename(columns=tex_cols)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Detection performance across fault severity levels.}")
    lines.append(r"  \label{tab:severity_detection}")
    lines.append(r"  \small")
    ncols = len(tex_cols)
    lines.append(r"  \begin{tabular}{" + "l" * 2 + "r" * (ncols - 2) + "}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(df_tex.columns) + r" \\")
    lines.append(r"    \midrule")

    for _, row in df_tex.iterrows():
        def fmt(v):
            if pd.isna(v):
                return "--"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        lines.append("    " + " & ".join(fmt(v) for v in row.values) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex_out = SEV_PLOTS / "table_severity_detection.tex"
    tex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved LaTeX: {tex_out}")

    # ── severity_metadata.json ────────────────────────────────────────────────
    # Provenance record consumed by Phase9d for cross-script validation.
    metadata = {
        "generated_by":   "Phase9a_severity_detection_sweep.py",
        "generated_at":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_rows_detection": len(df_out),
        "fault_types":    sorted(df_out["fault_type"].unique().tolist()),
        "severity_tags":  df_out[["fault_type", "severity_tag", "severity_value"]]
                          .to_dict(orient="records"),
        "best_configs_used": {},
    }
    # Record which model/threshold/persistence was chosen for each fault type.
    for ft in metadata["fault_types"]:
        best_dir = SEV_PLOTS.parent.parent / "plots" / f"fault_analysis_{ft.replace('_', '')}" 
        # Try all three canonical spellings
        for candidate in [
            PROJECT_ROOT / "plots" / f"fault_analysis_{ft.replace('_', '')}",
            PROJECT_ROOT / "plots" / f"fault_analysis_{ft.replace('stuck_closed','stuckclosed').replace('stuck_open','stuckopen').replace('supply_curve','supplycurve')}",
        ]:
            best_cfg_candidates = list(candidate.glob("best_config_*.json")) if candidate.exists() else []
            if best_cfg_candidates:
                bcfg_path = sorted(best_cfg_candidates)[-1]
                try:
                    bcfg = json.loads(bcfg_path.read_text())
                    metadata["best_configs_used"][ft] = {
                        "file":        str(bcfg_path.relative_to(PROJECT_ROOT)),
                        "model":       bcfg.get("model"),
                        "threshold_key": bcfg.get("threshold_key"),
                        "persistence": bcfg.get("persistence"),
                    }
                except Exception:
                    pass
                break

    meta_out = SEV_PLOTS / "severity_metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved metadata: {meta_out}")

    print("\nPhase 9a complete.")


if __name__ == "__main__":
    main()
