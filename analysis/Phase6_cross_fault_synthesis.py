# -*- coding: utf-8 -*-
"""
Phase 6: Cross-Fault Synthesis (Paper KPI Tables + Compensability Spectrum)
============================================================================
Reads the already-logged runtime CSVs and Phase 4b sweep outputs to produce:

1) Ablation KPI table (baseline vs faulty(no-comp) vs compensated) for each fault type
2) Best detection configuration metrics table (from Phase 4b outputs)
3) Compensation recovery table (% recovery of discomfort, plus energy deltas)
4) "Compensability spectrum" figure

Data sources (expected in the project tree):
- runs/*/fdc_runtime_log.csv          (7 runs: 1 baseline + 3 detect + 3 comp)
- runs/*/run_config_*.json            (fault_window definition)
- plots/fault_analysis*/best_config_*.json
- plots/fault_analysis*/baseline_fpr_sweep_*.csv

Key fix vs earlier draft:
  Fault-window intervals are derived from run_config JSON (fault_window dict),
  NOT from an 'in_fault' column, because the older Phase 4 detect logs
  (stuck_closed_detect, stuck_open_detect) don't have that column.

Author : Nima Monghasemi
Date   : 2026-03-03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Defaults / constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TIMESTEP_MIN = 10
CP_WATER_J_PER_KG_K = 4180.0

RUN_CONFIG_FILES = {
    "baseline_detect":     "run_config_detect_baseline.json",
    "stuckclosed_detect":  "run_config_detect_stuckclosed.json",
    "stuckopen_detect":    "run_config_detect_stuckopen.json",
    "supplycurve_detect":  "run_config_detect_supplycurve.json",
    "stuckclosed_comp":    "run_config_comp_stuckclosed.json",
    "stuckopen_comp":      "run_config_comp_stuckopen.json",
    "supplycurve_comp":    "run_config_comp_supplycurve.json",
}

FAULT_KEYS = ["stuck_closed", "stuck_open", "supply_curve"]

FAULT_TO_RUNKEYS = {
    "stuck_closed": {"faulty": "stuckclosed_detect", "comp": "stuckclosed_comp"},
    "stuck_open":   {"faulty": "stuckopen_detect",   "comp": "stuckopen_comp"},
    "supply_curve": {"faulty": "supplycurve_detect", "comp": "supplycurve_comp"},
}

FAULT_TO_P4B_DIR = {
    "stuck_closed": "plots/fault_analysis_stuckclosed",
    "stuck_open":   "plots/fault_analysis_stuckopen",
    "supply_curve": "plots/fault_analysis_supplycurve",
}

FAULT_TO_P4B_FILES = {
    "stuck_closed": {
        "best_cfg":     "best_config_stuckclosed.json",
        "baseline_fpr": "baseline_fpr_sweep_stuckclosed.csv",
        "sweep":        "sweep_results_stuckclosed.csv",
    },
    "stuck_open": {
        "best_cfg":     "best_config_stuckopen.json",
        "baseline_fpr": "baseline_fpr_sweep_stuckopen.csv",
        "sweep":        "sweep_results_stuckopen.csv",
    },
    "supply_curve": {
        "best_cfg":     "best_config_supplycurve.json",
        "baseline_fpr": "baseline_fpr_sweep_supplycurve.csv",
        "sweep":        "sweep_results_supplycurve.csv",
    },
}

# Pretty labels for the paper tables / figures
FAULT_LABELS = {
    "stuck_closed": "Stuck-Closed Valve",
    "stuck_open":   "Stuck-Open Valve",
    "supply_curve": "Supply-Curve Bias",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility: locate run folders
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_run_dir(runs_root: Path, config_filename: str) -> Path:
    """Return the run directory under *runs_root* that contains *config_filename*
    and an fdc_runtime_log.csv.  When multiple matches exist, pick the
    newest (by runtime-CSV mtime)."""
    matches: list[tuple[float, Path]] = []
    for cfg_path in runs_root.rglob(config_filename):
        run_dir = cfg_path.parent
        csv_path = run_dir / "fdc_runtime_log.csv"
        if csv_path.exists():
            matches.append((csv_path.stat().st_mtime, run_dir))
    if not matches:
        raise FileNotFoundError(
            f"No run dir under {runs_root} contains {config_filename} + fdc_runtime_log.csv"
        )
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


# ──────────────────────────────────────────────────────────────────────────────
# Utility: datetime handling
# ──────────────────────────────────────────────────────────────────────────────

def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Build a datetime column.  Convention: month >= 10 → 2017, else 2018.
    Handles hour == 24 via pd.Timedelta."""
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    required = {"month", "day", "hour", "minute"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Runtime log missing time columns: {sorted(missing)}")

    years = df["month"].apply(lambda m: 2017 if int(m) >= 10 else 2018)
    date_str = (
        years.astype(str) + "-"
        + df["month"].astype(int).astype(str).str.zfill(2) + "-"
        + df["day"].astype(int).astype(str).str.zfill(2)
    )
    base_dates = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    offsets = (
        pd.to_timedelta(df["hour"].astype(int), unit="h")
        + pd.to_timedelta(df["minute"].astype(int), unit="m")
    )
    df["datetime"] = base_dates + offsets
    return df


def load_runtime_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = add_datetime_column(df)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fault-window extraction (from run_config JSON)
# ──────────────────────────────────────────────────────────────────────────────

def fault_window_from_config(run_dir: Path) -> Optional[Dict]:
    """Try to load and return the fault_window dict from any run_config
    JSON found in *run_dir*.  Returns None for baseline (no fault_window key
    or fault_type == 'none')."""
    for p in sorted(run_dir.glob("run_config_*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        ft = cfg.get("fault_type", "none")
        fw = cfg.get("fault_window")
        if ft != "none" and fw is not None:
            return fw
    return None


def fault_window_to_intervals(fw: Dict) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert a fault_window dict (from run config) into a list of
    (start, end) datetime intervals.

    The dict has keys: start_month, start_day, end_month, end_day,
    start_hour, end_hour.  The year convention is month >= 10 → 2017
    else 2018.

    A window spanning multiple days with intra-day hours (e.g. 08:00–17:00
    on Jan 15–16) yields ONE interval PER DAY, so the overnight gap — when
    the fault is not active — stays out of the KPI window.  A continuous
    0:00–24:00 window still produces a single interval.
    """
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]),   int(fw["end_day"])
    sh, eh = int(fw["start_hour"]),  int(fw["end_hour"])

    sy = 2017 if sm >= 10 else 2018
    ey = 2017 if em >= 10 else 2018

    # If the window covers full days (0:00 to 24:00), return one interval
    if sh == 0 and eh == 24:
        start_dt = pd.Timestamp(year=sy, month=sm, day=sd, hour=0)
        end_dt = pd.Timestamp(year=ey, month=em, day=ed) + pd.Timedelta(days=1)
        return [(start_dt, end_dt)]

    # Otherwise, generate one interval per day so that overnight gaps
    # (when the fault is NOT active) are excluded from KPI windows.
    intervals = []
    current = pd.Timestamp(year=sy, month=sm, day=sd)
    final_day = pd.Timestamp(year=ey, month=em, day=ed)

    while current <= final_day:
        day_start = current + pd.Timedelta(hours=sh)
        if eh == 24:
            day_end = current + pd.Timedelta(days=1)
        else:
            day_end = current + pd.Timedelta(hours=eh)
        intervals.append((day_start, day_end))
        current += pd.Timedelta(days=1)

    return intervals


def mask_from_intervals(df: pd.DataFrame,
                        intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
                        ) -> pd.Series:
    """Return a boolean Series aligned with *df* that is True for rows
    whose datetime falls in any of the half-open intervals [start, end)."""
    mask = pd.Series(False, index=df.index)
    for start, end in intervals:
        mask |= (df["datetime"] >= start) & (df["datetime"] < end)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# KPI computations
# ──────────────────────────────────────────────────────────────────────────────

def flow_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float)
    w = weights.astype(float).clip(lower=0.0)
    ok = v.notna() & w.notna() & (w > 0)
    if ok.sum() == 0:
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def compute_kpis(df_window: pd.DataFrame,
                 timestep_min: int = DEFAULT_TIMESTEP_MIN) -> Dict[str, float]:
    """Compute the KPI bundle for a given time-window slice."""
    dt_hours   = timestep_min / 60.0
    dt_seconds = timestep_min * 60.0
    cols = set(df_window.columns)
    out: Dict[str, float] = {}

    out["n_rows"] = int(len(df_window))

    # ---------- masks ----------
    occ = df_window["occupied"] == 1 if "occupied" in cols else pd.Series(True, index=df_window.index)
    htg = df_window["heating_active"] == 1 if "heating_active" in cols else pd.Series(True, index=df_window.index)

    # ---------- temp_error ----------
    if "temp_error" in cols:
        te = df_window.loc[occ, "temp_error"].astype(float)
    elif {"zone_temp", "intended_sp"} <= cols:
        te = (df_window.loc[occ, "zone_temp"].astype(float)
              - df_window.loc[occ, "intended_sp"].astype(float))
    elif {"zone_temp", "htg_sp"} <= cols:
        te = (df_window.loc[occ, "zone_temp"].astype(float)
              - df_window.loc[occ, "htg_sp"].astype(float))
    else:
        te = pd.Series(dtype=float)

    out["n_occ"] = int(occ.sum())
    out["ddh_abs_C_h"]  = float(te.abs().sum()  * dt_hours) if len(te) else float("nan")
    out["ddh_cold_C_h"] = float(np.maximum(-te, 0).sum() * dt_hours) if len(te) else float("nan")
    out["ddh_hot_C_h"]  = float(np.maximum( te, 0).sum() * dt_hours) if len(te) else float("nan")

    # ---------- energy ----------
    if {"m_dot", "delta_T_hw"} <= cols:
        mdot = df_window.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT   = df_window.loc[htg, "delta_T_hw"].astype(float).fillna(0.0)
    elif {"m_dot", "t_inlet", "t_outlet"} <= cols:
        mdot = df_window.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT   = (df_window.loc[htg, "t_inlet"].astype(float)
                - df_window.loc[htg, "t_outlet"].astype(float)).fillna(0.0)
    else:
        mdot = pd.Series(dtype=float)
        dT   = pd.Series(dtype=float)

    out["n_htg_active"] = int(htg.sum())

    if len(mdot):
        q_dot_W = mdot * CP_WATER_J_PER_KG_K * np.maximum(dT, 0.0)
        out["dh_energy_kWh_est"]    = float((q_dot_W * dt_seconds).sum() / 3.6e6)
        out["avg_heat_power_kW_est"] = float(q_dot_W.mean() / 1000.0)
    else:
        out["dh_energy_kWh_est"]    = float("nan")
        out["avg_heat_power_kW_est"] = float("nan")

    # ---------- temperatures ----------
    if "zone_temp" in cols:
        out["zone_temp_mean_C"] = float(df_window.loc[occ, "zone_temp"].astype(float).mean()) if occ.sum() else float("nan")
    else:
        out["zone_temp_mean_C"] = float("nan")

    # intended_sp may not exist in older detect logs — fall back to htg_sp
    if "intended_sp" in cols:
        out["intended_sp_mean_C"] = float(df_window.loc[occ, "intended_sp"].astype(float).mean()) if occ.sum() else float("nan")
    elif "htg_sp" in cols:
        out["intended_sp_mean_C"] = float(df_window.loc[occ, "htg_sp"].astype(float).mean()) if occ.sum() else float("nan")
    else:
        out["intended_sp_mean_C"] = float("nan")

    if {"t_outlet", "m_dot"} <= cols:
        out["return_temp_flow_wtd_C"] = flow_weighted_mean(
            df_window.loc[htg, "t_outlet"], df_window.loc[htg, "m_dot"])
    else:
        out["return_temp_flow_wtd_C"] = float("nan")

    if {"t_inlet", "m_dot"} <= cols:
        out["inlet_temp_flow_wtd_C"] = flow_weighted_mean(
            df_window.loc[htg, "t_inlet"], df_window.loc[htg, "m_dot"])
    else:
        out["inlet_temp_flow_wtd_C"] = float("nan")

    if "t_supply" in cols:
        out["t_supply_mean_C"] = float(df_window.loc[htg, "t_supply"].astype(float).mean()) if htg.sum() else float("nan")
    else:
        out["t_supply_mean_C"] = float("nan")

    # ---------- compensation diagnostics ----------
    if "comp_active" in cols:
        out["comp_active_rate"] = float((df_window["comp_active"] == 1).mean())
    else:
        out["comp_active_rate"] = float("nan")

    if "supply_temp_cmd" in cols:
        out["supply_temp_cmd_mean_C"] = float(df_window["supply_temp_cmd"].astype(float).mean())
    else:
        out["supply_temp_cmd_mean_C"] = float("nan")

    return out


def recovery_percent(baseline: float, faulty: float, compensated: float) -> float:
    """Recovery % for a 'lower-is-better' metric:
    (faulty − compensated) / (faulty − baseline) × 100."""
    if any(np.isnan(x) for x in [baseline, faulty, compensated]):
        return float("nan")
    denom = faulty - baseline
    if abs(denom) < 1e-12:
        return float("nan")
    return float((faulty - compensated) / denom * 100.0)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4b detection metrics loader
# ──────────────────────────────────────────────────────────────────────────────

def load_best_detection_metrics(project_dir: Path, fault_type: str) -> Dict:
    p4b_dir = project_dir / FAULT_TO_P4B_DIR[fault_type]
    files   = FAULT_TO_P4B_FILES[fault_type]

    best_path    = p4b_dir / files["best_cfg"]
    basefpr_path = p4b_dir / files["baseline_fpr"]

    if not best_path.exists():
        raise FileNotFoundError(f"Missing Phase 4b best config: {best_path}")

    best_cfg = json.loads(best_path.read_text(encoding="utf-8"))

    # Attach baseline FPR if available
    if basefpr_path.exists():
        bf = pd.read_csv(basefpr_path)
        model = best_cfg.get("model")
        tk    = best_cfg.get("threshold_key")
        p     = best_cfg.get("persistence")
        bf_match = bf[
            (bf["model"] == model) &
            (bf["threshold_key"] == tk) &
            (bf["persistence"] == p)
        ]
        best_cfg["baseline_fpr"] = float(bf_match["baseline_fpr"].iloc[0]) if len(bf_match) else float("nan")
    else:
        best_cfg["baseline_fpr"] = float("nan")

    # Ensure numeric
    for k in ["precision", "recall", "f1", "fpr", "baseline_fpr",
              "latency_minutes", "latency_steps"]:
        if k in best_cfg and best_cfg[k] is not None:
            try:
                best_cfg[k] = float(best_cfg[k])
            except (ValueError, TypeError):
                pass

    return best_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_compensability_spectrum(recovery_df: pd.DataFrame, out_path: Path):
    """Scatter: x = energy penalty vs baseline, y = discomfort recovery %."""
    df = recovery_df.copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    colours = ["#1E88E5", "#43A047", "#F4511E"]
    x = df["energy_delta_vs_baseline_pct"].values
    y = df["ddh_recovery_pct"].values
    labels = [FAULT_LABELS.get(ft, ft) for ft in df["fault_type"].values]

    ax.scatter(x, y, s=120, c=colours[:len(x)], zorder=5, edgecolors="k", linewidths=0.5)

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(8, 8), fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.axhline(0,   color="gray", lw=0.8, ls="--", alpha=0.6, label="No recovery")
    ax.axhline(100, color="gray", lw=0.8, ls=":",  alpha=0.6, label="Full recovery")

    ax.set_xlabel("Energy delta vs baseline during fault window [%]", fontsize=11)
    ax.set_ylabel("Discomfort recovery [%]  (DDH$_{abs}$)", fontsize=11)
    ax.set_title("Compensability Spectrum", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-20, 140)
    fig.tight_layout()

    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Spectrum figure saved: {out_path.with_suffix('.png')}")


def plot_kpi_bar_chart(kpi_df: pd.DataFrame, out_path: Path):
    """Grouped bar chart: DDH and Energy side-by-side for each fault × scenario."""
    scenarios = ["baseline", "faulty_no_comp", "compensated"]
    scenario_labels = ["Baseline", "Faulty\n(no comp.)", "Compensated"]
    colours = ["#4CAF50", "#F44336", "#2196F3"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("ddh_abs_C_h",       "Degree-hours [°C·h]",  "Discomfort (DDH abs)"),
        ("dh_energy_kWh_est", "Energy [kWh]",          "Hydronic Heat Delivered"),
    ]):
        ax = axes[ax_idx]
        n_faults = len(FAULT_KEYS)
        bar_w = 0.22
        x_base = np.arange(n_faults)

        for s_idx, sc in enumerate(scenarios):
            vals = []
            for ft in FAULT_KEYS:
                row = kpi_df[(kpi_df["fault_type"] == ft) & (kpi_df["scenario"] == sc)]
                vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            ax.bar(x_base + s_idx * bar_w, vals, bar_w,
                   label=scenario_labels[s_idx], color=colours[s_idx],
                   edgecolor="k", linewidth=0.4)

        ax.set_xticks(x_base + bar_w)
        ax.set_xticklabels([FAULT_LABELS[ft] for ft in FAULT_KEYS], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Ablation KPI Comparison Across Fault Types",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table helper
# ──────────────────────────────────────────────────────────────────────────────

def write_latex_table(df: pd.DataFrame, caption: str, label: str,
                      out_path: Path, fmt: Dict[str, str] | None = None):
    """Write a simple LaTeX booktabs table to file."""
    if fmt is None:
        fmt = {}
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    col_fmt = "l" + "r" * (len(df.columns) - 1)
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"    \toprule")

    # header
    header = " & ".join(c.replace("_", r"\_") for c in df.columns) + r" \\"
    lines.append(f"    {header}")
    lines.append(r"    \midrule")

    # rows
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in fmt and isinstance(v, (int, float)) and not np.isnan(v):
                cells.append(f"{v:{fmt[c]}}")
            elif isinstance(v, float) and np.isnan(v):
                cells.append("--")
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Cross-fault synthesis (tables + figures)."
    )
    parser.add_argument("--project-dir", default=None,
                        help="Project root (defaults to parent of this script's folder).")
    parser.add_argument("--timestep-min", type=int, default=DEFAULT_TIMESTEP_MIN)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="plots/cross_fault_synthesis")
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent
    runs_root   = (project_dir / args.runs_dir).resolve()
    out_dir     = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 6: CROSS-FAULT SYNTHESIS")
    print(f"  Project dir : {project_dir}")
    print(f"  Runs root   : {runs_root}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 78)

    # ── 1. Locate run directories ──
    run_dirs: Dict[str, Path] = {}
    for key, cfg_name in RUN_CONFIG_FILES.items():
        rd = find_latest_run_dir(runs_root, cfg_name)
        run_dirs[key] = rd
        print(f"  {key:22s} -> {rd.name}")

    # ── 2. Load runtime logs ──
    logs: Dict[str, pd.DataFrame] = {}
    for key, rd in run_dirs.items():
        logs[key] = load_runtime_log(rd / "fdc_runtime_log.csv")
        print(f"  Loaded {key:22s}: {len(logs[key]):>7,} rows")

    # ── 3. Derive fault intervals from run_config JSON ──
    #    We read the fault_window from the *faulty-detect* run config.
    #    This is the ground-truth schedule, works even when `in_fault` column
    #    is absent.
    fault_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ft in FAULT_KEYS:
        faulty_key = FAULT_TO_RUNKEYS[ft]["faulty"]
        rd = run_dirs[faulty_key]
        fw = fault_window_from_config(rd)
        if fw is None:
            print(f"  WARNING: No fault_window in config for {ft}")
            fault_intervals[ft] = []
        else:
            intervals = fault_window_to_intervals(fw)
            fault_intervals[ft] = intervals
            for s, e in intervals:
                dur_h = (e - s).total_seconds() / 3600
                print(f"  Fault window [{ft}]: {s}  ->  {e}  ({dur_h:.1f} h)")

    # ── 4. Compute KPIs per (fault_type × scenario) ──
    baseline_key = "baseline_detect"
    kpi_rows = []

    for ft in FAULT_KEYS:
        faulty_key = FAULT_TO_RUNKEYS[ft]["faulty"]
        comp_key   = FAULT_TO_RUNKEYS[ft]["comp"]
        intervals  = fault_intervals[ft]

        for scenario, rk in [("baseline", baseline_key),
                              ("faulty_no_comp", faulty_key),
                              ("compensated", comp_key)]:
            df = logs[rk]
            win_mask = mask_from_intervals(df, intervals)
            df_w = df.loc[win_mask].copy()

            kpi = compute_kpis(df_w, timestep_min=args.timestep_min)
            kpi["fault_type"] = ft
            kpi["scenario"]   = scenario
            kpi["run_key"]    = rk
            kpi["run_dir"]    = str(run_dirs[rk])
            kpi_rows.append(kpi)

    kpi_df = pd.DataFrame(kpi_rows)

    # ── 5. Recovery & energy delta table ──
    rec_rows = []
    for ft in FAULT_KEYS:
        base   = kpi_df[(kpi_df["fault_type"] == ft) & (kpi_df["scenario"] == "baseline")].iloc[0]
        faulty = kpi_df[(kpi_df["fault_type"] == ft) & (kpi_df["scenario"] == "faulty_no_comp")].iloc[0]
        comp   = kpi_df[(kpi_df["fault_type"] == ft) & (kpi_df["scenario"] == "compensated")].iloc[0]

        ddh_b, ddh_f, ddh_c = [float(x["ddh_abs_C_h"]) for x in [base, faulty, comp]]
        e_b, e_f, e_c       = [float(x["dh_energy_kWh_est"]) for x in [base, faulty, comp]]

        rec_rows.append({
            "fault_type": ft,
            "ddh_baseline":   ddh_b,
            "ddh_faulty":     ddh_f,
            "ddh_compensated": ddh_c,
            "ddh_recovery_pct": recovery_percent(ddh_b, ddh_f, ddh_c),
            "energy_baseline_kWh":   e_b,
            "energy_faulty_kWh":     e_f,
            "energy_comp_kWh":       e_c,
            "energy_delta_vs_baseline_pct": (
                (e_c - e_b) / e_b * 100.0
                if not np.isnan(e_c) and not np.isnan(e_b) and abs(e_b) > 1e-12
                else float("nan")
            ),
            "energy_delta_fault_to_comp_pct": (
                (e_c - e_f) / e_f * 100.0
                if not np.isnan(e_c) and not np.isnan(e_f) and abs(e_f) > 1e-12
                else float("nan")
            ),
        })
    recovery_df = pd.DataFrame(rec_rows)

    # ── 6. Detection metrics table from Phase 4b ──
    det_rows = []
    for ft in FAULT_KEYS:
        try:
            best = load_best_detection_metrics(project_dir, ft)
            best["fault_type"] = ft
            det_rows.append(best)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
    det_df = pd.DataFrame(det_rows)

    # ── 7. Save CSVs ──
    kpi_csv = out_dir / "table_ablation_kpis.csv"
    det_csv = out_dir / "table_detection_best.csv"
    rec_csv = out_dir / "table_comp_recovery.csv"

    kpi_df.to_csv(kpi_csv, index=False)
    det_df.to_csv(det_csv, index=False)
    recovery_df.to_csv(rec_csv, index=False)

    # compact paper-view
    paper_cols = [c for c in [
        "fault_type", "scenario",
        "ddh_abs_C_h", "ddh_cold_C_h", "ddh_hot_C_h",
        "dh_energy_kWh_est", "return_temp_flow_wtd_C",
        "zone_temp_mean_C", "intended_sp_mean_C",
        "comp_active_rate",
    ] if c in kpi_df.columns]
    kpi_df[paper_cols].to_csv(out_dir / "table_ablation_kpis_compact.csv", index=False)

    # ── 8. Figures ──
    plot_compensability_spectrum(recovery_df, out_dir / "compensability_spectrum")
    plot_kpi_bar_chart(kpi_df, out_dir / "ablation_bar_chart")

    # ── 9. LaTeX tables ──
    # Detection table
    if len(det_df):
        det_latex_df = det_df[["fault_type", "model", "threshold_key", "persistence",
                                "precision", "recall", "f1", "fpr",
                                "baseline_fpr", "latency_minutes"]].copy()
        det_latex_df.rename(columns={
            "fault_type": "Fault", "model": "Model", "threshold_key": "Thr.",
            "persistence": "Pers.", "precision": "Prec.", "recall": "Rec.",
            "f1": "F1", "fpr": "FPR", "baseline_fpr": "BL-FPR",
            "latency_minutes": "Lat. [min]"
        }, inplace=True)
        write_latex_table(det_latex_df,
                          caption="Best detection configuration per fault type.",
                          label="tab:detection_best",
                          out_path=out_dir / "table_detection_best.tex",
                          fmt={"Prec.": ".3f", "Rec.": ".3f", "F1": ".3f",
                               "FPR": ".4f", "BL-FPR": ".4f", "Lat. [min]": ".0f"})

    # Recovery table
    rec_latex = recovery_df[["fault_type", "ddh_baseline", "ddh_faulty",
                              "ddh_compensated", "ddh_recovery_pct",
                              "energy_delta_vs_baseline_pct"]].copy()
    rec_latex.rename(columns={
        "fault_type": "Fault",
        "ddh_baseline": "DDH BL",
        "ddh_faulty": "DDH Faulty",
        "ddh_compensated": "DDH Comp.",
        "ddh_recovery_pct": "Recovery [\\%]",
        "energy_delta_vs_baseline_pct": "Energy $\\Delta$ [\\%]",
    }, inplace=True)
    write_latex_table(rec_latex,
                      caption="Compensation recovery and energy impact per fault type.",
                      label="tab:comp_recovery",
                      out_path=out_dir / "table_comp_recovery.tex",
                      fmt={"DDH BL": ".1f", "DDH Faulty": ".1f",
                           "DDH Comp.": ".1f", "Recovery [\\%]": ".1f",
                           "Energy $\\Delta$ [\\%]": ".1f"})

    # ── 10. Metadata ──
    meta = {
        "project_dir":  str(project_dir),
        "runs_root":    str(runs_root),
        "out_dir":      str(out_dir),
        "timestep_min": int(args.timestep_min),
        "run_dirs":     {k: str(v) for k, v in run_dirs.items()},
        "fault_intervals": {
            ft: [(str(s), str(e)) for s, e in fault_intervals[ft]]
            for ft in FAULT_KEYS
        },
    }
    (out_dir / "synthesis_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # ── Summary ──
    print("\n" + "=" * 78)
    print("PHASE 6 COMPLETE — outputs:")
    print(f"  KPI table (CSV):     {kpi_csv}")
    print(f"  Detection table:     {det_csv}")
    print(f"  Recovery table:      {rec_csv}")
    print(f"  Spectrum figure:     {out_dir / 'compensability_spectrum.png'}")
    print(f"  Bar chart figure:    {out_dir / 'ablation_bar_chart.png'}")
    print(f"  LaTeX tables:        {out_dir / 'table_detection_best.tex'}")
    print(f"                       {out_dir / 'table_comp_recovery.tex'}")
    print("=" * 78)

    # Print quick summary to console
    print("\n-- RECOVERY SUMMARY --")
    for _, row in recovery_df.iterrows():
        ft = row["fault_type"]
        print(f"  {FAULT_LABELS.get(ft, ft):25s}:  "
              f"DDH recovery = {row['ddh_recovery_pct']:6.1f}%   "
              f"Energy delta vs BL = {row['energy_delta_vs_baseline_pct']:+6.1f}%")


if __name__ == "__main__":
    main()
