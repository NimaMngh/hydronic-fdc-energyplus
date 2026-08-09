# -*- coding: utf-8 -*-
"""
Phase 9b: Severity Compensation KPI Analysis
=============================================
Computes compensation KPIs (DDH Recovery %, Energy Delta %) for every new
severity-level simulation pair (detect + comp) versus the shared baseline run.

KPI Formulas
------------
  DDH Recovery %  = (DDH_faulty − DDH_comp) / (DDH_faulty − DDH_baseline) × 100
  Energy Delta %  = (E_comp − E_baseline) / E_baseline × 100

Works entirely from already-logged fdc_runtime_log.csv files — no
re-simulation needed.

Produces
--------
severity/plots/
    table_severity_compensation.csv
    table_severity_compensation.tex

Author : Nima Monghasemi
Date   : March 2026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

HERE         = Path(__file__).resolve().parent   # severity/analysis/
SEV_DIR      = HERE.parent                       # severity/
PROJECT_ROOT = SEV_DIR.parent                    # project root

RUNS_DIR     = PROJECT_ROOT / "runs"             # existing Phase 4/5 runs
SEV_RUNS_DIR = SEV_DIR / "runs"
SEV_PLOTS    = SEV_DIR / "plots"
SEV_PLOTS.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TIMESTEP_MIN  = 10
DT_HOURS      = TIMESTEP_MIN / 60.0
DT_SECONDS    = TIMESTEP_MIN * 60.0
CP_WATER      = 4180.0  # J/(kg·K)

BASELINE_CONFIG_FNAME = "run_config_detect_baseline.json"

# Existing moderate-severity (S3) run config file names
EXISTING_DETECT_CFGS: Dict[str, str] = {
    "stuck_closed": "run_config_detect_stuckclosed.json",
    "stuck_open":   "run_config_detect_stuckopen.json",
    "supply_curve": "run_config_detect_supplycurve.json",
}
EXISTING_COMP_CFGS: Dict[str, str] = {
    "stuck_closed": "run_config_comp_stuckclosed.json",
    "stuck_open":   "run_config_comp_stuckopen.json",
    "supply_curve": "run_config_comp_supplycurve.json",
}

FAULT_LABELS: Dict[str, str] = {
    "stuck_closed": "Stuck-Closed",
    "stuck_open":   "Stuck-Open",
    "supply_curve": "Supply-Curve Bias",
}

FAULT_PARAM_LABEL: Dict[str, str] = {
    "stuck_closed": "Flow fraction φ",
    "stuck_open":   "Bias ΔT [°C]",
    "supply_curve": "OAT bias ΔT [K]",
}

# Severity tags for compensation (detect + comp pairs).
# Format: (tag, severity_value, det_existing, comp_existing)
#   det_existing=True  → look for detect run in RUNS_DIR (Phase-4 runs)
#   comp_existing=True → look for comp   run in RUNS_DIR (Phase-5 runs)
#   False              → look in SEV_RUNS_DIR (new severity sweep runs)
#
# IMPORTANT — Supply-curve asymmetry
# -----------------------------------
# The original project ran detect at −8 K and comp at −15 K (different
# severity levels). For the severity sweep we need *matched* pairs:
#   scu_k08: detect already exists in RUNS_DIR at −8 K, but we must run
#            a NEW comp at −8 K (in SEV_RUNS_DIR) — NOT reuse the −15 K
#            existing comp run. Hence det_existing=True, comp_existing=False.
#   scu_k15: both detect and comp are new runs in SEV_RUNS_DIR.
SEVERITY_PAIRS: Dict[str, List[Tuple[str, float, bool, bool]]] = {
    "stuck_closed": [
        ("sc_s10", 0.10, False, False),
        ("sc_s20", 0.20, False, False),
        ("sc_s30", 0.30, True,  True ),  # S3: both detect & comp exist in RUNS_DIR
        ("sc_s50", 0.50, False, False),
        ("sc_s70", 0.70, False, False),
    ],
    "stuck_open": [
        ("so_b05", 0.5,  False, False),
        ("so_b10", 1.0,  False, False),
        ("so_b20", 2.0,  True,  True ),  # S3: both exist in RUNS_DIR
        ("so_b40", 4.0,  False, False),
        ("so_b60", 6.0,  False, False),
    ],
    "supply_curve": [
        ("scu_k02",  2.0, False, False),
        ("scu_k05",  5.0, False, False),
        ("scu_k08",  8.0, True,  False),  # detect=existing (−8K), comp=new in SEV_RUNS
        ("scu_k12", 12.0, False, False),
        ("scu_k15", 15.0, False, False),  # both new in SEV_RUNS
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
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


def find_run_dir(root: Path, config_fname: str) -> Optional[Path]:
    """Return newest run dir containing config_fname + fdc_runtime_log.csv."""
    matches: List[Tuple[float, Path]] = []
    for p in root.rglob(config_fname):
        rd  = p.parent
        csv = rd / "fdc_runtime_log.csv"
        if csv.exists():
            matches.append((csv.stat().st_mtime, rd))
    if not matches:
        return None
    return max(matches, key=lambda x: x[0])[1]


def fault_window_from_run_dir(run_dir: Path) -> Optional[Dict]:
    """Extract fault_window dict from run config JSON in run_dir."""
    for p in sorted(run_dir.glob("run_config_*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        fw  = cfg.get("fault_window")
        ft  = cfg.get("fault_type", "none")
        if fw and ft != "none":
            return fw
    return None


def fault_window_intervals(fw: Dict) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert fault_window dict to list of (start, end) datetime intervals."""
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]),   int(fw["end_day"])
    sh, eh = int(fw["start_hour"]),  int(fw["end_hour"])
    sy = 2017 if sm >= 10 else 2018
    ey = 2017 if em >= 10 else 2018

    if sh == 0 and eh == 24:
        return [(
            pd.Timestamp(year=sy, month=sm, day=sd),
            pd.Timestamp(year=ey, month=em, day=ed) + pd.Timedelta(days=1),
        )]

    intervals = []
    cur = pd.Timestamp(year=sy, month=sm, day=sd)
    end_day = pd.Timestamp(year=ey, month=em, day=ed)
    while cur <= end_day:
        day_start = cur + pd.Timedelta(hours=sh)
        day_end   = cur + pd.Timedelta(hours=eh if eh < 24 else 24)
        intervals.append((day_start, day_end))
        cur += pd.Timedelta(days=1)
    return intervals


def mask_from_intervals(df: pd.DataFrame,
                         intervals: List[Tuple]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for s, e in intervals:
        mask |= (df["datetime"] >= s) & (df["datetime"] < e)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# KPI computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_kpis_window(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute DDH (degree-hours discomfort) and estimated hydronic energy
    for the rows in *df* (already subsetted to the fault window).
    """
    cols = set(df.columns)

    # Occupancy / heating masks
    occ = df["occupied"] == 1       if "occupied"       in cols else pd.Series(True, index=df.index)
    htg = df["heating_active"] == 1 if "heating_active" in cols else pd.Series(True, index=df.index)

    # Temperature error → DDH
    if "temp_error" in cols:
        te = df.loc[occ, "temp_error"].astype(float)
    elif {"zone_temp", "intended_sp"} <= cols:
        te = df.loc[occ, "zone_temp"].astype(float) - df.loc[occ, "intended_sp"].astype(float)
    elif {"zone_temp", "htg_sp"} <= cols:
        te = df.loc[occ, "zone_temp"].astype(float) - df.loc[occ, "htg_sp"].astype(float)
    else:
        te = pd.Series(dtype=float)

    ddh_abs = float(te.abs().sum() * DT_HOURS) if len(te) else float("nan")

    # Hydronic energy (kWh)
    if {"m_dot", "t_inlet", "t_outlet"} <= cols:
        mdot = df.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT   = (df.loc[htg, "t_inlet"].astype(float)
                - df.loc[htg, "t_outlet"].astype(float)).fillna(0.0)
    elif {"m_dot", "delta_T_hw"} <= cols:
        mdot = df.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT   = df.loc[htg, "delta_T_hw"].astype(float).fillna(0.0)
    else:
        mdot = pd.Series(dtype=float)
        dT   = pd.Series(dtype=float)

    if len(mdot) and len(dT):
        q_W   = mdot * CP_WATER * np.maximum(dT, 0.0)
        energy_kWh = float((q_W * DT_SECONDS).sum() / 3.6e6)
    else:
        energy_kWh = float("nan")

    return {"ddh_abs": ddh_abs, "energy_kWh": energy_kWh}


def compute_recovery_kpis(
    kpi_baseline: Dict,
    kpi_faulty:   Dict,
    kpi_comp:     Dict,
) -> Dict[str, float]:
    """
    Compute DDH Recovery % and Energy Delta % from three KPI bundles.

    DDH Recovery %  = (DDH_faulty − DDH_comp) / (DDH_faulty − DDH_baseline) × 100
    Energy Delta %  = (E_comp − E_baseline) / E_baseline × 100
    """
    ddh_bl    = kpi_baseline["ddh_abs"]
    ddh_fault = kpi_faulty["ddh_abs"]
    ddh_comp  = kpi_comp["ddh_abs"]

    e_bl   = kpi_baseline["energy_kWh"]
    e_comp = kpi_comp["energy_kWh"]

    denom_ddh = ddh_fault - ddh_bl
    ddh_recovery_pct = (
        (ddh_fault - ddh_comp) / denom_ddh * 100.0
        if abs(denom_ddh) > 1e-9 else float("nan")
    )

    energy_delta_pct = (
        (e_comp - e_bl) / e_bl * 100.0
        if (not np.isnan(e_bl)) and e_bl > 0 else float("nan")
    )

    return {
        "ddh_baseline":       ddh_bl,
        "ddh_faulty":         ddh_fault,
        "ddh_comp":           ddh_comp,
        "ddh_recovery_pct":   ddh_recovery_pct,
        "energy_baseline_kWh": e_bl,
        "energy_faulty_kWh":   kpi_faulty["energy_kWh"],
        "energy_comp_kWh":     e_comp,
        "energy_delta_pct":    energy_delta_pct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load one CSV and extract fault-window KPIs
# ──────────────────────────────────────────────────────────────────────────────

def load_window_kpis(csv_path: Path, intervals: List) -> Dict[str, float]:
    df = pd.read_csv(csv_path, low_memory=False)
    df = add_datetime(df)
    mask = mask_from_intervals(df, intervals)
    df_win = df[mask].copy()
    if len(df_win) == 0:
        return {"ddh_abs": float("nan"), "energy_kWh": float("nan")}
    return compute_kpis_window(df_win)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("Phase 9b — Severity Compensation KPI Analysis")
    print("=" * 68)

    # ── Baseline KPIs ────────────────────────────────────────────────────────
    baseline_dir = find_run_dir(RUNS_DIR, BASELINE_CONFIG_FNAME)
    if baseline_dir is None:
        print("ERROR: Baseline run not found.  Ensure the baseline simulation "
              "exists in the runs/ folder.")
        sys.exit(1)
    print(f"\n  Baseline run: {baseline_dir.name}")

    rows: List[Dict] = []

    for fault_type, sev_list in SEVERITY_PAIRS.items():
        print(f"\n  Fault: {FAULT_LABELS[fault_type]}")

        for tag, sev_value, det_existing, comp_existing in sev_list:

            # Resolve detect run dir — independently from comp
            if det_existing:
                det_dir = find_run_dir(RUNS_DIR, EXISTING_DETECT_CFGS[fault_type])
            else:
                det_dir = find_run_dir(SEV_RUNS_DIR, f"run_config_detect_{tag}.json")

            # Resolve comp run dir — independently from detect
            # (critical for supply-curve scu_k08 where detect is existing
            #  at −8K but the original comp was at a different severity −15K;
            #  we therefore always look for a matched new comp in SEV_RUNS_DIR)
            if comp_existing:
                cmp_dir = find_run_dir(RUNS_DIR, EXISTING_COMP_CFGS[fault_type])
            else:
                cmp_dir = find_run_dir(SEV_RUNS_DIR, f"run_config_comp_{tag}.json")

            if det_dir is None or cmp_dir is None:
                missing = []
                if det_dir is None: missing.append("detect")
                if cmp_dir is None: missing.append("comp")
                print(f"    [{tag}] missing: {missing} — skipping")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **{k: float("nan") for k in [
                        "ddh_baseline", "ddh_faulty", "ddh_comp",
                        "ddh_recovery_pct",
                        "energy_baseline_kWh", "energy_faulty_kWh",
                        "energy_comp_kWh", "energy_delta_pct",
                    ]},
                })
                continue

            # Get fault window from detect run config
            fw = fault_window_from_run_dir(det_dir)
            if fw is None:
                fw = fault_window_from_run_dir(cmp_dir)
            if fw is None:
                print(f"    [{tag}] fault_window not resolved — skipping")
                continue

            intervals = fault_window_intervals(fw)

            print(f"    [{tag}]  sev={sev_value} ...", end="  ")
            try:
                kpi_bl  = load_window_kpis(baseline_dir / "fdc_runtime_log.csv", intervals)
                kpi_det = load_window_kpis(det_dir / "fdc_runtime_log.csv", intervals)
                kpi_cmp = load_window_kpis(cmp_dir / "fdc_runtime_log.csv", intervals)

                kpis = compute_recovery_kpis(kpi_bl, kpi_det, kpi_cmp)
                print(f"DDH_rec={kpis['ddh_recovery_pct']:.1f}%  "
                      f"ΔE={kpis['energy_delta_pct']:+.1f}%")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **kpis,
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                rows.append({
                    "fault_type":    fault_type,
                    "fault_label":   FAULT_LABELS[fault_type],
                    "severity_tag":  tag,
                    "severity_value": sev_value,
                    "param_label":   FAULT_PARAM_LABEL[fault_type],
                    **{k: float("nan") for k in [
                        "ddh_baseline", "ddh_faulty", "ddh_comp",
                        "ddh_recovery_pct",
                        "energy_baseline_kWh", "energy_faulty_kWh",
                        "energy_comp_kWh", "energy_delta_pct",
                    ]},
                })

    df_out = pd.DataFrame(rows)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_out = SEV_PLOTS / "table_severity_compensation.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"\n  Saved CSV:  {csv_out}")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    tex_rename = {
        "fault_label":           "Fault",
        "severity_tag":          "Level",
        "severity_value":        "Sev.",
        "ddh_baseline":          r"DDH$_{\rm BL}$",
        "ddh_faulty":            r"DDH$_{\rm fault}$",
        "ddh_comp":              r"DDH$_{\rm comp}$",
        "ddh_recovery_pct":      r"Recovery [\%]",
        "energy_faulty_kWh":     r"E$_{\rm fault}$ [kWh]",
        "energy_comp_kWh":       r"E$_{\rm comp}$ [kWh]",
        "energy_delta_pct":      r"$\Delta$E [\%]",
    }
    df_tex = df_out[list(tex_rename.keys())].rename(columns=tex_rename)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Compensation KPIs across fault severity levels. "
                 r"DDH = degree-hours of discomfort [°C·h]; "
                 r"$\Delta$E = energy delta vs.\ baseline.}")
    lines.append(r"  \label{tab:severity_compensation}")
    lines.append(r"  \small")
    ncols = len(tex_rename)
    lines.append(r"  \begin{tabular}{" + "l" * 2 + "r" * (ncols - 2) + "}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(df_tex.columns) + r" \\")
    lines.append(r"    \midrule")
    for _, row in df_tex.iterrows():
        def fmt(v):
            if pd.isna(v): return "--"
            if isinstance(v, float): return f"{v:.2f}"
            return str(v)
        lines.append("    " + " & ".join(fmt(v) for v in row.values) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex_out = SEV_PLOTS / "table_severity_compensation.tex"
    tex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved LaTeX: {tex_out}")
    print("\nPhase 9b complete.")


if __name__ == "__main__":
    main()
