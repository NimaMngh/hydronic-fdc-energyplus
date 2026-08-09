#!/usr/bin/env python3
"""
Phase10_sweep_pareto.py — compensation-target and ramp-gain sweeps.

Computes the KPI bundle for every *compensated* run it finds (reference runs,
the compensation-target sweep, the ramp-gain sweep), joins the baseline and
faulty references, and produces:

    sweep_kpis.csv              one row per comp run (KPIs + J_D, J_E, J_R)
    pareto_<fault>.png          recovery vs energy trade-off per fault type
    pareto_all_faults.png       combined overview

KPI definitions replicate analysis/Phase6_cross_fault_synthesis.py exactly:
  * DDH over occupied fault-window steps, |zone_temp - intended_sp| * dt_h
  * energy = sum(m_dot * cp * max(dT,0)) * dt_s / 3.6e6 over heating-active steps
  * return temperature = flow-weighted mean of t_outlet over heating-active steps
  * fault-window intervals from run_config fault_window (per-day intervals for
    intra-day windows; single continuous interval for 0-24 h windows;
    year convention: month >= 10 -> 2017 else 2018)

Objective indices:
  J_D = (DDH_fdc - DDH_base) / (DDH_fault - DDH_base)   residual discomfort (0 = full recovery)
  recovery_pct = (1 - J_D) * 100                        identical to Phase6 recovery_percent
  J_E = E_fdc / E_base                                  energy ratio vs fault-free baseline
  J_R = Tret_fdc - Tret_base                            return-temperature shift (K)

Usage (from the project root that contains runs/ and table_ablation_kpis.csv):
    python analysis/Phase10_sweep_pareto.py
    python analysis/Phase10_sweep_pareto.py --runs-root runs \
        --refs table_ablation_kpis.csv --out sweep_results

Author : Nima Monghasemi
Date   : August 2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CP_WATER_J_PER_KG_K = 4180.0
DEFAULT_TIMESTEP_MIN = 10

FAULT_LABELS = {
    "stuck_closed": "Stuck-closed valve",
    "stuck_open": "Stuck-open valve",
    "supply_curve": "Supply-setpoint bias",
}

# ────────────────────────────────────────────────────────────────────────────
# Run discovery and log loading
# ────────────────────────────────────────────────────────────────────────────

def find_comp_runs(runs_root: Path) -> List[Tuple[Path, Dict]]:
    """Return (run_dir, config) for every run dir containing a run_config with
    compensation_enabled true and a real fault_type."""
    found = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for cfg_path in sorted(run_dir.glob("run_config_*.json")):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if cfg.get("compensation_enabled") and cfg.get("fault_type", "none") != "none":
                found.append((run_dir, cfg))
                break
    return found


def load_log(run_dir: Path) -> Optional[pd.DataFrame]:
    log_path = run_dir / "fdc_runtime_log.csv"
    if not log_path.exists():
        cands = list(run_dir.glob("*runtime_log*.csv"))
        if not cands:
            return None
        log_path = cands[0]
    df = pd.read_csv(log_path)
    if not {"month", "day", "hour", "minute"} <= set(df.columns):
        return None
    year = np.where(df["month"].astype(int) >= 10, 2017, 2018)
    base = pd.to_datetime(
        {"year": year, "month": df["month"].astype(int), "day": df["day"].astype(int)}
    )
    df["datetime"] = base + pd.to_timedelta(df["hour"].astype(int), unit="h") \
                          + pd.to_timedelta(df["minute"].astype(int), unit="m")
    return df

# ────────────────────────────────────────────────────────────────────────────
# Fault-window handling (identical semantics to Phase6)
# ────────────────────────────────────────────────────────────────────────────

def fault_window_to_intervals(fw: Dict) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]), int(fw["end_day"])
    sh, eh = int(fw["start_hour"]), int(fw["end_hour"])
    sy = 2017 if sm >= 10 else 2018
    ey = 2017 if em >= 10 else 2018

    if sh == 0 and eh == 24:
        start_dt = pd.Timestamp(year=sy, month=sm, day=sd, hour=0)
        end_dt = pd.Timestamp(year=ey, month=em, day=ed) + pd.Timedelta(days=1)
        return [(start_dt, end_dt)]

    intervals = []
    current = pd.Timestamp(year=sy, month=sm, day=sd)
    final_day = pd.Timestamp(year=ey, month=em, day=ed)
    while current <= final_day:
        day_start = current + pd.Timedelta(hours=sh)
        day_end = current + pd.Timedelta(days=1) if eh == 24 else current + pd.Timedelta(hours=eh)
        intervals.append((day_start, day_end))
        current += pd.Timedelta(days=1)
    return intervals


def mask_from_intervals(df: pd.DataFrame,
                        intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for start, end in intervals:
        mask |= (df["datetime"] >= start) & (df["datetime"] < end)
    return mask

# ────────────────────────────────────────────────────────────────────────────
# KPI computation (identical math to Phase6 compute_kpis, relevant subset)
# ────────────────────────────────────────────────────────────────────────────

def flow_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float)
    w = weights.astype(float).clip(lower=0.0)
    ok = v.notna() & w.notna() & (w > 0)
    if ok.sum() == 0:
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def compute_kpis(dfw: pd.DataFrame, timestep_min: int = DEFAULT_TIMESTEP_MIN) -> Dict[str, float]:
    dt_hours = timestep_min / 60.0
    dt_seconds = timestep_min * 60.0
    cols = set(dfw.columns)
    out: Dict[str, float] = {"n_rows": int(len(dfw))}

    occ = dfw["occupied"] == 1 if "occupied" in cols else pd.Series(True, index=dfw.index)
    htg = dfw["heating_active"] == 1 if "heating_active" in cols else pd.Series(True, index=dfw.index)

    if "temp_error" in cols:
        te = dfw.loc[occ, "temp_error"].astype(float)
    elif {"zone_temp", "intended_sp"} <= cols:
        te = dfw.loc[occ, "zone_temp"].astype(float) - dfw.loc[occ, "intended_sp"].astype(float)
    elif {"zone_temp", "htg_sp"} <= cols:
        te = dfw.loc[occ, "zone_temp"].astype(float) - dfw.loc[occ, "htg_sp"].astype(float)
    else:
        te = pd.Series(dtype=float)

    out["n_occ"] = int(occ.sum())
    out["ddh_abs_C_h"] = float(te.abs().sum() * dt_hours) if len(te) else float("nan")

    if {"m_dot", "delta_T_hw"} <= cols:
        mdot = dfw.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT = dfw.loc[htg, "delta_T_hw"].astype(float).fillna(0.0)
    elif {"m_dot", "t_inlet", "t_outlet"} <= cols:
        mdot = dfw.loc[htg, "m_dot"].astype(float).clip(lower=0.0).fillna(0.0)
        dT = (dfw.loc[htg, "t_inlet"].astype(float) - dfw.loc[htg, "t_outlet"].astype(float)).fillna(0.0)
    else:
        mdot, dT = pd.Series(dtype=float), pd.Series(dtype=float)

    if len(mdot):
        q_dot_W = mdot * CP_WATER_J_PER_KG_K * np.maximum(dT, 0.0)
        out["dh_energy_kWh_est"] = float((q_dot_W * dt_seconds).sum() / 3.6e6)
    else:
        out["dh_energy_kWh_est"] = float("nan")

    out["zone_temp_mean_C"] = (float(dfw.loc[occ, "zone_temp"].astype(float).mean())
                               if "zone_temp" in cols and occ.sum() else float("nan"))
    out["return_temp_flow_wtd_C"] = (flow_weighted_mean(dfw.loc[htg, "t_outlet"], dfw.loc[htg, "m_dot"])
                                     if {"t_outlet", "m_dot"} <= cols else float("nan"))

    # compensation activity
    if "comp_active" in cols:
        ca = dfw["comp_active"].astype(float).fillna(0).astype(int)
        out["comp_active_rate"] = float((ca == 1).mean()) if len(ca) else float("nan")
        rising = int(((ca.diff() == 1)).sum() + (1 if len(ca) and ca.iloc[0] == 1 else 0))
        out["n_comp_cycles"] = rising
        first_idx = ca[ca == 1].index.min() if (ca == 1).any() else None
        if first_idx is not None:
            t_first = dfw.loc[first_idx, "datetime"]
            t0 = dfw["datetime"].min()
            out["first_comp_min_after_window_start"] = float((t_first - t0).total_seconds() / 60.0)
        else:
            out["first_comp_min_after_window_start"] = float("nan")
    else:
        out["comp_active_rate"] = float("nan")
        out["n_comp_cycles"] = -1
        out["first_comp_min_after_window_start"] = float("nan")

    if "supply_temp_cmd" in cols:
        out["supply_temp_cmd_mean_C"] = float(dfw["supply_temp_cmd"].astype(float).mean())
    return out

# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs", type=Path)
    ap.add_argument("--refs", default="table_ablation_kpis.csv", type=Path,
                    help="Reference KPI table with baseline / faulty_no_comp rows per fault")
    ap.add_argument("--out", default="sweep_results", type=Path)
    ap.add_argument("--timestep-min", default=DEFAULT_TIMESTEP_MIN, type=int)
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)

    refs = pd.read_csv(args.refs)
    ref_lookup: Dict[Tuple[str, str], Dict] = {}
    for _, r in refs.iterrows():
        ref_lookup[(r["fault_type"], r["scenario"])] = r.to_dict()

    rows = []
    for run_dir, cfg in find_comp_runs(args.runs_root):
        fault = cfg["fault_type"]
        fw = cfg.get("fault_window")
        if fw is None:
            print(f"skip {run_dir.name}: no fault_window")
            continue
        df = load_log(run_dir)
        if df is None:
            print(f"skip {run_dir.name}: no runtime log")
            continue
        mask = mask_from_intervals(df, fault_window_to_intervals(fw))
        k = compute_kpis(df.loc[mask].copy(), args.timestep_min)

        base = ref_lookup.get((fault, "baseline"))
        faulty = ref_lookup.get((fault, "faulty_no_comp"))
        if base is None or faulty is None:
            print(f"warn {run_dir.name}: missing references for {fault}")
            continue

        ddh_b, ddh_f, ddh_c = base["ddh_abs_C_h"], faulty["ddh_abs_C_h"], k["ddh_abs_C_h"]
        e_b, e_f, e_c = base["dh_energy_kWh_est"], faulty["dh_energy_kWh_est"], k["dh_energy_kWh_est"]
        tr_b, tr_c = base["return_temp_flow_wtd_C"], k["return_temp_flow_wtd_C"]

        denom = ddh_f - ddh_b
        J_D = (ddh_c - ddh_b) / denom if denom else float("nan")
        rows.append({
            "run_dir": str(run_dir),
            "run_name": cfg.get("run_name", run_dir.name),
            "fault_type": fault,
            "target_C": cfg["compensation_config"]["targets"][fault],
            "alpha": cfg["compensation_config"].get("ramp_factor", np.nan),
            "ddh_fdc_C_h": ddh_c,
            "ddh_baseline_C_h": ddh_b,
            "ddh_faulty_C_h": ddh_f,
            "J_D_residual_discomfort": J_D,
            "recovery_pct": (1.0 - J_D) * 100.0 if np.isfinite(J_D) else float("nan"),
            "energy_fdc_kWh": e_c,
            "J_E_vs_baseline": e_c / e_b if e_b else float("nan"),
            "energy_delta_vs_baseline_pct": (e_c - e_b) / e_b * 100.0 if e_b else float("nan"),
            "energy_delta_vs_faulty_pct": (e_c - e_f) / e_f * 100.0 if e_f else float("nan"),
            "tret_fdc_C": tr_c,
            "J_R_return_shift_K": tr_c - tr_b if np.isfinite(tr_c) else float("nan"),
            "zone_temp_mean_C": k["zone_temp_mean_C"],
            "comp_active_rate": k["comp_active_rate"],
            "n_comp_cycles": k["n_comp_cycles"],
            "first_comp_min_after_window_start": k["first_comp_min_after_window_start"],
        })
        print(f"processed {run_dir.name}: {fault} target={rows[-1]['target_C']:g} "
              f"alpha={rows[-1]['alpha']:g} recovery={rows[-1]['recovery_pct']:.1f}%")

    if not rows:
        print("No compensated runs found under", args.runs_root)
        return

    out_df = pd.DataFrame(rows).sort_values(["fault_type", "target_C", "alpha"])
    csv_path = args.out / "sweep_kpis.csv"
    out_df.to_csv(csv_path, index=False)
    print("\nwrote", csv_path)

    # ── Pareto plots ────────────────────────────────────────────────────────
    def pareto_front(sub: pd.DataFrame) -> pd.Series:
        """Non-dominated: maximize recovery, minimize energy_delta_vs_baseline."""
        eff = pd.Series(True, index=sub.index)
        for i in sub.index:
            for j in sub.index:
                if i == j:
                    continue
                better_or_eq = (sub.loc[j, "recovery_pct"] >= sub.loc[i, "recovery_pct"] and
                                sub.loc[j, "energy_delta_vs_baseline_pct"] <= sub.loc[i, "energy_delta_vs_baseline_pct"])
                strictly = (sub.loc[j, "recovery_pct"] > sub.loc[i, "recovery_pct"] or
                            sub.loc[j, "energy_delta_vs_baseline_pct"] < sub.loc[i, "energy_delta_vs_baseline_pct"])
                if better_or_eq and strictly:
                    eff.loc[i] = False
                    break
        return eff

    for fault, sub in out_df.groupby("fault_type"):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sc = ax.scatter(sub["recovery_pct"], sub["energy_delta_vs_baseline_pct"],
                        c=sub["J_R_return_shift_K"], cmap="coolwarm", s=90,
                        edgecolors="k", zorder=3)
        eff = pareto_front(sub)
        front = sub[eff].sort_values("recovery_pct")
        ax.plot(front["recovery_pct"], front["energy_delta_vs_baseline_pct"],
                "--", color="gray", lw=1.2, zorder=2, label="Pareto front")
        for _, r in sub.iterrows():
            lbl = f"T={r['target_C']:g}"
            if np.isfinite(r["alpha"]) and abs(r["alpha"] - 0.2) > 1e-9:
                lbl += f", $\\alpha$={r['alpha']:g}"
            ax.annotate(lbl, (r["recovery_pct"], r["energy_delta_vs_baseline_pct"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Return-temperature shift vs baseline $J_R$ [K]")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("DDH recovery [%]")
        ax.set_ylabel("Energy vs fault-free baseline [%]")
        ax.set_title(f"{FAULT_LABELS.get(fault, fault)}: comfort-energy trade-off")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = args.out / f"pareto_{fault}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print("wrote", fig_path)

    # combined overview
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    markers = {"stuck_closed": "o", "stuck_open": "s", "supply_curve": "^"}
    for fault, sub in out_df.groupby("fault_type"):
        ax.scatter(sub["recovery_pct"], sub["energy_delta_vs_baseline_pct"],
                   marker=markers.get(fault, "o"), s=80, edgecolors="k",
                   label=FAULT_LABELS.get(fault, fault))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("DDH recovery [%]")
    ax.set_ylabel("Energy vs fault-free baseline [%]")
    ax.set_title("Compensation-target and ramp-gain sweeps: all compensated runs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "pareto_all_faults.png", dpi=200)
    plt.close(fig)
    print("wrote", args.out / "pareto_all_faults.png")


if __name__ == "__main__":
    main()
