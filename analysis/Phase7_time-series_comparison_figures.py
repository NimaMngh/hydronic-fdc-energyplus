# -*- coding: utf-8 -*-
"""
Phase 7: Publication-Quality Time-Series Comparison Figures
============================================================
For each fault type, generates multi-panel figures overlaying the three
scenarios (baseline, faulty-no-comp, compensated) aligned to the fault
window.  Designed for direct inclusion in the journal paper.

Panels per figure:
  (a) Zone temperature vs setpoint
  (b) Mass flow rate
  (c) Supply / return water temperatures
  (d) Detection & compensation state (comp run only)

Also produces:
  - A combined 3×4 "mega-figure" for the appendix / supplementary
  - Individual SVG + PNG at journal resolution (300 DPI)

Reads:
  - plots/cross_fault_synthesis/synthesis_metadata.json   (from Phase 6)
  - runs/*/fdc_runtime_log.csv

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FAULT_KEYS = ["stuck_closed", "stuck_open", "supply_curve"]

FAULT_LABELS = {
    "stuck_closed": "Stuck-Closed Valve",
    "stuck_open":   "Stuck-Open Valve",
    "supply_curve": "Supply-Setpoint Bias",
}

SCENARIO_STYLES = {
    "baseline":       {"color": "#4CAF50", "lw": 1.3, "ls": "-",  "alpha": 0.85, "label": "Baseline"},
    "faulty_no_comp": {"color": "#F44336", "lw": 1.3, "ls": "-",  "alpha": 0.85, "label": "Faulty (no comp.)"},
    "compensated":    {"color": "#2196F3", "lw": 1.3, "ls": "-",  "alpha": 0.85, "label": "Compensated"},
}

RUN_CONFIG_FILES = {
    "baseline_detect":     "run_config_detect_baseline.json",
    "stuckclosed_detect":  "run_config_detect_stuckclosed.json",
    "stuckopen_detect":    "run_config_detect_stuckopen.json",
    "supplycurve_detect":  "run_config_detect_supplycurve.json",
    "stuckclosed_comp":    "run_config_comp_stuckclosed.json",
    "stuckopen_comp":      "run_config_comp_stuckopen.json",
    "supplycurve_comp":    "run_config_comp_supplycurve.json",
}

FAULT_TO_RUNKEYS = {
    "stuck_closed": {"faulty": "stuckclosed_detect", "comp": "stuckclosed_comp"},
    "stuck_open":   {"faulty": "stuckopen_detect",   "comp": "stuckopen_comp"},
    "supply_curve": {"faulty": "supplycurve_detect", "comp": "supplycurve_comp"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (reused from Phase 6 — self-contained so this script is independent)
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_run_dir(runs_root: Path, config_filename: str) -> Path:
    matches = []
    for cfg_path in runs_root.rglob(config_filename):
        rd = cfg_path.parent
        csv = rd / "fdc_runtime_log.csv"
        if csv.exists():
            matches.append((csv.stat().st_mtime, rd))
    if not matches:
        raise FileNotFoundError(f"No run dir with {config_filename} under {runs_root}")
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    years = df["month"].apply(lambda m: 2017 if int(m) >= 10 else 2018)
    date_str = (years.astype(str) + "-"
                + df["month"].astype(int).astype(str).str.zfill(2) + "-"
                + df["day"].astype(int).astype(str).str.zfill(2))
    base = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    offset = (pd.to_timedelta(df["hour"].astype(int), unit="h")
              + pd.to_timedelta(df["minute"].astype(int), unit="m"))
    df["datetime"] = base + offset
    return df


def load_runtime_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = add_datetime_column(df)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fault_window_from_config(run_dir: Path) -> Optional[Dict]:
    for p in sorted(run_dir.glob("run_config_*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        ft = cfg.get("fault_type", "none")
        fw = cfg.get("fault_window")
        if ft != "none" and fw is not None:
            return fw
    return None


def fault_window_to_interval(fw: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]),   int(fw["end_day"])
    sh, eh = int(fw["start_hour"]),  int(fw["end_hour"])
    sy = 2017 if sm >= 10 else 2018
    ey = 2017 if em >= 10 else 2018
    start = pd.Timestamp(year=sy, month=sm, day=sd, hour=sh)
    if eh == 24:
        end = pd.Timestamp(year=ey, month=em, day=ed) + pd.Timedelta(days=1)
    else:
        end = pd.Timestamp(year=ey, month=em, day=ed, hour=eh)
    return start, end


# ──────────────────────────────────────────────────────────────────────────────
# Window extraction with padding
# ──────────────────────────────────────────────────────────────────────────────

def extract_window(df: pd.DataFrame,
                   start: pd.Timestamp, end: pd.Timestamp,
                   pad_hours: float = 6.0) -> pd.DataFrame:
    """Return rows from *df* in [start − pad, end + pad]."""
    pad = pd.Timedelta(hours=pad_hours)
    mask = (df["datetime"] >= start - pad) & (df["datetime"] <= end + pad)
    return df.loc[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Per-fault figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_fault_comparison(fault_type: str,
                          windows: Dict[str, pd.DataFrame],
                          fault_start: pd.Timestamp,
                          fault_end: pd.Timestamp,
                          out_path: Path,
                          pad_hours: float = 6.0):
    """
    4-panel figure for one fault type:
      (a) zone_temp + setpoint
      (b) m_dot
      (c) t_inlet / t_outlet (+ t_supply if available)
      (d) detection flags & compensation state
    """
    has_comp = "compensated" in windows
    n_panels = 4

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 10),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    # Fault-window shading on all panels
    for ax in axes:
        ax.axvspan(fault_start, fault_end, color="#FFCDD2", alpha=0.35, zorder=0)

    # ──── Panel (a): Zone temperature ────
    ax = axes[0]
    sp_plotted = False
    for sc, sdf in windows.items():
        sty = SCENARIO_STYLES[sc]
        t = sdf["datetime"]
        ax.plot(t, sdf["zone_temp"], color=sty["color"], lw=sty["lw"],
                ls=sty["ls"], alpha=sty["alpha"], label=sty["label"])
        # Plot setpoint once (from baseline, it's the same schedule)
        if not sp_plotted:
            sp_col = "intended_sp" if "intended_sp" in sdf.columns else "htg_sp"
            ax.plot(t, sdf[sp_col], color="k", lw=1.0, ls="--", alpha=0.6,
                    label="Heating setpoint")
            sp_plotted = True

    ax.set_ylabel("Temperature [°C]", fontsize=10)
    ax.set_title(f"(a) Zone air temperature — {FAULT_LABELS[fault_type]}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1, borderaxespad=0.0)
    ax.grid(True, alpha=0.2)

    # ──── Panel (b): Mass flow rate ────
    ax = axes[1]
    for sc, sdf in windows.items():
        sty = SCENARIO_STYLES[sc]
        ax.plot(sdf["datetime"], sdf["m_dot"],
                color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                alpha=sty["alpha"], label=sty["label"])
    ax.set_ylabel("Mass flow [kg/s]", fontsize=10)
    ax.set_title("(b) Hydronic mass flow rate", fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1, borderaxespad=0.0)
    ax.grid(True, alpha=0.2)

    # ──── Panel (c): Supply / return temperatures ────
    ax = axes[2]
    for sc, sdf in windows.items():
        sty = SCENARIO_STYLES[sc]
        ax.plot(sdf["datetime"], sdf["t_inlet"],
                color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                alpha=sty["alpha"])
        ax.plot(sdf["datetime"], sdf["t_outlet"],
                color=sty["color"], lw=0.9, ls=":", alpha=sty["alpha"] * 0.8)
        # t_supply if available (supply-curve fault)
        if "t_supply" in sdf.columns:
            ax.plot(sdf["datetime"], sdf["t_supply"],
                    color=sty["color"], lw=1.0, ls="-.", alpha=sty["alpha"] * 0.7)

    # Custom legend for panel (c)
    legend_elements = []
    for sc in windows:
        sty = SCENARIO_STYLES[sc]
        legend_elements.append(Line2D([0], [0], color=sty["color"], lw=sty["lw"],
                                      ls="-", label=f"{sty['label']} (inlet)"))
        legend_elements.append(Line2D([0], [0], color=sty["color"], lw=0.9,
                                      ls=":", label=f"{sty['label']} (outlet)"))
    if any("t_supply" in sdf.columns for sdf in windows.values()):
        for sc, sdf in windows.items():
            if "t_supply" in sdf.columns:
                sty = SCENARIO_STYLES[sc]
                legend_elements.append(Line2D([0], [0], color=sty["color"], lw=1.0,
                                              ls="-.", label=f"{sty['label']} (supply)"))

    ax.set_ylabel("Temperature [°C]", fontsize=10)
    ax.set_title("(c) Water temperatures (inlet / outlet / supply)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1, borderaxespad=0.0)
    ax.grid(True, alpha=0.2)

    # ──── Panel (d): Detection & compensation state ────
    ax = axes[3]
    comp_df = windows.get("compensated")
    faulty_df = windows.get("faulty_no_comp")

    y_offset = 0.0
    bar_h = 0.3
    yticks = []
    yticklabels = []

    # Detection flags from faulty run (if present)
    if faulty_df is not None:
        for model_name, flag_col in [("OCSVM", "flag_ocsvm"),
                                      ("iForest", "flag_iforest"),
                                      ("LOF", "flag_lof")]:
            if flag_col in faulty_df.columns:
                t = faulty_df["datetime"].values
                flags = faulty_df[flag_col].astype(float).fillna(0).values
                _plot_state_bar(ax, t, flags, y_offset, bar_h,
                                color_on="#FF7043", color_off="#E0E0E0",
                                label_prefix=f"Det-{model_name}")
                yticks.append(y_offset + bar_h / 2)
                yticklabels.append(f"Det {model_name}\n(faulty)")
                y_offset += bar_h + 0.1

    # Detection + compensation from comp run
    if comp_df is not None:
        if "vote_triggered" in comp_df.columns:
            t = comp_df["datetime"].values
            vt = comp_df["vote_triggered"].astype(float).fillna(0).values
            _plot_state_bar(ax, t, vt, y_offset, bar_h,
                            color_on="#FF9800", color_off="#E0E0E0")
            yticks.append(y_offset + bar_h / 2)
            yticklabels.append("Vote triggered\n(comp)")
            y_offset += bar_h + 0.1

        if "comp_active" in comp_df.columns:
            t = comp_df["datetime"].values
            ca = comp_df["comp_active"].astype(float).fillna(0).values
            _plot_state_bar(ax, t, ca, y_offset, bar_h,
                            color_on="#2196F3", color_off="#E0E0E0")
            yticks.append(y_offset + bar_h / 2)
            yticklabels.append("Compensation\nactive")
            y_offset += bar_h + 0.1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_ylim(-0.1, y_offset + 0.1)
    ax.set_title("(d) Detection flags & compensation state",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(True, axis="x", alpha=0.2)

    # X-axis formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.suptitle(f"Fault Scenario Comparison: {FAULT_LABELS[fault_type]}",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.with_suffix('.png')}")


def _plot_state_bar(ax, t, values, y_bottom, height,
                    color_on="#FF7043", color_off="#E0E0E0",
                    label_prefix=""):
    """Draw a horizontal bar that is coloured where values > 0.5."""
    t_dt = pd.to_datetime(t)
    on_mask = values > 0.5
    # Draw background
    if len(t_dt) >= 2:
        ax.barh(y_bottom + height / 2, width=(t_dt[-1] - t_dt[0]),
                left=t_dt[0], height=height, color=color_off, alpha=0.3,
                edgecolor="none")
    # Draw ON blocks
    changes = np.diff(on_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if on_mask[0]:
        starts = np.concatenate([[0], starts])
    if on_mask[-1]:
        ends = np.concatenate([ends, [len(on_mask)]])
    for s, e in zip(starts, ends):
        left = t_dt[s]
        right = t_dt[min(e, len(t_dt) - 1)]
        width = right - left
        if width.total_seconds() > 0:
            ax.barh(y_bottom + height / 2, width=width, left=left,
                    height=height, color=color_on, alpha=0.85,
                    edgecolor="none")


# ──────────────────────────────────────────────────────────────────────────────
# Combined mega-figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_summary(all_windows: Dict[str, Dict[str, pd.DataFrame]],
                          all_intervals: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
                          out_path: Path):
    """
    3-column (one per fault) × 3-row (zone_temp, m_dot, water temps) summary.
    Lighter weight than the individual 4-panel figures; good for paper body.
    """
    n_faults = len(FAULT_KEYS)
    fig, axes = plt.subplots(3, n_faults, figsize=(5.5 * n_faults, 9),
                             sharex="col")

    for col_idx, ft in enumerate(FAULT_KEYS):
        fault_start, fault_end = all_intervals[ft]
        windows = all_windows[ft]

        for ax_row in range(3):
            ax = axes[ax_row, col_idx]
            ax.axvspan(fault_start, fault_end, color="#FFCDD2", alpha=0.3, zorder=0)

        # Row 0: zone_temp
        ax = axes[0, col_idx]
        sp_done = False
        for sc, sdf in windows.items():
            sty = SCENARIO_STYLES[sc]
            ax.plot(sdf["datetime"], sdf["zone_temp"],
                    color=sty["color"], lw=sty["lw"], alpha=sty["alpha"],
                    label=sty["label"])
            if not sp_done:
                sp_col = "intended_sp" if "intended_sp" in sdf.columns else "htg_sp"
                ax.plot(sdf["datetime"], sdf[sp_col],
                        color="k", lw=0.8, ls="--", alpha=0.5, label="Setpoint")
                sp_done = True
        ax.set_title(FAULT_LABELS[ft], fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Zone temp [°C]", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

        # Row 1: m_dot
        ax = axes[1, col_idx]
        for sc, sdf in windows.items():
            sty = SCENARIO_STYLES[sc]
            ax.plot(sdf["datetime"], sdf["m_dot"],
                    color=sty["color"], lw=sty["lw"], alpha=sty["alpha"])
        if col_idx == 0:
            ax.set_ylabel("Flow [kg/s]", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Row 2: t_inlet + t_outlet
        ax = axes[2, col_idx]
        for sc, sdf in windows.items():
            sty = SCENARIO_STYLES[sc]
            ax.plot(sdf["datetime"], sdf["t_inlet"],
                    color=sty["color"], lw=sty["lw"], alpha=sty["alpha"])
            ax.plot(sdf["datetime"], sdf["t_outlet"],
                    color=sty["color"], lw=0.8, ls=":", alpha=sty["alpha"] * 0.8)
        if col_idx == 0:
            ax.set_ylabel("Water temp [°C]", fontsize=9)
        ax.grid(True, alpha=0.2)

        # X-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

    fig.suptitle("Cross-Fault Comparison: Baseline vs Faulty vs Compensated",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined summary saved: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Compensation detail figure (supply temp command vs actual)
# ──────────────────────────────────────────────────────────────────────────────

def plot_compensation_detail(fault_type: str,
                             comp_window: pd.DataFrame,
                             fault_start: pd.Timestamp,
                             fault_end: pd.Timestamp,
                             out_path: Path):
    """
    2-panel detail of the compensation run:
      (a) supply_temp_cmd vs t_inlet vs t_supply (if available)
      (b) diagnosed_fault + comp_active timeline
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                             gridspec_kw={"hspace": 0.15, "height_ratios": [2, 1]})

    for ax in axes:
        ax.axvspan(fault_start, fault_end, color="#FFCDD2", alpha=0.3, zorder=0)

    t = comp_window["datetime"]

    # Panel (a)
    ax = axes[0]
    if "supply_temp_cmd" in comp_window.columns:
        ax.plot(t, comp_window["supply_temp_cmd"], color="#E91E63", lw=1.5,
                label="Supply temp command", zorder=3)
    ax.plot(t, comp_window["t_inlet"], color="#1565C0", lw=1.2,
            label="t_inlet (measured)", alpha=0.8)
    if "t_supply" in comp_window.columns:
        ax.plot(t, comp_window["t_supply"], color="#FF8F00", lw=1.0, ls="-.",
                label="t_supply (node)", alpha=0.7)
    ax.set_ylabel("Temperature [°C]", fontsize=10)
    ax.set_title(f"Compensation Detail — {FAULT_LABELS[fault_type]}",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.2)

    # Panel (b)
    ax = axes[1]
    y_off = 0.0
    bh = 0.35
    yticks, yticklabs = [], []

    if "comp_active" in comp_window.columns:
        _plot_state_bar(ax, t.values, comp_window["comp_active"].astype(float).fillna(0).values,
                        y_off, bh, color_on="#2196F3")
        yticks.append(y_off + bh / 2)
        yticklabs.append("Comp. active")
        y_off += bh + 0.1

    if "diagnosed_fault" in comp_window.columns:
        # Show as text annotations at transitions
        diag = comp_window["diagnosed_fault"].fillna("none").astype(str)
        changes = diag != diag.shift()
        for idx in comp_window.index[changes]:
            val = diag.loc[idx]
            if val != "none" and val.lower() != "nan":
                ax.annotate(val.replace("_", " "),
                            (comp_window.loc[idx, "datetime"], y_off + bh / 2),
                            fontsize=7, ha="left", va="center",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="#FFF9C4", ec="#F9A825", lw=0.5))
        yticks.append(y_off + bh / 2)
        yticklabs.append("Diagnosis")
        y_off += bh + 0.1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs, fontsize=8)
    ax.set_ylim(-0.1, y_off + 0.1)
    ax.grid(True, axis="x", alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Comp. detail saved: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Publication-quality comparison figures."
    )
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="plots/paper_figures")
    parser.add_argument("--pad-hours", type=float, default=6.0,
                        help="Hours of padding before/after fault window (default: 6).")
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent
    runs_root   = (project_dir / args.runs_dir).resolve()
    out_dir     = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 7: PUBLICATION-QUALITY FIGURES")
    print(f"  Project dir : {project_dir}")
    print(f"  Runs root   : {runs_root}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 78)

    # ── Locate runs ──
    run_dirs: Dict[str, Path] = {}
    for key, cfg_name in RUN_CONFIG_FILES.items():
        run_dirs[key] = find_latest_run_dir(runs_root, cfg_name)
        print(f"  {key:22s} -> {run_dirs[key].name}")

    # ── Load logs ──
    logs: Dict[str, pd.DataFrame] = {}
    for key, rd in run_dirs.items():
        logs[key] = load_runtime_log(rd / "fdc_runtime_log.csv")
        print(f"  Loaded {key:22s}: {len(logs[key]):>7,} rows")

    # ── Derive fault intervals ──
    fault_intervals: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for ft in FAULT_KEYS:
        faulty_key = FAULT_TO_RUNKEYS[ft]["faulty"]
        fw = fault_window_from_config(run_dirs[faulty_key])
        if fw is None:
            raise RuntimeError(f"No fault_window in config for {ft}")
        fault_intervals[ft] = fault_window_to_interval(fw)
        s, e = fault_intervals[ft]
        print(f"  Fault window [{ft}]: {s} -> {e}  ({(e-s).total_seconds()/3600:.1f} h)")

    # ── Build windowed DataFrames ──
    baseline_key = "baseline_detect"
    all_windows: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ft in FAULT_KEYS:
        faulty_key = FAULT_TO_RUNKEYS[ft]["faulty"]
        comp_key   = FAULT_TO_RUNKEYS[ft]["comp"]
        fstart, fend = fault_intervals[ft]

        windows = {}
        for sc, rk in [("baseline", baseline_key),
                        ("faulty_no_comp", faulty_key),
                        ("compensated", comp_key)]:
            windows[sc] = extract_window(logs[rk], fstart, fend,
                                         pad_hours=args.pad_hours)
            print(f"    {ft} / {sc:16s}: {len(windows[sc]):,} rows in window")

        all_windows[ft] = windows

    # ── Generate figures ──
    print("\nGenerating figures...")

    # Individual 4-panel per fault type
    for ft in FAULT_KEYS:
        fstart, fend = fault_intervals[ft]
        fname = f"comparison_{ft.replace(' ', '_')}"
        plot_fault_comparison(ft, all_windows[ft], fstart, fend,
                              out_dir / fname, pad_hours=args.pad_hours)

    # Compensation detail per fault type
    for ft in FAULT_KEYS:
        fstart, fend = fault_intervals[ft]
        comp_win = all_windows[ft]["compensated"]
        fname = f"comp_detail_{ft.replace(' ', '_')}"
        plot_compensation_detail(ft, comp_win, fstart, fend, out_dir / fname)

    # Combined summary
    plot_combined_summary(all_windows, fault_intervals,
                          out_dir / "combined_cross_fault_summary")

    # ── Done ──
    print("\n" + "=" * 78)
    print("PHASE 7 COMPLETE")
    print(f"  All figures saved to: {out_dir}")
    n_files = len(list(out_dir.glob("*.png"))) + len(list(out_dir.glob("*.svg")))
    print(f"  Total files generated: {n_files}")
    print("=" * 78)


if __name__ == "__main__":
    main()
