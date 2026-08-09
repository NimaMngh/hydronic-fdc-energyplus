# -*- coding: utf-8 -*-
"""
Phase 8: Journal-Ready Figure Refinements & Results Narrative
==============================================================
Reads Phase 6 & 7 outputs and produces:

1) Refined versions of the key figures with proper formatting
   (fixes x-axis label overlap, consistent fonts, proper sizing)
2) Auto-generated results narrative (text file) summarizing key findings
3) Detection comparison bar chart
4) Refined compensability spectrum with annotations

Author : Nima Monghasemi
Date   : 2026-03-03
Revised: 2026-03 — figure sizing corrected for Energy and Buildings
         at 0.85\textwidth (5.4 in display width). All figsize values
         now match the rendered width so fonts appear at true size.
Revised: 2026-04 — added --kappa flag for MDU dissertation template
         (B5-like stock 169×239 mm, textwidth = 113 mm = 4.45 in).
         3-panel fault figures (FIG16A/B/E) use kappa-optimised sizing
         so fonts render at 1:1 when included with width=\textwidth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from textwrap import dedent

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FAULT_KEYS = ["stuck_closed", "stuck_open", "supply_curve"]

FAULT_LABELS = {
    "stuck_closed": "Stuck-Closed\nValve",
    "stuck_open":   "Stuck-Open\nValve",
    "supply_curve": "Supply-Setpoint\nBias",
}

FAULT_LABELS_INLINE = {
    "stuck_closed": "Stuck-Closed Valve",
    "stuck_open":   "Stuck-Open Valve",
    "supply_curve": "Supply-Setpoint Bias",
}

# ── Display widths ──
# Energy & Buildings textwidth ≈ 6.38 in (single column).
# Figures included at 0.85\textwidth → 5.42 in display width.
# figsize width is set to match so fonts render at 1:1.
FIG_W_085 = 5.4   # for \includegraphics[width=0.85\textwidth]  (journal)
FIG_W_100 = 6.4   # for \includegraphics[width=\textwidth]      (journal)

# ── MDU Kappa (B5-like: 169×239 mm stock) ──
# Page geometry:
#   \setstocksize{239mm}{169mm}
#   \usepackage[left=25mm,right=25mm,top=20mm,bottom=20mm,bindingoffset=6mm]{geometry}
# textwidth = 169 − 25 − 25 − 6 = 113 mm = 4.449 in
# textheight = 239 − 20 − 20     = 199 mm = 7.835 in
# Figures included at width=\textwidth → 4.45 in display width.
FIG_W_KAPPA = 4.45   # for \includegraphics[width=\textwidth]   (kappa)

# Journal-friendly matplotlib defaults (render at true size)
JOURNAL_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.0,
}

SCENARIO_COLORS = {
    "baseline":       "#2E7D32",
    "faulty_no_comp": "#C62828",
    "compensated":    "#1565C0",
}

SCENARIO_LABELS = {
    "baseline":       "Baseline",
    "faulty_no_comp": "Faulty (no comp.)",
    "compensated":    "Compensated",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (self-contained)
# ──────────────────────────────────────────────────────────────────────────────

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


def find_latest_run_dir(runs_root: Path, config_filename: str) -> Path:
    matches = []
    for cfg in runs_root.rglob(config_filename):
        rd = cfg.parent
        csv = rd / "fdc_runtime_log.csv"
        if csv.exists():
            matches.append((csv.stat().st_mtime, rd))
    if not matches:
        raise FileNotFoundError(f"No run with {config_filename} under {runs_root}")
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
        if cfg.get("fault_type", "none") != "none" and cfg.get("fault_window"):
            return cfg["fault_window"]
    return None


def fault_window_to_interval(fw: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    sm, sd = int(fw["start_month"]), int(fw["start_day"])
    em, ed = int(fw["end_month"]),   int(fw["end_day"])
    sh, eh = int(fw["start_hour"]),  int(fw["end_hour"])
    sy = 2017 if sm >= 10 else 2018
    ey = 2017 if em >= 10 else 2018
    start = pd.Timestamp(year=sy, month=sm, day=sd, hour=sh)
    end = (pd.Timestamp(year=ey, month=em, day=ed) + pd.Timedelta(days=1)
           if eh == 24 else pd.Timestamp(year=ey, month=em, day=ed, hour=eh))
    return start, end


def extract_window(df, start, end, pad_hours=6.0):
    pad = pd.Timedelta(hours=pad_hours)
    mask = (df["datetime"] >= start - pad) & (df["datetime"] <= end + pad)
    return df.loc[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Refined Compensability Spectrum
# ──────────────────────────────────────────────────────────────────────────────

def fig_compensability_spectrum(recovery_df: pd.DataFrame, out_path: Path):
    """Publication-quality compensability spectrum with quadrant annotations."""
    with plt.rc_context(JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(FIG_W_085, 3.8))

        x = recovery_df["energy_delta_vs_baseline_pct"].values
        y = recovery_df["ddh_recovery_pct"].values
        faults = recovery_df["fault_type"].values

        markers = ["s", "D", "o"]
        colours = ["#1565C0", "#2E7D32", "#E65100"]

        # Per-point annotation offsets: (x_offset, y_offset) in points
        offsets = {
            "stuck_closed": (-12, -16),
            "stuck_open":   (10, -18),
            "supply_curve": (-14, 12),
        }

        for i, (xi, yi, ft) in enumerate(zip(x, y, faults)):
            ax.scatter(xi, yi, s=100, c=colours[i], marker=markers[i],
                       zorder=5, edgecolors="k", linewidths=0.6,
                       label=FAULT_LABELS_INLINE[ft])
            ox, oy = offsets.get(ft, (10, 10))
            ax.annotate(
                f"{yi:.0f}%",
                (xi, yi),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=9, fontweight="bold",
                color=colours[i],
                arrowprops=dict(arrowstyle="-", color=colours[i], lw=0.7),
            )

        # Reference lines
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(100, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.5)

        # Quadrant labels
        quadrant_style = dict(fontsize=7.5, color="#555555",
                              fontweight="bold", style="italic")
        ax.text(27, 115, "Full recovery,\nenergy above baseline",
                ha="right", **quadrant_style)
        ax.text(-12, 115, "Full recovery,\nenergy below baseline",
                ha="left", **quadrant_style)
        ax.text(27, -15, "No recovery,\nenergy above baseline",
                ha="right", **quadrant_style)
        ax.text(-12, -15, "No recovery,\nenergy below baseline",
                ha="left", **quadrant_style)

        ax.set_xlabel("Energy change vs. baseline [%]")
        ax.set_ylabel("Discomfort recovery [%]")

        # Legend: outside the plot area, horizontal, below the figure
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=3, fontsize=7.5, framealpha=0.9,
                  handletextpad=0.6, columnspacing=1.5,
                  markerscale=1.0, edgecolor="#CCCCCC")

        ax.set_xlim(-15, 30)
        ax.set_ylim(-25, 130)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)

        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".svg"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Spectrum: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Detection Performance Comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig_detection_comparison(det_df: pd.DataFrame, out_path: Path):
    """Grouped bar chart of precision, recall, F1, FPR per fault type."""
    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_100, 3.5),
                                 gridspec_kw={"wspace": 0.32})

        faults = det_df["fault_type"].tolist()
        x = np.arange(len(faults))
        w = 0.22

        # Left panel: Prec / Rec / F1
        ax = axes[0]
        for i, (metric, lbl, col) in enumerate([
            ("precision", "Precision", "#1565C0"),
            ("recall",    "Recall",    "#2E7D32"),
            ("f1",        "F1-score",  "#E65100"),
        ]):
            vals = det_df[metric].values
            bars = ax.bar(x + i * w, vals, w, label=lbl, color=col,
                          edgecolor="k", linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + w)
        ax.set_xticklabels([FAULT_LABELS_INLINE[ft] for ft in faults],
                           fontsize=7.5, color="black")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.35)
        ax.legend(loc="upper right", fontsize=7)

        ax.text(0.5, -0.18, "(a)", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="black")

        # Right panel: FPR + latency (dual axis)
        ax = axes[1]
        ax2 = ax.twinx()

        fpr_vals = det_df["fpr"].values * 100
        lat_vals = det_df["latency_minutes"].values

        b1 = ax.bar(x - 0.15, fpr_vals, 0.3, label="FPR (%)",
                     color="#C62828", edgecolor="k", linewidth=0.4, alpha=0.85)
        b2 = ax2.bar(x + 0.15, lat_vals, 0.3, label="Latency (min)",
                      color="#FFA726", edgecolor="k", linewidth=0.4, alpha=0.85)

        for bar, v in zip(b1, fpr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=7,
                    color="black")
        for bar, v in zip(b2, lat_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                     color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([FAULT_LABELS_INLINE[ft] for ft in faults],
                           fontsize=7.5, color="black")
        ax.set_ylabel("False positive rate (%)")
        ax2.set_ylabel("Detection latency (min)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

        ax.text(0.5, -0.18, "(b)", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="black")

        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".svg"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Detection comparison: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Refined per-fault 3-panel (zone temp + flow + state bar)
# ──────────────────────────────────────────────────────────────────────────────

def fig_fault_refined(fault_type: str,
                      windows: Dict[str, pd.DataFrame],
                      fault_start: pd.Timestamp,
                      fault_end: pd.Timestamp,
                      out_path: Path,
                      kappa: bool = False):
    """Cleaner 3-panel figure with fixed formatting.

    Parameters
    ----------
    kappa : bool
        If True, size the figure for the MDU kappa template
        (B5-like, textwidth = 113 mm = 4.45 in, included at
        width=\\textwidth).  If False (default), size for
        Energy and Buildings at 0.85\\textwidth (5.4 in).
    """
    if kappa:
        # MDU kappa: 169×239 mm stock, textwidth = 113 mm = 4.45 in
        # Figures included at width=\textwidth → 4.45 in display
        # Height 5.0 in → panels (a) 2.68 in, (b) 1.79 in, (c) 0.54 in
        # Figure + caption ≈ 74 % of textheight — fits well on page
        fig_w = FIG_W_KAPPA   # 4.45 in
        fig_h = 5.0           # in
    else:
        # Energy & Buildings: textwidth ≈ 6.38 in, 0.85\tw = 5.42 in
        fig_w = FIG_W_085     # 5.4 in
        fig_h = 4.6           # in

    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(
            3, 1, figsize=(fig_w, fig_h), sharex=True,
            gridspec_kw={"hspace": 0.10, "height_ratios": [3, 2, 0.6]}
        )

        # Legend labels
        legend_labels = {
            "baseline":       "Baseline",
            "faulty_no_comp": "Faulty",
            "compensated":    "Compensated",
        }

        for ax in axes:
            ax.axvspan(fault_start, fault_end, color="#FFCDD2", alpha=0.25, zorder=0)

        # ── Panel (a): Zone temperature ──
        ax = axes[0]
        sp_done = False
        for sc in ["baseline", "faulty_no_comp", "compensated"]:
            if sc not in windows:
                continue
            sdf = windows[sc]
            ax.plot(sdf["datetime"], sdf["zone_temp"],
                    color=SCENARIO_COLORS[sc], label=legend_labels[sc])
            if not sp_done:
                sp_col = "intended_sp" if "intended_sp" in sdf.columns else "htg_sp"
                ax.plot(sdf["datetime"], sdf[sp_col],
                        color="k", lw=0.8, ls="--", alpha=0.5, label="Setpoint")
                sp_done = True
        ax.set_ylabel("Zone temperature (\u00b0C)")
        ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, borderaxespad=0.0)

        ax.text(0.01, 0.95, "(a)", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="black",
                fontweight="normal")

        # ── Panel (b): Mass flow ──
        ax = axes[1]
        for sc in ["baseline", "faulty_no_comp", "compensated"]:
            if sc not in windows:
                continue
            sdf = windows[sc]
            ax.plot(sdf["datetime"], sdf["m_dot"],
                    color=SCENARIO_COLORS[sc], label=legend_labels[sc])

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.35)

        ax.set_ylabel("Mass flow rate (kg/s)")
        ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, borderaxespad=0.0)

        ax.text(0.01, 0.95, "(b)", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="black",
                fontweight="normal")

        # ── Panel (c): State bar ──
        ax = axes[2]
        comp_df = windows.get("compensated")
        y_off = 0.0
        bh = 0.4
        yticks, ylabels = [], []

        if comp_df is not None and "comp_active" in comp_df.columns:
            t = comp_df["datetime"].values
            ca = comp_df["comp_active"].astype(float).fillna(0).values
            _draw_state_bar(ax, t, ca, y_off, bh, "#1565C0")
            yticks.append(y_off + bh / 2)
            ylabels.append("Compensation\nstatus")

        ax.axvspan(fault_start, fault_end, ymin=0, ymax=1,
                   color="#FFCDD2", alpha=0.25, zorder=0)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7.5, color="black")
        ax.set_ylim(-0.05, bh + 0.05)
        ax.tick_params(axis="y", labelcolor="black", length=0)
        ax.tick_params(axis="x", labelcolor="black")

        ax.text(0.01, 0.85, "(c)", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="black",
                fontweight="normal")

        # ── X-axis formatting ──
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".svg"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Fault figure [{fault_type}]: {out_path.with_suffix('.png')}")


def _draw_state_bar(ax, t, values, y_bottom, height, color_on, color_off="#E0E0E0"):
    """Draw horizontal on/off bar."""
    t_dt = pd.to_datetime(t)
    on = values > 0.5
    if len(t_dt) >= 2:
        ax.barh(y_bottom + height / 2, width=(t_dt[-1] - t_dt[0]),
                left=t_dt[0], height=height, color=color_off, alpha=0.25,
                edgecolor="none")
    changes = np.diff(on.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if on[0]:
        starts = np.concatenate([[0], starts])
    if on[-1]:
        ends = np.concatenate([ends, [len(on)]])
    for s, e in zip(starts, ends):
        left = t_dt[s]
        right = t_dt[min(e, len(t_dt) - 1)]
        w = right - left
        if w.total_seconds() > 0:
            ax.barh(y_bottom + height / 2, width=w, left=left,
                    height=height, color=color_on, alpha=0.8, edgecolor="none")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Refined combined cross-fault summary (fix x-axis)
# ──────────────────────────────────────────────────────────────────────────────

def fig_combined_cross_fault(all_windows, all_intervals, out_path):
    """3×2 grid: (zone_temp, m_dot) × 3 faults. Clean x-axis."""
    with plt.rc_context(JOURNAL_RC):
        n = len(FAULT_KEYS)
        fig, axes = plt.subplots(2, n, figsize=(FIG_W_100, 4.0), sharex="col")

        for col, ft in enumerate(FAULT_KEYS):
            fs, fe = all_intervals[ft]
            wins = all_windows[ft]

            for row in range(2):
                ax = axes[row, col]
                ax.axvspan(fs, fe, color="#FFCDD2", alpha=0.25, zorder=0)

            # Row 0: zone temp
            ax = axes[0, col]
            sp_done = False
            for sc in ["baseline", "faulty_no_comp", "compensated"]:
                if sc not in wins:
                    continue
                sdf = wins[sc]
                ax.plot(sdf["datetime"], sdf["zone_temp"],
                        color=SCENARIO_COLORS[sc], label=SCENARIO_LABELS[sc])
                if not sp_done:
                    sp_col = "intended_sp" if "intended_sp" in sdf.columns else "htg_sp"
                    ax.plot(sdf["datetime"], sdf[sp_col],
                            color="k", lw=0.7, ls="--", alpha=0.4, label="Setpoint")
                    sp_done = True
            ax.set_title(FAULT_LABELS_INLINE[ft], fontweight="bold", fontsize=8.5)
            if col == 0:
                ax.set_ylabel("Zone temp [\u00b0C]")
            ax.legend(fontsize=6, loc="lower left", ncol=2)

            # Row 1: m_dot
            ax = axes[1, col]
            for sc in ["baseline", "faulty_no_comp", "compensated"]:
                if sc not in wins:
                    continue
                sdf = wins[sc]
                ax.plot(sdf["datetime"], sdf["m_dot"],
                        color=SCENARIO_COLORS[sc])
            if col == 0:
                ax.set_ylabel("Flow rate [kg/s]")

            # X-axis (bottom row only)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center",
                     fontsize=6.5)

        fig.suptitle("Cross-Fault Comparison: Baseline vs Faulty vs Compensated",
                     fontweight="bold", fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".svg"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Combined cross-fault: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Ablation bar chart (refined)
# ──────────────────────────────────────────────────────────────────────────────

def fig_ablation_bars(kpi_df: pd.DataFrame, out_path: Path):
    """DDH and Energy side-by-side, with value labels."""
    with plt.rc_context(JOURNAL_RC):
        scenarios = ["baseline", "faulty_no_comp", "compensated"]
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_100, 3.5))

        for ax_i, (metric, ylabel, title) in enumerate([
            ("ddh_abs_C_h",       "Degree-hours [\u00b0C\u00b7h]",
             "(a) Discomfort (DDH abs)"),
            ("dh_energy_kWh_est", "Energy [kWh]",
             "(b) Hydronic heat delivered"),
        ]):
            ax = axes[ax_i]
            n_f = len(FAULT_KEYS)
            bw = 0.22
            x_base = np.arange(n_f)

            for s_i, sc in enumerate(scenarios):
                vals = []
                for ft in FAULT_KEYS:
                    row = kpi_df[(kpi_df["fault_type"] == ft)
                                 & (kpi_df["scenario"] == sc)]
                    vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
                bars = ax.bar(x_base + s_i * bw, vals, bw,
                              label=SCENARIO_LABELS[sc],
                              color=SCENARIO_COLORS[sc],
                              edgecolor="k", linewidth=0.3)
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.5,
                                f"{v:.1f}", ha="center", va="bottom",
                                fontsize=6)

            ax.set_xticks(x_base + bw)
            ax.set_xticklabels([FAULT_LABELS_INLINE[ft] for ft in FAULT_KEYS],
                               fontsize=7.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", loc="left", fontsize=8.5)
            ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".svg"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Ablation bars: {out_path.with_suffix('.png')}")


# ──────────────────────────────────────────────────────────────────────────────
# Automated Results Narrative
# ──────────────────────────────────────────────────────────────────────────────

def generate_narrative(kpi_df, recovery_df, det_df, out_path):
    """Write a structured text summarizing all key findings."""

    lines = []
    lines.append("=" * 72)
    lines.append("AUTO-GENERATED RESULTS NARRATIVE")
    lines.append("Phase 8 — Cross-Fault Synthesis")
    lines.append("=" * 72)

    # Detection
    lines.append("\n## 1. Detection Performance\n")
    for _, row in det_df.iterrows():
        ft = row["fault_type"]
        lines.append(f"**{FAULT_LABELS_INLINE[ft]}**: "
                     f"Best model = {row['model']} (threshold: {row['threshold_key']}, "
                     f"persistence: {row['persistence']}). "
                     f"F1 = {row['f1']:.3f}, Precision = {row['precision']:.3f}, "
                     f"Recall = {row['recall']:.3f}, FPR = {row['fpr']:.4f}, "
                     f"Baseline FPR = {row.get('baseline_fpr', float('nan')):.4f}, "
                     f"Latency = {row['latency_minutes']:.0f} min.")

    # Compensation
    lines.append("\n## 2. Compensation Recovery\n")
    for _, row in recovery_df.iterrows():
        ft = row["fault_type"]
        lines.append(f"**{FAULT_LABELS_INLINE[ft]}**: "
                     f"DDH baseline = {row['ddh_baseline']:.1f}, "
                     f"DDH faulty = {row['ddh_faulty']:.1f}, "
                     f"DDH compensated = {row['ddh_compensated']:.1f}. "
                     f"Recovery = {row['ddh_recovery_pct']:.1f}%. "
                     f"Energy \u0394 vs baseline = "
                     f"{row['energy_delta_vs_baseline_pct']:+.1f}%.")

    # Interpretation
    lines.append("\n## 3. Key Findings\n")

    sc_rec = recovery_df.set_index("fault_type")["ddh_recovery_pct"]
    best_ft = sc_rec.idxmax()
    worst_ft = sc_rec.idxmin()

    lines.append(f"- Highest recovery: {FAULT_LABELS_INLINE[best_ft]} "
                 f"at {sc_rec[best_ft]:.1f}%.")
    lines.append(f"- Lowest recovery: {FAULT_LABELS_INLINE[worst_ft]} "
                 f"at {sc_rec[worst_ft]:.1f}%.")

    if sc_rec["stuck_closed"] < 5:
        lines.append("\n- **NOTE**: Stuck-closed recovery is near zero. "
                     "Investigation shows the supply temperature actuator "
                     "command is issued but not executed by EnergyPlus — "
                     "likely the EMS actuator object is missing from the IDF. "
                     "This requires an IDF fix and re-simulation.")

    lines.append("\n## 4. Detection Difficulty Gradient\n")
    det_sorted = det_df.sort_values("f1", ascending=False)
    for rank, (_, row) in enumerate(det_sorted.iterrows(), 1):
        lines.append(f"  {rank}. {FAULT_LABELS_INLINE[row['fault_type']]} — "
                     f"F1 = {row['f1']:.3f} ({row['model']})")

    lines.append("\n" + "=" * 72)

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(f"  Narrative: {out_path}")
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Refined journal figures + results narrative."
    )
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--synthesis-dir", default="plots/cross_fault_synthesis")
    parser.add_argument("--out-dir", default="plots/paper_figures_refined")
    parser.add_argument("--pad-hours", type=float, default=6.0)
    parser.add_argument(
        "--kappa", action="store_true",
        help="Use MDU kappa template sizing (B5-like, textwidth=113 mm). "
             "Affects only the 3-panel fault figures (FIG16A/B/E). "
             "In LaTeX, include these with width=\\textwidth.",
    )
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    project_dir = (Path(args.project_dir).resolve()
                   if args.project_dir else script_dir.parent)
    runs_root   = (project_dir / args.runs_dir).resolve()
    synth_dir   = (project_dir / args.synthesis_dir).resolve()
    out_dir     = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 8: JOURNAL-READY FIGURES & NARRATIVE")
    print(f"  Project dir   : {project_dir}")
    print(f"  Synthesis dir : {synth_dir}")
    print(f"  Output dir    : {out_dir}")
    if args.kappa:
        print(f"  Mode          : KAPPA (MDU B5, textwidth=113 mm)")
        print(f"  3-panel figs  : figsize=({FIG_W_KAPPA}, 5.0) in")
        print(f"  LaTeX include : width=\\textwidth")
    else:
        print(f"  Mode          : JOURNAL (Energy & Buildings)")
        print(f"  3-panel figs  : figsize=({FIG_W_085}, 4.6) in")
    print("=" * 72)

    # ── Load Phase 6 tables ──
    kpi_df      = pd.read_csv(synth_dir / "table_ablation_kpis.csv")
    recovery_df = pd.read_csv(synth_dir / "table_comp_recovery.csv")
    det_df      = pd.read_csv(synth_dir / "table_detection_best.csv")

    print(f"  KPI rows: {len(kpi_df)}, Recovery rows: {len(recovery_df)}, "
          f"Detection rows: {len(det_df)}")

    # ── Load runs + build windows ──
    run_dirs = {}
    for key, cfg_name in RUN_CONFIG_FILES.items():
        run_dirs[key] = find_latest_run_dir(runs_root, cfg_name)

    logs = {}
    for key, rd in run_dirs.items():
        logs[key] = load_runtime_log(rd / "fdc_runtime_log.csv")

    fault_intervals = {}
    for ft in FAULT_KEYS:
        fk = FAULT_TO_RUNKEYS[ft]["faulty"]
        fw = fault_window_from_config(run_dirs[fk])
        fault_intervals[ft] = fault_window_to_interval(fw)

    baseline_key = "baseline_detect"
    all_windows = {}
    for ft in FAULT_KEYS:
        fk = FAULT_TO_RUNKEYS[ft]["faulty"]
        ck = FAULT_TO_RUNKEYS[ft]["comp"]
        fs, fe = fault_intervals[ft]
        all_windows[ft] = {
            "baseline":       extract_window(logs[baseline_key], fs, fe,
                                             args.pad_hours),
            "faulty_no_comp": extract_window(logs[fk], fs, fe, args.pad_hours),
            "compensated":    extract_window(logs[ck], fs, fe, args.pad_hours),
        }

    # ── Generate all figures ──
    print("\nGenerating refined figures...")

    fig_compensability_spectrum(recovery_df,
                               out_dir / "fig_compensability_spectrum")
    fig_detection_comparison(det_df, out_dir / "fig_detection_comparison")
    fig_ablation_bars(kpi_df, out_dir / "fig_ablation_bars")

    for ft in FAULT_KEYS:
        fs, fe = fault_intervals[ft]
        fig_fault_refined(ft, all_windows[ft], fs, fe,
                          out_dir / f"fig_comparison_{ft}",
                          kappa=args.kappa)

    fig_combined_cross_fault(all_windows, fault_intervals,
                             out_dir / "fig_combined_cross_fault")

    # ── Narrative ──
    print("\nGenerating narrative...")
    narrative = generate_narrative(kpi_df, recovery_df, det_df,
                                  out_dir / "results_narrative.txt")
    print(narrative)

    print("\n" + "=" * 72)
    print("PHASE 8 COMPLETE")
    n_files = (len(list(out_dir.glob("*.png")))
               + len(list(out_dir.glob("*.svg")))
               + len(list(out_dir.glob("*.pdf"))))
    print(f"  Files generated: {n_files}")
    print(f"  Output dir: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()