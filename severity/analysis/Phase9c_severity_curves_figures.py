# -*- coding: utf-8 -*-
"""
Phase 9c: Severity Curve Figures & LaTeX Tables
========================================================================
Builds the per-fault severity curves and the combined panel.

Axis convention for stuck-closed: after invert_xaxis() the physical axis runs
0.70 (left/mild) -> 0.10 (right/severe), so the label reads "(more severe -->)"
and severity increases to the RIGHT.

Plateau shading: grey hatched bands mark severity regions where consecutive
detection metrics are identical (saturation behaviour).

Annotations on the combined panel:
  - "Undetectable" at so_b05 (F1=0)
  - "Comp. saturation" at so_b20 (DDH_comp identical to so_b40/60)
  - "Recovery collapse" at sc_s10 (poorest DDH recovery)
  - "scu_k02: near-zero comp." for the supply-curve discontinuity

Author : Nima Monghasemi
Date   : March 2026
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

HERE         = Path(__file__).resolve().parent
SEV_DIR      = HERE.parent
PROJECT_ROOT = SEV_DIR.parent

SEV_PLOTS  = SEV_DIR / "plots"
PAPER_FIGS = PROJECT_ROOT / "plots" / "paper_figures_refined"
SEV_PLOTS.mkdir(parents=True, exist_ok=True)

DET_CSV  = SEV_PLOTS / "table_severity_detection.csv"
COMP_CSV = SEV_PLOTS / "table_severity_compensation.csv"

# -----------------------------------------------------------------------
# JOURNAL_RC
# -----------------------------------------------------------------------

JOURNAL_RC = {
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "legend.fontsize":  8,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.2,
    "lines.linewidth":  1.2,
}

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

FAULT_KEYS   = ["stuck_closed", "stuck_open", "supply_curve"]
FAULT_LABELS = {
    "stuck_closed": "Stuck-Closed Valve",
    "stuck_open":   "Stuck-Open Valve",
    "supply_curve": "Supply-Curve Bias",
}
FAULT_X_LABEL = {
    "stuck_closed": "Flow fraction phi",
    "stuck_open":   "Thermostat bias DeltaT [degC]",
    "supply_curve": "OAT sensor bias |DeltaT| [K]",
}
# For stuck-closed: increasing phi = DECREASING severity.
# Invert x-axis so RIGHT = most severe (phi=0.10).
# Arrow label: "(more severe -->)" because after inversion severity grows rightward.
INVERT_X = {"stuck_closed": True, "stuck_open": False, "supply_curve": False}

DET_COLORS  = {"f1": "#1565C0", "precision": "#FF6D00", "recall": "#2E7D32"}
DET_MARKERS = {"f1": "o", "precision": "s", "recall": "^"}

COMP_COLOR_DDH    = "#C62828"
COMP_COLOR_ENERGY = "#00695C"
COMP_MARKER_DDH   = "D"
COMP_MARKER_ENERGY = "v"
COMP_LABEL_DDH    = "DDH Recovery [%]"
COMP_LABEL_ENERGY = "Energy Delta vs. Baseline [%]"

PLATEAU_COLOR   = "#DDDDDD"   # light grey for saturation bands
PLATEAU_ALPHA   = 0.35
PLATEAU_HATCH   = "////"
ANNOT_FONTSIZE  = 7.5

# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p, name in [(DET_CSV, "Detection"), (COMP_CSV, "Compensation")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} table not found:\n  {p}\n"
                f"Run Phase9{'a' if name == 'Detection' else 'b'}_severity_*.py first."
            )
    return pd.read_csv(DET_CSV), pd.read_csv(COMP_CSV)


def subset(df: pd.DataFrame, fault_type: str) -> pd.DataFrame:
    return df[df["fault_type"] == fault_type].copy().sort_values("severity_value")


# -----------------------------------------------------------------------
# Plateau detection helper
# -----------------------------------------------------------------------

def get_plateau_spans(x: np.ndarray, y: np.ndarray,
                      tol: float = 1e-6) -> list[tuple[float, float]]:
    """
    Return list of (x_start, x_end) intervals where consecutive y values
    are identical within *tol* (saturation / plateau behaviour).
    Used to draw shaded bands on the detection panels.
    """
    spans = []
    i = 0
    while i < len(y) - 1:
        if abs(y[i] - y[i + 1]) < tol:
            j = i + 1
            while j < len(y) - 1 and abs(y[j] - y[j + 1]) < tol:
                j += 1
            spans.append((float(x[i]), float(x[j])))
            i = j
        else:
            i += 1
    return spans


def shade_plateaus(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                   invert: bool = False) -> None:
    """Draw hatched grey bands over plateau regions on *ax*."""
    spans = get_plateau_spans(x, y)
    for xs, xe in spans:
        lo, hi = (min(xs, xe), max(xs, xe))
        ax.axvspan(lo, hi,
                   color=PLATEAU_COLOR, alpha=PLATEAU_ALPHA,
                   hatch=PLATEAU_HATCH, linewidth=0,
                   label="_plateau")
    if spans:
        patch = mpatches.Patch(facecolor=PLATEAU_COLOR, alpha=0.6,
                               hatch=PLATEAU_HATCH, label="Saturation plateau")
        handles, labels = ax.get_legend_handles_labels()
        labels_clean = [l for l in labels if not l.startswith("_")]
        handles_clean = [h for h, l in zip(handles, labels) if not l.startswith("_")]
        ax.legend(handles_clean + [patch],
                  labels_clean + ["Saturation plateau"],
                  loc="lower right", fontsize=7)


# -----------------------------------------------------------------------
# Axis fill helpers
# -----------------------------------------------------------------------

def _fill_detection_ax(ax: plt.Axes, sub: pd.DataFrame, fault_type: str,
                        shade: bool = True) -> None:
    x = sub["severity_value"].values
    f1_vals = sub["f1"].values

    for metric in ["f1", "precision", "recall"]:
        ax.plot(x, sub[metric].values,
                marker=DET_MARKERS[metric], color=DET_COLORS[metric],
                label=metric.capitalize(), linewidth=1.2, markersize=5)

    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Detection metric")
    ax.set_title("Detection Performance", fontweight="bold")

    if shade:
        shade_plateaus(ax, x, f1_vals)
    else:
        ax.legend(loc="lower right")

    xlabel = FAULT_X_LABEL[fault_type]
    if INVERT_X[fault_type]:
        ax.invert_xaxis()
        xlabel = xlabel + "  (more severe -->)"   # FIX: arrow points RIGHT
    ax.set_xlabel(xlabel)


def _fill_compensation_ax(ax: plt.Axes, sub: pd.DataFrame,
                           fault_type: str) -> plt.Axes:
    x = sub["severity_value"].values

    ax.plot(x, sub["ddh_recovery_pct"].values,
            marker=COMP_MARKER_DDH, color=COMP_COLOR_DDH,
            label=COMP_LABEL_DDH, linewidth=1.2, markersize=5)
    ax.axhline(100, color=COMP_COLOR_DDH, lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel("DDH Recovery [%]", color=COMP_COLOR_DDH)
    ax.tick_params(axis="y", labelcolor=COMP_COLOR_DDH)
    ax.set_title("Compensation KPIs", fontweight="bold")

    ax_r = ax.twinx()
    ax_r.plot(x, sub["energy_delta_pct"].values,
              marker=COMP_MARKER_ENERGY, color=COMP_COLOR_ENERGY,
              label=COMP_LABEL_ENERGY, linewidth=1.2, markersize=5, linestyle="--")
    ax_r.axhline(0, color=COMP_COLOR_ENERGY, lw=0.6, ls=":", alpha=0.5)
    ax_r.set_ylabel("Energy Delta vs. Baseline [%]", color=COMP_COLOR_ENERGY)
    ax_r.tick_params(axis="y", labelcolor=COMP_COLOR_ENERGY)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    pairs = [(h, l) for h, l in zip(h1 + h2, l1 + l2) if not l.startswith("_")]
    ax.legend([p[0] for p in pairs], [p[1] for p in pairs],
              loc="upper left", fontsize=7)

    xlabel = FAULT_X_LABEL[fault_type]
    if INVERT_X[fault_type]:
        ax.invert_xaxis()
        ax_r.invert_xaxis()
        xlabel = xlabel + "  (more severe -->)"   # FIX
    ax.set_xlabel(xlabel)

    return ax_r


# -----------------------------------------------------------------------
# Annotation helpers  (key findings from the assessment)
# -----------------------------------------------------------------------

def _annotate_detection(ax: plt.Axes, sub: pd.DataFrame, fault_type: str) -> None:
    """Add text callouts for noteworthy points on a detection panel."""
    x = sub["severity_value"].values
    f1 = sub["f1"].values

    if fault_type == "stuck_open":
        # so_b05: F1=0 — undetectable threshold
        row = sub[sub["severity_tag"] == "so_b05"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            ax.annotate("Undetectable\n(F1=0)", xy=(xv, 0.0),
                        xytext=(xv + 0.3, 0.15),
                        fontsize=ANNOT_FONTSIZE, color="#A32D2D",
                        arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.8),
                        ha="left")

    if fault_type == "supply_curve":
        # scu_k05: F1 peak — counter-intuitive (not highest severity)
        row = sub[sub["severity_tag"] == "scu_k05"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            yv = float(row["f1"].values[0])
            ax.annotate("F1 peak\n(not at max. sev.)", xy=(xv, yv),
                        xytext=(xv + 1.5, yv - 0.12),
                        fontsize=ANNOT_FONTSIZE, color="#185FA5",
                        arrowprops=dict(arrowstyle="->", color="#185FA5", lw=0.8),
                        ha="left")

    if fault_type == "stuck_closed":
        # sc_s70: recall drops and latency doubles
        row = sub[sub["severity_tag"] == "sc_s70"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            yv = float(row["f1"].values[0])
            ax.annotate("Recall drop\n+2.4x latency", xy=(xv, yv),
                        xytext=(xv - 0.12, yv + 0.08),
                        fontsize=ANNOT_FONTSIZE, color="#BA7517",
                        arrowprops=dict(arrowstyle="->", color="#BA7517", lw=0.8),
                        ha="right")


def _annotate_compensation(ax: plt.Axes, sub: pd.DataFrame, fault_type: str) -> None:
    """Add text callouts for noteworthy points on a compensation panel."""
    if fault_type == "stuck_closed":
        # sc_s10: recovery collapse
        row = sub[sub["severity_tag"] == "sc_s10"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            yv = float(row["ddh_recovery_pct"].values[0])
            ax.annotate("Recovery\ncollapse", xy=(xv, yv),
                        xytext=(xv - 0.08, yv + 10),
                        fontsize=ANNOT_FONTSIZE, color="#A32D2D",
                        arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.8),
                        ha="right")

    if fault_type == "stuck_open":
        # so_b05: Only annotate if recovery is actually substantial (> 30%)
        row = sub[sub["severity_tag"] == "so_b05"]
        if not row.empty:
            yv = float(row["ddh_recovery_pct"].values[0])
            if yv > 30.0:
                xv = float(row["severity_value"].values[0])
                ax.annotate(f"F1=0 yet\n{yv:.1f}% recovery\n(see diagnostics)",
                            xy=(xv, yv),
                            xytext=(xv + 0.4, yv - 18),
                            fontsize=ANNOT_FONTSIZE, color="#A32D2D",
                            arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.8),
                            ha="left")
        # so_b20+: saturation (DDH_comp identical)
        row = sub[sub["severity_tag"] == "so_b20"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            yv = float(row["ddh_recovery_pct"].values[0])
            ax.annotate("Comp. saturation\n(DDH_comp fixed)", xy=(xv, yv),
                        xytext=(xv + 0.5, yv - 10),
                        fontsize=ANNOT_FONTSIZE, color="#185FA5",
                        arrowprops=dict(arrowstyle="->", color="#185FA5", lw=0.8),
                        ha="left")

    if fault_type == "supply_curve":
        # scu_k02: near-zero compensation
        row = sub[sub["severity_tag"] == "scu_k02"]
        if not row.empty:
            xv = float(row["severity_value"].values[0])
            yv = float(row["ddh_recovery_pct"].values[0])
            ax.annotate("Near-zero\ncompensation", xy=(xv, yv),
                        xytext=(xv + 1.5, yv + 12),
                        fontsize=ANNOT_FONTSIZE, color="#A32D2D",
                        arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.8),
                        ha="left")



# -----------------------------------------------------------------------
# Figure 1: Per-fault figures
# -----------------------------------------------------------------------

def make_per_fault_figures(det_df: pd.DataFrame, comp_df: pd.DataFrame,
                            out_dir: Path) -> None:
    print("\n  Generating per-fault severity curve figures ...")
    for fault_type in FAULT_KEYS:
        det_sub  = subset(det_df,  fault_type)
        comp_sub = subset(comp_df, fault_type)

        with plt.rc_context(JOURNAL_RC):
            fig, (ax_det, ax_comp) = plt.subplots(1, 2, figsize=(12, 4.8))

            _fill_detection_ax(ax_det, det_sub, fault_type, shade=True)
            _annotate_detection(ax_det, det_sub, fault_type)

            _fill_compensation_ax(ax_comp, comp_sub, fault_type)
            _annotate_compensation(ax_comp, comp_sub, fault_type)

            fig.suptitle(f"Severity Analysis -- {FAULT_LABELS[fault_type]}",
                         fontsize=13, fontweight="bold")
            fig.subplots_adjust(wspace=0.52)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            tag  = fault_type.replace("_", "")
            stem = out_dir / f"fig_severity_{tag}"
            for ext in [".png", ".svg", ".pdf"]:
                fig.savefig(stem.with_suffix(ext))
            plt.close(fig)
            print(f"    Saved: {stem}.{{png,svg,pdf}}")


# -----------------------------------------------------------------------
# Figure 2: Combined 3x2 panel
# -----------------------------------------------------------------------

def make_combined_panel(det_df: pd.DataFrame, comp_df: pd.DataFrame,
                         out_dir: Path) -> None:
    print("\n  Generating combined 3x2 severity panel figure ...")

    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(3, 2, figsize=(14, 11),
                                 constrained_layout=True)

        for row_idx, fault_type in enumerate(FAULT_KEYS):
            det_sub  = subset(det_df,  fault_type)
            comp_sub = subset(comp_df, fault_type)
            is_sc    = INVERT_X[fault_type]

            ax_det  = axes[row_idx, 0]
            ax_comp = axes[row_idx, 1]

            # ── Detection panel ──────────────────────────────────────
            x_det = det_sub["severity_value"].values
            f1_v  = det_sub["f1"].values
            for metric in ["f1", "precision", "recall"]:
                ax_det.plot(x_det, det_sub[metric].values,
                            marker=DET_MARKERS[metric], color=DET_COLORS[metric],
                            label=metric.capitalize(), linewidth=1.2, markersize=4)
            ax_det.set_ylim(-0.05, 1.10)
            ax_det.set_ylabel("Metric score")
            ax_det.set_title(f"{FAULT_LABELS[fault_type]} -- Detection",
                             fontsize=10, fontweight="bold")

            # Plateau shading
            spans = get_plateau_spans(x_det, f1_v)
            for xs, xe in spans:
                ax_det.axvspan(min(xs, xe), max(xs, xe),
                               color=PLATEAU_COLOR, alpha=PLATEAU_ALPHA,
                               hatch=PLATEAU_HATCH, linewidth=0)

            xlabel_det = FAULT_X_LABEL[fault_type]
            if is_sc:
                ax_det.invert_xaxis()
                xlabel_det += "  (more severe -->)"   # FIX
            ax_det.set_xlabel(xlabel_det)

            # Build legend including plateau patch if needed
            handles, labels = ax_det.get_legend_handles_labels()
            labels_clean  = [l for l in labels  if not l.startswith("_")]
            handles_clean = [h for h, l in zip(handles, labels) if not l.startswith("_")]
            if spans:
                handles_clean.append(mpatches.Patch(
                    facecolor=PLATEAU_COLOR, alpha=0.6,
                    hatch=PLATEAU_HATCH, label="Saturation"))
                labels_clean.append("Saturation")
            ax_det.legend(handles_clean, labels_clean,
                          loc="lower right", fontsize=7)

            # Annotations
            _annotate_detection(ax_det, det_sub, fault_type)

            # ── Compensation panel ────────────────────────────────────
            x_comp = comp_sub["severity_value"].values

            ax_comp.plot(x_comp, comp_sub["ddh_recovery_pct"].values,
                         marker=COMP_MARKER_DDH, color=COMP_COLOR_DDH,
                         label=COMP_LABEL_DDH, linewidth=1.2, markersize=4)
            ax_comp.axhline(100, color=COMP_COLOR_DDH,
                            lw=0.6, ls="--", alpha=0.4)
            ax_comp.set_ylabel("DDH Recovery [%]", color=COMP_COLOR_DDH)
            ax_comp.tick_params(axis="y", labelcolor=COMP_COLOR_DDH)
            ax_comp.set_title(f"{FAULT_LABELS[fault_type]} -- Compensation",
                              fontsize=10, fontweight="bold")

            ax_r = ax_comp.twinx()
            ax_r.plot(x_comp, comp_sub["energy_delta_pct"].values,
                      marker=COMP_MARKER_ENERGY, color=COMP_COLOR_ENERGY,
                      label=COMP_LABEL_ENERGY,
                      linewidth=1.2, markersize=4, linestyle="--")
            ax_r.axhline(0, color=COMP_COLOR_ENERGY, lw=0.6, ls=":", alpha=0.4)
            ax_r.set_ylabel("Energy Delta vs. Baseline [%]",
                            color=COMP_COLOR_ENERGY)
            ax_r.tick_params(axis="y", labelcolor=COMP_COLOR_ENERGY)

            h1, l1 = ax_comp.get_legend_handles_labels()
            h2, l2 = ax_r.get_legend_handles_labels()
            ax_comp.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

            xlabel_comp = FAULT_X_LABEL[fault_type]
            if is_sc:
                ax_comp.invert_xaxis()
                ax_r.invert_xaxis()
                xlabel_comp += "  (more severe -->)"   # FIX
            ax_comp.set_xlabel(xlabel_comp)

            # Annotations
            _annotate_compensation(ax_comp, comp_sub, fault_type)

        fig.suptitle(
            "Fault Severity Parametric Sweep -- Detection & Compensation KPIs",
            fontsize=13, fontweight="bold")

        stem = out_dir / "fig_severity_combined_panel"
        for ext in [".png", ".svg", ".pdf"]:
            fig.savefig(stem.with_suffix(ext))
        plt.close(fig)
        print(f"    Saved: {stem}.{{png,svg,pdf}}")

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".svg", ".pdf"]:
        src = stem.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, PAPER_FIGS / src.name)
    print(f"    Copied combined panel to: {PAPER_FIGS}/")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Phase 9c v2 -- Severity Curve Figures (with annotations)")
    print("=" * 68)
    try:
        det_df, comp_df = load_tables()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  Detection table  : {len(det_df)} rows")
    print(f"  Compensation table: {len(comp_df)} rows")

    make_per_fault_figures(det_df, comp_df, SEV_PLOTS)
    make_combined_panel(det_df, comp_df, SEV_PLOTS)

    print(f"\nPhase 9c complete.  Outputs in: {SEV_PLOTS}")


if __name__ == "__main__":
    main()
