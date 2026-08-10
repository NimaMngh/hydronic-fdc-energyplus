# -*- coding: utf-8 -*-
"""
Combined severity panel: 1x3 subplots, one per fault type
  - Left y-axis:  F1-score (detection)
  - Right y-axis: DDH Recovery % (compensation)
  - x-axis:       severity parameter
  - One key annotation per panel
  - Shared y-axis labels on outer edges only
  - Middle panel y-ticks hidden on both axes

figsize matches the rendered textwidth (6.4 in) so fonts come out at
true size without LaTeX downscaling.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# Paper style — sized for 1:1 rendering at textwidth
# -----------------------------------------------------------------------
JOURNAL_RC = {
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        8.5,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  7,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "figure.dpi":       150,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.15,
    "lines.linewidth":  1.2,
}

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
FAULT_KEYS = ["stuck_closed", "stuck_open", "supply_curve"]
FAULT_TITLES = {
    "stuck_closed": "Stuck-closed valve",
    "stuck_open":   "Stuck-open valve",
    "supply_curve": "Supply-curve bias",
}
FAULT_X_LABELS = {
    "stuck_closed": "Flow fraction $\\varphi$  (more severe \u2192)",
    "stuck_open":   "Setpoint bias $\\Delta T_{\\mathrm{bias}}$ [\u00b0C]",
    "supply_curve": "Supply bias $|\\Delta T_{\\mathrm{supply}}|$ [K]",
}
FAULT_SUBTAGS = {
    "stuck_closed": "(a)",
    "stuck_open":   "(b)",
    "supply_curve": "(c)",
}
INVERT_X = {"stuck_closed": True, "stuck_open": False, "supply_curve": False}

F1_COLOR  = "#1565C0"
DDH_COLOR = "#C62828"

# Display width: single-column textwidth ~ 6.38 in
FIG_W = 6.4

def get_plateau_spans(x, y, tol=1e-6, min_points=3):
    """Runs of equal values, requiring at least min_points members.

    A two-point flat spot is not a plateau: with the corrected
    stuck-closed data it would shade phi = 0.20-0.30 while the curve
    falls from 0.900 to 0.000 immediately after.
    """
    spans = []
    i = 0
    while i < len(y) - 1:
        if abs(y[i] - y[i + 1]) < tol:
            j = i + 1
            while j < len(y) - 1 and abs(y[j] - y[j + 1]) < tol:
                j += 1
            if (j - i + 1) >= min_points:
                spans.append((float(x[i]), float(x[j])))
            i = j
        else:
            i += 1
    return spans

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent
SEV_DIR   = HERE.parent
SEV_PLOTS = SEV_DIR / "plots"

det_df  = pd.read_csv(SEV_PLOTS / "table_severity_detection.csv")
comp_df = pd.read_csv(SEV_PLOTS / "table_severity_compensation.csv")


# -----------------------------------------------------------------------
# Create figure — width matches \textwidth for 1:1 font rendering
# -----------------------------------------------------------------------
with plt.rc_context(JOURNAL_RC):

    # Tight wspace: the middle panel carries no y-axis labels on either
    # side, so the panels can sit much closer than the default spacing
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.0),
                             layout="constrained",
                             gridspec_kw={"wspace": 0.04})

    twin_axes = []

    for idx, fault_type in enumerate(FAULT_KEYS):
        ax_f1 = axes[idx]

        det_sub  = det_df[det_df["fault_type"] == fault_type].sort_values("severity_value")
        comp_sub = comp_df[comp_df["fault_type"] == fault_type].sort_values("severity_value")

        x_det  = det_sub["severity_value"].values
        x_comp = comp_sub["severity_value"].values
        f1_vals  = det_sub["f1"].values
        ddh_vals = comp_sub["ddh_recovery_pct"].values

        # --- Left y-axis: F1-score (%) ---
        line_f1, = ax_f1.plot(x_det, f1_vals * 100,
                              marker="o", color=F1_COLOR, linewidth=1.2,
                              markersize=4, label="Detection F1", zorder=3)
        ax_f1.set_ylim(-5, 105)
        ax_f1.tick_params(axis="y", labelcolor=F1_COLOR)

        # Left y-axis: only panel (a) shows label and tick labels;
        # both middle and right panels hide left tick labels entirely
        if idx == 0:
            ax_f1.set_ylabel("Detection F1 (%)", color=F1_COLOR,
                             fontweight="bold")
        else:
            ax_f1.set_ylabel("")
            ax_f1.tick_params(axis="y", labelleft=False)

        # Plateau shading
        spans = get_plateau_spans(x_det, f1_vals)
        for xs, xe in spans:
            ax_f1.axvspan(min(xs, xe), max(xs, xe),
                          color="#DDDDDD", alpha=0.35, hatch="////",
                          linewidth=0, zorder=1)

        # --- Right y-axis: DDH Recovery ---
        ax_ddh = ax_f1.twinx()
        twin_axes.append(ax_ddh)
        line_ddh, = ax_ddh.plot(x_comp, ddh_vals,
                                marker="D", color=DDH_COLOR, linewidth=1.2,
                                markersize=4, linestyle="--",
                                label="DDH recovery", zorder=3)
        ax_ddh.set_ylim(-5, 105)
        ax_ddh.tick_params(axis="y", labelcolor=DDH_COLOR)

        # Right y-axis — only panel (c) shows label and tick labels;
        # panels (a) and (b) hide right tick labels
        if idx == 2:
            ax_ddh.set_ylabel("DDH recovery (%)", color=DDH_COLOR,
                              fontweight="bold")
        else:
            ax_ddh.set_ylabel("")
            ax_ddh.tick_params(axis="y", labelright=False)

        # Middle panel: inward ticks on both axes so the tick marks stay
        # visible as a reference without occupying label space
        if idx == 1:
            ax_f1.tick_params(axis="y", direction="in")
            ax_ddh.tick_params(axis="y", direction="in")

        # Reference line
        ax_f1.axhline(0, color="grey", lw=0.5, ls="-", alpha=0.3)

        # x-axis label with slightly smaller font and sub-tag on new line
        xlabel = FAULT_X_LABELS[fault_type]
        if INVERT_X[fault_type]:
            ax_f1.invert_xaxis()
        ax_f1.set_xlabel(xlabel + f"\n{FAULT_SUBTAGS[fault_type]}",
                         fontsize=7)

        # Combined legend
        lines = [line_f1, line_ddh]
        labels = [l.get_label() for l in lines]
        if spans:
            patch = mpatches.Patch(facecolor="#DDDDDD", alpha=0.6,
                                   hatch="////", label="Detection plateau")
            lines.append(patch)
            labels.append("Detection plateau")

        if fault_type == "stuck_closed":
            leg_loc = "lower right"
        elif fault_type == "stuck_open":
            leg_loc = "lower right"
        else:
            leg_loc = "upper right"

        ax_f1.legend(lines, labels, loc=leg_loc, fontsize=6.5,
                     framealpha=0.92)

        # ---------------------------------------------------------------
        # Annotations: one per panel, only where the curve shape alone
        # does not carry the point.
        #   "Recovery collapse"              — panel (a)
        #   "Undetectable (F1 = 0)"          — panel (b)
        #   "F1 peak (not at max. severity)" — panel (c)
        # ---------------------------------------------------------------
        annot_kw = dict(fontsize=6.5, ha="left",
                        arrowprops=dict(arrowstyle="->", lw=0.7))

        if fault_type == "stuck_open":
            row = det_sub[det_sub["severity_tag"] == "so_b05"]
            if not row.empty:
                xv = float(row["severity_value"].values[0])
                ax_f1.annotate("Undetectable\n(F1 = 0)", xy=(xv, 0),
                               xytext=(xv + 0.5, 18),
                               color="#A32D2D", **annot_kw)

        if fault_type == "stuck_closed":
            # phi = 0.70 is the mildest restriction and now sits below
            # the ensemble detection floor. Placed in the empty upper-left
            # region: the x axis is inverted here, so a slightly smaller
            # phi runs the text rightwards, clear of both curves.
            row_u = det_sub[det_sub["severity_tag"] == "sc_s70"]
            if not row_u.empty and float(row_u["f1"].values[0]) == 0.0:
                xv = float(row_u["severity_value"].values[0])
                ax_f1.annotate("Undetectable\n(F1 = 0)", xy=(xv, 0),
                               xytext=(xv - 0.015, 72),
                               color="#A32D2D", **annot_kw)

            row_c = comp_sub[comp_sub["severity_tag"] == "sc_s10"]
            if not row_c.empty:
                xv = float(row_c["severity_value"].values[0])
                yv = float(row_c["ddh_recovery_pct"].values[0])
                ax_ddh.annotate("Recovery\ncollapse", xy=(xv, yv),
                                xytext=(xv + 0.35, yv + 30),
                                color="#A32D2D", ha="left",
                                arrowprops=dict(arrowstyle="->",
                                               color="#A32D2D", lw=0.7),
                                fontsize=6.5,
                                annotation_clip=False)

        if fault_type == "supply_curve":
            row = det_sub[det_sub["severity_tag"] == "scu_k05"]
            if not row.empty:
                xv = float(row["severity_value"].values[0])
                yv = float(row["f1"].values[0]) * 100
                ax_f1.annotate("F1 peak\n(not at max.\nseverity)",
                               xy=(xv, yv),
                               xytext=(xv + 2, yv - 18),
                               color="#185FA5", **annot_kw)

    # Save
    out = SEV_PLOTS / "fig_severity_combined_panel_optionA"
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    print(f"Saved: {out}.{{png,pdf,svg}}")
