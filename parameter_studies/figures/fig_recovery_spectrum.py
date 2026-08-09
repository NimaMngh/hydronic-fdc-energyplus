# -*- coding: utf-8 -*-
"""
Figures built from the tabulated sweep KPIs.

  fig_compensability_spectrum.pdf  (Figure 7) - renamed fault label and
      neutral energy-axis wording ("below/above baseline" instead of
      "saving/penalty"), as promised in the R2.6 response.
  fig_target_sensitivity.pdf       (new, Section 4.5) - compensation-target
      sweep in the recovery-energy plane with the return-temperature shift
      colour-coded, plus the ramp-gain sweep.

Styling is copied from Phase8_paper_figures_refined.py so the new panels sit
consistently beside the existing ones.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOURNAL_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.2, "lines.linewidth": 1.0,
}
FIG_W_085 = 5.4
OUT = Path(__file__).resolve().parent / "output"

LABEL = {
    "stuck_closed": "Stuck-Closed Valve",
    "stuck_open":   "Stuck-Open Valve",
    "supply_bias":  "Supply-Setpoint Bias",
}
COLOUR = {"stuck_closed": "#1565C0", "stuck_open": "#2E7D32",
          "supply_bias": "#E65100"}
MARKER = {"stuck_closed": "s", "stuck_open": "D", "supply_bias": "o"}

# ---- reference configuration (Table: target sweep, dagger rows) -------------
REF = {                       # recovery %, dE_base %, dT_ret K
    "stuck_closed": (28.8, -7.1, -1.5),
    "stuck_open":   (68.8, +23.1, -4.2),
    "supply_bias":  (57.2, -0.8, +6.0),
}

# ---- Compensation-target sweep ---------------------------------
TARGET_SWEEP = {
    "stuck_closed": [(60, 0.0, -11.1, -8.3), (65, 17.7, -8.6, -5.0),
                     (70, 28.8, -7.1, -1.5)],
    "stuck_open":   [(50, 68.8, +23.1, -4.2), (55, 59.2, +24.8, -2.2),
                     (60, 51.5, +27.5, -0.6)],
    "supply_bias":  [(60, 41.6, -1.3, -1.1), (65, 51.5, -0.9, +2.5),
                     (70, 57.2, -0.8, +6.0)],
}
REF_TARGET = {"stuck_closed": 70, "stuck_open": 50, "supply_bias": 70}

# ---- Ramp-gain sweep -------------------------------------------
ALPHA_SWEEP = {
    "stuck_closed": [(0.1, 22.4), (0.2, 28.8), (0.4, 33.8)],
    "stuck_open":   [(0.1, 67.9), (0.2, 68.8), (0.4, 62.2)],
    "supply_bias":  [(0.1, 57.2), (0.2, 57.2), (0.4, 57.2)],
}


def fig_recovery_spectrum():
    with plt.rc_context(JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(FIG_W_085, 3.8))
        offsets = {"stuck_closed": (-12, -16), "stuck_open": (10, -18),
                   "supply_bias": (-14, 12)}
        for ft, (rec, de, _dt) in REF.items():
            ax.scatter(de, rec, s=100, c=COLOUR[ft], marker=MARKER[ft],
                       zorder=5, edgecolors="k", linewidths=0.6,
                       label=LABEL[ft])
            ox, oy = offsets[ft]
            ax.annotate(f"{rec:.0f}%", (de, rec), textcoords="offset points",
                        xytext=(ox, oy), fontsize=9, fontweight="bold",
                        color=COLOUR[ft],
                        arrowprops=dict(arrowstyle="-", color=COLOUR[ft],
                                        lw=0.7))
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(100, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.5)

        q = dict(fontsize=7.5, color="#555555", fontweight="bold",
                 style="italic")
        ax.text(27, 115, "Full recovery +\nenergy above baseline",
                ha="right", **q)
        ax.text(-12, 115, "Full recovery +\nenergy below baseline",
                ha="left", **q)
        ax.text(27, -15, "No recovery +\nenergy above baseline",
                ha="right", **q)
        ax.text(-12, -15, "No recovery +\nenergy below baseline",
                ha="left", **q)

        ax.set_xlabel("Energy change vs. baseline [%]")
        ax.set_ylabel("Discomfort recovery [%]")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
                  fontsize=7.5, framealpha=0.9, handletextpad=0.6,
                  columnspacing=1.5, markerscale=1.0, edgecolor="#CCCCCC")
        ax.set_xlim(-15, 30)
        ax.set_ylim(-25, 130)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)
        for ext in (".pdf", ".png"):
            fig.savefig(OUT / f"fig_compensability_spectrum{ext}")
        plt.close(fig)
        print(f"  saved fig_compensability_spectrum.pdf -> {OUT}")


def fig_target_sensitivity():
    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.25),
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})
        ax, bx = axes

        # ---- (a) target sweep in the recovery-energy plane --------------
        dts = [p[3] for ps in TARGET_SWEEP.values() for p in ps]
        vmax = max(abs(min(dts)), abs(max(dts)))
        norm = plt.Normalize(-vmax, vmax)
        cmap = plt.get_cmap("coolwarm")

        for ft, pts in TARGET_SWEEP.items():
            xs = [p[2] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-", color=COLOUR[ft], lw=0.9, alpha=0.55,
                    zorder=3)
            for tgt, rec, de, dt in pts:
                ref = tgt == REF_TARGET[ft]
                ax.scatter(de, rec, s=118 if ref else 74, c=[cmap(norm(dt))],
                           marker=MARKER[ft], zorder=5,
                           edgecolors="k" if not ref else COLOUR[ft],
                           linewidths=0.6 if not ref else 1.8)
                ax.annotate(f"{tgt}", (de, rec), textcoords="offset points",
                            xytext=(6, 5), fontsize=7, color="#333333")

        handles = [plt.Line2D([0], [0], marker=MARKER[ft], color="none",
                              markerfacecolor="#BBBBBB",
                              markeredgecolor="k", markeredgewidth=0.6,
                              markersize=7, label=LABEL[ft])
                   for ft in TARGET_SWEEP]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  framealpha=0.9, edgecolor="#CCCCCC", handletextpad=0.5)
        ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Energy change vs. baseline [%]")
        ax.set_ylabel("Discomfort recovery [%]")
        ax.set_title("(a) Compensation-target sweep", fontsize=9)
        ax.set_xlim(-14, 31)
        ax.set_ylim(-8, 82)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.055)
        cb.set_label(r"$\Delta T_\mathrm{ret}$ [K]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # ---- (b) ramp-gain sweep ---------------------------------------
        for ft, pts in ALPHA_SWEEP.items():
            a = [p[0] for p in pts]
            r = [p[1] for p in pts]
            bx.plot(a, r, "-", color=COLOUR[ft], lw=1.2, marker=MARKER[ft],
                    markersize=5.5, markeredgecolor="k",
                    markeredgewidth=0.5, label=LABEL[ft], zorder=4)
            i_ref = a.index(0.2)
            bx.scatter([a[i_ref]], [r[i_ref]], s=132, facecolors="none",
                       edgecolors=COLOUR[ft], linewidths=1.6, zorder=5)
        bx.set_xscale("log")
        bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        bx.set_xticks([0.1, 0.2, 0.4])
        bx.set_xticklabels(["0.1", "0.2", "0.4"])
        bx.set_xlim(0.085, 0.47)
        bx.set_xlabel(r"Ramp gain $\alpha$ [--]")
        bx.set_ylabel("Discomfort recovery [%]")
        bx.set_title("(b) Ramp-gain sweep", fontsize=9)
        bx.set_ylim(15, 78)
        bx.legend(loc="center right", fontsize=7, framealpha=0.9,
                  edgecolor="#CCCCCC", handletextpad=0.5)

        fig.tight_layout()
        for ext in (".pdf", ".png"):
            fig.savefig(OUT / f"fig_target_sensitivity{ext}")
        plt.close(fig)
        print(f"  saved fig_target_sensitivity.pdf -> {OUT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_recovery_spectrum()
    fig_target_sensitivity()
