#!/usr/bin/env python3
"""
Phase11_fig_target_sensitivity.py — compensation-parameter sensitivity figure.

Builds fig_target_sensitivity.pdf (Sec. 4.5, compensation-parameter
sensitivity) from the sweep KPI table written by
analysis/Phase10_sweep_pareto.py.

One panel per fault type in the recovery-energy plane:
  * circles  = compensation-target sweep at the reference gain (alpha = 0.2)
  * diamonds = ramp-gain sweep at the reference target (alpha = 0.1 / 0.4)
  * marker colour = flow-weighted return-temperature shift vs. baseline (K)
  * the reference configuration is labelled in bold

Usage (from the project root):
    python analysis\\Phase11_fig_target_sensitivity.py
    python analysis\\Phase11_fig_target_sensitivity.py ^
        --csv plots\\parameter_studies\\sweep_kpis.csv ^
        --out plots\\parameter_studies\\fig_target_sensitivity.pdf

Author : Nima Monghasemi
Date   : August 2026
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REF_ALPHA = 0.2
FAULTS = [
    ("stuck_closed", "(a) Stuck-closed valve", 70),
    ("stuck_open", "(b) Stuck-open valve", 50),
    ("supply_curve", "(c) Supply-setpoint bias", 70),
]
# Diverging scale centred on zero return-temperature shift; the limits span
# the observed range across all 15 runs (-8.3 K ... +6.1 K).
NORM = TwoSlopeNorm(vmin=-8.5, vcenter=0.0, vmax=6.5)
CMAP = plt.cm.coolwarm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("plots") / "parameter_studies" / "sweep_kpis.csv",
                    help="KPI table written by Phase10_sweep_pareto.py")
    ap.add_argument("--out", type=Path,
                    default=Path("plots") / "parameter_studies" / "fig_target_sensitivity.pdf",
                    help="Output PDF path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))
    for ax, (ft, title, ref_target) in zip(axes, FAULTS):
        sub = df[df.fault_type == ft]
        tgt = sub[np.isclose(sub.alpha, REF_ALPHA)].sort_values("target_C")
        alp = sub[~np.isclose(sub.alpha, REF_ALPHA)].sort_values("alpha")

        # target sweep: dashed connector + circles
        ax.plot(tgt.recovery_pct, tgt.energy_delta_vs_baseline_pct,
                "--", color="0.55", lw=1.0, zorder=2)
        ax.scatter(tgt.recovery_pct, tgt.energy_delta_vs_baseline_pct,
                   c=tgt.J_R_return_shift_K, cmap=CMAP, norm=NORM, s=110,
                   edgecolors="k", linewidths=0.8, zorder=3, marker="o")

        # gain sweep: diamonds
        ax.scatter(alp.recovery_pct, alp.energy_delta_vs_baseline_pct,
                   c=alp.J_R_return_shift_K, cmap=CMAP, norm=NORM, s=95,
                   edgecolors="k", linewidths=0.8, zorder=4, marker="D")

        # annotations
        for _, r in tgt.iterrows():
            is_ref = r.target_C == ref_target
            lbl = f"{int(r.target_C)}$^\\circ$C" + (" (ref.)" if is_ref else "")
            ax.annotate(lbl, (r.recovery_pct, r.energy_delta_vs_baseline_pct),
                        textcoords="offset points", xytext=(7, 7), fontsize=8.5,
                        fontweight="bold" if is_ref else "normal")
        if ft == "supply_curve" and len(alp):
            r = alp.iloc[0]
            ax.annotate("$\\alpha$=0.1, 0.4\n(coincident)",
                        (r.recovery_pct, r.energy_delta_vs_baseline_pct),
                        textcoords="offset points", xytext=(7, -24), fontsize=8)
        else:
            for _, r in alp.iterrows():
                ax.annotate(f"$\\alpha$={r.alpha:g}",
                            (r.recovery_pct, r.energy_delta_vs_baseline_pct),
                            textcoords="offset points", xytext=(7, -13), fontsize=8)

        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("DDH recovery [%]", fontsize=10)
        ax.grid(alpha=0.3)
        x0, x1 = ax.get_xlim(); ax.set_xlim(x0 - 3, x1 + 10)
        y0, y1 = ax.get_ylim(); pad = (y1 - y0) * 0.20
        ax.set_ylim(y0 - pad, y1 + pad)

    axes[0].set_ylabel("Energy vs. fault-free baseline [%]", fontsize=10)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                      ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("$\\Delta T_\\mathrm{ret}$ vs. baseline [K]", fontsize=10)

    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
