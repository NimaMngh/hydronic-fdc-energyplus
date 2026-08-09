# -*- coding: utf-8 -*-
"""
Signature-based fault diagnosis ladder (Figure 3).

Layout: a strict three-column grid
(priority tag | condition | action) with reserved, non-overlapping x ranges,
so no coloured action block can sit on top of the condition text. The two P3
entry paths converge through an explicit "or" junction. Font sizes are
auto-fitted per box and an overflow check is run before export.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

C = {
    "ink":         "#1F2937",
    "muted":       "#6B7280",
    "light_muted": "#C0C5CC",
    "entry_fill":  "#EEF2F7",
    "entry_edge":  "#94A3B8",
    "rule_fill":   "#FFFFFF",
    "rule_edge":   "#475569",
    "band_p1":     "#FEE2E2",
    "band_p2":     "#DBEAFE",
    "band_p3":     "#D1FAE5",
    "band_p4":     "#F3F4F6",
    "sc":          "#DC2626",
    "so":          "#2563EB",
    "scu":         "#16A34A",
    "unk":         "#9CA3AF",
    "arrow":       "#475569",
}

# ---------------------------------------------------------------- column grid
X_TAG_C,  X_TAG_W  = 0.62, 0.78     # priority tag column
X_CON_L,  X_CON_R  = 1.30, 6.60     # condition column (reserved)
X_ACT_L,  X_ACT_R  = 7.38, 10.45    # action column (reserved)
X_CON_C = (X_CON_L + X_CON_R) / 2
X_ACT_C = (X_ACT_L + X_ACT_R) / 2
X_CON_W = X_CON_R - X_CON_L
X_ACT_W = X_ACT_R - X_ACT_L
X_SPINE = X_CON_C

_texts = []          # (text_artist, x_left_limit, x_right_limit, label)


def box(ax, xy, w, h, txt, *, fc, ec, fs, fw="normal", tc="#111",
        lw=1.15, ls=1.20, ha="center", zo=3, tag=""):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.16",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo))
    tx = x if ha == "center" else x - w / 2 + 0.16
    t = ax.text(tx, y, txt, ha=ha, va="center", fontsize=fs, fontweight=fw,
                color=tc, zorder=zo + 1, linespacing=ls)
    _texts.append((t, x - w / 2, x + w / 2, tag))
    return t


def arrow(ax, s, e, *, c=None, lw=1.25):
    ax.add_patch(FancyArrowPatch(
        s, e, arrowstyle="-|>", color=c or C["arrow"], linewidth=lw,
        mutation_scale=10, zorder=2, shrinkA=0, shrinkB=0))


def band(ax, yc, h, col, label):
    ax.add_patch(Rectangle((0.16, yc - h / 2), X_ACT_R - 0.16 + 0.10, h,
                           facecolor=col, edgecolor="none", alpha=0.45,
                           zorder=0))
    box(ax, (X_TAG_C, yc), X_TAG_W, 0.34, label, fc="white",
        ec=C["rule_edge"], fs=8.5, fw="bold", tc=C["ink"], lw=0.9,
        tag=f"tag-{label}")


def make_figure():
    fig, ax = plt.subplots(figsize=(7.0, 5.75))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(1.15, 9.75)
    ax.axis("off")

    yE, yP1, yP2 = 9.25, 8.05, 6.50
    y3a, y3b, yP4 = 4.90, 3.15, 1.75
    y_join = (y3a + y3b) / 2 - 0.28
    y_scu = y_join

    band(ax, yP1, 1.14, C["band_p1"], "P1")
    band(ax, yP2, 1.34, C["band_p2"], "P2")
    band(ax, (y3a + y3b) / 2, (y3a - y3b) + 2.05, C["band_p3"], "P3")
    band(ax, yP4, 0.96, C["band_p4"], "P4")

    # ------------------------------------------------------------- entry
    box(ax, (X_CON_C, yE), 4.55, 0.58,
        "Persistent anomaly\n"
        r"($N_{\mathrm{persist}}$ consecutive ML flags)",
        fc=C["entry_fill"], ec=C["entry_edge"], fs=9.5, tc=C["ink"],
        lw=1.2, tag="entry")
    arrow(ax, (X_SPINE, yE - 0.31), (X_SPINE, yP1 + 0.44))

    # ------------------------------------------------------------- P1
    box(ax, (X_CON_C, yP1), X_CON_W, 0.82,
        "Low flow  +  sustained underheating\n"
        r"$r_{\dot{m}} < 0.6$" r"$\;$ and $\;$"
        r"$\bar{e}_{T,\,\mathrm{2h}} < -0.3\,^{\circ}$C",
        fc=C["rule_fill"], ec=C["rule_edge"], fs=9.3, tc=C["ink"],
        ls=1.25, tag="P1-cond")
    box(ax, (X_ACT_C, yP1), X_ACT_W, 0.86,
        "Stuck-closed\nphysical, zone\n"
        r"$T_{\mathrm{target}}=70\,^{\circ}$C  (boost)",
        fc=C["sc"], ec=C["sc"], fs=8.4, fw="bold", tc="white",
        lw=1.0, ls=1.18, tag="P1-act")
    arrow(ax, (X_CON_R + 0.06, yP1), (X_ACT_L - 0.06, yP1))
    arrow(ax, (X_SPINE, yP1 - 0.44), (X_SPINE, yP2 + 0.54))

    # ------------------------------------------------------------- P2
    box(ax, (X_CON_C, yP2), X_CON_W, 1.02,
        "Normal flow  +  overheating\n"
        r"($\bar{e}_{T,\,\mathrm{2h}} > 0.3\,^{\circ}$C"
        r"$\;$ or $\;$" r"$e_T > 0.5\,^{\circ}$C)" "\n"
        r"and $\;$ $r_{\dot{m}} \geq 0.6$",
        fc=C["rule_fill"], ec=C["rule_edge"], fs=9.3, tc=C["ink"],
        tag="P2-cond")
    box(ax, (X_ACT_C, yP2), X_ACT_W, 0.86,
        "Stuck-open\ncontrol, zone\n"
        r"$T_{\mathrm{target}}=50\,^{\circ}$C  (reduce)",
        fc=C["so"], ec=C["so"], fs=8.4, fw="bold", tc="white",
        lw=1.0, ls=1.18, tag="P2-act")
    arrow(ax, (X_CON_R + 0.06, yP2), (X_ACT_L - 0.06, yP2))
    arrow(ax, (X_SPINE, yP2 - 0.54), (X_SPINE, y3a + 0.70))

    # ------------------------------------------------------------- P3a
    box(ax, (X_CON_C, y3a), X_CON_W, 1.32,
        "Supply temperature depressed  +  normal flow\n"
        r"$T_{\mathrm{sup,\,baseline}} - T_{\mathrm{sup}} \geq 5\,$K"
        r"$\;$ and $\;$" r"$r_{\dot{m}} \geq 0.6$" "\n"
        r"$\bullet\;$ shortfall $\geq 10\,$K $\rightarrow$ classify immediately"
        "\n"
        r"$\bullet\;$ shortfall 5" "\u2013" r"10$\,$K $\rightarrow$ "
        r"$\geq\!1$ zone under-delivering",
        fc=C["rule_fill"], ec=C["rule_edge"], fs=8.4, tc=C["ink"],
        ls=1.20, ha="left", tag="P3a-cond")
    arrow(ax, (X_SPINE, y3a - 0.69), (X_SPINE, y3b + 0.58))

    # ------------------------------------------------------------- P3b
    box(ax, (X_CON_C, y3b), X_CON_W, 1.02,
        "Multi-zone underheating  +  normal flow\n"
        r"$\bar{e}_{T,\,\mathrm{2h}} < -0.3\,^{\circ}$C"
        r"$\;$ and $\;$" r"$r_{\dot{m}} \geq 0.6$" "\n"
        r"$\geq\!2$ additional zones under-delivering",
        fc=C["rule_fill"], ec=C["rule_edge"], fs=8.6, tc=C["ink"],
        ls=1.20, ha="left", tag="P3b-cond")

    # ---- explicit "or" junction feeding the shared P3 action -------------
    x_j = X_CON_R + 0.36
    ax.plot([X_CON_R + 0.06, x_j], [y3a, y3a], color=C["arrow"], lw=1.15,
            zorder=2, solid_capstyle="round")
    ax.plot([X_CON_R + 0.06, x_j], [y3b, y3b], color=C["arrow"], lw=1.15,
            zorder=2, solid_capstyle="round")
    ax.plot([x_j, x_j], [y3b, y3a], color=C["arrow"], lw=1.15, zorder=2,
            solid_capstyle="round")
    ax.text(x_j, (y3a + y_join) / 2, "or", ha="center", va="center",
            fontsize=8.2, style="italic", color=C["muted"], zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none"))
    arrow(ax, (x_j, y_join), (X_ACT_L - 0.06, y_join))

    box(ax, (X_ACT_C, y_scu), X_ACT_W, 0.94,
        "Supply-setpoint bias\ncontrol, system\n"
        r"$T_{\mathrm{target}}=70\,^{\circ}$C  (boost)",
        fc=C["scu"], ec=C["scu"], fs=8.4, fw="bold", tc="white",
        lw=1.0, ls=1.18, tag="P3-act")

    arrow(ax, (X_SPINE, y3b - 0.54), (X_SPINE, yP4 + 0.44))

    # ------------------------------------------------------------- P4
    box(ax, (X_CON_C, yP4), X_CON_W, 0.60, "No signature matched",
        fc=C["rule_fill"], ec=C["rule_edge"], fs=9.3, tc=C["ink"],
        tag="P4-cond")
    box(ax, (X_ACT_C, yP4), X_ACT_W, 0.68,
        "Unknown\nno compensation",
        fc=C["unk"], ec=C["unk"], fs=8.4, fw="bold", tc="white",
        lw=1.0, ls=1.18, tag="P4-act")
    arrow(ax, (X_CON_R + 0.06, yP4), (X_ACT_L - 0.06, yP4))

    return fig, ax


def autofit_and_check(fig, ax, min_fs=6.6):
    """Shrink any text that overruns its own box, then verify no overflow
    and no encroachment into the action column."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    report = []
    for t, xl, xr, tag in _texts:
        for _ in range(24):
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            p0 = inv.transform((bb.x0, bb.y0))
            p1 = inv.transform((bb.x1, bb.y1))
            over = max(xl - p0[0], p1[0] - xr)
            if over <= 0.02 or t.get_fontsize() <= min_fs:
                break
            t.set_fontsize(t.get_fontsize() - 0.2)
            fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        p0 = inv.transform((bb.x0, bb.y0))
        p1 = inv.transform((bb.x1, bb.y1))
        status = "OK" if (p0[0] >= xl - 0.02 and p1[0] <= xr + 0.02) else "OVERFLOW"
        enc = "" if not (tag.endswith("cond") and p1[0] > X_ACT_L) else "  <-- ENTERS ACTION COLUMN"
        report.append(f"  {status:9s} {tag:11s} fs={t.get_fontsize():.1f} "
                      f"x=[{p0[0]:.2f},{p1[0]:.2f}] box=[{xl:.2f},{xr:.2f}]{enc}")
    return report


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = make_figure()
    for line in autofit_and_check(fig, ax):
        print(line)
    fig.savefig(out / "fig_diagnosis_flowchart.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out / "fig_diagnosis_flowchart.png", dpi=300, bbox_inches="tight")
    print(f"saved fig_diagnosis_flowchart.pdf / .png -> {out}")
