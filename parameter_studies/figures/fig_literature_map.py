# -*- coding: utf-8 -*-
"""
Literature positioning map – Figure 1
Final: grid-snapped positions, no shaded region, clean layout.
Sized for Energy and Buildings (elsarticle, 0.95\textwidth ~ 6.0 in).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# -- Colours (matched to LaTeX definitions) ---
DH_BLUE      = np.array([31,  78, 121]) / 255
HVAC_ORANGE  = np.array([198, 124,  36]) / 255

DH_BLUE_60   = DH_BLUE   * 0.60 + np.array([1,1,1]) * 0.40
HVAC_OR_65   = HVAC_ORANGE * 0.65 + np.array([1,1,1]) * 0.35
DH_BLUE_70   = DH_BLUE   * 0.70 + np.array([1,1,1]) * 0.30

# -- Figure setup ---
fig, ax = plt.subplots(figsize=(6.0, 4.9))

# -- Axis limits & ticks ---
ax.set_xlim(-0.50, 5.75)
ax.set_ylim(-0.65, 4.80)

ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(
    ['No detection', 'Rule-based', 'Model-based',
     'Supervised ML', 'Unsupervised ML', 'ML-in-the-loop'],
    rotation=38, ha='right', fontsize=8.5
)

ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(
    ['Alarm only',
     'Advisory',
     'Generic\ncorrection',
     'Fault-specific\ncorrection',
     'Supervisory\ncompensation'],
    fontsize=8.5
)

ax.set_xlabel('Degree of automated fault diagnosis', fontsize=9.5)
ax.set_ylabel('Degree of corrective autonomy',       fontsize=9.5)

# -- Grid ---
ax.grid(which='both', color='gray', alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)

# -- DH/hydronic studies (navy circles) ---
#   Djuric (2,1) | Sarran (2,1) | Hakansson (2,1) | Losi (3,0) | Lee (4,0) | Leiria (4,0)
# Small jitter where studies share a cell
dh_xs = [1.78, 2.22, 2.00,  3.0,   3.82,  4.18]
dh_ys = [0.86, 0.86, 1.20,  0.0,   0.10, -0.10]
ax.scatter(dh_xs, dh_ys,
           s=80, marker='o',
           facecolors=DH_BLUE_60, edgecolors=DH_BLUE,
           linewidths=0.8, zorder=5)

ax.annotate('Djuric et al. (2008)',
            xy=(1.78, 0.86), xytext=(1.30, 0.48),
            fontsize=7.5, color=DH_BLUE, va='top', ha='center',
            arrowprops=dict(arrowstyle='-', color=DH_BLUE,
                            lw=0.5, shrinkA=0, shrinkB=3))

ax.annotate('Sarran et al. (2022)',
            xy=(2.22, 0.86), xytext=(2.78, 0.52),
            fontsize=7.5, color=DH_BLUE, va='top', ha='center',
            arrowprops=dict(arrowstyle='-', color=DH_BLUE,
                            lw=0.5, shrinkA=0, shrinkB=3))

ax.annotate('Håkansson et al. (2025)',
            xy=(2.00, 1.20), xytext=(2.00, 1.48),
            fontsize=7.5, color=DH_BLUE, va='bottom', ha='center',
            arrowprops=dict(arrowstyle='-', color=DH_BLUE,
                            lw=0.5, shrinkA=0, shrinkB=3))

ax.annotate('Losi et al. (2024)',
            xy=(3.0, 0.0), xytext=(2.55, -0.35),
            fontsize=7.5, color=DH_BLUE, va='top', ha='center')

ax.annotate('Lee et al. (2023)',
            xy=(3.82, 0.10), xytext=(3.15, 0.55),
            fontsize=7.5, color=DH_BLUE, va='bottom', ha='left',
            arrowprops=dict(arrowstyle='-', color=DH_BLUE,
                            lw=0.5, shrinkA=0, shrinkB=3))

ax.annotate('Leiria et al. (2025)',
            xy=(4.18, -0.10), xytext=(4.55, -0.45),
            fontsize=7.5, color=DH_BLUE, va='top', ha='center',
            arrowprops=dict(arrowstyle='-', color=DH_BLUE,
                            lw=0.5, shrinkA=0, shrinkB=3))

# -- Air-side HVAC studies (amber circles) ---
#   Pritoni (1,2) | Masdoua (3,3) | Kim & Kim (4,3)
hvac_xs = [1.0, 3.0, 4.0]
hvac_ys = [2.0, 3.0, 3.0]
ax.scatter(hvac_xs, hvac_ys,
           s=80, marker='o',
           facecolors=HVAC_OR_65, edgecolors=HVAC_ORANGE,
           linewidths=0.8, zorder=5)

ax.annotate('Pritoni et al. (2022)',
            xy=(1.0, 2.0), xytext=(1.15, 2.25),
            fontsize=7.5, color=HVAC_ORANGE, va='bottom', ha='left')

ax.annotate('Masdoua et al. (2025)',
            xy=(3.0, 3.0), xytext=(2.70, 3.30),
            fontsize=7.5, color=HVAC_ORANGE, va='bottom', ha='right')

ax.annotate('Kim & Kim (2025)',
            xy=(4.0, 3.0), xytext=(4.25, 3.30),
            fontsize=7.5, color=HVAC_ORANGE, va='bottom', ha='left')

# -- This work (filled navy star) ---
ax.scatter([5.0], [4.0],
           s=200, marker='*',
           facecolors=DH_BLUE, edgecolors=DH_BLUE_70,
           linewidths=0.9, zorder=6)
ax.annotate('This work',
            xy=(5.0, 4.0), xytext=(5.0, 4.18),
            fontsize=9, fontweight='bold', color=DH_BLUE,
            va='bottom', ha='center')

# -- Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=DH_BLUE_60, markeredgecolor=DH_BLUE,
           markeredgewidth=0.8, markersize=7,
           label='DH / hydronic systems'),
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=HVAC_OR_65, markeredgecolor=HVAC_ORANGE,
           markeredgewidth=0.8, markersize=7,
           label='Air-side HVAC'),
    Line2D([0], [0], marker='*', color='none',
           markerfacecolor=DH_BLUE, markeredgecolor=DH_BLUE_70,
           markeredgewidth=0.9, markersize=10,
           label='This work'),
]
legend = ax.legend(
    handles=legend_elements,
    loc='upper left', fontsize=8,
    framealpha=0.92, edgecolor='gray',
    handletextpad=0.5
)

# -- Layout & export ---
plt.tight_layout()

from pathlib import Path
OUT = Path(__file__).resolve().parent / 'output'
OUT.mkdir(parents=True, exist_ok=True)
pdf_path = str(OUT / 'lit_positioning.pdf')
png_path = str(OUT / 'lit_positioning.png')

plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
plt.savefig(png_path, dpi=600, bbox_inches='tight')
print(f"saved lit_positioning.pdf / .png -> {OUT}")
