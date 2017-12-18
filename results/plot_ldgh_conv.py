import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt
from mpltools import annotation


FONTSIZE = 12
# MARKERSIZE = 10
# LINEWIDTH = 3

tau_h_data = ["LDG-H/LDG-H-d%d-tau_order-h.csv" % i
              for i in range(1, 4)]
tau_hneg1_data = ["LDG-H/LDG-H-d%d-tau_order-hneg1.csv" % i
                  for i in range(1, 4)]

for data in tau_h_data + tau_hneg1_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

h_dfs = pd.concat(pd.read_csv(data) for data in tau_h_data)
hneg1_dfs = pd.concat(pd.read_csv(data) for data in tau_hneg1_data)
h_groups = h_dfs.groupby(["Degree"], as_index=False)
hneg1_groups = hneg1_dfs.groupby(["Degree"], as_index=False)

seaborn.set(style="ticks")

# Gather all mesh parameters and compute h=1/r^2
# (they are the same for both runs)
r_values = [r[1] for r in h_dfs["Mesh"].drop_duplicates().items()]
h_array = [1.0/2**r for r in r_values]

# Gather number of mesh cells (also same for both)
num_cells = [n[1] for n in h_dfs["NumCells"].drop_duplicates().items()]

# FiveThirtyEight color scheme
colors = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
# colors = seaborn.cubehelix_palette(3, start=.5, rot=-.75, light=.65)
markers = ["o", "s", "^"]

fig, axes = plt.subplots(2, 2, figsize=(7, 5), squeeze=False)
axes = axes.flatten()
ax1, ax2, ax3, ax4 = axes

ymin1 = 0.5e-10
ymin2 = 1.0e-8
ymax = 1.0
ax1.set_ylim(ymin1, ymax)
ax2.set_ylim(ymin1, ymax)
ax3.set_ylim(ymin2, ymax)
ax4.set_ylim(ymin2, ymax)

ax1.spines["left"].set_bounds(ymin1, ymax)
ax2.spines["left"].set_bounds(ymin1, ymax)
ax3.spines["left"].set_bounds(ymin2, ymax)
ax4.spines["left"].set_bounds(ymin2, ymax)

for ax in [ax1, ax2, ax3, ax4]:
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(h_array)
    ax.invert_xaxis()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()

ax1.set_ylabel("$L^2$ error", fontsize=FONTSIZE)
ax3.set_ylabel("$L^2$ error", fontsize=FONTSIZE)

for h_group, hneg1_group in zip(h_groups, hneg1_groups):
    h_degree, h_df = h_group
    hneg1_degree, hneg1_df = hneg1_group
    h_label = "Degree %d $\\left(\\tau = \\mathcal{O}(h)\\right)$" % h_degree
    hneg1_label = (
        "Degree %d $\\left(\\tau = "
        "\\mathcal{O}\\left(\\frac{1}{h}\\right)\\right)$"
        % hneg1_degree
    )

    ax1.plot(h_array, h_df.ScalarErrors,
             label=h_label,
             # linewidth=LINEWIDTH,
             linestyle="solid",
             # markersize=MARKERSIZE,
             marker=markers[h_degree - 1],
             color=colors[h_degree - 1],
             clip_on=False)

    ax1.plot(h_array, hneg1_df.ScalarErrors,
             label=hneg1_label,
             # linewidth=LINEWIDTH,
             linestyle="dotted",
             # markersize=MARKERSIZE,
             marker=markers[hneg1_degree - 1],
             color=colors[hneg1_degree - 1],
             clip_on=False)

    ax2.plot(h_array, h_df.PostProcessedScalarErrors,
             label=h_label,
             # linewidth=LINEWIDTH,
             linestyle="solid",
             # markersize=MARKERSIZE,
             marker=markers[h_degree - 1],
             color=colors[h_degree - 1],
             clip_on=False)

    ax2.plot(h_array, hneg1_df.PostProcessedScalarErrors,
             label=hneg1_label,
             # linewidth=LINEWIDTH,
             linestyle="dotted",
             # markersize=MARKERSIZE,
             marker=markers[hneg1_degree - 1],
             color=colors[hneg1_degree - 1],
             clip_on=False)

    ax3.plot(h_array, h_df.FluxErrors,
             label=h_label,
             # linewidth=LINEWIDTH,
             linestyle="solid",
             # markersize=MARKERSIZE,
             marker=markers[h_degree - 1],
             color=colors[h_degree - 1],
             clip_on=False)

    ax3.plot(h_array, hneg1_df.FluxErrors,
             label=hneg1_label,
             # linewidth=LINEWIDTH,
             linestyle="dotted",
             # markersize=MARKERSIZE,
             marker=markers[hneg1_degree - 1],
             color=colors[hneg1_degree - 1],
             clip_on=False)

    ax4.plot(h_array, h_df.PostProcessedFluxErrors,
             label=h_label,
             # linewidth=LINEWIDTH,
             linestyle="solid",
             # markersize=MARKERSIZE,
             marker=markers[h_degree - 1],
             color=colors[h_degree - 1],
             clip_on=False)

    ax4.plot(h_array, hneg1_df.PostProcessedFluxErrors,
             label=hneg1_label,
             # linewidth=LINEWIDTH,
             linestyle="dotted",
             # markersize=MARKERSIZE,
             marker=markers[hneg1_degree - 1],
             color=colors[hneg1_degree - 1],
             clip_on=False)

# Slope markers for scalar
annotation.slope_marker((0.12, 0.2), 1, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.070, 0.15)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.035, 2e-3), 2, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[1],
                                     'position': (0.02, 1e-3)},
                        poly_kwargs={'facecolor': colors[1]})

annotation.slope_marker((0.02, 2e-7), 3, ax=ax1,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.035, 2.75e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[2]})

# Slope markers for post-processed scalar
annotation.slope_marker((0.025, 1e-5), 3, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.014, 0.55e-5)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.025, 0.75e-7), 4, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[1],
                                     'position': (0.0125, 0.275e-7)},
                        poly_kwargs={'facecolor': colors[1]})

annotation.slope_marker((0.035, 0.25e-9), 5, ax=ax2,
                        invert=True,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.065, 0.55e-9)},
                        poly_kwargs={'facecolor': colors[2]})

# Slope markers for flux
annotation.slope_marker((0.025, 2e-3), 2, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.014, 1e-3)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.025, 2e-5), 3, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[1],
                                     'position': (0.014, 1e-5)},
                        poly_kwargs={'facecolor': colors[1]})

annotation.slope_marker((0.035, 1e-7), 4, ax=ax3,
                        invert=True,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.065, 2e-7)},
                        poly_kwargs={'facecolor': colors[2]})

# Slope markers for post-processed flux
annotation.slope_marker((0.025, 2e-3), 2, ax=ax4,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.014, 1e-3)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.025, 1e-5), 3, ax=ax4,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[1],
                                     'position': (0.014, 0.5e-5)},
                        poly_kwargs={'facecolor': colors[1]})

annotation.slope_marker((0.035, 0.5e-7), 4, ax=ax4,
                        invert=True,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.065, 1e-7)},
                        poly_kwargs={'facecolor': colors[2]})

ax1.set_title("Scalar", fontsize=FONTSIZE)
ax2.set_title("Post-processed scalar", fontsize=FONTSIZE)
ax3.set_title("Flux", fontsize=FONTSIZE)
ax4.set_title("Post-processed flux", fontsize=FONTSIZE)

for ax in [ax2, ax4]:
    ax.set_yticklabels([])
    ax.tick_params(direction="inout", which="both", axis="y")

for ax in [ax1, ax2]:
    ax.set_xticklabels([])
    ax.tick_params(direction="inout", which="both", axis="x")

for ax in [ax3, ax4]:
    ax.set_xticklabels(["$2^{-%d}$" % r
                        for r in r_values])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)

for ax in [ax1, ax3]:
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE)

fig.subplots_adjust(wspace=0.1, hspace=0.25)
xlabel = fig.text(0.5, -0.0275,
                  "Mesh size $h=2^{-r}$",
                  ha='center',
                  fontsize=FONTSIZE)

handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.075),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)
fig.savefig("LDGH-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
