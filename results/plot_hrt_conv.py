import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt
from mpltools import annotation


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3

rt_data = ["hybrid-mixed/H-RT-degree-%d.csv" % i for i in range(0, 5)]

for data in rt_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

dfs = pd.concat(pd.read_csv(data) for data in rt_data)
groups = dfs.groupby(["Degree"], as_index=False)

seaborn.set(style="ticks")

# Gather all mesh parameters and compute h=1/r^2
r_values = [r[1] for r in dfs["Mesh"].drop_duplicates().items()]
h_array = [1.0/2**r for r in r_values]

# Gather number of mesh cells
num_cells = [n[1] for n in dfs["NumCells"].drop_duplicates().items()]

colors = seaborn.cubehelix_palette(5, start=.5, rot=-.75, light=.65)
markers = ["o", "s", "^", "D", "v"]
linestyles = ["solid", "dashed", "dashdot", "dotted", "solid"]

fig, (axes,) = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
(ax1, ax2, ax3) = axes
ymin = 1.0e-12
ymax = 1.5
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)

ax1.spines["left"].set_bounds(ymin, ymax)
ax2.spines["left"].set_bounds(ymin, ymax)
ax3.spines["left"].set_bounds(ymin, ymax)

for ax in [ax1, ax2, ax3]:
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

for group in groups:
    degree, df = group
    label = "RT order %d" % degree

    ax1.plot(h_array, df.ScalarErrors,
             label=label,
             linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

    ax2.plot(h_array, df.PostProcessedScalarErrors,
             label=label,
             linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

    ax3.plot(h_array, df.FluxErrors,
             label=label,
             linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

# Slope markers for scalar
annotation.slope_marker((0.1, 0.15), 1, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[0],
                                     'position': (0.062, 0.125)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.0625, 0.0001), 3, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.04, 0.000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-7), 5, ax=ax1,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

# Slope markers for post-processed scalar
annotation.slope_marker((0.1, 0.015), 2, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[0],
                                     'position': (0.062, 0.01)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.05, 0.000001), 4, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.0325, 0.00000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-8), 6, ax=ax2,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-8)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

# Slope markers for flux
annotation.slope_marker((0.1, 0.50), 1, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[0],
                                     'position': (0.062, 0.475)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.05, 0.0001), 3, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.0325, 0.000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-7), 5, ax=ax3,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

ax1.set_title("Scalar", fontsize=FONTSIZE)
ax2.set_title("Post-processed scalar", fontsize=FONTSIZE)
ax3.set_title("Flux", fontsize=FONTSIZE)

for ax in [ax1, ax2, ax3]:
    ax.set_xticklabels(["$2^{-%d}$\n(%d)" % (r, n)
                        for (r, n) in zip(r_values, num_cells)])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)

for ax in [ax2, ax3]:
    ax.set_yticklabels([])
    ax.tick_params(direction="inout", which="both", axis="y")

for tick in ax1.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

fig.subplots_adjust(wspace=0.075)
xlabel = fig.text(0.5, -0.2,
                  "Mesh size $h=2^{-r}$\n(Number of cells)",
                  ha='center',
                  fontsize=FONTSIZE)

handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=5,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)
fig.savefig("HRT-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
