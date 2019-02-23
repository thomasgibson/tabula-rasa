import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt
from mpltools import annotation


FONTSIZE = 12

rt_data = ["results/H-RT-degree-%d.csv" % i for i in range(0, 4)]

for data in rt_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

dfs = pd.concat(pd.read_csv(data) for data in rt_data)
groups = dfs.groupby(["Degree"], as_index=False)


# Gather all mesh parameters and compute h=1/r^2
r_values = [r[1] for r in dfs["Mesh"].drop_duplicates().items()]
h_array = [1.0/2**r for r in r_values]

# Gather number of mesh cells
num_cells = [n[1] for n in dfs["NumCells"].drop_duplicates().items()]

markers = ["o", "s", "^", "v", ">", "<", "D", "p", "h", "*"]
colors = seaborn.color_palette(n_colors=4)
linestyles = ["solid", "dashed", "dashdot", "dotted", "solid"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 3, figsize=(9.5, 3), squeeze=False)
(ax1, ax2, ax3) = axes
ymin = 1.0e-13
ymax = 2.0
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)

ax1.spines["left"].set_bounds(ymin, ymax)
ax2.spines["left"].set_bounds(ymin, ymax)
ax3.spines["left"].set_bounds(ymin, ymax)

for ax in [ax1, ax2, ax3]:
    # ax.spines["left"].set_position(("outward", 10))
    # ax.spines["bottom"].set_position(("outward", 10))
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(h_array)
    ax.invert_xaxis()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()


ax1.set_ylabel("$L^2$ error", fontsize=FONTSIZE+2)

for group in groups:
    degree, df = group
    label = "RT order %d" % degree

    ax1.plot(h_array, df.ScalarErrors,
             label=label,
             # linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             # markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

    ax2.plot(h_array, df.PostProcessedScalarErrors,
             label=label,
             # linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             # markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

    ax3.plot(h_array, df.FluxErrors,
             label=label,
             # linewidth=LINEWIDTH,
             linestyle=linestyles[degree],
             # markersize=MARKERSIZE,
             marker=markers[degree],
             color=colors[degree],
             clip_on=False)

# Slope markers for scalar
annotation.slope_marker((0.1, 0.15), 1, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.05, 0.125)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.0625, 0.0001), 3, ax=ax1,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.0325, 0.0000525)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.0575, 1e-7), 4, ax=ax1,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[3],
                                     'position': (0.11, 1.75e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[3]})

# Slope markers for post-processed scalar
annotation.slope_marker((0.1, 0.015), 2, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.05, 0.01)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.05, 0.0000015), 4, ax=ax2,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.0265, 0.0000006)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.075, 1e-8), 5, ax=ax2,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[3],
                                     'position': (0.15, 2.0e-8)},
                        invert=True, poly_kwargs={'facecolor': colors[3]})

# Slope markers for flux
annotation.slope_marker((0.1, 0.50), 1, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.05, 0.475)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.05, 0.0001), 3, ax=ax3,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[2],
                                     'position': (0.0265, 0.00005)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.0525, 1e-7), 4, ax=ax3,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[3],
                                     'position': (0.1, 1.75e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[3]})

ax1.set_title("Scalar", fontsize=FONTSIZE)
ax2.set_title("Post-processed scalar", fontsize=FONTSIZE)
ax3.set_title("Flux", fontsize=FONTSIZE)

for ax in [ax1, ax2, ax3]:
    ax.set_xticklabels(["$2^{-%d}$" % r
                        for r in r_values])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE-2)

for ax in [ax2, ax3]:
    ax.set_yticklabels([])
    ax.tick_params(direction="inout", which="both", axis="y")

for tick in ax1.get_yticklabels():
    tick.set_fontsize(FONTSIZE-2)

for ax in axes:
    ax.grid(b=True, which='major', linestyle='-.')

fig.subplots_adjust(wspace=0.075)
xlabel = fig.text(0.5, -0.075,
                  "Mesh size $2^{-r}$",
                  ha='center',
                  fontsize=FONTSIZE)

handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.125),
                    bbox_transform=fig.transFigure,
                    ncol=5,
                    handlelength=1.5,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

# seaborn.despine(fig)
fig.savefig("HRT-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
