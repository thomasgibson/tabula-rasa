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


p4_data = "helmholtz-results/helmholtz_conv-d-4.csv"
p5_data = "helmholtz-results/helmholtz_conv-d-5.csv"
p6_data = "helmholtz-results/helmholtz_conv-d-6.csv"
p7_data = "helmholtz-results/helmholtz_conv-d-7.csv"
data_set = [p4_data, p5_data, p6_data, p7_data]


for data in data_set:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

dfs = pd.concat(pd.read_csv(data) for data in data_set)
groups = dfs.groupby(["Degree"], as_index=False)

seaborn.set(style="ticks")

# Gather all mesh parameters and compute h=1/r^2
r_values = [r[1] for r in dfs["Mesh"].drop_duplicates().items()]
h_array = [1.0/2**r for r in r_values]

# Gather number of mesh cells (directly coincide with mesh parameter
num_cells = [n[1] for n in dfs["NumCells"].drop_duplicates().items()]

# FiveThirtyEight color scheme
colors = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
# colors = seaborn.cubehelix_palette(4, start=.5, rot=-.75, light=.65)
markers = iter(["o", "s", "^", "D"])
linestyles = iter(["solid", "dashed", "dashdot", "dotted"])
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
ax, = axes
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE+2)
ax.set_ylim([dfs.L2Errors.min()/2, dfs.L2Errors.max()*2])

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

for group in groups:
    degree, df = group

    ax.plot(h_array, df.L2Errors,
            label="Degree %d" % degree,
            linewidth=LINEWIDTH,
            linestyle=next(linestyles),
            markersize=MARKERSIZE,
            marker=next(markers),
            color=colors[degree - 4],
            clip_on=False)

annotation.slope_marker((0.375, 0.5), 5, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[0],
                                     'position': (0.2175, 0.2)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.13, 1e-7), 8, ax=ax,
                        text_kwargs={'fontsize': FONTSIZE-2,
                                     'color': colors[3],
                                     'position': (0.225, 3.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[3]})

ax.set_title("3D Helmholtz convergence", fontsize=FONTSIZE)

ax.set_xticklabels(["$2^{-%d}$" % r for r in r_values])

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

xlabel = fig.text(0.5, -0.1,
                  "Mesh size $h=2^{-r}$",
                  ha='center',
                  fontsize=FONTSIZE+2)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.175),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)
plt.title("3D Helmholtz convergence", fontsize=FONTSIZE)
fig.savefig("helmholtz-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
