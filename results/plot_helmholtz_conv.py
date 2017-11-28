import os
import sys
import pandas as pd
import seaborn

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

fig = plt.figure(figsize=(7, 5), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlabel("Mesh size $h=2^{-r}$\n(Number of cells)", fontsize=FONTSIZE)
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE)
ax.set_ylim([dfs.L2Errors.min()/2,
             dfs.L2Errors.max()*2])
ax.loglog()
ax.invert_xaxis()

# Gather all mesh parameters and compute h=1/r^2
r_values = [r[1] for r in dfs["Mesh"].drop_duplicates().items()]
h_array = [1.0/2**r for r in r_values]

# Gather number of mesh cells (directly coincide with mesh parameter
num_cells = [n[1] for n in dfs["NumCells"].drop_duplicates().items()]

colors = seaborn.cubehelix_palette(4, start=.5, rot=-.75, light=.65)
markers = iter(["o", "s", "^", "D"])
linestyles = iter(["solid", "dashed", "dashdot", "dotted"])
for group in groups:
    degree, df = group
    # All dfs have the same h array
    ax.plot(h_array, df.L2Errors,
            label="Degree %d" % degree,
            linewidth=LINEWIDTH,
            linestyle=next(linestyles),
            markersize=MARKERSIZE,
            marker=next(markers),
            color=colors[degree - 4],
            clip_on=False)

ax.set_xticks(h_array)
ax.set_xticklabels(["$2^{-%d}$\n(%d)" % (r, n)
                    for (r, n) in zip(r_values, num_cells)])

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.15),
                    bbox_transform=fig.transFigure,
                    ncol=4,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

annotation.slope_marker((0.375, 0.5), 5, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[0],
                                     'position': (0.2275, 0.2)},
                        poly_kwargs={'facecolor': colors[0]})
annotation.slope_marker((0.13, 1e-7), 8, ax=ax,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[3],
                                     'position': (0.2125, 3.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[3]})

seaborn.despine(fig)
plt.title("3D Helmholtz convergence", fontsize=FONTSIZE)
fig.savefig("helmholtz-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])
