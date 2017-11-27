import os
import sys
import pandas as pd
import seaborn

from matplotlib import pyplot as plt
from mpltools import annotation


FONTSIZE = 16
MARKERSIZE = 12
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


# Gather number of mesh cells (directly coincide with mesh parameter
num_cells = [n[1] for n in dfs["NumCells"].drop_duplicates().items()]

colors = seaborn.color_palette("cubehelix", n_colors=8)
markers = ["o", "s", "^", "D", "v"]
linestyles = ["solid", "dashed", "dashdot", "dotted", "solid"]

# scaling factor for y-axis
scale = 2.0
# Scalar errors
fig = plt.figure(figsize=(7, 6), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlabel("Mesh size $h=2^{-r}$\n"
              "(Number of cells)\n", fontsize=FONTSIZE)
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE)
ax.set_ylim([dfs.ScalarErrors.min()/scale,
             dfs.ScalarErrors.max()*scale])
ax.loglog()
ax.invert_xaxis()
for group in groups:
    degree, df = group
    label = "H-RT%d-DG%d" % (degree, degree)
    ax.plot(h_array, df.ScalarErrors,
            label=label,
            linewidth=LINEWIDTH,
            linestyle=linestyles[degree],
            markersize=MARKERSIZE,
            marker=markers[degree],
            color=colors[degree],
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
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

annotation.slope_marker((0.0625, 0.0001), 3, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.04, 0.000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-7), 5, ax=ax,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

seaborn.despine(fig)
plt.title("H-RT scalar convergence", fontsize=FONTSIZE)
fig.savefig("HRT-scalar-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])


# Flux errors
fig = plt.figure(figsize=(7, 6), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlabel("Mesh size $h=2^{-r}$\n"
              "(Number of cells)\n", fontsize=FONTSIZE)
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE)
ax.set_ylim([dfs.FluxErrors.min()/scale,
             dfs.FluxErrors.max()*scale])
ax.loglog()
ax.invert_xaxis()
for group in groups:
    degree, df = group
    label = "H-RT%d-DG%d" % (degree, degree)
    ax.plot(h_array, df.FluxErrors,
            label=label,
            linewidth=LINEWIDTH,
            linestyle=linestyles[degree],
            markersize=MARKERSIZE,
            marker=markers[degree],
            color=colors[degree],
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
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

annotation.slope_marker((0.05, 0.0001), 3, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.0325, 0.000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-7), 5, ax=ax,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-7)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

seaborn.despine(fig)
plt.title("H-RT flux convergence", fontsize=FONTSIZE)
fig.savefig("HRT-flux-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])


fig = plt.figure(figsize=(7, 6), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlabel("Mesh size $h=2^{-r}$\n"
              "(Number of cells)\n", fontsize=FONTSIZE)
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE)
ax.set_ylim([dfs.PostProcessedScalarErrors.min()/scale,
             dfs.PostProcessedScalarErrors.max()*scale])
ax.loglog()
ax.invert_xaxis()
for group in groups:
    degree, df = group
    label = "H-RT%d-DG%d" % (degree, degree)
    ax.plot(h_array, df.PostProcessedScalarErrors,
            label=label,
            linewidth=LINEWIDTH,
            linestyle=linestyles[degree],
            markersize=MARKERSIZE,
            marker=markers[degree],
            color=colors[degree],
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
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=2,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

annotation.slope_marker((0.1, 0.015), 2, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[0],
                                     'position': (0.062, 0.01)},
                        poly_kwargs={'facecolor': colors[0]})

annotation.slope_marker((0.05, 0.000001), 4, ax=ax,
                        invert=False,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[2],
                                     'position': (0.0325, 0.00000065)},
                        poly_kwargs={'facecolor': colors[2]})
annotation.slope_marker((0.13, 1e-8), 6, ax=ax,
                        text_kwargs={'fontsize': FONTSIZE,
                                     'color': colors[4],
                                     'position': (0.2000, 2.25e-8)},
                        invert=True, poly_kwargs={'facecolor': colors[4]})

seaborn.despine(fig)
plt.title("H-RT post-processed scalar convergence", fontsize=FONTSIZE)
fig.savefig("HRT-pp-scalar-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])
