import os
import pandas as pd
import seaborn

from matplotlib import pyplot as plt


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3


hdg_params = [(4, 1), (4, 2), (4, 3),
              (8, 1), (8, 2), (8, 3),
              (16, 1), (16, 2), (16, 3),
              (32, 1), (32, 2), (32, 3),
              (64, 1), (64, 2), (64, 3)]
hdg_data = ["HDG_CG_comp/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(4, 2), (4, 3), (4, 4),
             (8, 2), (8, 3), (8, 4),
             (16, 2), (16, 3), (16, 4),
             (32, 2), (32, 3), (32, 4),
             (64, 2), (64, 3), (64, 4)]
cg_data = ["HDG_CG_comp/CG_data_N%d_deg%d.csv" % param
           for param in cg_params]

for d in hdg_data + cg_data:
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)

# FiveThirtyEight color scheme
colors = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
# colors = seaborn.cubehelix_palette(4, start=.5, rot=-.75, light=.65)
markers = ["o", "s", "^", "D"]
linestyles = ["solid", "dashdot"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
(ax1, ax2) = axes
ax1.set_ylabel("$L^2$ error", fontsize=FONTSIZE+2)
ymin = 1.0e-10
ymax = 1
for ax in [ax1, ax2]:
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

label_dct = {}

for group in cg_groups:
    degree, df = group

    if degree == 2:
        solve_times = df.KSPSolve.values
        errors = df.true_err.values
        label = "$CG_%d(p_h)$" % degree
        ax1.plot(solve_times, errors,
                 label=label,
                 color=colors[degree - 2],
                 marker=markers[degree - 2],
                 linestyle="solid",
                 clip_on=False)
        label_dct[label] = degree

    if degree == 3:
        solve_times = df.KSPSolve.values
        errors = df.true_err.values
        label = "$CG_%d(p_h)$" % degree
        ax2.plot(solve_times, errors,
                 label=label,
                 color=colors[degree - 2],
                 marker=markers[degree - 2],
                 linestyle="solid",
                 clip_on=False)
        label_dct[label] = degree

for group in hdg_groups:
    degree, df = group

    if degree == 2:
        solve_times = df.HDGTotal.values
        errors = df.true_err_u.values
        label = "$HDG_%d(p_h)$" % degree
        ax1.plot(solve_times, errors,
                 label=label,
                 color=colors[degree - 2],
                 marker=markers[degree - 2],
                 linestyle="dotted",
                 clip_on=False)
        label_dct[label] = degree + 2

    if degree == 3:
        solve_times = df.HDGTotal.values
        errors = df.true_err_u.values
        label = "$HDG_%d(p_h)$" % degree
        ax2.plot(solve_times, errors,
                 label=label,
                 color=colors[degree - 2],
                 marker=markers[degree - 2],
                 linestyle="dotted",
                 clip_on=False)
        label_dct[label] = degree + 2

for group in hdg_groups:
    degree, df = group

    if degree == 2:
        solve_times = df.HDGTotal.values
        errors = df.ErrorPP.values
        label = "$HDG_%d(p_h^\\star)$" % degree
        ax1.plot(solve_times, errors,
                 label=label,
                 color=colors[degree],
                 marker=markers[degree],
                 linestyle="dashdot",
                 clip_on=False)
        label_dct[label] = degree + 4

    if degree == 3:
        solve_times = df.HDGTotal.values
        errors = df.ErrorPP.values
        label = "$HDG_%d(p_h^\\star)$" % degree
        ax2.plot(solve_times, errors,
                 label=label,
                 color=colors[degree],
                 marker=markers[degree],
                 linestyle="dashdot",
                 clip_on=False)
        label_dct[label] = degree + 4


for ax in [ax1, ax2]:
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)

    if ax is not ax2:
        for tick in ax.get_yticklabels():
            tick.set_fontsize(FONTSIZE)
    else:
        ax.set_yticklabels([])
        ax.tick_params(direction="inout", which="both", axis="y")

xlabel = fig.text(0.5, -0.1,
                  "Time (s)",
                  ha='center',
                  fontsize=FONTSIZE+2)

title = fig.text(0.5, 1,
                 "CG vs HDG error budget",
                 ha="center",
                 fontsize=FONTSIZE+2)

handles = []
labels = []
for ax in [ax1, ax2]:
    handle, label = ax.get_legend_handles_labels()
    handles.extend(handle)
    labels.extend(label)

labels, handles = zip(*sorted(zip(labels, handles),
                              key=lambda t: label_dct[t[0]]))

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.3),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=3,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)
fig.savefig("budget.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, title, legend])
