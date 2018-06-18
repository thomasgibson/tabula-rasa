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
              (64, 1), (64, 2), (64, 3),
              (128, 1)]
hdg_data = ["HDG_CG_comp/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(4, 2), (4, 3), (4, 4),
             (8, 2), (8, 3), (8, 4),
             (16, 2), (16, 3), (16, 4),
             (32, 2), (32, 3), (32, 4),
             (64, 2), (64, 3), (64, 4),
             (128, 2)]
cg_data = ["HDG_CG_comp/CG_data_N%d_deg%d.csv" % param
           for param in cg_params]

for d in hdg_data + cg_data:
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)

markers = ["o", "s", "^", "v", ">", "<", "D", "p", "h", "*"]
colors = list(seaborn.color_palette(n_colors=3))
linestyles = ["solid", "dashdot"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
ax, = axes
ax.set_ylabel("$L^2$ error", fontsize=FONTSIZE+2)
ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xscale('log')
ax.set_yscale('log')

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

for group in cg_groups:
    degree, df = group

    if degree == 2:
        solve_times = df.SNESSolve.values
        errors = df.true_err.values
        label = "$CG_%d(p_h)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 2],
                marker=markers[degree - 2],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="solid",
                clip_on=False)

    if degree == 3:
        solve_times = df.SNESSolve.values
        errors = df.true_err.values
        label = "$CG_%d(p_h)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 2],
                marker=markers[degree - 2],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="solid",
                clip_on=False)

    if degree == 4:
        solve_times = df.SNESSolve.values
        errors = df.true_err.values
        label = "$CG_%d(p_h)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 2],
                marker=markers[degree - 2],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="solid",
                clip_on=False)

for group in hdg_groups:
    degree, df = group

    if degree == 1:
        solve_times = df.SNESSolve.values + df.HDGPPTime.values
        errors = df.ErrorPP.values
        label = "$HDG_%d(p_h^\\star)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 1],
                marker=markers[degree - 1],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="dotted",
                clip_on=False)

    if degree == 2:
        solve_times = df.SNESSolve.values + df.HDGPPTime.values
        errors = df.ErrorPP.values
        label = "$HDG_%d(p_h^\\star)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 1],
                marker=markers[degree - 1],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="dotted",
                clip_on=False)

    if degree == 3:
        solve_times = df.SNESSolve.values + df.HDGPPTime.values
        errors = df.ErrorPP.values
        label = "$HDG_%d(p_h^\\star)$" % degree
        ax.plot(solve_times, errors,
                label=label,
                color=colors[degree - 1],
                marker=markers[degree - 1],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                linestyle="dotted",
                clip_on=False)


for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

ax.grid(b=True, which='major', linestyle='-.')

xlabel = fig.text(0.5, -0.1,
                  "Execution time (s)",
                  ha='center',
                  fontsize=FONTSIZE+2)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=2,
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
            bbox_extra_artists=[xlabel, legend])
