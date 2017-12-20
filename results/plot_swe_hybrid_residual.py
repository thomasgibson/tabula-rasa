import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt


FONTSIZE = 12
MARKERSIZE = 10
LINEWIDTH = 3


diagnostic_data = ["williamson-test-case-5/hybrid_diagnostics_W5_ref7.csv",
                   "williamson-test-case-5/diagnostics_W5_ref7.csv"]


for data in diagnostic_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

df = pd.read_csv(diagnostic_data[0])
df1 = pd.read_csv(diagnostic_data[1])

seaborn.set(style="ticks")

colors = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']

fig, (axes,) = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
ax, = axes

ytitle = "$\\frac{\parallel b-Ax^*\parallel_2}{\parallel b\parallel_2}$"
ax.set_ylabel(ytitle, fontsize=FONTSIZE+2)

ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_yscale('log')
ax.set_xticks(df["SimTime"].values)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

t_array = list(df["SimTime"].values)
hybrid_r = list(df["ResidualReductions"].values)
sc_r = list(df1["ResidualReductions"].values)
# Remove t=0 values (they're just 1)
# t_array.pop(0)
# hybrid_r.pop(0)
# sc_r.pop(0)
ax.set_ylim([min(hybrid_r)/2, max(sc_r)])

ax.plot(t_array, hybrid_r,
        label="ksp: preonly\npc: hybridization",
        linewidth=LINEWIDTH,
        linestyle="solid",
        markersize=MARKERSIZE,
        marker="o",
        color=colors[0],
        clip_on=False)

ax.plot(t_array, sc_r,
        label="ksp: gmres\npc: approx-sc",
        linewidth=LINEWIDTH,
        linestyle="solid",
        markersize=MARKERSIZE,
        marker="d",
        color=colors[1],
        clip_on=False)

ax.plot(df["SimTime"].values, [1e-8]*len(df["SimTime"].values),
        label="1.0e-8",
        linewidth=LINEWIDTH,
        linestyle="dotted",
        markersize=MARKERSIZE,
        marker=None,
        color=colors[2],
        clip_on=False)

ax.set_xticklabels([t for t in df["SimTime"].values], rotation=45)

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE-4)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE-2)

xlabel = fig.text(0.5, -0.125,
                  "Time (s): $t_{n+1} = t_{n} + \Delta t$",
                  ha='center',
                  fontsize=FONTSIZE)
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

seaborn.despine(fig)
plt.title("Residual reductions (avg) over 20 timesteps", fontsize=FONTSIZE)
fig.savefig("hybrid_w5_reductions.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
