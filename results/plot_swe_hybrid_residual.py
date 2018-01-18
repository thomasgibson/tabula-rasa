import os
import sys
import pandas as pd
import numpy as np
import seaborn
import matplotlib

from matplotlib import pyplot as plt


FONTSIZE = 12
MARKERSIZE = 8
LINEWIDTH = 2.5


diagnostic_data = ["williamson-test-case-5/hybrid_profile_W5_ref7.csv",
                   "williamson-test-case-5/profile_W5_ref7.csv"]


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
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

hybrid_r = list(df["ResidualReductions"].values)
sc_r = list(df1["ResidualReductions"].values)

hybrid_groups = df.groupby(["SimTime"], as_index=False)
app_sc_groups = df1.groupby(["SimTime"], as_index=False)

t_array = []
avg_hybrid_r = []
avg_app_sc_r = []
max_hybrid_r = []
min_hybrid_r = []
max_app_sc_r = []
min_app_sc_r = []

for group in hybrid_groups:
    t, df = group
    reductions = list(df["ResidualReductions"].values)
    avg = sum(reductions)/len(reductions)
    max_r = max(reductions)
    min_r = min(reductions)

    t_array.append(t)
    avg_hybrid_r.append(avg)
    max_hybrid_r.append(max_r)
    min_hybrid_r.append(min_r)

for group in app_sc_groups:
    _, df = group
    reductions = list(df["ResidualReductions"].values)
    avg = sum(reductions)/len(reductions)
    max_r = max(reductions)
    min_r = min(reductions)

    avg_app_sc_r.append(avg)
    max_app_sc_r.append(max_r)
    min_app_sc_r.append(min_r)

ax.set_ylim([min(min_hybrid_r), max(max_app_sc_r)])

avg_hybrid_r = np.array(avg_hybrid_r)
avg_app_sc_r = np.array(avg_app_sc_r)
max_hybrid_r = np.array(max_hybrid_r)
max_app_sc_r = np.array(max_app_sc_r)
min_hybrid_r = np.array(min_hybrid_r)
min_app_sc_r = np.array(min_app_sc_r)

ax.plot(t_array, avg_hybrid_r,
        label="ksp: preonly\npc: hybridization",
        linewidth=LINEWIDTH,
        linestyle="solid",
        markersize=MARKERSIZE,
        marker="o",
        color=colors[0],
        clip_on=False)

ax.fill_between(t_array, min_hybrid_r, max_hybrid_r,
                alpha=0.2, facecolor=colors[0],
                edgecolor=colors[0], linewidth=1,
                antialiased=True)

ax.plot(t_array, avg_app_sc_r,
        label="ksp: gmres\npc: approx-sc",
        linewidth=LINEWIDTH,
        linestyle="solid",
        markersize=MARKERSIZE,
        marker="d",
        color=colors[1],
        clip_on=False)

ax.fill_between(t_array, min_app_sc_r, max_app_sc_r,
                alpha=0.2, facecolor=colors[1],
                edgecolor=colors[1], linewidth=1,
                antialiased=True)

t_array.insert(0, 0.0)
ax.set_xticks(t_array)
ax.set_xticklabels(t_array, rotation=45)

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
                    ncol=2,
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
