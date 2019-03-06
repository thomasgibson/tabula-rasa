import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt


FONTSIZE = 14
MARKERSIZE = 10
LINEWIDTH = 3


cfl_range = [2, 4, 6, 8, 10, 12, 16, 32, 64]
lo_rtcf_data = ["results/hybrid_RTCF1_GW_ref6_nlayers85_CFL%d.csv" %
                cfl for cfl in cfl_range]
nlo_rtcf_data = ["results/hybrid_RTCF2_GW_ref5_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]
nlo_bdfm_data = ["results/hybrid_BDFM2_GW_ref4_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]

for data in lo_rtcf_data + nlo_rtcf_data + nlo_bdfm_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


lo_rtcf_dfs = pd.concat(pd.read_csv(data) for data in lo_rtcf_data)
nlo_rtcf_dfs = pd.concat(pd.read_csv(data) for data in nlo_rtcf_data)
nlo_bdfm_dfs = pd.concat(pd.read_csv(data) for data in nlo_bdfm_data)

lo_rtcf_groups = lo_rtcf_dfs.groupby(["CFL"], as_index=False)
nlo_rtcf_groups = nlo_rtcf_dfs.groupby(["CFL"], as_index=False)
nlo_bdfm_groups = nlo_bdfm_dfs.groupby(["CFL"], as_index=False)

colors = seaborn.color_palette(n_colors=2)
linestyles = ["solid", "dotted"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 1, figsize=(7, 6), squeeze=False)
ax, = axes
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_ylabel("DOFs per second", fontsize=FONTSIZE+2)
ax.set_xlim(1, 128)
ax.set_xscale('log')
# ax.set_ylim(0, 30)
ax.set_xticks(cfl_range)
ax.set_xticklabels(cfl_range)


lo_rtcf_cfls = []
lo_rtcf_dps = []
for group in lo_rtcf_groups:
    cfl, df = group
    lo_rtcf_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    dofs = df.up_dofs.values[0]
    lo_rtcf_dps.append(dofs/time)

nlo_rtcf_cfls = []
nlo_rtcf_dps = []
for group in nlo_rtcf_groups:
    cfl, df = group
    nlo_rtcf_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    dofs = df.up_dofs.values[0]
    nlo_rtcf_dps.append(dofs/time)

nlo_bdfm_cfls = []
nlo_bdfm_dps = []
for group in nlo_bdfm_groups:
    cfl, df = group
    nlo_bdfm_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    dofs = df.up_dofs.values[0]
    nlo_bdfm_dps.append(dofs/time)


ax.plot(lo_rtcf_cfls, lo_rtcf_dps,
        label="$RTCF_1$",
        color=colors[0],
        marker="o",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[0],
        clip_on=False)

ax.plot(nlo_rtcf_cfls, nlo_rtcf_dps,
        label="$RTCF_2$",
        color=colors[0],
        marker="o",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[1],
        clip_on=False)

ax.plot(nlo_bdfm_cfls, nlo_bdfm_dps,
        label="$BDFM_2$",
        color=colors[1],
        marker="^",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[0],
        clip_on=False)


for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

ax.grid(b=True, which='major', linestyle='-.')

xlabel = fig.text(0.5, 0,
                  "Horizontal CFL number",
                  ha='center',
                  fontsize=FONTSIZE+2)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=1.5,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)


fig.savefig("hybrid_dps_vs_cfl.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
