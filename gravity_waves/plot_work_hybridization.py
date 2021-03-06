import os
import sys
import pandas as pd
import seaborn

from matplotlib import pyplot as plt


FONTSIZE = 14
MARKERSIZE = 10
LINEWIDTH = 3


cfl_range = [2, 4, 6, 8, 10, 12, 16, 24, 32, 64]

lo_rt_data = ["results/hybrid_RT1_GW_ref6_nlayers85_CFL%d.csv" %
              cfl for cfl in cfl_range]
nlo_rt_data = ["results/hybrid_RT2_GW_ref4_nlayers85_CFL%d.csv" %
               cfl for cfl in cfl_range]
lo_rtcf_data = ["results/hybrid_RTCF1_GW_ref7_nlayers85_CFL%d.csv" %
                cfl for cfl in cfl_range]
nlo_rtcf_data = ["results/hybrid_RTCF2_GW_ref5_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]
nlo_bdfm_data = ["results/hybrid_BDFM2_GW_ref4_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]

for data in (lo_rtcf_data + nlo_rtcf_data +
             lo_rt_data + nlo_rt_data + nlo_bdfm_data):
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


lo_rt_dfs = pd.concat(pd.read_csv(data) for data in lo_rt_data)
nlo_rt_dfs = pd.concat(pd.read_csv(data) for data in nlo_rt_data)
lo_rtcf_dfs = pd.concat(pd.read_csv(data) for data in lo_rtcf_data)
nlo_rtcf_dfs = pd.concat(pd.read_csv(data) for data in nlo_rtcf_data)
nlo_bdfm_dfs = pd.concat(pd.read_csv(data) for data in nlo_bdfm_data)

lo_rt_groups = lo_rt_dfs.groupby(["CFL"], as_index=False)
nlo_rt_groups = nlo_rt_dfs.groupby(["CFL"], as_index=False)
lo_rtcf_groups = lo_rtcf_dfs.groupby(["CFL"], as_index=False)
nlo_rtcf_groups = nlo_rtcf_dfs.groupby(["CFL"], as_index=False)
nlo_bdfm_groups = nlo_bdfm_dfs.groupby(["CFL"], as_index=False)

colors = seaborn.color_palette(n_colors=3)
linestyles = ["solid", "dotted"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 1, figsize=(7, 6), squeeze=False)
ax, = axes
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_ylabel("Execution time [s] / Execution time (CFL 2) [s]", fontsize=FONTSIZE+2)
ax.set_xlim(0, 64)
ax.set_ylim(0.75, 2.0)
ax.set_xticks(cfl_range)
ax.set_xticklabels(cfl_range)
ax.axvline(10, color='k')
ax.axvline(2, color='k')
ax.axvspan(2, 10, ymin=0, ymax=2.0, alpha=0.5, color='gray')


lo_rt_cfls = []
lo_rt_time_ratios = []
lo_rt_cfl_to_time = {}
for group in lo_rt_groups:
    cfl, df = group
    lo_rt_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    lo_rt_cfl_to_time[cfl] = time
    lo_rt_time_ratios.append(time / lo_rt_cfl_to_time[2])

nlo_rt_cfls = []
nlo_rt_time_ratios = []
nlo_rt_cfl_to_time = {}
for group in nlo_rt_groups:
    cfl, df = group
    nlo_rt_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    nlo_rt_cfl_to_time[cfl] = time
    nlo_rt_time_ratios.append(time / nlo_rt_cfl_to_time[2])

lo_rtcf_cfls = []
lo_rtcf_time_ratios = []
lo_rtcf_cfl_to_time = {}
for group in lo_rtcf_groups:
    cfl, df = group
    lo_rtcf_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    lo_rtcf_cfl_to_time[cfl] = time
    lo_rtcf_time_ratios.append(time / lo_rtcf_cfl_to_time[2])

nlo_rtcf_cfls = []
nlo_rtcf_time_ratios = []
nlo_rtcf_cfl_to_time = {}
for group in nlo_rtcf_groups:
    cfl, df = group
    nlo_rtcf_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    nlo_rtcf_cfl_to_time[cfl] = time
    nlo_rtcf_time_ratios.append(time / nlo_rtcf_cfl_to_time[2])

nlo_bdfm_cfls = []
nlo_bdfm_time_ratios = []
nlo_bdfm_cfl_to_time = {}
for group in nlo_bdfm_groups:
    cfl, df = group
    nlo_bdfm_cfls.append(cfl)
    time = df.HybridFullSolveTime.values[0]
    nlo_bdfm_cfl_to_time[cfl] = time
    nlo_bdfm_time_ratios.append(time / nlo_bdfm_cfl_to_time[2])


ax.plot(lo_rt_cfls, lo_rt_time_ratios,
        label="$RT_1$",
        color=colors[2],
        marker="s",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[0],
        clip_on=False)

ax.plot(nlo_rt_cfls, nlo_rt_time_ratios,
        label="$RT_2$",
        color=colors[2],
        marker="s",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[1],
        clip_on=False)


ax.plot(lo_rtcf_cfls, lo_rtcf_time_ratios,
        label="$RTCF_1$",
        color=colors[0],
        marker="o",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[0],
        clip_on=False)

ax.plot(nlo_rtcf_cfls, nlo_rtcf_time_ratios,
        label="$RTCF_2$",
        color=colors[0],
        marker="o",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[1],
        clip_on=False)

ax.plot(nlo_bdfm_cfls, nlo_bdfm_time_ratios,
        label="$BDFM_2$",
        color=colors[1],
        marker="^",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle=linestyles[1],
        clip_on=False)


for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

ax.grid(b=True, which='major', linestyle='-.')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

xlabel = fig.text(0.5, 0,
                  "Horizontal CFL number",
                  ha='center',
                  fontsize=FONTSIZE+2)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.4, 0.875),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=2,
                    fontsize=FONTSIZE-2,
                    numpoints=1,
                    frameon=True)


fig.savefig("hybrid_work_vs_cfl.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
