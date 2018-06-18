import os
import pandas as pd
import numpy as np
import seaborn
from collections import defaultdict

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

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

markers = ["o", "s", "^", "v", ">", "<", "D", "p", "h", "*"]
colors = seaborn.color_palette(n_colors=3)

fig, (axes,) = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
ax1, = axes


ax1.spines["left"].set_position(("outward", 10))
ax1.spines["bottom"].set_position(("outward", 10))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.xaxis.set_ticks_position("bottom")
ax1.yaxis.set_ticks_position("left")


cg_dofs = defaultdict(list)
cg_iters = defaultdict(list)
hdg_trace_dofs = defaultdict(list)
hdg_iters = defaultdict(list)

num_cells_cg = defaultdict(list)
num_cells_hdg = defaultdict(list)

for group in cg_groups:
    degree, df = group

    dofs = list(df.dofs.values)
    iters = list(df.ksp_iters.values)
    cg_dofs[degree].extend(dofs)
    cg_iters[degree].extend(iters)
    num_cells_cg[degree].extend(list(df.num_cells.values))


for group in hdg_groups:
    degree, df = group

    dofs = list(df.trace_dofs.values)
    iters = list(df.ksp_iters.values)
    hdg_trace_dofs[degree].extend(dofs)
    hdg_iters[degree].extend(iters)
    num_cells_hdg[degree].extend(list(df.num_cells.values))

for cg_k, hdg_k in zip(cg_dofs, hdg_trace_dofs):

    dof_ratios = [Ntrace / Ncg for Ncg, Ntrace in
                  zip(cg_dofs[cg_k], hdg_trace_dofs[hdg_k])]

    # DOF ratios
    # ax1.plot(num_cells_cg[cg_k], dof_ratios,
    #          color=colors[cg_k-2],
    #          marker=markers[cg_k-2],
    #          # markersize=MARKERSIZE,
    #          linestyle="dashed",
    #          # linewidth=LINEWIDTH,
    #          label="k = %s" % cg_k)

    # KSP iterations
    ax1.plot(num_cells_cg[cg_k], cg_iters[cg_k],
             color=colors[cg_k-2],
             marker=markers[cg_k-2],
             markersize=MARKERSIZE,
             linestyle="solid",
             linewidth=LINEWIDTH,
             label="$CG_%s$" % cg_k)
    ax1.plot(num_cells_hdg[hdg_k], hdg_iters[hdg_k],
             color=colors[hdg_k - 1],
             marker=markers[hdg_k - 1],
             markersize=MARKERSIZE,
             linestyle="dotted",
             linewidth=LINEWIDTH,
             label="$HDG_%s$" % hdg_k)

# num_cells = num_cells_cg[3] + [num_cells_cg[2][-1]]
# ones = np.ones(len(num_cells))
# ax1.plot(num_cells, ones, color="k",
#          linestyle="dotted",
#          linewidth=LINEWIDTH,
#          label="Ratio 1")

# ax1.set_ylim([1, 5.5])
# ax1.set_ylabel("DOF ratio ($HDG_{k-1} / CG_k$)",
#                fontsize=FONTSIZE)
ax1.set_ylim([0, 60])
ax1.set_ylabel("Krylov iterations", fontsize=FONTSIZE+2)

for ax in axes:
    ax.set_xscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              fontsize=FONTSIZE,
              handlelength=3,
              numpoints=1,
              frameon=False)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE)

ax.grid(b=True, which='major', linestyle='-.')

# fig.subplots_adjust(wspace=0.25)
xlabel = fig.text(0.5, -0.1,
                  "Number of cells",
                  ha='center',
                  fontsize=FONTSIZE+2)
seaborn.despine(fig)
fig.savefig("data.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel])
