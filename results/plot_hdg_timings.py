import os
import pandas as pd
import numpy as np
import seaborn

from matplotlib import pyplot as plt

FONTSIZE = 12

hdg_params = [(8, 1), (8, 2), (8, 3),
              (16, 1), (16, 2), (16, 3),
              (32, 1), (32, 2), (32, 3),
              (64, 1), (64, 2), (64, 3)]
hdg_data = ["HDG_CG_comp/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(8, 2), (8, 3), (8, 4),
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

seaborn.set(style="ticks")

fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
axes = axes.flatten()
ax1, ax2, ax3, ax4 = axes

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

ind = np.arange(4)
width = 0.35
for ax in axes:
    ax.set_xticks(ind)
    ax.set_yscale('log')

for group in cg_groups:

    degree, df = group

    if degree == 3:

        ksp_solve = df.KSPSolve.values
        setup = df.PCSetUp.values
        dofs = df.dofs.values

        a1 = ax1.bar(ind, ksp_solve, width)
        a2 = ax1.bar(ind, setup, width,
                     bottom=ksp_solve)

        ax1.set_xticklabels(["%s" % dof for dof in dofs])
        ax1.legend((a1[0], a2[0]), ("Linear solve", "Setup"))
        ax1.set_title("$CG_%s$" % degree, fontsize=FONTSIZE)
        ax1.set_xlabel("Degrees of freedom", fontsize=FONTSIZE)
        ax1.set_ylabel("Time (s)", fontsize=FONTSIZE)

    if degree == 4:

        ksp_solve = df.KSPSolve.values
        setup = df.PCSetUp.values
        dofs = df.dofs.values

        a1 = ax3.bar(ind, ksp_solve, width)
        a2 = ax3.bar(ind, setup, width,
                     bottom=ksp_solve)

        ax3.set_xticklabels(["%s" % dof for dof in dofs])
        ax3.legend((a1[0], a2[0]), ("Linear solve", "Setup"))
        ax3.set_title("$CG_%s$" % degree, fontsize=FONTSIZE)
        ax3.set_xlabel("Degrees of freedom", fontsize=FONTSIZE)
        ax3.set_ylabel("Time (s)", fontsize=FONTSIZE)

for group in hdg_groups:

    degree, df = group

    if degree == 2:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        update = df.HDGUpdate.values
        rhs_setup = rhs_time
        dofs = df.trace_dofs.values
        pcsetup = df.PCSetUp.values

        a1 = ax2.bar(ind, trace_solves, width)
        a2 = ax2.bar(ind, rhs_setup, width,
                     bottom=trace_solves)
        a3 = ax2.bar(ind, recovery_times, width,
                     bottom=trace_solves + rhs_setup)
        a4 = ax2.bar(ind, pp_times, width,
                     bottom=trace_solves + rhs_setup + recovery_times)
        a5 = ax2.bar(ind, pcsetup, width,
                     bottom=(trace_solves + rhs_setup +
                             recovery_times + pp_times))

        ax2.set_xticklabels(["%s" % dof for dof in dofs])
        ax2.legend((a1[0], a2[0], a3[0], a4[0], a5[0]), ("Linear solve",
                                                         "RHS assembly",
                                                         "Local recovery",
                                                         "Post-processing",
                                                         "Setup"))
        ax2.set_title("$HDG_%s$" % degree, fontsize=FONTSIZE)
        ax2.set_xlabel("Trace degrees of freedom", fontsize=FONTSIZE)

    if degree == 3:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        update = df.HDGUpdate.values
        rhs_setup = rhs_time
        dofs = df.trace_dofs.values
        pcsetup = df.PCSetUp.values

        a1 = ax4.bar(ind, trace_solves, width)
        a2 = ax4.bar(ind, rhs_setup, width,
                     bottom=trace_solves)
        a3 = ax4.bar(ind, recovery_times, width,
                     bottom=trace_solves + rhs_setup)
        a4 = ax4.bar(ind, pp_times, width,
                     bottom=trace_solves + rhs_setup + recovery_times)
        a5 = ax4.bar(ind, pcsetup, width,
                     bottom=(trace_solves + rhs_setup +
                             recovery_times + pp_times))

        ax4.set_xticklabels(["%s" % dof for dof in dofs])
        ax4.legend((a1[0], a2[0], a3[0], a4[0], a5[0]), ("Linear solve",
                                                         "RHS assembly",
                                                         "Local recovery",
                                                         "Post-processing",
                                                         "Setup"))
        ax4.set_title("$HDG_%s$" % degree, fontsize=FONTSIZE)
        ax4.set_xlabel("Trace degrees of freedom", fontsize=FONTSIZE)

for ax in [ax1, ax2, ax3, ax4]:
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE-2)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE-2)

fig.subplots_adjust(wspace=0.15, hspace=0.375)
seaborn.despine(fig)
fig.savefig("timings.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight")
