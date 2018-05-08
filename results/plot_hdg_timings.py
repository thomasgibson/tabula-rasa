import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

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

sns.set(style="ticks")

fig, axes = plt.subplots(2, 1, figsize=(8, 12), squeeze=False)
axes = axes.flatten()
ax1, ax2 = axes

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

ind = np.arange(4)
width = 0.35
for ax in axes:
    ax.set_xticks(ind + width/2)
    ax.set_yscale('log')

ax1.set_ylim([1.0e-2, 70])
ax2.set_ylim([1.0e-2, 330])

cg3_dofs = []
cg4_dofs = []
ax1bars = []
ax2bars = []
yticks1 = []
yticks2 = []
for group in cg_groups:

    degree, df = group
    # n_e = df.num_cells.values[0]
    # kd = degree ** 3

    if degree == 3:

        ksp_solve = df.KSPSolve.values
        setup = df.PCSetUp.values
        dofs = df.dofs.values

        yticks1.extend(list(ksp_solve + setup))

        a1 = ax1.bar(ind, ksp_solve, width,
                     color="#96595A")
        a2 = ax1.bar(ind, setup, width,
                     bottom=ksp_solve,
                     color="#DA897C")

        ax1bars.extend([(a1, "CG solve"), (a2, "CG setup")])
        cg3_dofs.extend(list(dofs))
        ax1.set_ylabel("Time [s] (log)", fontsize=FONTSIZE)

    if degree == 4:

        ksp_solve = df.KSPSolve.values
        setup = df.PCSetUp.values
        dofs = df.dofs.values

        yticks2.extend(list(ksp_solve + setup))

        a1 = ax2.bar(ind, ksp_solve, width,
                     color="#96595A")
        a2 = ax2.bar(ind, setup, width,
                     bottom=ksp_solve,
                     color="#DA897C")

        ax2bars.extend([(a1, "CG solve"), (a2, "CG setup")])
        cg4_dofs.extend(list(dofs))
        ax2.set_ylabel("Time [s] (log)", fontsize=FONTSIZE)

hdg2_dofs = []
hdg3_dofs = []
for group in hdg_groups:

    degree, df = group
    # n_e = df.num_cells.values[0]
    # kd = (degree + 1) ** 3

    if degree == 2:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        dofs = df.trace_dofs.values
        pcsetup = df.PCSetUp.values

        total = (recovery_times +
                 pp_times +
                 rhs_time +
                 trace_solves +
                 pcsetup)

        yticks1.extend(list(total))

        local_ops = (np.array(rhs_time) +
                     np.array(recovery_times) +
                     np.array(pp_times))

        a1 = ax1.bar(ind + width, trace_solves, width,
                     color="#0D6A82")
        a2 = ax1.bar(ind + width, local_ops, width,
                     bottom=trace_solves,
                     color="#B2E4CF")
        a3 = ax1.bar(ind + width, pcsetup, width,
                     bottom=trace_solves + local_ops,
                     color="#E4E4B2")

        ax1bars.extend([(a1, "HDG trace solve"),
                        (a2, "HDG local ops"),
                        (a3, "HDG setup")])
        hdg2_dofs.extend(list(dofs))

    if degree == 3:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        dofs = df.trace_dofs.values
        pcsetup = df.PCSetUp.values

        total = (recovery_times +
                 pp_times +
                 rhs_time +
                 trace_solves +
                 pcsetup)

        yticks2.extend(list(total))

        local_ops = (np.array(rhs_time) +
                     np.array(recovery_times) +
                     np.array(pp_times))

        a1 = ax2.bar(ind + width, trace_solves, width,
                     color="#0D6A82")
        a2 = ax2.bar(ind + width, local_ops, width,
                     bottom=trace_solves,
                     color="#B2E4CF")
        a3 = ax2.bar(ind + width, pcsetup, width,
                     bottom=trace_solves + local_ops,
                     color="#E4E4B2")

        ax2bars.extend([(a1, "HDG trace solve"),
                        (a2, "HDG local ops"),
                        (a3, "HDG setup")])
        hdg3_dofs.extend(list(dofs))

ax1.set_xticklabels(["%s\n(%s)" % (x, y)
                     for x, y in zip(cg3_dofs, hdg2_dofs)])
ax2.set_xticklabels(["%s\n(%s)" % (x, y)
                     for x, y in zip(cg4_dofs, hdg3_dofs)])
ax2.set_xlabel("CG DoFs\n(HDG trace DoFs)",
               fontsize=FONTSIZE)

for ax in [ax1, ax2]:
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE-2)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE-2)

    ax.yaxis.grid(True)

ax1.set_title("$CG_3$ and $HDG_2$")
ax2.set_title("$CG_4$ and $HDG_3$")

ax1.set_yticks(yticks1)
ax2.set_yticks(yticks2)
for ax in [ax1, ax2]:
    plt.setp(ax.get_yminorticklabels(), visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))

ax1.legend([a[0] for a, b in ax1bars],
           [b for a, b in ax1bars])
ax2.legend([a[0] for a, b in ax2bars],
           [b for a, b in ax2bars])

fig.subplots_adjust(hspace=0.25)
fig.savefig("timings.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight")
