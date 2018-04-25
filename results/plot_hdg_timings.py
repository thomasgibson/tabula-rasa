import os
import pandas as pd
import matplotlib
import numpy as np

from matplotlib import pyplot as plt

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

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
ax1, ax2, ax3, ax4 = axes

cg_dfs = pd.concat(pd.read_csv(d) for d in cg_data)
cg_groups = cg_dfs.groupby(["degree"], as_index=False)

hdg_dfs = pd.concat(pd.read_csv(d) for d in hdg_data)
hdg_groups = hdg_dfs.groupby(["degree"], as_index=False)

ind = np.arange(5)
width = 0.35
for ax in axes:
    ax.set_xticks(ind)
    ax.set_yscale('log')

for group in cg_groups:

    degree, df = group

    if degree == 3:

        ksp_solve = df.KSPSolve.values
        dofs = df.dofs.values

        a1 = ax1.bar(ind, ksp_solve, width)

        ax1.set_xticklabels(["%s" % dof for dof in dofs])
        ax1.legend((a1[0],), ("Execution time",))
        ax1.set_title("CG %s" % degree)
        ax1.set_xlabel("Degrees of freedom")
        ax1.set_ylabel("Log time (s)")

    if degree == 4:

        ksp_solve = df.KSPSolve.values
        dofs = df.dofs.values

        a1 = ax2.bar(ind, ksp_solve, width)

        ax2.set_xticklabels(["%s" % dof for dof in dofs])
        ax2.legend((a1[0],), ("Execution solve",))
        ax2.set_title("CG %s" % degree)
        ax2.set_xlabel("Degrees of freedom")
        ax2.set_ylabel("Log time (s)")

for group in hdg_groups:

    degree, df = group

    if degree == 2:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        update = df.HDGUpdate.values
        setup = rhs_time + update
        dofs = df.trace_dofs.values

        a1 = ax3.bar(ind, trace_solves, width)
        a2 = ax3.bar(ind, setup, width,
                     bottom=trace_solves)
        a3 = ax3.bar(ind, recovery_times, width,
                     bottom=trace_solves + setup)
        a4 = ax3.bar(ind, pp_times, width,
                     bottom=trace_solves + setup + recovery_times)

        ax3.set_xticklabels(["%s" % dof for dof in dofs])
        ax3.legend((a1[0], a2[0], a3[0], a4[0]), ("Linear solve", "RHS Assembly",
                                                  "Local recovery", "Post-processing"))
        ax3.set_title("HDG %s" % degree)
        ax3.set_xlabel("Trace degrees of freedom")
        ax3.set_ylabel("Log time (s)")

    if degree == 3:

        recovery_times = df.HDGRecover.values
        pp_times = df.HDGPPTime.values
        rhs_time = df.HDGRhs.values
        trace_solves = df.HDGSolve.values
        update = df.HDGUpdate.values
        setup = rhs_time + update
        dofs = df.trace_dofs.values

        a1 = ax4.bar(ind, trace_solves, width)
        a2 = ax4.bar(ind, setup, width,
                     bottom=trace_solves)
        a3 = ax4.bar(ind, recovery_times, width,
                     bottom=trace_solves + setup)
        a4 = ax4.bar(ind, pp_times, width,
                     bottom=trace_solves + setup + recovery_times)

        ax4.set_xticklabels(["%s" % dof for dof in dofs])
        ax4.legend((a1[0], a2[0], a3[0], a4[0]), ("Linear solve", "RHS Assembly",
                                                  "Local recovery", "Post-processing"))
        ax4.set_title("HDG %s" % degree)
        ax4.set_xlabel("Trace degrees of freedom")
        ax4.set_ylabel("Log time (s)")

plt.show()
