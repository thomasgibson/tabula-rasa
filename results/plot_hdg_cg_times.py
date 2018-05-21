import os
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

FONTSIZE = 12

hdg_params = [(64, 1), (64, 2), (64, 3)]
hdg_data = ["HDG_CG_comp/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(64, 2), (64, 3), (64, 4)]
cg_data = ["HDG_CG_comp/CG_data_N%d_deg%d.csv" % param
           for param in cg_params]

for d in hdg_data + cg_data:
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)


sns.set(style="ticks")

fig, axes = plt.subplots(1, 1, figsize=(8, 6), squeeze=False)
axes = axes.flatten()
ax, = axes


cg_dfs = [pd.read_csv(d) for d in cg_data]
hdg_dfs = [pd.read_csv(d) for d in hdg_data]


ind = np.arange(3)
width = 0.35
for ax in axes:
    ax.set_xticks(ind + width/2)

ax.set_ylim([0.0, 3.5])

labels = []
for i, (cg_df, hdg_df) in enumerate(zip(cg_dfs, hdg_dfs)):

    cg_k = cg_df.degree.values[0]
    hdg_k = hdg_df.degree.values[0]

    cg_dofs = cg_df.dofs.values[0]
    tr_dofs = hdg_df.trace_dofs.values[0]

    cg_solve = cg_df.KSPSolve.values[0]
    cg_assembly = cg_df.SNESJacobianEval.values[0]
    cg_residual = cg_df.SNESFunctionEval.values[0]
    cg_total = cg_solve + cg_assembly + cg_residual

    norm_cg_t = cg_solve / cg_total
    norm_cg_assembly = cg_assembly / cg_total
    norm_cg_res = cg_residual / cg_total

    a1 = ax.bar(ind[i], norm_cg_t, width,
                edgecolor="k",
                linewidth=1,
                color="#96595A")
    a2 = ax.bar(ind[i], norm_cg_assembly, width,
                bottom=norm_cg_t,
                edgecolor="k",
                linewidth=1,
                color="#DA897C")
    a3 = ax.bar(ind[i], norm_cg_res, width,
                bottom=norm_cg_t + norm_cg_assembly,
                edgecolor="k",
                linewidth=1,
                color="#DAA520")

    trace_solve = hdg_df.HDGTraceSolve.values[0] / cg_total
    back_sub = hdg_df.HDGRecover.values[0] / cg_total
    rhs_assembly = hdg_df.HDGRhs.values[0] / cg_total
    pp_time = hdg_df.HDGPPTime.values[0] / cg_total
    trace_assemble = hdg_df.HDGUpdate.values[0] / cg_total

    a4 = ax.bar(ind[i] + width, trace_solve, width,
                edgecolor="k",
                linewidth=1,
                color="#0D6A82")
    a5 = ax.bar(ind[i] + width, trace_assemble, width,
                color="#B2E4CF",
                edgecolor="k",
                linewidth=1,
                bottom=trace_solve)
    a6 = ax.bar(ind[i] + width, rhs_assembly, width,
                bottom=trace_solve + trace_assemble,
                edgecolor="k",
                linewidth=1,
                color="#E4E4B2")
    a7 = ax.bar(ind[i] + width, back_sub, width,
                bottom=trace_solve + trace_assemble + rhs_assembly,
                edgecolor="k",
                linewidth=1,
                color="#122330")
    a8 = ax.bar(ind[i] + width, pp_time, width,
                bottom=(trace_solve + trace_assemble + rhs_assembly
                        + back_sub),
                edgecolor="k",
                linewidth=1,
                color="#008000")

    ax.legend((a1[0], a2[0], a3[0],
               a4[0], a5[0], a6[0], a7[0], a8[0]),
              ("CG solve", "CG assembly", "CG other",
               "HDG trace solve", "HDG assembly",
               "HDG forward elim.", "HDG back sub.",
               "HDG post-processing"),
              loc=9, ncol=3,
              bbox_to_anchor=(0.5, 1))

    labels.append("$CG_%d$ and $HDG_%d$\n%d (%d)" % (cg_k, hdg_k,
                                                     cg_dofs, tr_dofs))

ax.set_ylabel("Execution time (s) / CG execution time (s)")
xlabel = fig.text(0.5, -0.05,
                  "$CG_k$ and $HDG_{k-1}$\n CG dofs (HDG trace dofs)",
                  ha="center",
                  fontsize=FONTSIZE)

ax.set_xticklabels(labels)
fig.savefig("cg_hdg_compare.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel])
