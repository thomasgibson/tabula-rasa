import os
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

FONTSIZE = 12

hdg_params = [(64, 1), (64, 2), (64, 3)]
hdg_data = ["results/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(64, 2), (64, 3), (64, 4)]
cg_data = ["results/CG_data_N%d_deg%d.csv" % param
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

ax.set_ylim([0.0, 4.0])

labels = []
for i, (cg_df, hdg_df) in enumerate(zip(cg_dfs, hdg_dfs)):

    cg_k = cg_df.degree.values[0]
    hdg_k = hdg_df.degree.values[0]

    cg_dofs = cg_df.dofs.values[0]
    tr_dofs = hdg_df.trace_dofs.values[0]

    cg_solve = cg_df.KSPSolve.values[0]
    cg_assembly = cg_df.SNESJacobianEval.values[0]
    cg_total = cg_solve + cg_assembly

    norm_cg_t = cg_solve / cg_total
    norm_cg_assembly = cg_assembly / cg_total

    a1 = ax.bar(ind[i], norm_cg_t, width,
                edgecolor="k",
                linewidth=1,
                color="#96595A")
    a3 = ax.bar(ind[i], norm_cg_assembly, width,
                bottom=norm_cg_t,
                edgecolor="k",
                linewidth=1,
                color="#DA897C")

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
                color="#A9A9A9")

    labels.append("$CG_%d$ and $HDG_%d$\n%d (%d)" % (cg_k, hdg_k,
                                                     cg_dofs, tr_dofs))

ax.set_ylabel("Time (s) / CG total time (s)")
xlabel = fig.text(0.5, -0.05,
                  "$CG_k$ and $HDG_{k-1}$\n CG dofs (HDG trace dofs)",
                  ha="center",
                  fontsize=FONTSIZE)

legend1 = plt.legend((a3[0], a1[0]),
                     ("CG assembly", "CG solve"),
                     ncol=1,
                     bbox_to_anchor=(0.3, 1))
legend2 = plt.legend((a8[0], a7[0], a6[0], a5[0], a4[0]),
                     ("HDG post-processing", "HDG back sub.",
                      "HDG forward elim.", "HDG assembly",
                      "HDG trace solve"),
                     ncol=1,
                     bbox_to_anchor=(0.75, 1))

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE-2)

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE-2)

ax.set_xticklabels(labels)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
fig.savefig("cg_hdg_compare.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel])
