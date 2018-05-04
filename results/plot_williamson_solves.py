import os
import sys
import pandas as pd
import numpy as np
import seaborn

from matplotlib import pyplot as plt


params = [("RTCF", 1, 8, 28.125),
          ("RTCF", 2, 8, 28.125),
          ("BDM", 2, 8, 28.125)]
gmres_times = ["SWE/gmres_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
               for param in params]
hybrid_times = ["SWE/hybrid_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
                for param in params]

for data in hybrid_times:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


seaborn.set(style="ticks")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
axes = axes.flatten()
ax0, ax = axes

hmm_dfs = [pd.read_csv(d) for d in hybrid_times]
gmres_dfs = [pd.read_csv(d) for d in gmres_times]

ind = np.arange(len(params))
width = 0.67

for ax in axes:
    ax.set_xticks(ind)
    # ax.set_yscale("log")

methods = []
degrees_of_freedom = []
method = {("RT", 1): "$RT_1 \\times DG_0$",
          ("RT", 2): "$RT_2 \\times DG_1$",
          ("BDM", 2): "$BDM_2 \\times DG_1$",
          ("RTCF", 1): "$RTCF_1 \\times DQ_0$",
          ("RTCF", 2): "$RTCF_2 \\times DQ_1$"}

ax0.set_ylabel("Time (s)")

schur_times = []
schur_low_times = []
mass_inverse_times = []
gmres_other_times = []
for i, df in enumerate(gmres_dfs):
    schur_solve = df.KSPSchur.values[0]
    schur_low = df.KSPFSLow.values[0]
    mass_invert = df.KSPF0.values[0]
    other = df.PETScLogKSPSolve.values[0] - (schur_solve + schur_low
                                             + mass_invert)

    schur_times.append(schur_solve)
    schur_low_times.append(schur_low)
    mass_inverse_times.append(mass_invert)
    gmres_other_times.append(other)

a1 = ax0.bar(ind, schur_times, width,
             color='#30a2da')
a2 = ax0.bar(ind, mass_inverse_times, width,
             bottom=schur_times,
             color='#fc4f30')
a3 = ax0.bar(ind, schur_low_times, width,
             bottom=(np.array(schur_times) +
                     np.array(mass_inverse_times)),
             color='#e5ae38')
a4 = ax0.bar(ind, gmres_other_times, width,
             bottom=(np.array(schur_times) +
                     np.array(mass_inverse_times) +
                     np.array(schur_low_times)),
             color="k")

ax0.legend((a1[0], a2[0], a3[0], a4[0]), ("Schur solve",
                                          "Velocity mass",
                                          "Apply velocity mass",
                                          "GMRES other"))

rhs_assembly_times = []
trace_solve_times = []
local_recovery_times = []
projection_times = []

for i, df in enumerate(hmm_dfs):
    methods.append((df.method.values[0], df.model_degree.values[0]))
    degrees_of_freedom.append(df.total_dofs.values[0])

    break_time = df.HybridBreak.values[0]
    rhs_time = df.HybridRHS.values[0]
    rhs_assembly = break_time + rhs_time
    rhs_assembly_times.append(rhs_assembly)

    trace_solve_time = df.HybridTraceSolve.values[0]
    trace_solve_times.append(trace_solve_time)

    recon_time = df.HybridReconstruction.values[0]
    proj_time = df.HybridProjection.values[0]
    local_recovery_times.append(recon_time)
    projection_times.append(proj_time)

a1 = ax.bar(ind, trace_solve_times, width)
a2 = ax.bar(ind, rhs_assembly_times, width,
            bottom=trace_solve_times)
a3 = ax.bar(ind, local_recovery_times, width,
            bottom=np.array(rhs_assembly_times) + np.array(trace_solve_times))
a4 = ax.bar(ind, projection_times, width,
            bottom=(np.array(rhs_assembly_times) +
                    np.array(trace_solve_times) +
                    np.array(local_recovery_times)))

ax.legend((a1[0], a2[0], a3[0], a4[0]), ("Linear solve",
                                         "RHS assembly",
                                         "Local recovery",
                                         "Projection"))

ax0.set_xticklabels(["%s\n(%d)" % (method[mth], dofs) for (mth, dofs) in
                     zip(methods, degrees_of_freedom)])
ax.set_xticklabels(["%s\n(%d)" % (method[mth], dofs) for (mth, dofs) in
                    zip(methods, degrees_of_freedom)])

xlabel = fig.text(0.5, -0.1,
                  "Discretization\n(Degrees of freedom)",
                  ha='center')

title = fig.text(0.5, 1,
                 "Approx. Schur Comp. vs Hybrid-mixed methods",
                 ha="center")

seaborn.despine(fig)
fig.savefig("williamson_timings.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight")
