import os
import sys
import pandas as pd
import numpy as np
import seaborn
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


FONTSIZE = 12


params = [("RT", 1, 8, 28.125),
          # ("RT", 2, 8, 28.125),
          # ("RTCF", 1, 8, 28.125),
          # ("RTCF", 2, 8, 28.125),
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
ax1, ax2 = axes

hmm_dfs = [pd.read_csv(d) for d in hybrid_times]
gmres_dfs = [pd.read_csv(d) for d in gmres_times]

ind = np.arange(2)
width = 0.5

for ax in axes:
    ax.set_xticks(ind)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

method = {("RT", 1): "$RT_1 \\times DG_0$",
          ("RT", 2): "$RT_2 \\times DG_1$",
          ("BDM", 2): "$BDM_2 \\times DG_1$",
          ("RTCF", 1): "$RTCF_1 \\times DQ_0$",
          ("RTCF", 2): "$RTCF_2 \\times DQ_1$"}

# RT on triangles
rtlo_schur_times = []
rtlo_schur_low_times = []
rtlo_mass_inverse_times = []
rtlo_gmres_other_times = []
rtnlo_schur_times = []
rtnlo_schur_low_times = []
rtnlo_mass_inverse_times = []
rtnlo_gmres_other_times = []

# RT on quads
rtcflo_schur_times = []
rtcflo_schur_low_times = []
rtcflo_mass_inverse_times = []
rtcflo_gmres_other_times = []
rtcfnlo_schur_times = []
rtcfnlo_schur_low_times = []
rtcfnlo_mass_inverse_times = []
rtcfnlo_gmres_other_times = []

# BDM2 on triangles
bdm_schur_times = []
bdm_schur_low_times = []
bdm_mass_inverse_times = []
bdm_gmres_other_times = []

# DOFs
rtlo_dofs = []
rtcflo_dofs = []
rtnlo_dofs = []
rtcfnlo_dofs = []
bdm_dofs = []

# Time (KSPSolve)
rtlo_time = []
rtcflo_time = []
rtnlo_time = []
rtcfnlo_time = []
bdm_time = []

for df in gmres_dfs:

    schur_solve = df.KSPSchur.values[0]
    schur_low = df.KSPFSLow.values[0]
    mass_invert = df.KSPF0.values[0]
    other = df.PETScLogKSPSolve.values[0] - (schur_solve + schur_low
                                             + mass_invert)
    dofs = df.total_dofs.values[0]
    kspsolve = df.PETScLogKSPSolve.values[0]

    if df.method.values[0] == "RT":

        if df.model_degree.values[0] == 1:

            rtlo_schur_times.append(schur_solve)
            rtlo_schur_low_times.append(schur_low)
            rtlo_mass_inverse_times.append(mass_invert)
            rtlo_gmres_other_times.append(other)
            rtlo_dofs.append(dofs)
            rtlo_time.extend([schur_solve, kspsolve])

        else:

            assert df.model_degree.values[0] == 2

            rtnlo_schur_times.append(schur_solve)
            rtnlo_schur_low_times.append(schur_low)
            rtnlo_mass_inverse_times.append(mass_invert)
            rtnlo_gmres_other_times.append(other)
            rtnlo_dofs.append(dofs)
            rtnlo_time.extend([schur_solve])

    elif df.method.values[0] == "RTCF":

        if df.model_degree.values[0] == 1:

            rtcflo_schur_times.append(schur_solve)
            rtcflo_schur_low_times.append(schur_low)
            rtcflo_mass_inverse_times.append(mass_invert)
            rtcflo_gmres_other_times.append(other)
            rtcflo_dofs.append(dofs)
            rtcflo_time.extend([schur_solve, kspsolve])

        else:

            assert df.model_degree.values[0] == 2

            rtcfnlo_schur_times.append(schur_solve)
            rtcfnlo_schur_low_times.append(schur_low)
            rtcfnlo_mass_inverse_times.append(mass_invert)
            rtcfnlo_gmres_other_times.append(other)
            rtcfnlo_dofs.append(dofs)
            rtcfnlo_time.extend([schur_solve, kspsolve])

    else:

        assert df.method.values[0] == "BDM"
        assert df.model_degree.values[0] == 2

        bdm_schur_times.append(schur_solve)
        bdm_schur_low_times.append(schur_low)
        bdm_mass_inverse_times.append(mass_invert)
        bdm_gmres_other_times.append(other)
        bdm_dofs.append(dofs)
        bdm_time.extend([schur_solve, kspsolve])

# H-RT on triangles
hrtlo_rhs_assembly_times = []
hrtlo_trace_solve_times = []
hrtlo_local_recovery_times = []
hrtlo_projection_times = []
hrtnlo_rhs_assembly_times = []
hrtnlo_trace_solve_times = []
hrtnlo_local_recovery_times = []
hrtnlo_projection_times = []

# H-RTCF on triangles
hrtcflo_rhs_assembly_times = []
hrtcflo_trace_solve_times = []
hrtcflo_local_recovery_times = []
hrtcflo_projection_times = []
hrtcfnlo_rhs_assembly_times = []
hrtcfnlo_trace_solve_times = []
hrtcfnlo_local_recovery_times = []
hrtcfnlo_projection_times = []

# H-BDM2 on triangles
hbdm_rhs_assembly_times = []
hbdm_trace_solve_times = []
hbdm_local_recovery_times = []
hbdm_projection_times = []

# DOFs
hrtlo_dofs = []
hrtcflo_dofs = []
hrtnlo_dofs = []
hrtcfnlo_dofs = []
hbdm_dofs = []

# Time (KSPSolve)
hrtlo_time = []
hrtcflo_time = []
hrtnlo_time = []
hrtcfnlo_time = []
hbdm_time = []

for df in hmm_dfs:

    break_time = df.HybridBreak.values[0]
    rhs_time = df.HybridRHS.values[0]
    rhs_assembly = break_time + rhs_time

    trace_solve_time = df.HybridTraceSolve.values[0]

    recon_time = df.HybridReconstruction.values[0]
    proj_time = df.HybridProjection.values[0]

    ksptime = df.PETScLogKSPSolve.values[0]
    dofs = df.total_dofs.values[0]

    if df.method.values[0] == "RT":

        if df.model_degree.values[0] == 1:

            hrtlo_rhs_assembly_times.append(rhs_assembly)
            hrtlo_trace_solve_times.append(trace_solve_time)
            hrtlo_local_recovery_times.append(recon_time)
            hrtlo_projection_times.append(proj_time)
            hrtlo_dofs.append(dofs)
            hrtlo_time.extend([trace_solve_time,
                               trace_solve_time + rhs_assembly,
                               ksptime])

        else:

            assert df.model_degree.values[0] == 2

            hrtnlo_rhs_assembly_times.append(rhs_assembly)
            hrtnlo_trace_solve_times.append(trace_solve_time)
            hrtnlo_local_recovery_times.append(recon_time)
            hrtnlo_projection_times.append(proj_time)
            hrtnlo_dofs.append(dofs)
            hrtnlo_time.extend([trace_solve_time,
                                trace_solve_time + rhs_assembly,
                                ksptime])

    elif df.method.values[0] == "RTCF":

        if df.model_degree.values[0] == 1:

            hrtcflo_rhs_assembly_times.append(rhs_assembly)
            hrtcflo_trace_solve_times.append(trace_solve_time)
            hrtcflo_local_recovery_times.append(recon_time)
            hrtcflo_projection_times.append(proj_time)
            hrtcflo_dofs.append(dofs)
            hrtcflo_time.extend([trace_solve_time,
                                 trace_solve_time + rhs_assembly,
                                 ksptime])

        else:

            assert df.model_degree.values[0] == 2

            hrtcfnlo_rhs_assembly_times.append(rhs_assembly)
            hrtcfnlo_trace_solve_times.append(trace_solve_time)
            hrtcfnlo_local_recovery_times.append(recon_time)
            hrtcfnlo_projection_times.append(proj_time)
            hrtcfnlo_dofs.append(dofs)
            hrtcfnlo_time.extend([trace_solve_time,
                                  trace_solve_time + rhs_assembly,
                                  ksptime])

    else:

        assert df.method.values[0] == "BDM"
        assert df.model_degree.values[0] == 2

        hbdm_rhs_assembly_times.append(rhs_assembly)
        hbdm_trace_solve_times.append(trace_solve_time)
        hbdm_local_recovery_times.append(recon_time)
        hbdm_projection_times.append(proj_time)
        hbdm_time.extend([trace_solve_time,
                          trace_solve_time + rhs_assembly,
                          ksptime])

# LO RT
ax21 = ax1.twinx()
ax1.set_yscale('log')
# ax21.set_yscale('log')

ag1 = ax1.bar(ind[0], rtlo_schur_times[0],
              width,
              color='#30a2da')
ah1 = ax21.bar(ind[1], hrtlo_trace_solve_times[0],
               width)
ag2 = ax1.bar(ind[0], rtlo_mass_inverse_times[0],
              width,
              bottom=rtlo_schur_times[0],
              color='#fc4f30')
ah2 = ax21.bar(ind[1], hrtlo_rhs_assembly_times[0],
               width,
               bottom=hrtlo_trace_solve_times[0])
ag3 = ax1.bar(ind[0], rtlo_schur_low_times[0],
              width,
              bottom=(rtlo_schur_times[0] +
                      rtlo_mass_inverse_times[0]),
              color='#e5ae38')
ah3 = ax21.bar(ind[1], hrtlo_local_recovery_times[0],
               width,
               bottom=(hrtlo_trace_solve_times[0] +
                       hrtlo_rhs_assembly_times[0]))
ag4 = ax1.bar(ind[0], rtlo_gmres_other_times[0],
              width,
              bottom=(rtlo_schur_times[0] +
                      rtlo_mass_inverse_times[0] +
                      rtlo_schur_low_times[0]),
              color='k')
ah4 = ax21.bar(ind[1], hrtlo_projection_times[0],
               width,
               bottom=(hrtlo_trace_solve_times[0] +
                       hrtlo_rhs_assembly_times[0] +
                       hrtlo_local_recovery_times[0]))

ax1.set_yticks(rtlo_time)
ax21.set_yticks(hrtlo_time)
for ax in [ax1, ax21]:
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.setp(ax.get_yminorticklabels(), visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))

ax1.set_ylabel('gmres time [s] (log)', fontsize=FONTSIZE)
ax21.set_ylabel('hybridization time [s]', fontsize=FONTSIZE)

ax1.set_xticklabels(["%s\n%s" % (method[mth], pc) for (mth, pc) in
                     zip([("RT", 1), ("RT", 1)], ["gmres", "hybridization"])])

ax1.legend((ag1[0], ag2[0], ag3[0], ag4[0]), ("Schur solve",
                                              "Velocity mass",
                                              "Apply velocity mass",
                                              "GMRES other"),
           bbox_to_anchor=(0.5, 1.375),
           fontsize=FONTSIZE-2)
ax21.legend((ah1[0], ah2[0], ah3[0], ah4[0]), ("Trace solve",
                                               "RHS assembly",
                                               "Local recovery",
                                               "Projection"),
            bbox_to_anchor=(3.0, 1.375),
            fontsize=FONTSIZE-2)

# BDM2 DG1
ax22 = ax2.twinx()
ax2.set_yscale('log')
# ax22.set_yscale('log')

ax2.bar(ind[0], bdm_schur_times[0],
        width,
        color='#30a2da')
ax22.bar(ind[1], hbdm_trace_solve_times[0],
         width)
ax2.bar(ind[0], bdm_mass_inverse_times[0],
        width,
        bottom=bdm_schur_times[0],
        color='#fc4f30')
ax22.bar(ind[1], hbdm_rhs_assembly_times[0],
         width,
         bottom=hbdm_trace_solve_times[0])
ax2.bar(ind[0], bdm_schur_low_times[0],
        width,
        bottom=(bdm_schur_times[0] +
                bdm_mass_inverse_times[0]),
        color='#e5ae38')
ax22.bar(ind[1], hbdm_local_recovery_times[0],
         width,
         bottom=(hbdm_trace_solve_times[0] +
                 hbdm_rhs_assembly_times[0]))
ax2.bar(ind[0], bdm_gmres_other_times[0],
        width,
        bottom=(bdm_schur_times[0] +
                bdm_mass_inverse_times[0] +
                bdm_schur_low_times[0]),
        color='k')
ax22.bar(ind[1], hbdm_projection_times[0],
         width,
         bottom=(hbdm_trace_solve_times[0] +
                 hbdm_rhs_assembly_times[0] +
                 hbdm_local_recovery_times[0]))

ax2.set_yticks(bdm_time)
ax22.set_yticks(hbdm_time)
for ax in [ax2, ax22]:
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.setp(ax.get_yminorticklabels(), visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))

ax2.set_ylabel('gmres time [s] (log)', fontsize=FONTSIZE)
ax22.set_ylabel('hybridization time [s]', fontsize=FONTSIZE)

ax2.set_xticklabels(["%s\n%s" % (method[mth], pc) for (mth, pc) in
                     zip([("BDM", 2), ("BDM", 2)],
                         ["gmres", "hybridization"])])


xlabel = fig.text(0.5, -0.075,
                  "Discretization",
                  ha='center',
                  fontsize=FONTSIZE)

title = fig.text(0.5, 1.025,
                 "Approx. Schur-comp. vs hybrid-mixed methods",
                 ha="center",
                 fontsize=FONTSIZE)

for ax in [ax1, ax2]:
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE-2)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE-2)

for ax in [ax21, ax22]:
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE-2)

fig.subplots_adjust(wspace=0.75)
# seaborn.despine(fig)
fig.savefig("williamson_timings.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight")
