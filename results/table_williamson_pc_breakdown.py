import os
import sys
import pandas as pd


params = [("RT", 1, 8, 28.125),
          ("BDM", 2, 8, 28.125)]

gmres_times = ["SWE/gmres_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
               for param in params]
hybrid_times = ["SWE/hybrid_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
                for param in params]

rt_data = "SWE/gmres_RT1_data_W5_ref8_Dt28.125_NS100.csv"
bdm_data = "SWE/gmres_BDM2_data_W5_ref8_Dt28.125_NS100.csv"

for data in gmres_times + hybrid_times + [rt_data, bdm_data]:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

gmres_time_dfs = [pd.read_csv(d) for d in gmres_times]
hybrid_time_dfs = [pd.read_csv(d) for d in hybrid_times]

df_rt_data = pd.read_csv(rt_data)
df_bdm_data = pd.read_csv(bdm_data)

table = r"""\begin{tabular}{llcccc}
\hline
\multirow{2}{*}{Preconditioner} &
\multirow{2}{*}{Stage} &
\multicolumn{2}{c}{$RT_1 \times DG_0$} &
\multicolumn{2}{c}{$BDM_2 \times DG_1$} \\
& & $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$ \\ \hline
"""

lformat = r"""& {stage} & {rt_time: .3f} & {rt_p: .2f} \% & {bdm_time: .3f} & {bdm_p: .2f} \% \\
"""

# For PC GMRES, we report time per iteration
gmres_iter_rt = df_rt_data.OuterIters.values.sum()
gmres_iter_bdm = df_bdm_data.OuterIters.values.sum()

# NOTE: for hybridization, we report time per picard iteration
# since there are no outer GMRES iterations.
npicard = 400  # 4 per time-step, for 100 time-steps

table += r"""
\multirow{4}{*}{approx. Schur}
"""

rt_df, bdm_df = gmres_time_dfs
hrt_df, hbdm_df = hybrid_time_dfs

rt_schur_solve = rt_df.KSPSchur.values[0]/gmres_iter_rt
rt_schur_low = rt_df.KSPFSLow.values[0]/gmres_iter_rt
rt_mass_invert = rt_df.KSPF0.values[0]/gmres_iter_rt
rt_total = rt_df.PETScLogKSPSolve.values[0]/gmres_iter_rt
rt_other = rt_total - (rt_schur_solve
                       + rt_schur_low
                       + rt_mass_invert)

bdm_schur_solve = bdm_df.KSPSchur.values[0]/gmres_iter_bdm
bdm_schur_low = bdm_df.KSPFSLow.values[0]/gmres_iter_bdm
bdm_mass_invert = bdm_df.KSPF0.values[0]/gmres_iter_bdm
bdm_total = bdm_df.PETScLogKSPSolve.values[0]/gmres_iter_bdm
bdm_other = bdm_total - (bdm_schur_solve
                         + bdm_schur_low
                         + bdm_mass_invert)

table += lformat.format(stage="Schur solve",
                        rt_time=rt_schur_solve,
                        rt_p=rt_schur_solve/rt_total * 100,
                        bdm_time=bdm_schur_solve,
                        bdm_p=bdm_schur_solve/bdm_total * 100)
table += lformat.format(stage="invert velocity mass: $A$",
                        rt_time=rt_mass_invert,
                        rt_p=rt_mass_invert/rt_total * 100,
                        bdm_time=bdm_mass_invert,
                        bdm_p=bdm_mass_invert/bdm_total * 100)
table += lformat.format(stage="apply inverse: $A^{-1}$",
                        rt_time=rt_schur_low,
                        rt_p=rt_schur_low/rt_total * 100,
                        bdm_time=bdm_schur_low,
                        bdm_p=bdm_schur_low/bdm_total * 100)
table += lformat.format(stage="gmres other",
                        rt_time=rt_other,
                        rt_p=rt_other/rt_total * 100,
                        bdm_time=bdm_other,
                        bdm_p=bdm_other/bdm_total * 100)
table += r"""\hline
"""

table += r"""& Total & %.3f & & %.3f & \\ \hline""" % (rt_total, bdm_total)

hrt_break_time = hrt_df.HybridBreak.values[0]/npicard
hrt_rhs_time = hrt_df.HybridRHS.values[0]/npicard
hrt_trace_solve_time = hrt_df.HybridTraceSolve.values[0]/npicard
hrt_recon_time = hrt_df.HybridReconstruction.values[0]/npicard
hrt_proj_time = hrt_df.HybridProjection.values[0]/npicard
hrt_total = hrt_df.PETScLogKSPSolve.values[0]/npicard

hbdm_break_time = hbdm_df.HybridBreak.values[0]/npicard
hbdm_rhs_time = hbdm_df.HybridRHS.values[0]/npicard
hbdm_trace_solve_time = hbdm_df.HybridTraceSolve.values[0]/npicard
hbdm_recon_time = hbdm_df.HybridReconstruction.values[0]/npicard
hbdm_proj_time = hbdm_df.HybridProjection.values[0]/npicard
hbdm_total = hbdm_df.PETScLogKSPSolve.values[0]/npicard

table += r"""
\multirow{5}{*}{hybridization}
"""

table += lformat.format(stage=r"Transfer: $R_{\Delta X}\rightarrow\widehat{R}_{\Delta X}$",
                        rt_time=hrt_break_time,
                        rt_p=hrt_break_time/hrt_total * 100,
                        bdm_time=hbdm_break_time,
                        bdm_p=hbdm_break_time/hbdm_total * 100)
table += lformat.format(stage=r"Forward elim.: $-C\widehat{A}^{-1}\widehat{R}_{\Delta X}$",
                        rt_time=hrt_rhs_time,
                        rt_p=hrt_rhs_time/hrt_total * 100,
                        bdm_time=hbdm_rhs_time,
                        bdm_p=hbdm_rhs_time/hbdm_total * 100)
table += lformat.format(stage="Trace solve",
                        rt_time=hrt_trace_solve_time,
                        rt_p=hrt_trace_solve_time/hrt_total * 100,
                        bdm_time=hbdm_trace_solve_time,
                        bdm_p=hbdm_trace_solve_time/hbdm_total * 100)
table += lformat.format(stage="Back sub.",
                        rt_time=hrt_recon_time,
                        rt_p=hrt_recon_time/hrt_total * 100,
                        bdm_time=hbdm_recon_time,
                        bdm_p=hbdm_recon_time/hbdm_total * 100)
table += lformat.format(stage=r"Projection: $\Pi_{\text{div}}\Delta\widehat{U}$",
                        rt_time=hrt_proj_time,
                        rt_p=hrt_proj_time/hrt_total * 100,
                        bdm_time=hbdm_proj_time,
                        bdm_p=hbdm_proj_time/hbdm_total * 100)


table += r"""\hline
"""

table += r"""& Total & %.2f & & %.2f & \\ \hline""" % (hrt_total, hbdm_total)

table += r"""
\end{tabular}"""

print(table)
