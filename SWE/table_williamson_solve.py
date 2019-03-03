import os
import sys
import pandas as pd


params = [("RT", 7, 100.0),
          ("BDM", 7, 100.0)]

gmres_times = ["results/gmres_%s_profile_W5_ref%d_Dt%s_NS25.csv" % param
               for param in params]
hybrid_times = ["results/hybrid_%s_profile_W5_ref%d_Dt%s_NS25.csv" % param
                for param in params]

rt_data = "results/gmres_RT_data_W5_ref7_Dt100.0_NS25.csv"
bdm_data = "results/gmres_BDM_data_W5_ref7_Dt100.0_NS25.csv"
hrt_data = "results/hybrid_RT_data_W5_ref7_Dt100.0_NS25.csv"
hbdm_data = "results/hybrid_BDM_data_W5_ref7_Dt100.0_NS25.csv"

for data in gmres_times + hybrid_times + [rt_data, bdm_data,
                                          hrt_data, hbdm_data]:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

df_rt_data = pd.read_csv(rt_data)
df_bdm_data = pd.read_csv(bdm_data)
df_hrt_data = pd.read_csv(hrt_data)
df_hbdm_data = pd.read_csv(hbdm_data)

gmres_data_map = {("RT", 1): df_rt_data,
                  ("BDM", 2): df_bdm_data}
hybrid_data_map = {("RT", 1): df_hrt_data,
                   ("BDM", 2): df_hbdm_data}

gmres_time_dfs = [pd.read_csv(d) for d in gmres_times]
hybrid_time_dfs = [pd.read_csv(d) for d in hybrid_times]

method_map = {("RT", 1): "{$\\text{RT}_1 \\times \\text{DG}_0$}",
              ("BDM", 2): "{$\\text{BDM}_2 \\times \\text{DG}_1$}"}


table = r"""\begin{tabular}{cccccc}\hline
\multicolumn{6}{c}{\textbf{Preconditioner and solver details}} \\
\multirow{2}{*}{Mixed method} &
\multirow{2}{*}{Preconditioner} &
\multirow{2}{*}{$t_{\text{total}}$ (s)} &
Avg. outer & Avg. inner &
\multirow{2}{*}{$\frac{t_{\text{total}}^{\text{gmres}}}{t_{\text{total}}^{\text{hybrid.}}}$}\\
& & & its. & its. & \\ \hline
"""

lformat = r"""& {pc} & {total_time: .3f} & {outerits} & {innerits} &
"""

for gmres_df, hybrid_df in zip(gmres_time_dfs, hybrid_time_dfs):
    # Make sure we have the right ordering
    method_gmres = gmres_df.method.values[0]
    method_hybrid = hybrid_df.method.values[0]
    assert method_gmres == method_hybrid
    method = method_gmres
    deg = gmres_df.model_degree.values[0]

    table += r"""\multirow{2}{*}%s""" % method_map[(method, deg)]

    gmres_total = gmres_df.PETSCLogKSPSolve.values[0]
    hybrid_total = hybrid_df.PETSCLogKSPSolve.values[0]

    data_gmres = gmres_data_map[(method, deg)]
    data_hybrid = hybrid_data_map[(method, deg)]

    gmres_outerits = int(round(data_gmres.OuterIters.values.mean(), 0))
    gmres_innerits = int(round(data_gmres.InnerIters.values.mean(), 0))
    hybrid_outerits = "None"
    hybrid_innerits = int(round(data_hybrid.InnerIters.values.mean(), 0))

    # Gmres first
    table += lformat.format(pc="approx. Schur.",
                            total_time=gmres_total,
                            outerits=gmres_outerits,
                            innerits=gmres_innerits)

    ratio = gmres_total/hybrid_total
    table += r"""
\multirow{2}{*}{%.3f} \\""" % ratio

    # Hybridization
    table += lformat.format(pc="hybridization",
                            total_time=hybrid_total,
                            outerits=hybrid_outerits,
                            innerits=hybrid_innerits)
    table += r"""\\
\hline"""

table += r"""
\end{tabular}"""

print(table)
