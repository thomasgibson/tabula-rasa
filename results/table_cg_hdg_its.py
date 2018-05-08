import os
import pandas as pd

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

table = r"""\begin{tabular}{ccccccc}
\hline
\multicolumn{7}{c}{\textbf{Solver details}} \\
$k_{HDG}$ & HDG trace its. &
$\norm{p - p_{\text{HDG}}}_{L^2}$ &
$\norm{p - p^\star_{\text{HDG}}}_{L^2}$ &
CG its.& $\norm{p - p_{\text{CG}}}_{L^2}$ &
$k_{CG}$ \\ \hline
"""

hdg_dfs = [pd.read_csv(d) for d in hdg_data]
cg_dfs = [pd.read_csv(d) for d in cg_data]

lformat = r"""{hdg_deg} & {hdg_its} & {hdg_err: .3g} &
{hdg_pperr: .3g} & {cg_its} & {cg_err: .3g} & {cg_deg} \\
"""

for cg, hdg in zip(cg_dfs, hdg_dfs):

    k_cg = cg.degree.values[0]
    k_hdg = hdg.degree.values[0]
    cg_its = cg.ksp_iters.values[0]
    hdg_its = hdg.ksp_iters.values[0]
    cg_err = cg.true_err.values[0]
    hdg_err = hdg.true_err_u.values[0]
    hdg_pperr = hdg.ErrorPP.values[0]

    table += lformat.format(hdg_deg=k_hdg,
                            hdg_its=hdg_its,
                            hdg_err=hdg_err,
                            hdg_pperr=hdg_pperr,
                            cg_its=cg_its,
                            cg_err=cg_err,
                            cg_deg=k_cg)

table += r"""\hline
\end{tabular}
"""

print(table)
