import os
import sys
import pandas as pd


data_set = ["hybrid-mixed/H-RTCF-degree-0.csv",
            "hybrid-mixed/H-RTCF-degree-1.csv",
            "hybrid-mixed/H-RTCF-degree-2.csv",
            "hybrid-mixed/H-RTCF-degree-3.csv"]

for data in data_set:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

dfs = pd.concat(pd.read_csv(data) for data in data_set)
groups = dfs.groupby(["Degree"], as_index=False)

table = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{| l | c| c | c | c | c | c | c |}
\hline
\multicolumn{8}{|c|}{RTCF-H method} \\
\hline
\multirow{2}{*}{$k$} & mesh &
\multicolumn{2}{|c|}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
\multicolumn{2}{|c|}{
$\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
\multicolumn{2}{|c|}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} \\
\cline{2-8}
& $r$ & $L^2$-error & rate & $L^2$-error & rate & $L^2$-error & rate \\
"""

lformat = r"""& {mesh: d} & {ScalarErrors:.3e} & {ScalarRates} & {FluxErrors:.3e} & {FluxRates} & {PPScalarErrors:.3e} & {PPScalarRates} \\
"""


def rate(s):
    if s == '---':
        return s
    else:
        return "{s:.3f}".format(s=float(s))


for data in data_set:
    df = pd.read_csv(data)
    df = df.sort_values("Mesh")
    degree = df.Degree.values[0]
    table += r"""
    \hline
    \multirow{5}{*}{%d}
    """ % degree
    for k in df.Mesh:
        sliced = df.loc[lambda x: x.Mesh == k]
        table += lformat.format(mesh=k,
                                ScalarErrors=sliced.ScalarErrors.values[0],
                                ScalarRates=rate(sliced.ScalarConvRates.values[0]),
                                FluxErrors=sliced.FluxErrors.values[0],
                                FluxRates=rate(sliced.FluxConvRates.values[0]),
                                PPScalarErrors=sliced.PostProcessedScalarErrors.values[0],
                                PPScalarRates=rate(sliced.PostProcessedScalarRates.values[0]))

table += r"""\hline
\end{tabular}}
"""
print(table)
