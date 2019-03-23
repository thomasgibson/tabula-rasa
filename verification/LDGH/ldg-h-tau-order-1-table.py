import os
import sys
import pandas as pd


data_set = ["results/LDG-H-d1-tau_order-1.csv",
            "results/LDG-H-d2-tau_order-1.csv",
            "results/LDG-H-d3-tau_order-1.csv"]

for data in data_set:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


table = r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{| l | c | c | c | c | c | c | c | c | c |}
\hline
\multicolumn{10}{|c|}{LDG-H method ($\tau = \mathcal{O}(1)$)} \\
\hline
\multirow{2}{*}{$k$} & mesh &
\multicolumn{2}{|c|}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
\multicolumn{2}{|c|}{
$\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
\multicolumn{2}{|c|}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} &
\multicolumn{2}{|c|}{
$\norm{\boldsymbol{u}-\boldsymbol{u}_h^{\star}}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} \\
\cline{2-10}
& $r$ & $L^2$-error & rate & $L^2$-error & rate & $L^2$-error & rate & $L^2$-error & rate \\
"""

lformat = r"""& {mesh: d} & {ScalarErrors:.3e} & {ScalarRates} & {FluxErrors:.3e} & {FluxRates} & {PPScalarErrors:.3e} & {PPScalarRates} & {PPFluxErrors:.3e} & {PPFluxRates} \\
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
    \multirow{6}{*}{%d}
    """ % degree
    for k in df.Mesh:
        sliced = df.loc[lambda x: x.Mesh == k]
        table += lformat.format(mesh=k,
                                ScalarErrors=sliced.ScalarErrors.values[0],
                                ScalarRates=rate(sliced.ScalarConvRates.values[0]),
                                FluxErrors=sliced.FluxErrors.values[0],
                                FluxRates=rate(sliced.FluxConvRates.values[0]),
                                PPScalarErrors=sliced.PostProcessedScalarErrors.values[0],
                                PPScalarRates=rate(sliced.PostProcessedScalarRates.values[0]),
                                PPFluxErrors=sliced.PostProcessedFluxErrors.values[0],
                                PPFluxRates=rate(sliced.PostProcessedFluxRates.values[0]))

table += r"""\hline
\end{tabular}}
"""
print(table)
