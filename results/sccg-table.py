import os
import sys
import pandas as pd


p4_data = "helmholtz-results/helmholtz_conv-d-4.csv"
p5_data = "helmholtz-results/helmholtz_conv-d-5.csv"
p6_data = "helmholtz-results/helmholtz_conv-d-6.csv"
p7_data = "helmholtz-results/helmholtz_conv-d-7.csv"
data_set = [p4_data, p5_data, p6_data, p7_data]

for data in data_set:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

table = r"""\begin{tabular}{| l | c | c | c |}
\hline
\multicolumn{4}{|c|}{$H^1$ Helmholtz} \\
\hline
\multirow{2}{*}{$k$} & mesh &
\multicolumn{2}{|c|}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} \\
\cline{2-4}
& $r$ & $L^2$-error & rate \\
"""

lformat = r"""& {mesh: d} & {L2Errors:.3e} & {ConvRates} \\
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
                                L2Errors=sliced.L2Errors.values[0],
                                ConvRates=rate(sliced.ConvRates.values[0]))

table += r"""\hline
\end{tabular}
"""
print(table)
