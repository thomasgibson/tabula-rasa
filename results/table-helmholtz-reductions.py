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

dfs = pd.concat(pd.read_csv(data) for data in data_set)
groups = dfs.groupby(["Degree"], as_index=False)

table = r"""Degree & Reductions (avg.): $\frac{\norm{b - A x^*}_2}{\norm{b - A x^0}_2}$\\
\hline
"""

lformat = r"""{degree: d} & {reductions:.4g}\\
"""

for group in groups:
    degree, df = group
    table += lformat.format(degree=degree,
                            reductions=max(df["ResidualReductions"].values))

print(table)
