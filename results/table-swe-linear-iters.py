import os
import sys
import pandas as pd
import math


diagnostic_data = ["williamson-test-case-5/hybrid_profile_W5_ref7.csv",
                   "williamson-test-case-5/profile_W5_ref7.csv"]


for data in diagnostic_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)

df = pd.read_csv(diagnostic_data[0])
df1 = pd.read_csv(diagnostic_data[1])

table = r"""\begin{tabular}{|l||c|c|}
\hline
\multirow{2}{*}{\textbf{Stage}}
& \multicolumn{1}{c|}{\textbf{Hybridization}} 
& \multicolumn{1}{c|}{\textbf{Approx. Schur}} \\ \cline{2-3}
& Krylov iterations & Krylov iterations \\
\hline
"""

lformat = r"""{stage} & {Hybriditer} & {SCiter}\\
\hline
"""


table += lformat.format(stage="Outer KSP (GMRES)",
                        Hybriditer="None",  # HybridPC doesn't have an outerKSP
                        SCiter=int(round(df1["OuterIters"].mean(), 0)))
table += lformat.format(stage="Inner KSP (GAMG+CG)",
                        Hybriditer=int(round(df["InnerIters"].mean(), 0)),
                        SCiter=int(round(df1["InnerIters"].mean(), 0)))

table += r"""\end{tabular}
"""

print(table)
