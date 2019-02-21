import os
import pandas as pd

hdg_params = [(64, 1), (64, 2), (64, 3)]
hdg_data = ["results/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

cg_params = [(64, 2), (64, 3), (64, 4)]
cg_data = ["results/CG_data_N%d_deg%d.csv" % param
           for param in cg_params]

for d in hdg_data + cg_data:
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)

table = r"""\begin{tabular}{lcccccc}
\hline
\multirow{2}{*}{Stage}
& \multicolumn{2}{c}{$HDG_1$}
& \multicolumn{2}{c}{$HDG_2$}
& \multicolumn{2}{c}{$HDG_3$}
\\
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$ \\ \hline
"""

hdg_dfs = [pd.read_csv(d) for d in hdg_data]

lformat = r"""{stage} & {t1: .2f} & {p1: .2f} \% & {t2: .2f} & {p2: .2f} \% & {t3: .2f} & {p3: .2f} \% \\
"""

df1, df2, df3 = hdg_dfs

recovery1 = df1.HDGRecover.values[0]
pp1 = df1.HDGPPTime.values[0]
rhs1 = df1.HDGRhs.values[0]
trace1 = df1.HDGTraceSolve.values[0]
assembly1 = df1.HDGUpdate.values[0]
# residual1 = df1.SNESFunctionEval.values[0]
total1 = recovery1 + pp1 + rhs1 + trace1 + assembly1  # + residual1)
snes1 = df1.SNESSolve.values[0]

recovery2 = df2.HDGRecover.values[0]
pp2 = df2.HDGPPTime.values[0]
rhs2 = df2.HDGRhs.values[0]
trace2 = df2.HDGTraceSolve.values[0]
assembly2 = df2.HDGUpdate.values[0]
# residual2 = df2.SNESFunctionEval.values[0]
total2 = recovery2 + pp2 + rhs2 + trace2 + assembly2  # + residual2)
snes2 = df2.SNESSolve.values[0]

recovery3 = df3.HDGRecover.values[0]
pp3 = df3.HDGPPTime.values[0]
rhs3 = df3.HDGRhs.values[0]
trace3 = df3.HDGTraceSolve.values[0]
assembly3 = df3.HDGUpdate.values[0]
# residual3 = df3.SNESFunctionEval.values[0]
total3 = recovery3 + pp3 + rhs3 + trace3 + assembly3  # + residual3
snes3 = df3.SNESSolve.values[0]

table += lformat.format(stage="Matrix assembly (static cond.)",
                        t1=assembly1,
                        p1=assembly1/total1 * 100.,
                        t2=assembly2,
                        p2=assembly2/total2 * 100.,
                        t3=assembly3,
                        p3=assembly3/total3 * 100.)

table += lformat.format(stage="Forward elimination",
                        t1=rhs1,
                        p1=rhs1/total1 * 100.,
                        t2=rhs2,
                        p2=rhs2/total2 * 100.,
                        t3=rhs3,
                        p3=rhs3/total3 * 100.)

table += lformat.format(stage="Trace solve",
                        t1=trace1,
                        p1=trace1/total1 * 100.,
                        t2=trace2,
                        p2=trace2/total2 * 100.,
                        t3=trace3,
                        p3=trace3/total3 * 100.)

table += lformat.format(stage="Back substitution",
                        t1=recovery1,
                        p1=recovery1/total1 * 100.,
                        t2=recovery2,
                        p2=recovery2/total2 * 100.,
                        t3=recovery3,
                        p3=recovery3/total3 * 100.)

table += lformat.format(stage="Post processing",
                        t1=pp1,
                        p1=pp1/total1 * 100.,
                        t2=pp2,
                        p2=pp2/total2 * 100.,
                        t3=pp3,
                        p3=pp3/total3 * 100.)
table += r"""\hline
"""

table += r"""HDG Total & %.2f & & %.2f & & %.2f & \\ \hline""" % (total1,
                                                                  total2,
                                                                  total3)

table += r"""
& \multicolumn{2}{c}{$CG_2$}
& \multicolumn{2}{c}{$CG_3$}
& \multicolumn{2}{c}{$CG_4$}
\\
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$ \\ \hline
"""

cg_dfs = [pd.read_csv(d) for d in cg_data]
df1, df2, df3 = cg_dfs

ksp_solve1 = df1.KSPSolve.values[0]
assembly1 = df1.SNESJacobianEval.values[0]
total1 = assembly1 + ksp_solve1

ksp_solve2 = df2.KSPSolve.values[0]
assembly2 = df2.SNESJacobianEval.values[0]
total2 = assembly2 + ksp_solve2

ksp_solve3 = df3.KSPSolve.values[0]
assembly3 = df3.SNESJacobianEval.values[0]
total3 = assembly3 + ksp_solve3

table += lformat.format(stage="Matrix assembly (monolithic)",
                        t1=assembly1,
                        p1=assembly1/total1 * 100.,
                        t2=assembly2,
                        p2=assembly2/total2 * 100.,
                        t3=assembly3,
                        p3=assembly3/total3 * 100.)

table += lformat.format(stage="Solve",
                        t1=ksp_solve1,
                        p1=ksp_solve1/total1 * 100.,
                        t2=ksp_solve2,
                        p2=ksp_solve2/total2 * 100.,
                        t3=ksp_solve3,
                        p3=ksp_solve3/total3 * 100.)

table += r"""\hline
"""

table += r"""CG Total & %.2f & & %.2f & & %.2f & \\ \hline""" % (total1,
                                                                 total2,
                                                                 total3)

table += r"""
\end{tabular}"""

print(table)
