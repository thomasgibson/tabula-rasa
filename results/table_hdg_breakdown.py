import os
import pandas as pd

hdg_params = [(8, 1), (8, 2), (8, 3),
              (16, 1), (16, 2), (16, 3),
              (32, 1), (32, 2), (32, 3),
              (64, 1), (64, 2), (64, 3)]
hdg_data = ["HDG_CG_comp/HDG_data_N%d_deg%d.csv" % param
            for param in hdg_params]

for d in hdg_data:
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)

table = r"""\begin{tabular}{llcccccc}
\hline
\multirow{2}{*}{Mesh}
& \multirow{2}{*}{Stage}
& \multicolumn{2}{c}{$HDG_1$}
& \multicolumn{2}{c}{$HDG_2$}
& \multicolumn{2}{c}{$HDG_3$}
\\
& & $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$
& $t_{\text{stage}}$ (s) & \% $t_{\text{total}}$ \\ \hline
"""

hdg_dfs = [pd.read_csv(d) for d in hdg_data]

hdg_dfs_N8 = [df for df in hdg_dfs if df.num_cells.values[0] == 6*8**3]
hdg_dfs_N16 = [df for df in hdg_dfs if df.num_cells.values[0] == 6*16**3]
hdg_dfs_N32 = [df for df in hdg_dfs if df.num_cells.values[0] == 6*32**3]
hdg_dfs_N64 = [df for df in hdg_dfs if df.num_cells.values[0] == 6*64**3]

lformat = r"""& {stage} & {t1: .3g} & {p1: .3g} \% & {t2: .3g} & {p2: .3g} \% & {t3: .3g} & {p3: .3g} \% \\
"""

for df1, df2, df3 in [hdg_dfs_N8, hdg_dfs_N16, hdg_dfs_N32, hdg_dfs_N64]:
    # Make sure everything is organized correctly
    nc1 = df1.num_cells.values[0]
    nc2 = df2.num_cells.values[0]
    nc3 = df3.num_cells.values[0]
    assert nc1 == nc2
    assert nc2 == nc3

    nc = nc1
    if nc == 6*8**3:
        N = 8
    elif nc == 6*16**3:
        N = 16
    elif nc == 6*32**3:
        N = 32
    else:
        assert nc == 6*64**3
        N = 64

    table += r"""
\multirow{6}{*}{$N = %s$}
""" % N

    recovery1 = df1.HDGRecover.values[0]
    pp1 = df1.HDGPPTime.values[0]
    rhs1 = df1.HDGRhs.values[0]
    trace1 = df1.HDGSolve.values[0]
    setup1 = df1.PCSetUp.values[0]
    assembly1 = df1.HDGUpdate.values[0]
    other1 = setup1 - assembly1
    total1 = (recovery1 + pp1 + rhs1 + trace1 + setup1)

    recovery2 = df2.HDGRecover.values[0]
    pp2 = df2.HDGPPTime.values[0]
    rhs2 = df2.HDGRhs.values[0]
    trace2 = df2.HDGSolve.values[0]
    setup2 = df2.PCSetUp.values[0]
    assembly2 = df2.HDGUpdate.values[0]
    other2 = setup2 - assembly2
    total2 = (recovery2 + pp2 + rhs2 + trace2 + setup2)

    recovery3 = df3.HDGRecover.values[0]
    pp3 = df3.HDGPPTime.values[0]
    rhs3 = df3.HDGRhs.values[0]
    trace3 = df3.HDGSolve.values[0]
    setup3 = df3.PCSetUp.values[0]
    assembly3 = df3.HDGUpdate.values[0]
    other3 = setup3 - assembly3
    total3 = (recovery3 + pp3 + rhs3 + trace3 + setup3)

    table += lformat.format(stage="PC setup (assembly)",
                            t1=assembly1,
                            p1=assembly1/total1 * 100.,
                            t2=assembly2,
                            p2=assembly2/total2 * 100.,
                            t3=assembly3,
                            p3=assembly3/total3 * 100.)

    table += lformat.format(stage="PC setup (other)",
                            t1=other1,
                            p1=other1/total1 * 100.,
                            t2=other2,
                            p2=other2/total2 * 100.,
                            t3=other3,
                            p3=other3/total3 * 100.)

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

    table += lformat.format(stage="Post-processing",
                            t1=pp1,
                            p1=pp1/total1 * 100.,
                            t2=pp2,
                            p2=pp2/total2 * 100.,
                            t3=pp3,
                            p3=pp3/total3 * 100.)
    table += r"""\hline
"""

    table += r"""& Total & %.3f & & %.3f & & %.3f & \\ \hline""" % (total1,
                                                                    total2,
                                                                    total3)

table += r"""
\end{tabular}"""

print(table)
