import os
import sys
import pandas as pd


cfl_range = [2, 8]

lo_rt_data = ["results/hybrid_RT1_GW_ref6_nlayers85_CFL%d.csv" %
              cfl for cfl in cfl_range]
nlo_rt_data = ["results/hybrid_RT2_GW_ref4_nlayers85_CFL%d.csv" %
               cfl for cfl in cfl_range]
lo_rtcf_data = ["results/hybrid_RTCF1_GW_ref7_nlayers85_CFL%d.csv" %
                cfl for cfl in cfl_range]
nlo_rtcf_data = ["results/hybrid_RTCF2_GW_ref5_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]
nlo_bdfm_data = ["results/hybrid_BDFM2_GW_ref4_nlayers85_CFL%d.csv" %
                 cfl for cfl in cfl_range]

for data in (lo_rtcf_data + nlo_rtcf_data +
             lo_rt_data + nlo_rt_data + nlo_bdfm_data):
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


lo_rt2, lo_rt8 = [pd.read_csv(d) for d in lo_rt_data]
nlo_rt2, nlo_rt8 = [pd.read_csv(d) for d in nlo_rt_data]
lo_rtcf2, lo_rtcf8 = [pd.read_csv(d) for d in lo_rtcf_data]
nlo_rtcf2, nlo_rtcf8 = [pd.read_csv(d) for d in nlo_rtcf_data]
nlo_bdfm2, nlo_bdfm8 = [pd.read_csv(d) for d in nlo_bdfm_data]


labels = ["$\\text{RT}_1$", "$\\text{RT}_2$",
          "$\\text{BDFM}_2$",
          "$\\text{RTCF}_1$", "$\\text{RTCF}_2$"]


table = r"""\begin{tabular}{c|ccccc}\hline
\multicolumn{6}{c}{\textbf{Execution time (s)}} \\
CFL & $\text{RT}_1$ & $\text{RT}_2$ & $\text{BDFM}_2$ & $\text{RTCF}_1$ & $\text{RTCF}_2$ \\
\hline
"""

formater = r"""{cfl} & {rtlo_time}s & {rtnlo_time}s & {bdfm_time}s & {rtcflo_time}s & {rtcfnlo_time}s \\
"""

cfl2 = lo_rt2.CFL.values[0]
assert 2 == cfl2
assert cfl2 == nlo_rt2.CFL.values[0]
assert cfl2 == nlo_bdfm2.CFL.values[0]
assert cfl2 == lo_rtcf2.CFL.values[0]
assert cfl2 == nlo_rtcf2.CFL.values[0]

table += formater.format(cfl=cfl2,
                         rtlo_time=lo_rt2.HybridFullSolveTime.values[0],
                         rtnlo_time=nlo_rt2.HybridFullSolveTime.values[0],
                         bdfm_time=nlo_bdfm2.HybridFullSolveTime.values[0],
                         rtcflo_time=lo_rtcf2.HybridFullSolveTime.values[0],
                         rtcfnlo_time=nlo_rtcf2.HybridFullSolveTime.values[0])

cfl8 = lo_rt8.CFL.values[0]
assert 8 == cfl8
assert cfl8 == nlo_rt8.CFL.values[0]
assert cfl8 == nlo_bdfm8.CFL.values[0]
assert cfl8 == lo_rtcf8.CFL.values[0]
assert cfl8 == nlo_rtcf8.CFL.values[0]

table += formater.format(cfl=cfl8,
                         rtlo_time=lo_rt8.HybridFullSolveTime.values[0],
                         rtnlo_time=nlo_rt8.HybridFullSolveTime.values[0],
                         bdfm_time=nlo_bdfm8.HybridFullSolveTime.values[0],
                         rtcflo_time=lo_rtcf8.HybridFullSolveTime.values[0],
                         rtcfnlo_time=nlo_rtcf8.HybridFullSolveTime.values[0])


table += r"""\hline
\end{tabular}
"""

print(table)
