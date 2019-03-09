import os
import sys
import pandas as pd


cfl_range = [2]

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


lo_rt, = [pd.read_csv(d) for d in lo_rt_data]
nlo_rt, = [pd.read_csv(d) for d in nlo_rt_data]
lo_rtcf, = [pd.read_csv(d) for d in lo_rtcf_data]
nlo_rtcf, = [pd.read_csv(d) for d in nlo_rtcf_data]
nlo_bdfm, = [pd.read_csv(d) for d in nlo_bdfm_data]


labels = ["$\\text{RT}_1$", "$\\text{RT}_2$",
          "$\\text{BDFM}_2$",
          "$\\text{RTCF}_1$", "$\\text{RTCF}_2$"]


# Grid and discretization information
table = r"""\begin{tabular}{ccccccc}\hline
\multicolumn{7}{c}{\textbf{Discretizations and grid information}} \\
Mixed method & \# horiz. cells & \# vert. layers & $\Delta x$ & $\Delta z$ & $U$-$P$ dofs & Hybrid. dofs \\
\hline
"""

mtd_format = r"""{method} & {ncells} & 85 & {dx} km & {dz} m & {total} M & {hybrid_total} M \\
"""

table += mtd_format.format(method=labels[0],
                           ncells=lo_rt.num_cells.values[0],
                           dx=int(lo_rt.DxMax.values[0] / 1000),
                           dz=int(lo_rt.Dz.values[0]),
                           total=lo_rt.up_dofs.values[0] / 1e6,
                           hybrid_total=(lo_rt.HybridUPDOFS.values[0] +
                                         lo_rt.HybridTraceDOFS.values[0]) / 1e6)
table += mtd_format.format(method=labels[1],
                           ncells=nlo_rt.num_cells.values[0],
                           dx=int(nlo_rt.DxMax.values[0] / 1000),
                           dz=int(nlo_rt.Dz.values[0]),
                           total=nlo_rt.up_dofs.values[0] / 1e6,
                           hybrid_total=(nlo_rt.HybridUPDOFS.values[0] +
                                         nlo_rt.HybridTraceDOFS.values[0]) / 1e6)
table += mtd_format.format(method=labels[2],
                           ncells=nlo_bdfm.num_cells.values[0],
                           dx=int(nlo_bdfm.DxMax.values[0] / 1000),
                           dz=int(nlo_bdfm.Dz.values[0]),
                           total=nlo_bdfm.up_dofs.values[0] / 1e6,
                           hybrid_total=(nlo_bdfm.HybridUPDOFS.values[0] +
                                         nlo_bdfm.HybridTraceDOFS.values[0]) / 1e6)
table += mtd_format.format(method=labels[3],
                           ncells=lo_rtcf.num_cells.values[0],
                           dx=int(lo_rtcf.DxMax.values[0] / 1000),
                           dz=int(lo_rtcf.Dz.values[0]),
                           total=lo_rtcf.up_dofs.values[0] / 1e6,
                           hybrid_total=(lo_rtcf.HybridUPDOFS.values[0] +
                                         lo_rtcf.HybridTraceDOFS.values[0]) / 1e6)
table += mtd_format.format(method=labels[4],
                           ncells=nlo_rtcf.num_cells.values[0],
                           dx=int(nlo_rtcf.DxMax.values[0] / 1000),
                           dz=int(nlo_rtcf.Dz.values[0]),
                           total=nlo_rtcf.up_dofs.values[0] / 1e6,
                           hybrid_total=(nlo_rtcf.HybridUPDOFS.values[0] +
                                         nlo_rtcf.HybridTraceDOFS.values[0]) /1e6)
table += r"""\hline
\end{tabular}
"""

print(table)
