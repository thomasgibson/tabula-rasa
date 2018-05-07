import os
import sys
import pandas as pd
from firedrake import *


params = [("RT", 1, 8, 28.125),
          ("BDM", 2, 8, 28.125)]

gmres_data = ["SWE/gmres_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
              for param in params]
hybrid_data = ["SWE/hybrid_%s%s_profile_W5_ref%d_Dt%s_NS100.csv" % param
               for param in params]

for data in gmres_data + hybrid_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


method_map = {("RT", 1): "{$RT_1 \\times DG_0$}",
              ("BDM", 2): "{$BDM_2 \\times DG_1$}"}


gmres_dfs = [pd.read_csv(d) for d in gmres_data]
hybrid_dfs = [pd.read_csv(d) for d in hybrid_data]

# Grid and discretization information
table = r"""\begin{tabular}{ccccc}
\hline
\multicolumn{5}{c}{\textbf{Grid information}} \\
\multirow{2}{*}{Refinements} &
Number of & $\Delta x_{\text{min}}$ & $\Delta x_{\text{max}}$ &
\multirow{2}{*}{$\Delta t$ (s)}\\
& cells & (km) & (km) & \\ \hline
"""

# All have the same grid information recorded
df = gmres_dfs[0]

grid_format = r"""{refinements} & {cells} & {dxmin: .3f} & {dxmax: .3f} & {dt}\\
\hline
"""

table += grid_format.format(refinements=df.refinement_level.values[0],
                            cells=df.num_cells.values[0],
                            dxmin=df.DxMin.values[0]/1000,
                            dxmax=df.DxMax.values[0]/1000,
                            dt=df.Dt.values[0])

# Discretization information
# build the mesh and function spaces to get u and D dofs
R = 6371220.0
mesh = OctahedralSphereMesh(R, 8, degree=2, hemisphere="both")
RT1 = FunctionSpace(mesh, "RT", 1)
DG0 = FunctionSpace(mesh, "DG", 0)
BDM2 = FunctionSpace(mesh, "BDM", 2)
DG1 = FunctionSpace(mesh, "DG", 1)
uRT1 = Function(RT1)
dDG0 = Function(DG0)
uBDM2 = Function(BDM2)
dDG1 = Function(DG1)

table += r"""\multicolumn{5}{c}{\textbf{Discretization properties}} \\
\multicolumn{2}{c}{\multirow{2}{*}{Mixed method}} &
Velocity & Depth & \multirow{2}{*}{Total} \\
& & unknowns & unknowns & \\ \hline
"""


mtd_format = r"""{method} & {udofs} & {ddofs} & {total} \\
"""
for df in gmres_dfs:
    method = df.method.values[0]
    deg = df.model_degree.values[0]
    if method == "RT":
        udofs = uRT1.dof_dset.layout_vec.getSize()
        ddofs = dDG0.dof_dset.layout_vec.getSize()
    else:
        assert method == "BDM"
        udofs = uBDM2.dof_dset.layout_vec.getSize()
        ddofs = dDG1.dof_dset.layout_vec.getSize()

    total_dofs = df.total_dofs.values[0]
    assert total_dofs == udofs + ddofs

    table += r"""\multicolumn{2}{c}"""
    table += mtd_format.format(method=method_map[(method, deg)],
                               udofs=udofs,
                               ddofs=ddofs,
                               total=total_dofs)

table += r"""\hline
\end{tabular}
"""

print(table)
