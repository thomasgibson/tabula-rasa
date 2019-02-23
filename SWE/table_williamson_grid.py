import os
import sys
import pandas as pd
from firedrake import *


params = [("RT", 8, 62.5),
          ("BDM", 8, 62.5)]

gmres_data = ["results/gmres_%s_profile_W5_ref%d_Dt%s_NS20.csv" % param
              for param in params]
hybrid_data = ["results/hybrid_%s_profile_W5_ref%d_Dt%s_NS20.csv" % param
               for param in params]

for data in gmres_data + hybrid_data:
    if not os.path.exists(data):
        print("Cannot find data file '%s'" % data)
        sys.exit(1)


method_map = {("RT", 1): "{$\\text{RT}_1 \\times \\text{DG}_0$}",
              ("BDM", 2): "{$\\text{BDM}_2 \\times \\text{DG}_1$}"}


gmres_dfs = [pd.read_csv(d) for d in gmres_data]
hybrid_dfs = [pd.read_csv(d) for d in hybrid_data]

# Grid and discretization information
table = r"""\begin{tabular}{ccccccc}
\hline
"""

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

table += r"""\multicolumn{7}{c}{\textbf{Discretization properties}} \\
\multicolumn{2}{c}{\multirow{2}{*}{Mixed method}} & \multirow{2}{*}{\# cells} &
\multirow{2}{*}{$\Delta x$} & Velocity & Depth & \multirow{2}{*}{Total}  \\
& & &  unknowns & unknowns & \\ \hline
"""


mtd_format = r"""{method} & {cells} & {dx} & {udofs} & {ddofs} & {total} \\
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
    ncells = df.num_cells.values[0]
    dxmax = df.DxMax.values[0]

    table += r"""\multicolumn{2}{c}"""
    table += mtd_format.format(method=method_map[(method, deg)],
                               cells=ncells,
                               dx=dxmax/1000,
                               udofs=udofs,
                               ddofs=ddofs,
                               total=total_dofs)

table += r"""\hline
\end{tabular}
"""

print(table)
