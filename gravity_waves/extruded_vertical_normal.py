from firedrake import *


__all__ = ['VerticalNormal']


class VerticalNormal(object):

    def __init__(self, mesh):

        self._mesh = mesh
        self._build_khat()

    @property
    def khat(self):

        return self._khat

    def _build_khat(self):

        fs = VectorFunctionSpace(self._mesh, 'DG', 0)
        self._khat = Function(fs)
        host_mesh = self._mesh._base_mesh
        kernel_code = '''build_normal(double **base_coords,
                                          double **normals) {
              const int ndim=3;
              const int nvert=3;
              double dx[2][ndim];
              double xavg[ndim];
              double n[ndim];
              // Calculate vector between the two points
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                for (int j=0;j<2;++j) {
                  dx[j][i] = base_coords[1+j][i] - base_coords[0][i];
                }
              }
              // Calculate normal
              for (int i=0;i<ndim;++i) {
                n[i] = dx[0][(1+i)%3]*dx[1][(2+i)%3]
                     - dx[0][(2+i)%3]*dx[1][(1+i)%3];
              }
              // Calculate vector at centre of edge
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                xavg[i] = 0.0;
                for (int j=0;j<nvert;++j) { // Loop over vertices
                  xavg[i] += base_coords[j][i];
                }
              }
              // Calculate ||n|| and n.x_avg
              double nrm = 0.0;
              double n_dot_xavg = 0.0;
              for (int i=0;i<ndim;++i) {
                nrm += n[i]*n[i];
                n_dot_xavg += n[i]*xavg[i];
              }
              nrm = sqrt(nrm);
              // Orient correctly
              nrm *= (n_dot_xavg<0?-1:+1);
              for (int i=0;i<ndim;++i) {
                normals[0][i] = n[i]/nrm;
              }
            }'''

        kernel = op2.Kernel(kernel_code, 'build_normal')
        base_coords = host_mesh.coordinates
        op2.par_loop(kernel,
                     self._khat.cell_set,
                     base_coords.dat(op2.READ, base_coords.cell_node_map()),
                     self._khat.dat(op2.WRITE, self._khat.cell_node_map()))
