import pandas as pd
from firedrake import COMM_WORLD
import hdg_problem as HDG
import cg_problem as CG


rtols = [1.0e-1, 1.0e-2,
         1.0e-3, 1.0e-4,
         1.0e-5, 1.0e-6,
         1.0e-7, 1.0e-8]


def run_solvers(degree, size, rtol):

    cgprb = CG.CGProblem(degree=degree, N=size)
    hdgprb = HDG.HDGProblem(degree=degree, N=size)

    cg_params = {'ksp_type': 'cg',
                 'pc_type': 'gamg',
                 'ksp_rtol': rtol,
                 'ksp_monitor_true_residual': True,
                 'mg_levels': {'ksp_type': 'chebyshev',
                               'ksp_max_it': 2,
                               'pc_type': 'bjacobi',
                               'sub_pc_type': 'ilu'}}

    cg_solver = cgprb.solver(parameters=cg_params)

    hdg_params = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'pc_python_type': 'scpc.HybridSCPC',
                  'hybrid_sc': cg_params}

    hdg_solver = hdgprb.solver(parameters=hdg_params)

    cg_solver.solve()
    hdg_solver.solve()

    cg_disc_err = cgprb.err
    hdg_disc_err = hdgprb.err[1]

    true_err_cg = cgprb.true_err
    true_err_hdg = hdgprb.true_err[0]

    return (cg_disc_err, hdg_disc_err, true_err_cg, true_err_hdg)


for degree in [1, 2]:
    for size in [8, 16, 32, 64]:

        alg_errs_cg = []
        alg_errs_hdg = []
        l2_err_cg = []
        l2_err_hdg = []
        for rtol in rtols:

            errs = run_solvers(degree, size, rtol)
            cg_disc_err, hdg_disc_err, true_err_cg, true_err_hdg = errs
            alg_errs_cg.append(cg_disc_err)
            alg_errs_hdg.append(hdg_disc_err)
            l2_err_cg.append(true_err_cg)
            l2_err_hdg.append(true_err_hdg)

        if COMM_WORLD.rank == 0:

            data = {"degree": [degree] * len(l2_err_cg),
                    "size": [size] * len(l2_err_hdg),
                    "L2errorCG": l2_err_cg,
                    "L2errorHDG": l2_err_hdg,
                    "AlgErrCG": alg_errs_cg,
                    "AlgErrHDG": alg_errs_hdg,
                    "rtols": rtols}

            result = "degree_%s_size_%s_conv_data.csv" % (degree, size)
            df = pd.DataFrame(data)
            df.to_csv(result, index=False, mode="w")
