#### Tabula-Rasa
Experimentation repository for the manuscript:

> Thomas H. Gibson, Lawrence Mitchell, David A. Ham, and Colin J. Cotter.
> "A domain-specific language for the hybridization and static condensation of finite element methods".

#### Usage

This repository requires a latest version of Firedrake (https://github.com/firedrakeproject/firedrake).
The paper above contains documentation and references to specific versions of Firedrake used to generate
the results in the paper and its supplement.

Once Firedrake has been obtained and installed, start the virtual environment in each shell from which
you use Firedrake:

```
source firedrake/bin/activate
```

Once the venv is activated, you may run the python scripts in this repository via the standard way:

```
python3 script.py
```

Or in parallel:

```
mpiexec -n N python3 script.py
```

To reproduce the results, you will need to clone the repository SCPC: https://github.com/thomasgibson/scpc,
which contains the python implementations of the static condensation and hybridization preconditioners.
Once SCPC is cloned, simply run `python3 setup.py install` while the Firedrake venv is active. Once it
has finished, all scripts should run correctly.

#### Collecting data

There are four directorys containing scripts for generating and collecting data. The folders
and corresponding run scripts are:

- `HMM/run_hybrid_mixed_mtd_convergence.py` (Convergence tests for hybridized mixed methods)
- `LDGH/run_LDGH_convergence.py` (Convergence tests for the hybridized DG method)
- `HDG_CG_Comp/run_cg.py` (Profiling script for the CG discretization of the 3D elliptic problem) 
- `HDG_CG_Comp/run_hdg.py` (Profiling script for the HDG discretization of the 3D elliptic problem) 
- `SWE/swe_williamson5.py` (Nonlinear shallow water system on a sphere with an isolated mountain)

Run `python3 scripy.py --help` for a summary of script arguments. The `results` directory contains raw
data used in the paper. The python scripts in that directory can be used to reproduce the plots and
tables used in the manuscript.
