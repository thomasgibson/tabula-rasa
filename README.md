# Tabula-Rasa
Experimentation repository for the manuscript:
> Thomas H. Gibson, Lawrence Mitchell, David A. Ham, and Colin J. Cotter.
> "Slate: extending Firedrake's domain-specific abstraction to hybridized solvers for geoscience and beyond".

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

For code-generation verification, see the documentation and experiment in `verification`.

### Collecting data

There are three main directorys containing scripts for generating and collecting data. The folders and corresponding run scripts are:

- `HDG_CG_Comp/run_cg.py` (Profiling script for the CG discretization of the 3D elliptic problem) 
- `HDG_CG_Comp/run_hdg.py` (Profiling script for the HDG discretization of the 3D elliptic problem) 
- `SWE/swe_williamson5.py` (Nonlinear shallow water system on a sphere with an isolated mountain)
- `gravity_waves/run_profiler.py` (three-dimensional linear Boussinesq system)

Run `python3 scripy.py --help` for a summary of script arguments. The `results` directory contains raw data used in the paper.
The python scripts in that directory can be used to reproduce the plots and tables used in the manuscript.
