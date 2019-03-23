import HMM

from argparse import ArgumentParser
from firedrake.petsc import PETSc


# Test cases determined by (degree, mixed method)
test_cases = [(0, "RT"), (1, "RT"), (2, "RT"), (3, "RT"),
              (0, "RTCF"), (1, "RTCF"), (2, "RTCF"), (3, "RTCF"),
              (1, "BDM"), (2, "BDM"), (3, "BDM")]


parser = ArgumentParser(description="""Run hybrid-mixed convergence test.""",
                        add_help=False)

parser.add_argument("--start", action="store", default=1, type=int,
                    help="Starting resolution parameter.")

parser.add_argument("--end", action="store", default=7, type=int,
                    help="Resolution parameter to end at.")

parser.add_argument("--help", action="store_true", help="Show help.")


args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)


# Generates all CSV files by call the convergence test script
for test_case in test_cases:
    assert args.start < args.end
    degree, method = test_case
    HMM.run_mixed_hybrid_convergence(degree, method,
                                     start=args.start,
                                     end=args.end)
