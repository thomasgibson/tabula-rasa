import HMM

from argparse import ArgumentParser
from firedrake.petsc import PETSc


parser = ArgumentParser(description="""Plot Hybrid-Mixed solutions.""",
                        add_help=False)

parser.add_argument("--r", action="store", default=3, type=int,
                    help="Resolution parameter.")

parser.add_argument("--order", action="store", default=1, type=int,
                    help="Approximation order.")

parser.add_argument("--method", action="store", default="RT",
                    choices=["RT", "RTCF", "BDM"],
                    help="Tau order.")

parser.add_argument("--help", action="store_true", help="Show help.")


args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)


HMM.run_single_test(r=args.r,
                    degree=args.order,
                    method=args.method)
