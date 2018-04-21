import LDGH

from argparse import ArgumentParser
from firedrake.petsc import PETSc


parser = ArgumentParser(description="""Plot HDG solutions.""",
                        add_help=False)

parser.add_argument("--r", action="store", default=3, type=int,
                    help="Resolution parameter.")

parser.add_argument("--degree", action="store", default=1, type=int,
                    help="Approximation degree.")

parser.add_argument("--tau", action="store", default="1",
                    choices=["1", "h", "1/h"],
                    help="Tau order.")

parser.add_argument("--help", action="store_true", help="Show help.")


args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(0)


LDGH.run_single_test(r=args.r,
                     degree=args.degree,
                     tau_order=args.tau,
                     write=True)
