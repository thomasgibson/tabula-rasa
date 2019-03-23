import LDGH

from argparse import ArgumentParser
from firedrake.petsc import PETSc


# Test cases determined by (degree, tau order)
test_cases = [(1, '1'), (1, 'h'), (1, '1/h'),
              (2, '1'), (2, 'h'), (2, '1/h'),
              (3, '1'), (3, 'h'), (3, '1/h')]


parser = ArgumentParser(description="""Run HDG convergence test.""",
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
    degree, tau_order = test_case
    LDGH.run_LDG_H_convergence(degree, tau_order,
                               start=args.start, end=args.end)
