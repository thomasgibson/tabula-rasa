import LDGH
import sys


r = int(sys.argv[1])
degree = int(sys.argv[2])
tau_order = sys.argv[3]


LDGH.run_single_test(r=r,
                     degree=degree,
                     tau_order=tau_order,
                     write=True)
