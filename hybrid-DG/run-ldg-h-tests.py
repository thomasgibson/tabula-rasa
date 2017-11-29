import LDGH


# Test cases determined by (degree, tau order)
test_cases = [(1, '1'), (1, 'h'), (1, '1/h'),
              (2, '1'), (2, 'h'), (2, '1/h'),
              (3, '1'), (3, 'h'), (3, '1/h')]


# Generates all CSV files by call the convergence test script
for test_case in test_cases:
    degree, tau_order = test_case
    LDGH.run_LDG_H_convergence(degree, tau_order)
