import hybridmixed


# Test cases determined by (degree, mixed method)
test_cases = [(0, "RT"), (1, "RT"), (2, "RT"), (3, "RT"), (4, "RT"),
              (0, "RTCF"), (1, "RTCF"), (2, "RTCF"), (3, "RTCF"), (4, "RTCF"),
              (1, "BDM"), (2, "BDM"), (3, "BDM"), (4, "BDM")]


# Generates all CSV files by call the convergence test script
for test_case in test_cases:
    degree, method = test_case
    hybridmixed.run_mixed_hybrid_convergence(degree, method)
