from __future__ import absolute_import, print_function, division


# TODO: Add detailed documentation about the various eigen
# parameters
eigen_factorizations = [
    "householderQr",
    "colPivHouseholderQr",
    "fullPivHouseholderQr",
    "partialPivLu",
    "fullPivLu",
    "llt",
    "ldlt",
    "jacobiSvd"
]

parameters = {

    # By default, inverses are computed directly. You may
    # not want to do this.
    "inverse_factor": None,

    # Here, colPivHouseholderQR is a QR decomposition with
    # column pivoting. It's a good default, as it works for
    # all matrices while being reasonably fast.
    "local_solve": "colPivHouseholderQr"
}


def default_parameters():
    return parameters.copy()
