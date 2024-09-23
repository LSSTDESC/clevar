import sys
import pytest
import importlib
import os
from clevar.cosmology import AstroPyCosmology, CCLCosmology
from clevar import optional_libs


@pytest.fixture(scope="module", params=[AstroPyCosmology, CCLCosmology])
def CosmoClass(request):
    param = request.param
    if optional_libs.ccl is None and param == CCLCosmology:
        pytest.skip(f"Missing backend '{param}'.")
    return param


@pytest.fixture(
    scope="module",
    params=[
        {},
        {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.0},
        {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.01},
    ],
)
def cosmo_init(request):
    return request.param
