import sys
import pytest
import importlib
import os


@pytest.fixture(scope="module", params=[{}, {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.0}, {"H0": 67.0, "Omega_b0": 0.06, "Omega_dm0": 0.22, "Omega_k0": 0.01}])

def cosmo_init(request):
    param = request.param

    return param


