"""Tests for cosmo"""

import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

import clevar
from clevar.cosmology.parent_class import Cosmology


def test_installation_pyccl():
    ccl_safe = clevar.optional_libs.ccl
    clevar.optional_libs.ccl = None
    assert_raises(ImportError, clevar.cosmology.CCLCosmology)
    clevar.optional_libs.ccl = ccl_safe


def test_class():
    """Unit tests abstract class and unimplemented methdods"""
    # Test basic
    assert_raises(TypeError, Cosmology.__getitem__, None, None)
    assert_raises(TypeError, Cosmology.__setitem__, None, None, None)
    # Unimplemented methods
    assert_raises(NotImplementedError, Cosmology._init_from_cosmo, None, None)
    assert_raises(NotImplementedError, Cosmology._init_from_params, None)
    assert_raises(NotImplementedError, Cosmology._set_param, None, None, None)
    assert_raises(NotImplementedError, Cosmology._get_param, None, None)
    assert_raises(AttributeError, Cosmology.set_be_cosmo, None, None)
    assert_raises(NotImplementedError, Cosmology.get_Omega_m, None, None)
    assert_raises(NotImplementedError, Cosmology.eval_da_z1z2, None, None, None)
    assert_raises(AttributeError, Cosmology.eval_da, None, None)
    assert_raises(NotImplementedError, Cosmology.get_E2, None, None)


TOLERANCE = {"rtol": 1.0e-15}


def test_z_and_a(CosmoClass):
    """Unit tests abstract class z and a methdods"""

    cosmo = CosmoClass()

    z = np.linspace(0.0, 10.0, 1000)

    assert_raises(ValueError, cosmo._get_a_from_z, z - 1.0)

    a = cosmo._get_a_from_z(z)

    assert_raises(ValueError, cosmo._get_z_from_a, a * 2.0)

    z_cpy = cosmo._get_z_from_a(a)

    assert_allclose(z_cpy, z, **TOLERANCE)

    a_cpy = cosmo._get_a_from_z(z_cpy)

    assert_allclose(a_cpy, a, **TOLERANCE)

    # Convert from a to z - scalar, list, ndarray
    assert_allclose(cosmo._get_a_from_z(0.5), 2.0 / 3.0, **TOLERANCE)
    assert_allclose(
        cosmo._get_a_from_z([0.1, 0.2, 0.3, 0.4]),
        [10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0],
        **TOLERANCE
    )
    assert_allclose(
        cosmo._get_a_from_z(np.array([0.1, 0.2, 0.3, 0.4])),
        np.array([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0]),
        **TOLERANCE
    )

    # Convert from z to a - scalar, list, ndarray
    assert_allclose(cosmo._get_z_from_a(2.0 / 3.0), 0.5, **TOLERANCE)
    assert_allclose(
        cosmo._get_z_from_a([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0]),
        [0.1, 0.2, 0.3, 0.4],
        **TOLERANCE
    )
    assert_allclose(
        cosmo._get_z_from_a(np.array([10.0 / 11.0, 5.0 / 6.0, 10.0 / 13.0, 5.0 / 7.0])),
        np.array([0.1, 0.2, 0.3, 0.4]),
        **TOLERANCE
    )

    # Some potential corner-cases for the two funcs
    assert_allclose(
        cosmo._get_a_from_z(np.array([0.0, 1300.0])), np.array([1.0, 1.0 / 1301.0]), **TOLERANCE
    )
    assert_allclose(
        cosmo._get_z_from_a(np.array([1.0, 1.0 / 1301.0])), np.array([0.0, 1300.0]), **TOLERANCE
    )

    # Test for exceptions when outside of domains
    assert_raises(ValueError, cosmo._get_a_from_z, -5.0)
    assert_raises(ValueError, cosmo._get_a_from_z, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_a_from_z, np.array([-5.0, 5.0]))
    assert_raises(ValueError, cosmo._get_z_from_a, 5.0)
    assert_raises(ValueError, cosmo._get_z_from_a, [-5.0, 5.0])
    assert_raises(ValueError, cosmo._get_z_from_a, np.array([-5.0, 5.0]))

    # Convert from a to z to a (and vice versa)
    testval = 0.5
    assert_allclose(cosmo._get_a_from_z(cosmo._get_z_from_a(testval)), testval, **TOLERANCE)
    assert_allclose(cosmo._get_z_from_a(cosmo._get_a_from_z(testval)), testval, **TOLERANCE)


def test_cosmo_basic(CosmoClass, cosmo_init):
    """Unit tests abstract class z and a methdods"""
    cosmo = CosmoClass(**cosmo_init)
    assert_raises(NotImplementedError, cosmo.__setitem__, "h", 0.5)
    assert isinstance(cosmo.get_desc(), str)
    # Test get_<PAR>(z)
    Omega_m0 = cosmo["Omega_m0"]
    assert_allclose(cosmo.get_Omega_m(0.0), Omega_m0, **TOLERANCE)
    assert_allclose(cosmo.get_E2(0.0), 1.0, **TOLERANCE)
    # Test getting all parameters
    for param in ("Omega_m0", "Omega_b0", "Omega_dm0", "Omega_k0", "h", "H0"):
        cosmo[param]
    # Test params values
    for param in cosmo_init.keys():
        assert_allclose(cosmo_init[param], cosmo[param], **TOLERANCE)
    assert_raises(NotImplementedError, cosmo._set_param, "nonexistent", 0.0)
    # Test missing parameter
    assert_raises(ValueError, cosmo._get_param, "nonexistent")
    # Test da(z) = da12(0, z)
    z = np.linspace(0.0, 10.0, 1000)
    assert_allclose(cosmo.eval_da(z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    assert_allclose(cosmo.eval_da_z1z2(0.0, z), cosmo.eval_da_z1z2(0.0, z), rtol=8.0e-15)
    # Test initializing cosmo
    test_cosmo = CosmoClass(be_cosmo=cosmo.be_cosmo)
    # Test mass2radius
    cosmo.eval_mass2radius(1e14, 0, delta=200, mass_type="background")  # To add test here
    assert_raises(ValueError, cosmo.eval_mass2radius, 1e14, 0, delta=200, mass_type="nonexistent")


def _rad2mpc_helper(dist, redshift, cosmo, do_inverse):
    """Helper function to clean up test_convert_rad_to_mpc. Truth is computed using
    astropy so this test is very circular. Once we swap to CCL very soon this will be
    a good source of truth."""
    d_a = cosmo.eval_da(redshift)  # Mpc
    if do_inverse:
        assert_allclose(cosmo.mpc2rad(dist, redshift), dist / d_a, **TOLERANCE)
    else:
        assert_allclose(cosmo.rad2mpc(dist, redshift), dist * d_a, **TOLERANCE)


def test_convert_rad_to_mpc(CosmoClass):
    """Test conversion between physical and angular units and vice-versa."""
    # Set some default values if I want them
    redshift = 0.25
    cosmo = CosmoClass(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)
    # Test basic conversions each way
    _rad2mpc_helper(0.003, redshift, cosmo, do_inverse=False)
    _rad2mpc_helper(1.0, redshift, cosmo, do_inverse=True)
    # Convert back and forth and make sure I get the same answer
    midtest = cosmo.rad2mpc(0.003, redshift)
    assert_allclose(cosmo.mpc2rad(midtest, redshift), 0.003, **TOLERANCE)
    # Test some different redshifts
    for onez in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        _rad2mpc_helper(0.33, onez, cosmo, do_inverse=False)
        _rad2mpc_helper(1.0, onez, cosmo, do_inverse=True)
    # Test some different H0
    for oneh0 in [30.0, 50.0, 67.3, 74.7, 100.0]:
        _rad2mpc_helper(
            0.33, 0.5, CosmoClass(H0=oneh0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045), do_inverse=False
        )
        _rad2mpc_helper(
            1.0, 0.5, CosmoClass(H0=oneh0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045), do_inverse=True
        )
    # Test some different Omega_M
    for oneomm in [0.1, 0.3, 0.5, 1.0]:
        _rad2mpc_helper(
            0.33,
            0.5,
            CosmoClass(H0=70.0, Omega_dm0=oneomm - 0.045, Omega_b0=0.045),
            do_inverse=False,
        )
        _rad2mpc_helper(
            1.0, 0.5, CosmoClass(H0=70.0, Omega_dm0=oneomm - 0.045, Omega_b0=0.045), do_inverse=True
        )
