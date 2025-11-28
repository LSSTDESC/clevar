# pylint: disable=no-member, protected-access
"""Tests for geometry.py"""
import numpy as np
from numpy.testing import assert_raises, assert_allclose

from clevar.geometry import convert_units

TOLERANCE = {"rtol": 1.0e-6, "atol": 0}


def test_convert_units(CosmoClass):
    """Test the wrapper function to convert units. Corner cases should be tested in the
    individual functions. This function should test one case for all supported conversions
    and the error handling.
    """
    # Make an astropy cosmology object for testing
    cosmo = CosmoClass(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045)

    # Test that each unit is supported
    convert_units(1.0, "radians", "degrees")
    convert_units(1.0, "arcmin", "arcsec")
    convert_units(1.0, "Mpc", "kpc")
    convert_units(1.0, "Mpc", "kpc")

    # Error checking
    assert_raises(ValueError, convert_units, 1.0, "radians", "CRAZY")
    assert_raises(ValueError, convert_units, 1.0, "CRAZY", "radians")
    assert_raises(TypeError, convert_units, 1.0, "arcsec", "Mpc")
    assert_raises(TypeError, convert_units, 1.0, "arcsec", "Mpc", None, cosmo)
    assert_raises(TypeError, convert_units, 1.0, "arcsec", "Mpc", 0.5, None)
    assert_raises(ValueError, convert_units, 1.0, "arcsec", "Mpc", -0.5, cosmo)
    assert_raises(ValueError, convert_units, np.ones(2), "arcsec", "Mpc", -0.5 * np.ones(2), cosmo)

    # Test cases to make sure angular -> angular is fitting together
    assert_allclose(convert_units(np.pi, "radians", "degrees"), 180.0, **TOLERANCE)
    assert_allclose(convert_units(180.0, "degrees", "radians"), np.pi, **TOLERANCE)
    assert_allclose(convert_units(1.0, "degrees", "arcmin"), 60.0, **TOLERANCE)
    assert_allclose(convert_units(1.0, "degrees", "arcsec"), 3600.0, **TOLERANCE)

    # Test cases to make sure physical -> physical is fitting together
    assert_allclose(convert_units(1.0, "Mpc", "kpc"), 1.0e3, **TOLERANCE)
    assert_allclose(convert_units(1000.0, "kpc", "Mpc"), 1.0, **TOLERANCE)
    assert_allclose(convert_units(1.0, "Mpc", "pc"), 1.0e6, **TOLERANCE)

    # Test conversion from angular to physical
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_arcmin, redshift = 20.0, 0.5
    d_a = cosmo.eval_da(redshift) * 1.0e3  # kpc
    truth = r_arcmin * (1.0 / 60.0) * (np.pi / 180.0) * d_a
    assert_allclose(convert_units(r_arcmin, "arcmin", "kpc", redshift, cosmo), truth, **TOLERANCE)

    # Test conversion both ways between angular and physical units
    # Using astropy, circular now but this will be fine since we are going to be
    # swapping to CCL soon and then its kosher
    r_kpc, redshift = 20.0, 0.5
    #    d_a = cosmo.angular_diameter_distance(redshift).to('kpc').value
    d_a = cosmo.eval_da(redshift) * 1.0e3  # kpc
    truth = r_kpc * (1.0 / d_a) * (180.0 / np.pi) * 60.0
    assert_allclose(convert_units(r_kpc, "kpc", "arcmin", redshift, cosmo), truth, **TOLERANCE)
