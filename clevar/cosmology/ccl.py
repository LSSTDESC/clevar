# Functions to model halo profiles

import numpy as np
import warnings

from ..constants import Constants as const

from .parent_class import Cosmology

__all__ = []


def _patch_rho_crit_to_cd2018(rho_crit_external):
    r"""Convertion factor for rho_crit of any external module to
    CODATA 2018+IAU 2015

    rho_crit_external: float
        Critical density of the Universe in units of :math:`M_\odot\ Mpc^{-3}`
    """

    rhocrit_mks = 3.0 * 100.0 * 100.0 / (8.0 * np.pi * const.GNEWT.value)
    rhocrit_cd2018 = (
        rhocrit_mks * 1000.0 * 1000.0 * const.PC_TO_METER.value * 1.0e6 / const.SOLAR_MASS.value
    )

    return rhocrit_cd2018 / rho_crit_external


class CCLCosmology(Cosmology):
    def __init__(self, **kwargs):
        import pyccl as ccl

        self.ccl = ccl

        super(CCLCosmology, self).__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = "ccl"

        assert isinstance(self.be_cosmo, self.ccl.Cosmology)

        # cor factor for sigma_critical
        self.cor_factor = _patch_rho_crit_to_cd2018(self.ccl.physical_constants.RHO_CRITICAL)

    def _init_from_cosmo(self, be_cosmo):
        assert isinstance(be_cosmo, self.ccl.Cosmology)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):
        self.be_cosmo = self.ccl.Cosmology(
            Omega_c=Omega_dm0,
            Omega_b=Omega_b0,
            Omega_k=Omega_k0,
            h=H0 / 100.0,
            sigma8=0.8,
            n_s=0.96,
            T_CMB=0.0,
            Neff=0.0,
            transfer_function="bbks",
            matter_power_spectrum="linear",
        )

    def _set_param(self, key, value):
        raise NotImplementedError("CCL do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            return self.ccl.omega_x(self.be_cosmo, 1.0, "matter")
        elif key == "Omega_b0":
            return self.be_cosmo["Omega_b"]
        elif key == "Omega_dm0":
            return self.be_cosmo["Omega_c"]
        elif key == "Omega_k0":
            return self.be_cosmo["Omega_k"]
        elif key == "h":
            return self.be_cosmo["h"]
        elif key == "H0":
            return self.be_cosmo["h"] * 100.0
        else:
            raise ValueError(f"Unsupported parameter {key}")

    def get_Omega_m(self, z):
        return self.ccl.omega_x(self.be_cosmo, 1.0 / (1.0 + z), "matter")

    def get_E2(self, z):
        a = 1.0 / (1.0 + z)
        return (self.ccl.h_over_h0(self.be_cosmo, a)) ** 2

    def eval_da_z1z2(self, z1, z2):
        a1 = 1.0 / (1.0 + z1)
        a2 = 1.0 / (1.0 + z2)
        return np.vectorize(self.ccl.angular_diameter_distance)(self.be_cosmo, a1, a2)
