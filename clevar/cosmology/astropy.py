# Functions to model halo profiles

import numpy as np
import warnings

from astropy import units
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from .. constants import Constants as const

from .parent_class import Cosmology

__all__ = []


class AstroPyCosmology(Cosmology):

    def __init__(self, **kwargs):
        super(AstroPyCosmology, self).__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = 'ct'

        assert isinstance(self.be_cosmo, LambdaCDM)

    def _init_from_cosmo(self, be_cosmo):

        assert isinstance(be_cosmo, LambdaCDM)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):

        Om0 = Omega_b0+Omega_dm0
        Ob0 = Omega_b0
        Ode0 = 1.0-Om0-Omega_k0

        self.be_cosmo = LambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Ode0=Ode0)

    def _set_param(self, key, value):
        raise NotImplementedError("Astropy do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            return self.be_cosmo.Om0
        elif key == "Omega_b0":
            return self.be_cosmo.Ob0
        elif key == "Omega_dm0":
            return self.be_cosmo.Odm0
        elif key == "Omega_k0":
            return self.be_cosmo.Ok0
        elif key == 'h':
            return self.be_cosmo.H0.to_value()/100.0
        elif key == 'H0':
            return self.be_cosmo.H0.to_value()
        else:
            raise ValueError(f"Unsupported parameter {key}")

    def get_Omega_m(self, z):
        return self.be_cosmo.Om(z)

    def get_E2Omega_m(self, z):
        return self.be_cosmo.Om(z)*(self.be_cosmo.H(z)/self.be_cosmo.H0)**2

    def eval_da_z1z2(self, z1, z2):
        return self.be_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.Mpc)

    def eval_sigma_crit(self, z_len, z_src):
        if np.any(np.array(z_src)<=z_len):
            warnings.warn(f'Some source redshifts are lower than the cluster redshift. Returning Sigma_crit = np.inf for those galaxies.')
        # Constants
        clight_pc_s = const.CLIGHT_KMS.value*1000./const.PC_TO_METER.value
        gnewt_pc3_msun_s2 = const.GNEWT.value*const.SOLAR_MASS.value/const.PC_TO_METER.value**3

        d_l = self.eval_da_z1z2(0, z_len)
        d_s = self.eval_da_z1z2(0, z_src)
        d_ls = self.eval_da_z1z2(z_len, z_src)

        beta_s = np.maximum(0., d_ls/d_s)
        return clight_pc_s**2/(4.0*np.pi*gnewt_pc3_msun_s2)*1/d_l*np.divide(1., beta_s)*1.0e6
