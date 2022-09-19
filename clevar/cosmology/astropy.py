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
        self.backend = 'astropy'

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

    def get_E2(self, z):
        return (self.be_cosmo.H(z)/self.be_cosmo.H0)**2

    def eval_da_z1z2(self, z1, z2):
        return self.be_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.Mpc)
