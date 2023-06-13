"""@file astropy.py
Cosmology using AstroPy
"""
from astropy import units
from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from .parent_class import Cosmology

__all__ = []


class AstroPyCosmology(Cosmology):
    """
    Cosmology object

    Attributes
    ----------
    backend: str
        Name of back-end used
    be_cosmo: cosmology library
        Cosmology library used in the back-end
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # this tag will be used to check if the cosmology object is accepted by the modeling
        self.backend = "astropy"

        assert isinstance(self.be_cosmo, LambdaCDM)

    def _init_from_cosmo(self, be_cosmo):
        assert isinstance(be_cosmo, LambdaCDM)
        self.be_cosmo = be_cosmo

    def _init_from_params(self, H0, Omega_b0, Omega_dm0, Omega_k0):
        # pylint: disable=arguments-differ
        kwargs = {
            "H0": H0,
            "Om0": Omega_b0 + Omega_dm0,
            "Ob0": Omega_b0,
            "Tcmb0": 2.7255,
            "Neff": 3.046,
            "m_nu": ([0.06, 0.0, 0.0] * units.eV),
        }
        self.be_cosmo = FlatLambdaCDM(**kwargs)
        if Omega_k0 != 0.0:
            kwargs["Ode0"] = self.be_cosmo.Ode0 - Omega_k0
            self.be_cosmo = LambdaCDM(**kwargs)

    def _set_param(self, key, value):
        raise NotImplementedError("Astropy do not support changing parameters")

    def _get_param(self, key):
        if key == "Omega_m0":
            return self.be_cosmo.Om0
        if key == "Omega_b0":
            return self.be_cosmo.Ob0
        if key == "Omega_dm0":
            return self.be_cosmo.Odm0
        if key == "Omega_k0":
            return self.be_cosmo.Ok0
        if key == "h":
            return self.be_cosmo.H0.to_value() / 100.0
        if key == "H0":
            return self.be_cosmo.H0.to_value()
        raise ValueError(f"Unsupported parameter {key}")

    def get_Omega_m(self, z):
        return self.be_cosmo.Om(z)

    def get_E2(self, z):
        return (self.be_cosmo.H(z) / self.be_cosmo.H0) ** 2

    def eval_da_z1z2(self, z1, z2):
        return self.be_cosmo.angular_diameter_distance_z1z2(z1, z2).to_value(units.Mpc)
