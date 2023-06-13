"""@file nfw_funcs.py
NFW functions for footprints
"""
import numpy as np


# pylint: disable=invalid-name
def nfw2D_profile_unnorm(radius, r_scale):
    """
    Approximation for NFW 2D Profile, non normalized

    Parameters
    ----------
    radius: array, float
        Radius position on the profile in :math:`Mpc`
    r_scale: float
        Scale radius for normalization in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`

    Notes
    -----
        taken from section 3.1 of arXiv:1104.2089
    """
    xvals = radius / r_scale
    func = np.ones_like(xvals)
    xpls, xmns = xvals[xvals > 1], xvals[xvals < 1]
    func[xvals > 1] -= (2.0 / (xpls**2 - 1.0) ** 0.5) * np.arctan(
        (-(1.0 - xpls) / (1.0 + xpls)) ** 0.5
    )
    func[xvals < 1] -= (2.0 / (1.0 - xmns**2) ** 0.5) * np.arctanh(
        ((1.0 - xmns) / (1.0 + xmns)) ** 0.5
    )
    sigma = np.ones_like(xvals) / 3.0
    sigma[xvals != 1] = func[xvals != 1] / (xvals[xvals != 1] ** 2 - 1.0)
    return sigma


def nfw2D_profile(radius, r_cluster, r_scale):
    """
    Approximation for NFW 2D Profile

    Parameters
    ----------
    radius: array, float
        Radius position on the profile in :math:`Mpc`
    r_cluster: float
        Cluster radius in :math:`Mpc`
    r_scale: float
        Scale radius for normalization in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`

    Notes
    -----
        taken from section 3.1 of arXiv:1104.2089, validated with r_scale = 0.15 :math:`Mpc/h`
        (0.214 :math:`Mpc`) and r_core = 0.1 :math:`Mpc/h` (0.142 :math:`Mpc`).
    """
    rho = np.log(r_cluster)
    c_nfw = np.exp(
        1.6517
        - 0.5479 * rho
        + 0.1382 * rho**2
        - 0.0719 * rho**3
        - 0.01582 * rho**4
        - 0.00085499 * rho**5
    )  # valid for given values of r_scale and r_core  / and 0.001<r_cluster<3.
    return nfw2D_profile_unnorm(radius, r_scale) * c_nfw


def _flatcore_comp(xvals, xmin, func, *args):
    """
    Compute distribution with flat core


    Parameters
    ----------
    xvals: array
        Values to compute the distribution at
    xmin: float
        Size of the core
    func: function
        Function of the distrubution, it must have the form `f(xvals, *args)`
    *args
        other parameters of the function

    Returns
    -------
    array
        Function `f` computed at `xvals` with `f(xvals<xmin)=f(xmin)`.
    """
    out = func(xmin * np.ones(1), *args)[0] * np.ones_like(xvals)
    out[xvals > xmin] = func(xvals[xvals > xmin], *args)
    return out


def nfw2D_profile_flatcore(radius, r_cluster, r_scale, r_core):
    """
    Approximation for NFW 2D Profile with a top-hat core

    Parameters
    ----------
    radius: array, float
        Radius position on the profile in :math:`Mpc`
    r_cluster: float
        Cluster radius in :math:`Mpc`
    r_scale: float
        Scale radius for normalization in :math:`Mpc`
    r_core: float
        Core radius of the profile in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`
    """
    return _flatcore_comp(radius, r_core, nfw2D_profile, r_cluster, r_scale)


def nfw2D_profile_flatcore_unnorm(radius, r_scale, r_core):
    """
    Approximation for NFW 2D Profile with a top-hat core

    Parameters
    ----------
    radius: array, float
        Radius position on the profile in :math:`Mpc`
    r_scale: float
        Scale radius for normalization in :math:`Mpc`
    r_core: float
        Core radius of the profile in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`
    """
    return _flatcore_comp(radius, r_core, nfw2D_profile_unnorm, r_scale)
