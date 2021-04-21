#!/usr/bin/env python
import numpy as np

def nfw2D_profile(R, Rc, Rs):
    '''
    Approximation for NFW 2D Profile

    Parameters
    ----------
    R: array, float
        Radius position on the profile in :math:`Mpc`
    Rc: float
        Central radius parameter of the profile
    Rs: float
        Scale radius for normalization in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`

    Notes
    -----
        taken from section 3.1 of arXiv:1104.2089,
        validated with Rs = 0.15 Mpc/h (0.214 :math:`Mpc`) and Rcore = 0.1 Mpc/h (0.142 :math:`Mpc`).
    '''
    x = R / Rs
    rho = np.log(Rc)
    f = np.ones(x.size)
    xp, xm = x[x>1], x[x<1]
    f[x>1] -= (2./(xp**2-1.)**0.5) * np.arctan((-(1.-xp)/(1.+xp))**0.5)
    f[x<1] -= (2./(1.-xm**2)**0.5) * np.arctanh(((1.-xm)/(1.+xm))**0.5)
    kNFW = np.exp(1.6517-0.5479*rho+0.1382*rho**2
            -0.0719*rho**3-0.01582*rho**4
            -0.00085499*rho**5) # valid for given values of Rs and Rcore  / and 0.001<Rc<3.
    Sigma = np.ones(x.size) * kNFW/12.
    Sigma[x!=1] = kNFW*f[x!=1]/(x[x!=1]**2-1.)
    return Sigma
def nfw2D_profile_flatcore(R, Rc, Rs, Rcore):
    '''
    Approximation for NFW 2D Profile with a top-hat core

    Parameters
    ----------
    R: array, float
        Radius position on the profile in :math:`Mpc`
    Rc: float
        Central radius parameter of the profile
    Rs: float
        Scale radius for normalization in :math:`Mpc`
    Rc: float
        Core radius of the profile in :math:`Mpc`

    Returns
    -------
    Sigma: array, float
        Surface density in units of :math:`1/Mpc^2`
    '''
    Sigma = nfw2D_profile(R, Rc, Rs)
    Sigma_core = nfw2D_profile(Rcore*np.ones(1), Rc, Rs)[0]
    Sigma[R<Rcore] = Sigma_core
    return Sigma
