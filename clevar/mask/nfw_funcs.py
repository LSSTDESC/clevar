#!/usr/bin/env python
import numpy as np

def nfw2D_profile(R, Rc, Rs):
    '''
    NFW 2D Profile

    Parameters
    ----------
    R: array, float
        Radius position on the profile in Mpc
    Rc: float
        Central radius parameter of the profile
    Rs: float
        Scale radius for normalization in Mpc

    Returns
    -------
    Sigma: array, float
        Surface density in units of 1/Mpc^2

    Notes
    -----
        taken from Rykoff 2012 - section 3.1 & Bartelmann 1996 A&A
        kNFW valid only for    Rs = 0.15/h # 0.214Mpc and Rcore = 0.1/h # 0.142Mpc
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
    NFW 2D Profile with a top-hat core

    Parameters
    ----------
    R: array, float
        Radius position on the profile in Mpc
    Rc: float
        Central radius parameter of the profile
    Rs: float
        Scale radius for normalization in Mpc
    Rc: float
        Core radius of the profile in Mpc

    Returns
    -------
    Sigma: array, float
        Surface density in units of 1/Mpc^2
    '''
    Sigma = nfw2D_profile(R, Rc, Rs)
    Sigma_core = nfw2D_profile(Rcore*np.ones(1), Rc, Rs)[0]
    Sigma[R<Rcore] = Sigma_core
    return Sigma
def nfw_flatcore_window_func(pix_list, ra, dec, z, r,
                             deg1mpc, ftpt, h=0.7):
    '''
    Get window function for NFW 2D Profile with a top-hat core

    Parameters
    ----------
    pix_list: list
        List of pixels in the window
    ra: float
        RA in deg
    dec: float
        DEC in deg
    z: float
        Redshift
    r: float
        Cluster radius in Mpc
    deg1mpc: float
        Size of 1 Mpc in degrees at the cluster redshift
    ftpt: FootprintZmax
        Footprint object
    h: float
        Hubble parameter

    Returns
    -------
    array
        Value of the window function at each pixel
    '''
    Rs = 0.15/h # 0.214Mpc
    Rcore = 0.1/h # 0.142Mpc
    sk = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    R = convert_units(sk.separation(ftpt['SkyCoord'][pix_list]).value, 'degrees', 'mpc')
    return nfw2D_profile_flatcore(R, r, Rs, Rcore)
