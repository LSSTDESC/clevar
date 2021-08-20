"""@file examples/support.py

Support functions for notebooks.
"""
import numpy as np
from astropy.table import Table

def _rand_from_dist(ndata, vmin, vmax, dist, nscale=10):
    """
    Generate random data given a distribution.

    Parameters
    ----------
    ndata: int
        Number of objects.
    vmin: float
        Minimin value
    vmax: float
        Maximum value
    dist: func
        Distribution function
    nscale: int
        Multipling factor for initial number of objects.

    Returns
    -------
    array
        Values with required distribution.
    """
    values_random_large = vmin+(vmax-vmin)*np.random.rand(ndata*nscale)
    weights = dist(values_random_large)
    return np.random.choice(
        values_random_large, ndata,
        p=weights/weights.sum(), replace=False)


def gen_cluster(N_clusters=1000, ra_min=0, ra_max=360,
                dec_min=-90, dec_max=90, z_min=0.05, z_max=2,
                logm_min=13, logm_max=15,
                z_scatter=0.03, lnm_scatter=0.4):
    """
    Generate two cluster catalogs with some relative completeness and purity.
    IMPORTANT: The output catalog distributions are not realistic.

    Parameters
    ----------
    N_clusters: int
        Goal number of clusters to be generated.
    ra_min: float
        Min Ra in degrees
    ra_max: float
        Max Ra ax degrees
    dec_min: float
        Min Dec in degrees
    dec_max: float
        Max Dec in degrees
    z_min: float
        Min redshift in degrees
    z_max: float
        Max redshift ax degrees
    logm_min: float
        Min value for log(M)
    logm_max: float
        Max value for log(M)
    z_scatter: float
        Scatter in redshift (will be multiplied by 1+z)
    lnm_scatter: float
        Scatter in ln(M)

    Returns
    -------
    input1, input2: astropy.table.Table
        Cluster catalogs with relative completeness and purity.
    """
    if z_min<.05:
        raise ValueError('Minimum redshift must be >=0.05')
    if z_max>2.3:
        raise ValueError('Maximum redshift must be <2.3')
    # Approximated fit for DC2, 1e12<M200<1e15.
    dn_dlogm = lambda x: 10**np.poly1d([ -0.4102,   9.6586, -52.4729])(x)
    n_logm = lambda x: 10**np.poly1d([ -0.4882,  11.5986, -63.8458])(x)
    dn_dz = np.poly1d([ -0.56,   4.99, -14.54,  14.14,  -0.62])
    # Create cluster with logm_min0 = logm_min-3*logm_scatter
    logm_min0 = logm_min-3*lnm_scatter/np.log(10)
    if logm_min0<12:
        raise ValueError('Minimum mass in computation ({logm_min0}) is too low.'
                         ' Increase logm_min or decrease lnm_scatter.')
    N_clusters0 = int(N_clusters*n_logm(logm_min0)/n_logm(logm_min))
    print(f'Initial number of clusters (logM>{logm_min0:.2f}): {N_clusters0:,}')
    Data0 = Table({
        'RA': ra_min+np.random.rand(N_clusters0)*(ra_max-ra_min),
        'DEC': dec_min+np.random.rand(N_clusters0)*(dec_max-dec_min),
        'Z': _rand_from_dist(N_clusters0, z_min, z_max, dn_dz, nscale=10),
        'RADIUS_ARCMIN': np.random.rand(N_clusters0),
        'MASS': 10**_rand_from_dist(N_clusters0, logm_min0, logm_max, dn_dlogm, nscale=10),
    })
    Data0['MASS_ERR'] = Data0['MASS']*np.random.normal(loc=lnm_scatter+0*Data0['MASS'], scale=.05)
    Data0['Z_ERR'] = (1+Data0['Z'])*np.random.normal(loc=z_scatter+0*Data0['Z'], scale=.005)
    # Crop catalog 1 with logm>logm_min
    input1 = Data0[Data0['MASS']>=10**logm_min]
    # Crop catalog 2 with logm+noise>logm_min and add noise to other quantities
    mass2 = np.exp(np.random.normal(loc=np.log(Data0['MASS']), scale=lnm_scatter))
    mask2 = mass2>=10**logm_min
    input2 = Table({
        'RA': np.random.normal(loc=Data0['RA'][mask2], scale=1/3600.),
        'DEC': np.random.normal(loc=Data0['DEC'][mask2], scale=1/3600.),
        'Z': np.random.normal(loc=Data0['Z'][mask2], scale=z_scatter),
        'RADIUS_ARCMIN': Data0['RADIUS_ARCMIN'][mask2],
        'MASS': mass2[mask2],
        'MASS_ERR': Data0['MASS_ERR'][mask2],
        'Z_ERR': Data0['Z_ERR'][mask2],
    })
    print(f'Clusters in catalog1: {len(input1):,}')
    print(f'Clusters in catalog2: {len(input2):,}')
    return input1, input2
