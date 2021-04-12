import numpy as np
import pylab as plt
from astropy.table import Table

def gen_cluster(N_clusters=1000, f1=0.7, f2=0.7, ra_min=0, ra_max=360,
                dec_min=-90, dec_max=90, z_min=0.05, z_max=2, logm_min=13):
    """
    Generate two cluster catalogs with some relative completeness and purity.
    IMPORTANT: The output catalog distributions are not realistic.

    Parameters
    ----------
    N_clusters: int
        Number of initial clusters to be generated
    f1: float
        Fraction to be assigned to the first catalog
    f2: float
        Fraction to be assigned to the second catalog
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

    Returns
    -------
    input1, input2: astropy.table.Table
        Cluster catalogs with relative completeness and purity.
    """
    Data0 = Table({
        'RA': ra_min+np.random.rand(N_clusters)*(ra_max-ra_min),
        'DEC': dec_min+np.random.rand(N_clusters)*(dec_max-dec_min),
        'Z': z_min+np.random.rand(N_clusters)*(z_max-z_min),
        'RADIUS_ARCMIN': np.random.rand(N_clusters),
    })
    m00 = logm_min+2*np.random.rand(N_clusters*10)
    prob_m = np.exp(-5*(m00-logm_min)**2)
    Data0['MASS'] = 10**np.random.choice(m00, N_clusters,
                                         p=prob_m/prob_m.sum(),
                                         replace=False)
    Data0['MASS_ERR'] = Data0['MASS']*np.random.normal(loc=.2+0*Data0['MASS'], scale=.05)
    Data0['Z_ERR'] = (1+Data0['Z'])*np.random.normal(loc=.03+0*Data0['Z'], scale=.005)

    cls1 = np.random.choice(range(N_clusters), int(N_clusters*f1), replace=False)
    input1 = Data0[cls1]
    x = (Data0['MASS']/1e14)
    comp = (x**3)/(x**3+1)
    pur = (x)/(x+1)
    prob = 1-pur
    prob[cls1] = comp[cls1]
    prob = prob/prob.sum()

    input2 = Data0[np.random.choice(range(N_clusters), int(N_clusters*f2), replace=False, p=prob)]
    input2['RA'] = np.random.normal(loc=input2['RA'], scale=1/3600.)
    input2['DEC'] = np.random.normal(loc=input2['DEC'], scale=1/3600.)
    input2['Z'] = np.random.normal(loc=input2['Z'], scale=.01)
    return input1, input2
