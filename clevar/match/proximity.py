import numpy as np
from .parent import Match
from ..geometry import units_bank, convert_units
from ..catalog import ClData
from ..utils import veclen


class ProximityMatch(Match):
    def __init__(self, ):
        self.type = 'Proximity'
    def multiple(self, cat1, cat2):
        """
        Make the one way multiple matching

        Parameters
        ----------
        cat1, cat2: ClusterObject
            Catalogs to be matched
        nradius: float
            Multiplier of the radius to be used
        radius_use: str (optional)
            Case of radius to be used, can be: max, each, other.
        """
        ra2, dec2, sk2 = (cat2.data[c] for c in ('ra', 'dec', 'SkyCoord'))
        ang2, z2min, z2max = (cat2.mt_input[c] for c in ('ang', 'zmin', 'zmax'))
        ang2max = ang2.max()
        for i, (ra1, dec1, sk1, ang1, z1min, z1max) in enumerate(zip(*(
            [cat1.data[c] for c in ('ra', 'dec', 'SkyCoord')]+
            [cat1.mt_input[c] for c in ('ang', 'zmin', 'zmax')]
            ))):
            # crop in redshift range
            mask = (z2max>=z1min)*(z2min<=z1max)
            if mask.any():
                # makes square crop with radius
                Da = max(ang2max, ang1)
                DaCos = Da / np.cos(np.radians(dec1))
                mask *= (ra2>=ra1-DaCos)*(ra2<ra1+DaCos)*(dec2>=dec1-Da)*(dec2<dec1+Da)
                if mask.any():
                    # makes circular crop
                    dist = sk1.separation(sk2[mask]).value
                    max_dist = self._max_mt_distance(ang1, ang2[mask], TYPE_R='max')
                    for id2 in cat2.data['id'][mask][dist<=max_dist]:
                        cat1.match['multi_self'][i].append(id2)
                        i2 = int(cat2.id_dict[id2])
                        cat2.match['multi_other'][i2].append(cat1.data['id'][i])
        print(f'* {len(cat1.match[veclen(cat1.match["multi_self"])>0]):,}/{cat1.size:,} objects matched.')
    def prep_cat_for_match(self, cat, delta_z, match_radius, n_delta_z=1, n_match_radius=1,
        cosmo=None):
        """
        Adds zmin, zmax and radius to cat.mt_input

        Parameters
        ----------
        cat: clevar.Catalog
            Input Catalog
        delta_z: float, string
            Defines the zmin, zmax for matching. If 'cat' uses redshift properties of the catalog,
            if 'spline.filename' interpolates data in 'filename' (z, zmin, zmax) fmt,
            if float uses dist_z*(1+z),
            if None does not use z.
        match_radius: string
            Defines the radius for matching (in degrees). If 'cat' uses the radius in the catalog,
            else must be in format 'value unit'. (ex: '1 arcsec', '1 Mpc')
        n_delta_z: float
            Number of delta_z to be used in the matching
        n_match_radius: float
            Multiplies the radius of the matchingi
        cosmo: clevar.Cosmology object
            Cosmology object for when radius has physical units
        """
        print('## Prep mt_cols')
        cat.mt_input = ClData()
        # Set zmin, zmax
        if delta_z is None:
            # No z use
            print('* zmin|zmax set to -1|10')
            cat.mt_input['zmin'] = -1*np.ones(cat.size)
            cat.mt_input['zmax'] = 10*np.ones(cat.size)
        elif delta_z == 'cat':
            # Values from catalog
            if 'zmin' in cat.data.colnames and 'zmax' in cat.data.colnames:
                print('* zmin|zmax from cat cols (zmin, zmax)')
                cat.mt_input['zmin'] = self._rescale_z(cat.data['z'], cat.data['zmin'], n_delta_z)
                cat.mt_input['zmax'] = self._rescale_z(cat.data['z'], cat.data['zmax'], n_delta_z)
            # create zmin/zmax if not there
            elif 'z_err' in cat.data.colnames:
                print('* zmin|zmax from cat cols (err)')
                cat.mt_input['zmin'] = cat.data['z']-n_delta_z*cat.data['z_err']
                cat.mt_input['zmax'] = cat.data['z']+n_delta_z*cat.data['z_err']
            else:
                raise ValueError('Catalog must contain zmin, zmax or z_err for this matching.')
        elif isinstance(delta_z, str):
            # zmin/zmax in auxiliar file
            print('* zmin|zmax from aux file')
            k = 3
            zv, zvmin, zvmax = np.loadtxt(delta_z)
            zvmin = self._rescale_z(zv, zvmin, nz)
            zvmax = self._rescale_z(zv, zvmax, nz)
            cat.mt_input['zmin'] = spline(zv, zvmin, k=k)(cat.data['z'])
            cat.mt_input['zmax'] = spline(zv, zvmax, k=k)(cat.data['z'])
            mt_cols['zmin'], mt_cols['zmax'] = zmin_func(mt_cols['z']),  zmax_func(mt_cols['z'])
        elif isinstance(delta_z, (int, float)):
            # zmin/zmax from sigma_z*(1+z)
            print('* zmin|zmax from config value')
            cat.mt_input['zmin'] = cat.data['z']-delta_z*(1.0+cat.data['z'])
            cat.mt_input['zmax'] = cat.data['z']+delta_z*(1.0+cat.data['z'])

        # Set angular radius
        if match_radius == 'cat':
            print('* ang radius from cat')
            in_rad, in_rad_unit = cat.data['rad'], cat.radius_unit
        else:
            print('* ang radius from set scale')
            in_rad = None
            for unit in units_bank:
                if unit in match_radius.lower():
                    try:
                        in_rad = float(match_radius.lower().replace(unit, ''))*np.ones(cat.size)
                        in_rad_unit = unit
                        break
                    except:
                        pass
            if in_rad is None:
                raise ValueError(f'Unknown radius unit in {match_radius}, must be in {units_bank.keys()}')
        # convert to degrees
        cat.mt_input['ang'] = convert_units(in_rad, in_rad_unit, 'degrees',
                                redshift=cat.data['z'] if 'z' in cat.data.colnames else None,
                                cosmo=cosmo)
    def _rescale_z(self, z, zlim, n):
        """Rescale zmin/zmax by a factor n
        
        Parameters
        ----------
        z: float, array
            Redshift value
        zlim: float, array
            Redshift limit
        n: float
            Value to rescale zlim

        Returns
        -------
        float, array
            Rescaled z limit
        """
        return z+n*(zlim-z)
    def _max_mt_distance(self, radius1, radius2, TYPE_R):
        """Get maximum angular distance allowed for the matching

        Parameters
        ----------
        radius1: float, array
            Radius to be used for catalog 1
        radius2: float, array
            Radius to be used for catalog 2
        TYPE_R: str
            Case of radius to be used, can be: own, other, min, max.

        Returns
        -------
        float, array
            Maximum angular distance allowed for matching
        """
        if TYPE_R=='own':
            f1 = np.ones(radius1.size)
            f2 = np.zeros(radius2.size)
        elif TYPE_R=='other':
            f1 = np.zeros(radius1.size)
            f2 = np.ones(radius2.size)
        elif TYPE_R=='max':
            f1 = (radius1 >= radius2)
            f2 = (radius1 < radius2)
        elif TYPE_R=='min':
            f1 = (radius1 < radius2)
            f2 = (radius1 >= radius2)
        return f1 * radius1 + f2 * radius2
