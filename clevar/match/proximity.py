import numpy as np
from .parent import Match
from ..geometry import units_bank, convert_units, square_mask
from ..catalog import ClData


class ProximityMatch(Match):
    def __init__(self, ):
        self.type = 'Proximity'
    def multiple(self, cat1, cat2):
        '''
        Make the one way multiple matching

        Parameters
        ----------
        cat1, cat2: ClusterObject
            Catalogs to be matched
        nradius: float
            Multiplier of the radius to be used
        radius_use: str (optional)
            Case of radius to be used, can be: max, each, other.
        '''
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
                mask *= square_mask(ra2, dec2, ra1-DaCos, ra1+DaCos, dec1-Da, dec1+Da)
                # makes circular crop
                if mask.any():
                    ########### get candidates in cyllinder ############
                    dist = sk1.separation(sk2[mask]).value
                    max_dist = self._max_mt_distance(ang1, ang2[mask], TYPE_R='max')
                    ########### print(multiple matches ############)
                    for id2 in cat2.data['id'][mask][dist<=max_dist]:
                        cat1.match['multi_self'][i].append(id2)
                        i2 = int(cat2.id_dict[id2])
                        cat2.match['multi_other'][i2].append(cat1.data['id'][i])
        print(f'* {len(cat1.match[cat1.match["multi_self"]!=None])} objects matched.')
        return True
    def _prep_for_match(self, cat, config):
        '''
        Parameters
        ----------
        z: float, array
            Redshift of catalog
        rad_cat: float, array
            Radius of catalog
        rad_unit: str
            Units of cluster radius
        sigz: float, string
            Defines the redshift matching. If 'incat' uses redshift
            properties of the catalogs, if 'scat_z' uses measured
            properties of redshift, if float uses dist_z*(1+z),
            if None does not use z.
        n_dz: float
            Multiplier to compute zmin, zmax
        rad_use: str, float
            If incat, uses cluster/halo radius, else uses as value for radius
        rad_use_units: str, None
            Units of cluster radius used if rad_use is not incat
        aux: Dict
            Dictionary with auxiliary columns
        cf: Cosmology object
            Cosmology object for when lengh has unit [Mpc]
        '''
        print('## Prep mt_cols')
        cat.mt_input = ClData()
        # Set zmin, zmax
        if config['delta_z'] is None:
            # No z use
            print('* zmin|zmax set to -1|10')
            cat.mt_input['zmin'] = -1*np.ones(cat.size)
            cat.mt_input['zmax'] = 10*np.ones(cat.size)
        elif config['delta_z'] == 'cat':
            # Values from catalog
            print('* zmin|zmax from cat cols')
            cat.mt_input['zmin'] = self._rescale_z(cat.data['z'], cat.data['zmin'], nz)
            cat.mt_input['zmax'] = self._rescale_z(cat.data['z'], cat.data['zmax'], nz)
        elif isinstance(config['delta_z'], str):
            # zmin/zmax in auxiliar file
            print('* zmin|zmax from aux file')
            k = 3
            zv, zvmin, zvmax = np.loadtxt(config['delta_z'])
            zvmin = self._rescale_z(zv, zvmin, nz)
            zvmax = self._rescale_z(zv, zvmax, nz)
            cat.mt_input['zmin'] = spline(zv, zvmin, k=k)(cat.data['z'])
            cat.mt_input['zmax'] = spline(zv, zvmax, k=k)(cat.data['z'])
            mt_cols['zmin'], mt_cols['zmax'] = zmin_func(mt_cols['z']),  zmax_func(mt_cols['z'])
        elif isinstance(config['delta_z'], (int, float)):
            # zmin/zmax from sigma_z*(1+z)
            print('* zmin|zmax from config value')
            cat.mt_input['zmin'] = cat.data['z']-config['delta_z']*(1.0+cat.data['z'])
            cat.mt_input['zmax'] = cat.data['z']+config['delta_z']*(1.0+cat.data['z'])

        # Set angular radius
        if config['match_radius'] == 'cat':
            print('* ang radius from cat')
            in_rad, in_rad_unit = cat.data['rad'], cat.data_unit['rad']
        else:
            print('* ang radius from set scale')
            in_rad = None
            for unit in units_bank:
                if unit in config['match_radius'].lower():
                    in_rad = float(config['match_radius'].lower().replace(unit, ''))*np.ones(cat.size)
                    in_rad_unit = unit
                    break
            if in_rad is None:
                return ValueError(f'Unknown radius unit in {config["match_radius"]}, must be in {units_bank.keys()}')
        # convert to degrees
        cat.mt_input['ang'] = convert_units(in_rad, in_rad_unit, 'degrees',
                                redshift=cat.data['z'] if 'z' in cat.data.colnames else None,
                                cosmo=None)
    def _rescale_z(z, zlim, n):
        return z+n*(zlim-z)
    def _max_mt_distance(self, ang1, ang2, TYPE_R):
        if TYPE_R=='own':
            f1 = np.ones(ang1.size)
            f2 = np.zeros(ang2.size)
        elif TYPE_R=='other':
            f1 = np.zeros(ang1.size)
            f2 = np.ones(ang2.size)
        elif TYPE_R=='max':
            f1 = (ang1 >= ang2)
            f2 = (ang1 < ang2)
        elif TYPE_R=='min':
            f1 = (ang1 < ang2)
            f2 = (ang1 >= ang2)
        return f1 * ang1 + f2 * ang2
