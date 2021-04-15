"""@file match_proximity.py
Proximity matching functions for command line execution
"""
import numpy as np
import pylab as plt
import os
import warnings

import clevar
from .helper_funcs import loadconf, make_catalog, make_cosmology, get_input_loop
def artificial(config_file, overwrite_config, overwrite_matching, case):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_config: bool
        Forces overwrite of matching output files
    case: str
        Run for which catalog. Options: 1, 2, both
    """
    # Create clevar objects from yml config
    config = loadconf(config_file,
        consistency_configs=['catalog1', 'catalog2'],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    if case in ('1', 'both'):
        print("\n# Creating footprint 1")
        save = True
        ftpt_cfg1 = config['catalog1']['footprint']
        if os.path.isfile(ftpt_cfg1['file']) and not overwrite_matching:
            print(f"\n*** File '{ftpt_cfg1['file']}' already exist! ***")
            save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
        if save:
            print("\n# Reading Catalog 1")
            c1 = make_catalog(config['catalog1'])
            ftpt1 = clevar.footprint.create_artificial_footprint(
                c1['ra'], c1['dec'], nside=ftpt_cfg1['nside'],
                nest=ftpt_cfg1['nest']) #min_density=2, neighbor_fill=None
            ftpt1[['pixel']].write(ftpt_cfg1['file'], overwrite=True)
    if case in ('2', 'both'):
        print("\n# Creating footprint 2")
        save = True
        ftpt_cfg2 = config['catalog2']['footprint']
        if os.path.isfile(ftpt_cfg2['file']) and not overwrite_matching:
            print(f"\n*** File '{ftpt_cfg2['file']}' already exist! ***")
            save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
        if save:
            print("\n# Reading Catalog 2")
            c2 = make_catalog(config['catalog2'])
            ftpt2 = clevar.footprint.create_artificial_footprint(
                c2['ra'], c2['dec'], nside=ftpt_cfg2['nside'],
                nest=ftpt_cfg2['nest']) #min_density=2, neighbor_fill=None
            out = ftpt2[['pixel']].write(ftpt_cfg2['file'], overwrite=True)
