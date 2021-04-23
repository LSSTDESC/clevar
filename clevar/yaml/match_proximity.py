"""@file match_proximity.py
Proximity matching functions for command line execution
"""
import numpy as np
import pylab as plt
import os
import warnings

import clevar
from .helper_funcs import loadconf, make_catalog, make_cosmology, get_input_loop
def run(config_file, overwrite_config, overwrite_files):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of matching output files
    """
    # Create clevar objects from yml config
    config = loadconf(config_file,
        consistency_configs=['catalog1', 'catalog2','proximity_match'],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    print("\n# Reading Catalog 1")
    c1 = make_catalog(config['catalog1'])
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config['cosmology'])
    # Run matching
    mt = clevar.match.ProximityMatch()
    for match_step in sorted([c for c in config.keys()
                            if 'proximity_match' in c]):
        prt_msg = '# Start matching' if match_step=='proximity_match'\
            else f'# Run step {match_step.replace("proximity_match_", "")}'
        print(f'\n{"#"*20}\n{prt_msg}\n{"#"*20}')
        cosmo_ = make_cosmology(config[match_step]['cosmology']) \
                    if 'cosmology' in config[match_step] else cosmo
        if cosmo_!=cosmo:
            warn_msg = ('replacing default cosmology in matching with:\n    '+
                    '\n    '.join([f'{k}: {v}' for k, v in config[match_step]['cosmology'].items()]))
            warnings.warn(warn_msg)
        mt.match_from_config(c1, c2, config[match_step], cosmo=cosmo_)
    out1, out2 = f'{config["outpath"]}/match1.fits', f'{config["outpath"]}/match2.fits'
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    save = True
    if (os.path.isfile(out1) or os.path.isfile(out2)) and not overwrite_files:
        print(f"\n*** File '{out1}' or '{out2}' already exist! ***")
        save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if save:
        mt.save_matches(c1, c2, out_dir=config['outpath'], overwrite=True)