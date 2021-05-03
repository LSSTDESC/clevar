"""@file match_proximity.py
Proximity matching functions for command line execution
"""
import numpy as np
import pylab as plt
import os
import warnings

import clevar
from .helper_funcs import loadconf, make_catalog, make_cosmology, get_input_loop
def proximity(config_file, overwrite_config, overwrite_files):
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
        load_configs=['catalog1', 'catalog2', 'cosmology', 'proximity_match', 'masks'],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    if config is None:
        return
    print("\n# Reading Catalog 1")
    c1 = make_catalog(config['catalog1'])
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config['cosmology'])
    # Run matching
    mt = clevar.match.ProximityMatch()
    steps = sorted([c for c in config['proximity_match'] if 'step' in c])
    for step in steps:
        match_conf = {'type': config['proximity_match']['type']}
        match_conf.update(config['proximity_match'][step])
        prt_msg = '# Start matching' if len(steps)==1\
            else f'# Run step {step.replace("step", "")}'
        print(f'\n{"#"*20}\n{prt_msg}\n{"#"*20}')
        cosmo_ = make_cosmology(match_conf['cosmology']) \
                    if 'cosmology' in match_conf else cosmo
        if cosmo_!=cosmo:
            warn_msg = ('replacing default cosmology in matching with:\n    '+
                    '\n    '.join([f'{k}: {v}' for k, v in match_conf['cosmology'].items()]))
            warnings.warn(warn_msg)
        mt.match_from_config(c1, c2, match_conf, cosmo=cosmo_)
    out1, out2 = f'{config["outpath"]}/match1.fits', f'{config["outpath"]}/match2.fits'
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    save = True
    if (os.path.isfile(out1) or os.path.isfile(out2)) and not overwrite_files:
        print(f"\n*** File '{out1}' or '{out2}' already exist! ***")
        save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if save:
        mt.save_matches(c1, c2, out_dir=config['outpath'], overwrite=True)

def write_output(config_file, overwrite_config, overwrite_files,
                 matching_method='proximity_match'):
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
        load_configs=['catalog1', 'catalog2', matching_method],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    print("\n# Reading Catalog 1")
    c1 = make_catalog(config['catalog1'])
    ftpt_quantities_file1 = f"{config['outpath']}/ftpt_quantities1.fits"
    if os.path.isfile(ftpt_quantities_file1):
        c1.load_footprint_quantities(ftpt_quantities_file1)
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    ftpt_quantities_file2 = f"{config['outpath']}/ftpt_quantities2.fits"
    if os.path.isfile(ftpt_quantities_file2):
        c2.load_footprint_quantities(ftpt_quantities_file2)
    print("\n# Adding Matching Info")
    mt = clevar.match.parent.Match()
    mt.load_matches(c1, c2, out_dir=config['outpath'])
    # Print outputs
    print(f"\n# Prep full {c1.name}")
    c1_full = clevar.ClData.read(config['catalog1']['file'])
    for col in [c_ for c_ in c1.data.colnames if c_[:3] in ('mt_', 'ft_', 'cf_')]:
        if col in ('mt_self', 'mt_other', 'mt_cross'):
            c1_full[col] = [c if c else '' for c in c1[col]]
        elif col in ('mt_multi_self', 'mt_multi_other'):
            c1_full[col] = [','.join(c) if c else '' for c in c1[col]]
        else:
            c1_full[col] = c1[col]
    print(f"\n# Prep full {c2.name}")
    c2_full = clevar.ClData.read(config['catalog2']['file'])
    for col in [c_ for c_ in c2.data.colnames if c_[:3] in ('mt_', 'ft_', 'cf_')]:
        if col in ('mt_self', 'mt_other', 'mt_cross'):
            c2_full[col] = [c if c else '' for c in c2[col]]
        elif col in ('mt_multi_self', 'mt_multi_other'):
            c2_full[col] = [','.join(c) if c else '' for c in c2[col]]
        else:
            c2_full[col] = c2[col]
    print(f"\n# Prep Matched catalog")
    m1, m2 = clevar.match.MatchedPairs.matching_masks(None, c1, c2, config['proximity_match']['type'])
    c_matched = clevar.ClData()
    for col in c1_full.colnames:
        c_matched[f'c1_{col}'] = c1_full[col][m1]
    for col in c2_full.colnames:
        c_matched[f'c2_{col}'] = c2_full[col][m2]
    # Save files
    out1, out2 = f'{config["outpath"]}/catalog1.fits', f'{config["outpath"]}/catalog2.fits'
    out_matched = f'{config["outpath"]}/catalog_matched.fits'
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    save = True
    if any(os.path.isfile(out) for out in (out1, out2, out_matched)) and not overwrite_files:
        print(f"\n*** File '{out1}' or '{out2}' or '{out_matched}' already exist! ***")
        save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if save:
        c1_full.write(out1, overwrite=True)
        c2_full.write(out2, overwrite=True)
        c_matched.write(out_matched, overwrite=True)
