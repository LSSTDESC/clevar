"""@file match_proximity.py
Proximity matching functions for command line execution
"""
import numpy as np
import pylab as plt
import os
import warnings

import clevar
from .helper_funcs import yaml, loadconf, make_catalog, add_mem_catalog,\
make_cosmology, get_input_loop

def match_general(config_file, overwrite_config, overwrite_files):
    """General matching function.

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of matching output files
    """
    match_functions = {'proximity': proximity, 'membership': membership}
    matching_mode = yaml.read(config_file)['matching_mode']
    if matching_mode not in match_functions:
        raise ValueError(f'matching_mode (={matching_mode}) must be in {list(match_functions.keys())}')
    match_functions[matching_mode](config_file, overwrite_config, overwrite_files)

def proximity(config_file, overwrite_config, overwrite_files):
    """Proximity matching.

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
        prt_msg = '# Start proximity matching' if len(steps)==1\
            else f'# Run step {step.replace("step", "")}'
        print(f'\n{"#"*len(prt_msg)}\n{prt_msg}\n{"#"*len(prt_msg)}')
        cosmo_ = make_cosmology(match_conf['cosmology']) \
                    if 'cosmology' in match_conf else cosmo
        if cosmo_!=cosmo:
            warn_msg = ('replacing default cosmology in matching with:\n    '+
                    '\n    '.join([f'{k}: {v}' for k, v in match_conf['cosmology'].items()]))
            warnings.warn(warn_msg)
        mt.match_from_config(c1, c2, match_conf, cosmo=cosmo_)
    save_matching_files(config, mt, c1, c2, overwrite_files)

def membership(config_file, overwrite_config, overwrite_files):
    """Membership matching.

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
        load_configs=['catalog1', 'catalog2', 'cosmology', 'membership_match', 'masks'],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    if config is None:
        return
    match_config = config['membership_match']
    print("\n# Reading Cluster Catalog 1")
    c1 = make_catalog(config['catalog1'])
    print("\n# Reading Cluster Catalog 2")
    c2 = make_catalog(config['catalog2'])
    print("\n# Reading Members Catalog 1")
    add_mem_catalog(c1, config['catalog1']['members'])
    print("\n# Reading Members Catalog 2")
    add_mem_catalog(c2, config['catalog2']['members'])
    mem_mt_radius = match_config['match_members_kwargs'].get('radius', '').lower()
    if any(unit in mem_mt_radius for unit in clevar.geometry.physical_bank):
        print("\n# Creating Cosmology")
        match_config['match_members_kwargs']['cosmo'] = make_cosmology(config['cosmology'])
    # Run matching
    mt = clevar.match.MembershipMatch()
    prt_msg = '# Start membership matching'
    print(f'\n{"#"*len(prt_msg)}\n{prt_msg}\n{"#"*len(prt_msg)}')
    mt.match_from_config(c1, c2,  match_config)
    save_matching_files(config, mt, c1, c2, overwrite_files)

def save_matching_files(config, mt, c1, c2, overwrite_files):
    """Saves internal matching files

    Parameters
    ----------
    config: dict
        Configuration of matching
    mt: clevar.match.Match
        Matching object
    c1: clevar.ClCatalog
        Catalog with matching results
    c2: clevar.ClCatalog
        Catalog with matching results
    overwrite_files: bool
        Forces overwrite of matching output files
    """
    out1, out2 = f'{config["outpath"]}/match1.fits', f'{config["outpath"]}/match2.fits'
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    save = True
    if (os.path.isfile(out1) or os.path.isfile(out2)) and not overwrite_files:
        print(f"\n*** File '{out1}' or '{out2}' already exist! ***")
        save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if save:
        mt.save_matches(c1, c2, out_dir=config['outpath'], overwrite=True)

def write_output(config_file, overwrite_config, overwrite_files):
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
    # read matching method
    matching_modes = ('proximity', 'membership')
    matching_mode = yaml.read(config_file)['matching_mode']
    if matching_mode not in matching_modes:
        raise ValueError(f'matching_mode (={matching_mode}) must be in {matching_modes}')
    matching_method = f'{matching_mode}_match'
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
    # Save files
    out1, out2 = f'{config["outpath"]}/catalog1.fits', f'{config["outpath"]}/catalog2.fits'
    out_matched = f'{config["outpath"]}/catalog_matched.fits'
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    save = True
    if any(os.path.isfile(out) for out in (out1, out2, out_matched)) and not overwrite_files:
        print(f"\n*** File '{out1}' or '{out2}' or '{out_matched}' already exist! ***")
        save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if save:
        print(f"\n# Saving full {c1.name}")
        clevar.match.output_catalog_with_matching(
            config['catalog1']['file'], out1, c1, overwrite=True)
        print(f"\n# Saving full {c2.name}")
        clevar.match.output_catalog_with_matching(
            config['catalog2']['file'], out2, c2, overwrite=True)
        print(f"\n# Saving Matched catalog")
        clevar.match.output_matched_catalog(
            config['catalog1']['file'], config['catalog2']['file'],
            out_matched, c1, c2, config[matching_method]['type'],
            overwrite=True)
