"""@file footprint.py
Footprint functions for command line execution
"""
import numpy as np
import pylab as plt
import os
import warnings

import clevar
from .helper_funcs import loadconf, make_catalog, make_cosmology, get_input_loop
def artificial(config_file, overwrite_config, overwrite_files, case):
    """Function to create footprint

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of output files
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
        if os.path.isfile(ftpt_cfg1['file']) and not overwrite_files:
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
        if os.path.isfile(ftpt_cfg2['file']) and not overwrite_files:
            print(f"\n*** File '{ftpt_cfg2['file']}' already exist! ***")
            save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
        if save:
            print("\n# Reading Catalog 2")
            c2 = make_catalog(config['catalog2'])
            ftpt2 = clevar.footprint.create_artificial_footprint(
                c2['ra'], c2['dec'], nside=ftpt_cfg2['nside'],
                nest=ftpt_cfg2['nest']) #min_density=2, neighbor_fill=None
            out = ftpt2[['pixel']].write(ftpt_cfg2['file'], overwrite=True)
def make_masks(config_file, overwrite_config, overwrite_files, case):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of output files
    case: str
        Run for which catalog. Options: 1, 2, both
    """
    # Create clevar objects from yml config
    config = loadconf(config_file,
        consistency_configs=['catalog1', 'catalog2'],
        fail_action='orverwrite' if overwrite_config else 'ask'
        )
    check_actions = {'o': (lambda : True, [], {}), 'q': (lambda :False, [], {}),}
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config['cosmology'])
    # Read footprints
    ftpt_cfg1 = config['catalog1']['footprint']
    args1 = {k: v if v!='None' else None for k, v in
             config['catalog1']['footprint'].items() if k!='file'}
    ftpt1 = clevar.Footprint.read(ftpt_cfg1['file'], **args1) if ftpt_cfg1['file'] is not None else None
    ftpt_cfg2 = config['catalog2']['footprint']
    args2 = {k: v if v!='None' else None for k, v in
             config['catalog2']['footprint'].items() if k!='file'}
    ftpt2 = clevar.Footprint.read(ftpt_cfg2['file'], **args2) if ftpt_cfg2['file'] is not None else None
    # Catalog 1
    ftpt_quantities_file1 = f"{config['outpath']}/ftpt_quantities1.fits"
    if case in ('1', 'both'):
        print("\n# Creating masks for catalog 1")
        save = True
        print("\n# Reading Catalog 1")
        c1 = make_catalog(config['catalog1'])
        for cf_name, mask_cfg in config['masks']['catalog1'].items():
            if cf_name[:12]=='in_footprint':
                print(f"\n# Adding footprint mask: {mask_cfg}")
                ftpt = {'self':ftpt1, 'other':ftpt2}[mask_cfg['which_footprint']]
                c1._add_ftpt_mask(ftpt, maskname=mask_cfg['name'])
            if cf_name[:13]=='coverfraction':
                aperture, aperture_unit = clevar.utils.str2dataunit(
                    mask_cfg['aperture'], clevar.geometry.units_bank)
                ftpt = {'self':ftpt1, 'other':ftpt2}[mask_cfg['which_footprint']]
                print(f"\n# Adding coverfrac: {mask_cfg}")
                c1.add_ftpt_coverfrac(ftpt, aperture, aperture_unit,
                    window=mask_cfg['window_function'], colname=mask_cfg['name'],
                    cosmo=cosmo)
        if os.path.isfile(ftpt_quantities_file1) and not overwrite_files:
            print(f"\n*** File '{ftpt_quantities_file1}' already exist! ***")
            save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
        if save:
            c1.save_footprint_quantities(ftpt_quantities_file1, overwrite=True)
    # Catalog 2
    ftpt_quantities_file2 = f"{config['outpath']}/ftpt_quantities2.fits"
    if case in ('2', 'both'):
        print("\n# Creating masks for catalog 2")
        save = True
        print("\n# Reading Catalog 2")
        c2 = make_catalog(config['catalog2'])
        for cf_name, mask_cfg in config['masks']['catalog2'].items():
            if cf_name[:12]=='in_footprint':
                print(f"\n# Adding footprint mask: {mask_cfg}")
                ftpt = {'self':ftpt2, 'other':ftpt1}[mask_cfg['which_footprint']]
                c2._add_ftpt_mask(ftpt, maskname=mask_cfg['name'])
            if cf_name[:13]=='coverfraction':
                aperture, aperture_unit = clevar.utils.str2dataunit(
                    mask_cfg['aperture'], clevar.geometry.units_bank)
                ftpt = {'self':ftpt2, 'other':ftpt1}[mask_cfg['which_footprint']]
                print(f"\n# Adding coverfrac: {mask_cfg}")
                c2.add_ftpt_coverfrac(ftpt, aperture, aperture_unit,
                    window=mask_cfg['window_function'], colname=mask_cfg['name'],
                    cosmo=cosmo)
        if os.path.isfile(ftpt_quantities_file2) and not overwrite_files:
            print(f"\n*** File '{ftpt_quantities_file2}' already exist! ***")
            save = get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
        if save:
            c2.save_footprint_quantities(ftpt_quantities_file2, overwrite=True)
