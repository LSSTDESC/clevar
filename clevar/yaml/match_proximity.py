import argparse
import numpy as np
import pylab as plt
import os

import clevar
from . import helper_funcs as hf
def run():
    """Main plot function"""
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config_file', help='Configuration yaml file')
    parser.add_argument('-oc','--overwrite-config', dest='overwrite_config', default=False, action='store_true', help='Overwrites config.log.yml file')
    parser.add_argument('-om','--overwrite-matching', dest='overwrite_matching', default=False, action='store_true', help='Overwrites matching output files')
    args = parser.parse_args()
    # Create clevar objects from yml config
    config = hf.loadconf(
        config_file=args.config_file,
        consistency_configs=['catalog1', 'catalog2','proximity_match'],
        fail_action='orverwrite' if args.overwrite_config else 'ask'
        )
    print("\n# Reading Catalog 1")
    c1 = hf.make_catalog(config['catalog1'])
    print("\n# Reading Catalog 2")
    c2 = hf.make_catalog(config['catalog2'])
    print("\n# Creating Cosmology")
    cosmo = hf.make_cosmology(config['cosmology'])
    # Run matching
    mt = clevar.match.ProximityMatch()
    for match_step in sorted([c for c in config.keys()
                            if 'proximity_match' in c]):
        prt_msg = '# Start matching' if match_step=='proximity_match'\
            else f'# Run step {match_step.replace("proximity_match_", "")}'
        print(f'\n{"#"*20}\n{prt_msg}\n{"#"*20}')
        cosmo_ = hf.make_cosmology(config[match_step]['cosmology']) \
                    if 'cosmology' in config[match_step] else cosmo
        if cosmo_!=cosmo:
            warn_msg = ('replacing default cosmology in matching with:\n    '+
                    '\n    '.join([f'{k}: {v}' for k, v in config[match_step]['cosmology'].items()]))
            warnings.warn(warn_msg)
        mt.match_from_config(c1, c2, config[match_step], cosmo=cosmo_)
    out1, out2 = f'{config["outpath"]}/match1.fits', f'{config["outpath"]}/match2.fits'
    check_actions = {'o': (lambda : None, [], {}), 'q': (exit, [], {}),}
    if os.path.isfile(out1) and not args.overwrite_matching:
        print(f"\n*** File '{out1}' already exist! ***")
        hf.get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    if os.path.isfile(out2) and not args.overwrite_matching:
        print(f"\n*** File '{out2}' already exist! ***")
        hf.get_input_loop('Overwrite(o) and proceed or Quit(q)?', check_actions)
    mt.save_matches(c1, c2, out_dir=config['outpath'], overwrite=True)
