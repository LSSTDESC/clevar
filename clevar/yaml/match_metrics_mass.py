import argparse
import numpy as np
import pylab as plt

import clevar
from . import helper_funcs as hf
def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    # Create clevar objects from yml config
    config = hf.loadconf(config_file,
        consistency_configs=['catalog1', 'catalog2','proximity_match'],
        )
    print("\n# Reading Catalog 1")
    c1 = hf.make_catalog(config['catalog1'])
    c1.load_match(f"{config['outpath']}/match1.fits")
    print("\n# Reading Catalog 2")
    c2 = hf.make_catalog(config['catalog2'])
    c2.load_match(f"{config['outpath']}/match2.fits")
    # Scaling Relations
    from clevar.match_metrics import scaling
    # prep configurations
    mass_conf = {
        'plot_case': 'all',
        'matching_type': 'cross',
        'add_redshift_label': True,
        'add_err': True,
        'add_cb': True,
        'xlabel': config['catalog1'].get('labels', {}).get('mass', None),
        'ylabel': config['catalog2'].get('labels', {}).get('mass', None),
        'log_mass': True,
        'ax_rotation': 0,
        'rotation_resolution': 30,
        'figsize': config.get('match_metrics', {}).get('figsize', '20 20'),
        'dpi': config.get('match_metrics', {}).get('dpi', '150'),
        }
    mass_conf.update(config.get('match_metrics', {}).get('mass', {}))
    mass_conf['figsize'] = np.array(mass_conf['figsize'].split(' '), dtype=float)/2.54
    mass_conf['dpi'] = int(mass_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        mass_conf[cat] = {
            'redshift_bins': 10,
            'redshift_num_fmt': '.1f',
            'mass_bins': 5,
            }
        mass_conf[cat].update(config.get('match_metrics', {}).get('mass', {}).get(cat, {}))
        # Format values
        mass_conf[cat]['redshift_bins'] = hf.make_bins(mass_conf[cat]['redshift_bins'])
        mass_conf[cat]['mass_bins'] = hf.make_bins(mass_conf[cat]['mass_bins'], mass_conf['log_mass'])
        mass_conf[cat] = {k: str_none(v) for k, v in mass_conf[cat].items()}
    ### Plots
    kwargs = {k:mass_conf[k] for k in ('matching_type', 'log_mass', 'add_err',
                                       'add_cb', 'xlabel', 'ylabel')}
    mass_name = f'{config["outpath"]}/mass'
    # Density Plot
    if any(case in mass_conf['plot_case'] for case in ('density', 'all')):
        print(f"\n# Mass density colors")
        plt.clf()
        fig = plt.figure(figsize=mass_conf['figsize'])
        ax = plt.axes()
        scaling.mass_density(c1, c2, **kwargs, ax=ax,
            bins1=mass_conf['catalog1']['mass_bins'],
            bins2=mass_conf['catalog2']['mass_bins'],
            ax_rotation=mass_conf['ax_rotation'],
            rotation_resolution=mass_conf['rotation_resolution'],
            )
        plt.savefig(f'{mass_name}_density.png', dpi=mass_conf['dpi'])
    for i in ('1', '2'):
        mass_conf_cat = mass_conf[f'catalog{i}']
        # z Color Plot
        if any(case in mass_conf['plot_case'] for case in ('zcolor', 'all')):
            print(f"\n# Mass (catalog {i} z colors)")
            plt.clf()
            fig = plt.figure(figsize=mass_conf['figsize'])
            ax = plt.axes()
            scaling.mass_zcolor(c1, c2, **kwargs, ax=ax, color1=i=='1')
            plt.savefig(f'{mass_name}_cat{i}zcolor.png', dpi=mass_conf['dpi'])
        # Panel density Plot
        if any(case in mass_conf['plot_case'] for case in ('density_panel', 'all')):
            print(f"\n# Mass density (catalog {i} z panel)")
            plt.clf()
            fig, axes = scaling.mass_density_zpanel(c1, c2, **kwargs, panel_cat1=i=='1',
                bins1=mass_conf['catalog1']['mass_bins'],
                bins2=mass_conf['catalog2']['mass_bins'],
                ax_rotation=mass_conf['ax_rotation'],
                rotation_resolution=mass_conf['rotation_resolution'],
                redshift_bins=mass_conf[f'catalog{i}']['redshift_bins'],
                label_fmt=mass_conf[f'catalog{i}']['redshift_num_fmt'],
                fig_kwargs={'figsize': mass_conf['figsize']},
                )
            plt.savefig(f'{mass_name}_density_cat{i}zpanel.png', dpi=mass_conf['dpi'])
