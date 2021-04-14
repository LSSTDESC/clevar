import argparse
import numpy as np
import pylab as plt

import clevar
from . import helper_funcs as hf
def run():
    """Main plot function"""
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config_file', help='Configuration yaml file')
    args = parser.parse_args()
    # Create clevar objects from yml config
    config = hf.loadconf(
        config_file=args.config_file,
        consistency_configs=['catalog1', 'catalog2','proximity_match'],
        )
    print("\n# Reading Catalog 1")
    c1 = hf.make_catalog(config['catalog1'])
    c1.load_match(f"{config['outpath']}/match1.fits")
    print("\n# Reading Catalog 2")
    c2 = hf.make_catalog(config['catalog2'])
    c2.load_match(f"{config['outpath']}/match2.fits")
    print("\n# Creating Cosmology")
    cosmo = hf.make_cosmology(config['cosmology'])
    # Scaling Relations
    from clevar.match_metrics import scaling
    # prep configurations
    z_conf = {
        'plot_case': 'all',
        'matching_type': 'cross',
        'add_mass_label': True,
        'add_err': True,
        'add_cb': True,
        'xlabel': config['catalog1'].get('labels', {}).get('z', None),
        'ylabel': config['catalog2'].get('labels', {}).get('z', None),
        'log_mass': True,
        'ax_rotation': 45,
        'rotation_resolution': 30,
        'figsize': config.get('match_metrics', {}).get('figsize', '20 20'),
        'dpi': config.get('match_metrics', {}).get('dpi', '150'),
        }
    z_conf.update(config.get('match_metrics', {}).get('redshift', {}))
    z_conf['figsize'] = np.array(z_conf['figsize'].split(' '), dtype=float)/2.54
    z_conf['dpi'] = int(z_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        z_conf[cat] = {
            'mass_bins': 4,
            'mass_num_fmt': '.2f',
            'redshift_bins': 10,
            }
        z_conf[cat].update(config.get('match_metrics', {}).get('redshift', {}).get(cat, {}))
        # Format values
        z_conf[cat]['redshift_bins'] = hf.make_bins(z_conf[cat]['redshift_bins'])
        z_conf[cat]['mass_bins'] = hf.make_bins(z_conf[cat]['mass_bins'], z_conf['log_mass'])
        z_conf[cat] = {k: str_none(v) for k, v in z_conf[cat].items()}
    ### Plots
    kwargs = {k:z_conf[k] for k in ('matching_type', 'add_err',
                                    'add_cb', 'xlabel', 'ylabel')}
    z_name = f'{config["outpath"]}/redshift'
    # Density Plot
    if any(case in z_conf['plot_case'] for case in ('density', 'all')):
        print(f"\n# Redshift density colors")
        plt.clf()
        fig = plt.figure(figsize=z_conf['figsize'])
        ax = plt.axes()
        scaling.redshift_density(c1, c2, **kwargs, ax=ax,
            bins1=z_conf['catalog1']['redshift_bins'],
            bins2=z_conf['catalog2']['redshift_bins'],
            ax_rotation=z_conf['ax_rotation'],
            rotation_resolution=z_conf['rotation_resolution'],
            )
        plt.savefig(f'{z_name}_density.png', dpi=z_conf['dpi'])
    for i in ('1', '2'):
        z_conf_cat = z_conf[f'catalog{i}']
        # z Color Plot
        if any(case in z_conf['plot_case'] for case in ('masscolor', 'all')):
            print(f"\n# Redshift (catalog {i} z colors)")
            plt.clf()
            fig = plt.figure(figsize=z_conf['figsize'])
            ax = plt.axes()
            scaling.redshift_masscolor(c1, c2, **kwargs, ax=ax, color1=i=='1',
                                            log_mass=z_conf['log_mass'])
            plt.savefig(f'{z_name}_cat{i}zcolor.png', dpi=z_conf['dpi'])
        # Panel density Plot
        if any(case in z_conf['plot_case'] for case in ('density_panel', 'all')):
            print(f"\n# Redshift density (catalog {i} z panel)")
            plt.clf()
            fig, axes = scaling.redshift_density_masspanel(c1, c2, **kwargs, panel_cat1=i=='1',
                bins1=z_conf['catalog1']['redshift_bins'],
                bins2=z_conf['catalog2']['redshift_bins'],
                ax_rotation=z_conf['ax_rotation'],
                rotation_resolution=z_conf['rotation_resolution'],
                mass_bins=z_conf[f'catalog{i}']['mass_bins'],
                label_fmt=z_conf[f'catalog{i}']['mass_num_fmt'],
                fig_kwargs={'figsize': z_conf['figsize']},
                )
            plt.savefig(f'{z_name}_density_cat{i}masspanel.png', dpi=z_conf['dpi'])
