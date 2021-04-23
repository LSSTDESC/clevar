"""@file match_metrics_distances.py
Matching metrics - distances functions for command line execution
"""
import numpy as np
import pylab as plt

import clevar
from .helper_funcs import loadconf, make_catalog, make_cosmology, make_bins
def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    # Create clevar objects from yml config
    config = loadconf(config_file,
        consistency_configs=['catalog1', 'catalog2','proximity_match'],
        )
    print("\n# Reading Catalog 1")
    c1 = make_catalog(config['catalog1'])
    c1.load_match(f"{config['outpath']}/match1.fits")
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    c2.load_match(f"{config['outpath']}/match2.fits")
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config['cosmology'])
    # Print metrics
    from clevar.match_metrics import distances
    # prep configurations
    dist_conf = {
        'matching_type': 'cross',
        'line_type':'steps',
        'add_mass_label': True,
        'add_redshift_label': True,
        'radial_bins': 20,
        'radial_bin_units': 'arcmin',
        'delta_redshift_bins': 20,
        'figsize': config.get('match_metrics', {}).get('figsize', '20 20'),
        'dpi': config.get('match_metrics', {}).get('dpi', '150'),
        }
    dist_conf.update(config.get('match_metrics', {}).get('distances', {}))
    dist_conf['figsize'] = np.array(dist_conf['figsize'].split(' '), dtype=float)/2.54
    dist_conf['dpi'] = int(dist_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        dist_conf[cat] = {
            'log_mass': True,
            'mass_num_fmt': '.2f',
            'redshift_num_fmt': '.1f',
            'redshift_bins': 10,
            'mass_bins': 5,
            }
        dist_conf[cat].update(config.get('match_metrics', {}).get('distances', {}).get(cat, {}))
        # Format values
        dist_conf[cat]['redshift_bins'] = make_bins(dist_conf[cat]['redshift_bins'])
        dist_conf[cat]['mass_bins'] = make_bins(dist_conf[cat]['mass_bins'], dist_conf[cat]['log_mass'])
        dist_conf[cat] = {k: str_none(v) for k, v in dist_conf[cat].items()}
    ### Plots
    # Central distances
    kwargs = dict(
        matching_type=dist_conf['matching_type'],
        shape=dist_conf['line_type'],
        radial_bins=dist_conf['radial_bins'],
        radial_bin_units=dist_conf['radial_bin_units'],
        cosmo=cosmo,
        )
    dist_cent_name = f'{config["outpath"]}/dist_cent_{dist_conf["radial_bin_units"]}'
    print("\n# Central distace plot (no bins)")
    fig = plt.figure(figsize=dist_conf['figsize'])
    ax = plt.axes()
    distances.central_position(c1, c2, **kwargs, ax=ax)
    plt.savefig(f'{dist_cent_name}.png', dpi=dist_conf['dpi'])
    for cats, i in zip([(c1, c2), (c2, c1)], ('1', '2')):
        dist_conf_cat = dist_conf[f'catalog{i}']
        print(f"\n# Central distace plot (catalog {i} mass bins)")
        fig = plt.figure(figsize=dist_conf['figsize'])
        ax = plt.axes()
        distances.central_position(*cats, **kwargs, ax=ax,
            quantity_bins='mass',
            bins=dist_conf_cat['mass_bins'],
            log_quantity=dist_conf_cat['log_mass'],
            add_legend=dist_conf['add_mass_label'],
            legend_fmt=dist_conf_cat['mass_num_fmt'])
        plt.savefig(f'{dist_cent_name}_cat{i}mass.png', dpi=dist_conf['dpi'])
        plt.close(fig)
        print(f"\n# Central distace plot (catalog {i} redshift bins)")
        fig = plt.figure(figsize=dist_conf['figsize'])
        ax = plt.axes()
        distances.central_position(*cats, **kwargs, ax=ax,
                quantity_bins='z',
                bins=dist_conf_cat['redshift_bins'],
                add_legend=dist_conf['add_redshift_label'],
                legend_fmt=dist_conf_cat['redshift_num_fmt'])
        plt.savefig(f'{dist_cent_name}_cat{i}redshift.png', dpi=dist_conf['dpi'])
        plt.close(fig)
    # Redshift distances
    kwargs.pop('radial_bins')
    kwargs.pop('radial_bin_units')
    kwargs.pop('cosmo')
    kwargs['redshift_bins'] = dist_conf['delta_redshift_bins']
    dist_z_name = f'{config["outpath"]}/dist_z'
    print("\n# Redshift distance plot (no bins)")
    fig = plt.figure(figsize=dist_conf['figsize'])
    ax = plt.axes()
    distances.redshift(c1, c2, ax=ax, **kwargs)
    plt.savefig(f'{dist_z_name}.png', dpi=dist_conf['dpi'])
    plt.close(fig)
    for cats, i in zip([(c1, c2), (c2, c1)], ('1', '2')):
        dist_conf_cat = dist_conf[f'catalog{i}']
        print(f"\n# Redshift distance plot (catalog {i} mass bins)")
        fig = plt.figure(figsize=dist_conf['figsize'])
        ax = plt.axes()
        distances.redshift(*cats, **kwargs, ax=ax,
            quantity_bins='mass',
            bins=dist_conf_cat['mass_bins'],
            log_quantity=dist_conf_cat['log_mass'],
            add_legend=dist_conf['add_mass_label'],
            legend_fmt=dist_conf_cat['mass_num_fmt'])
        plt.savefig(f'{dist_z_name}_cat{i}mass.png', dpi=dist_conf['dpi'])
        plt.close(fig)
        print(f"\n# Redshift distance plot (catalog {i} redshift bins)")
        fig = plt.figure(figsize=dist_conf['figsize'])
        ax = plt.axes()
        distances.redshift(*cats, **kwargs, ax=ax,
            quantity_bins='z',
            bins=dist_conf_cat['redshift_bins'],
            add_legend=dist_conf['add_redshift_label'],
            legend_fmt=dist_conf_cat['redshift_num_fmt'])
        plt.savefig(f'{dist_z_name}_cat{i}redshift.png', dpi=dist_conf['dpi'])
        plt.close(fig)
