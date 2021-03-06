"""@file match_metrics_redshift.py
Matching metrics - redshift rate functions for command line execution
"""
import numpy as np
import pylab as plt

import clevar
from .helper_funcs import loadconf, make_catalog, make_bins
def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    # Create clevar objects from yml config
    config = loadconf(config_file,
        load_configs=['catalog1', 'catalog2', 'cosmology',
                      'proximity_match', 'mt_metrics_redshift'],
        )
    if config is None:
        return
    print("\n# Reading Catalog 1")
    c1 = make_catalog(config['catalog1'])
    c1.load_match(f"{config['outpath']}/match1.fits")
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    c2.load_match(f"{config['outpath']}/match2.fits")
    # Scaling Relations
    from clevar.match_metrics import scaling
    # prep configurations
    z_conf = {}
    z_conf.update(config['mt_metrics_redshift'])
    # Format values
    z_conf['figsize'] = np.array(z_conf['figsize'].split(' '), dtype=float)/2.54
    z_conf['dpi'] = int(z_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        z_conf[cat]['redshift_bins'] = make_bins(z_conf[cat]['redshift_bins'])
        z_conf[cat]['mass_bins'] = make_bins(z_conf[cat]['mass_bins'], z_conf['log_mass'])
        z_conf[cat] = {k: str_none(v) for k, v in z_conf[cat].items()}
    ### Plots
    kwargs = {k:z_conf[k] for k in ('matching_type', 'add_err', 'add_cb')}
    z_name = f'{config["outpath"]}/redshift'
    # Density Plot
    if any(case in z_conf['plot_case'] for case in ('density', 'all')):
        print(f"\n# Redshift density colors")
        fig = plt.figure(figsize=z_conf['figsize'])
        ax = plt.axes()
        scaling.redshift_density(c1, c2, **kwargs, ax=ax,
            bins1=z_conf['catalog1']['redshift_bins'],
            bins2=z_conf['catalog2']['redshift_bins'],
            ax_rotation=z_conf['ax_rotation'],
            rotation_resolution=z_conf['rotation_resolution'],
            )
        plt.savefig(f'{z_name}_density.png', dpi=z_conf['dpi'])
        plt.close(fig)
    if any(case in z_conf['plot_case'] for case in ('scaling_metrics', 'all')):
        print(f"\n# Redshift metrics")
        fig, axes = scaling.redshift_metrics(c1, c2,
            bins1=z_conf['catalog1']['redshift_bins'],
            bins2=z_conf['catalog2']['redshift_bins'],
            matching_type=z_conf['matching_type'],
            fig_kwargs={'figsize': z_conf['figsize']},
            )
        plt.savefig(f'{z_name}_metrics.png', dpi=z_conf['dpi'])
        plt.close(fig)
    if any(case in z_conf['plot_case'] for case in ('density_metrics', 'all')):
        print(f"\n# Redshift density metrics")
        fig, axes = scaling.redshift_density_metrics(c1, c2, **kwargs,
            bins1=z_conf['catalog1']['redshift_bins'],
            bins2=z_conf['catalog2']['redshift_bins'],
            ax_rotation=z_conf['ax_rotation'],
            rotation_resolution=z_conf['rotation_resolution'],
            fig_kwargs={'figsize': z_conf['figsize']},
            )
        plt.savefig(f'{z_name}_density_metrics.png', dpi=z_conf['dpi'])
        plt.close(fig)
    for i in ('1', '2'):
        z_conf_cat = z_conf[f'catalog{i}']
        # z Color Plot
        if any(case in z_conf['plot_case'] for case in ('masscolor', 'all')):
            print(f"\n# Redshift (catalog {i} z colors)")
            fig = plt.figure(figsize=z_conf['figsize'])
            ax = plt.axes()
            scaling.redshift_masscolor(c1, c2, **kwargs, ax=ax, color1=i=='1',
                                            log_mass=z_conf['log_mass'])
            plt.savefig(f'{z_name}_cat{i}zcolor.png', dpi=z_conf['dpi'])
            plt.close(fig)
        # Panel density Plot
        if any(case in z_conf['plot_case'] for case in ('density_panel', 'all')):
            print(f"\n# Redshift density (catalog {i} z panel)")
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
            plt.close(fig)
