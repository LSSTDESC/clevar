"""@file match_metrics_mass.py
Matching metrics - mass functions for command line execution
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
                      'proximity_match', 'mt_metrics_mass'],
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
    mass_conf = {}
    mass_conf.update(config['mt_metrics_mass'])
    # Format values
    mass_conf['figsize'] = np.array(mass_conf['figsize'].split(' '), dtype=float)/2.54
    mass_conf['dpi'] = int(mass_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        mass_conf[cat]['redshift_bins'] = make_bins(mass_conf[cat]['redshift_bins'])
        mass_conf[cat]['mass_bins'] = make_bins(mass_conf[cat]['mass_bins'], mass_conf['log_mass'])
        mass_conf[cat]['fit_mass_bins'] = make_bins(mass_conf[cat]['fit_mass_bins'], mass_conf['log_mass'])
        mass_conf[cat]['fit_mass_bins_dist'] = make_bins(mass_conf[cat]['fit_mass_bins_dist'], mass_conf['log_mass'])
        mass_conf[cat] = {k: str_none(v) for k, v in mass_conf[cat].items()}
    ### Plots
    # config
    kwargs = {k:mass_conf[k] for k in ('matching_type', 'log_mass', 'add_err',
                                       'add_cb')}
    fit_kwargs = {k:mass_conf[k] for k in ('add_bindata', 'add_fit', 'add_fit_err', 'fit_statistics')}
    fit_kwargs_cat = {i:{
        'fit_bins1': mass_conf[f'catalog{i}']['fit_mass_bins'],
        'fit_bins2': mass_conf[f'catalog{i}']['fit_mass_bins_dist'],
    } for i in '12'}
    mass_name = f'{config["outpath"]}/mass'
    # Density Plot
    if any(case in mass_conf['plot_case'] for case in ('density', 'all')):
        print(f"\n# Mass density colors")
        fig = plt.figure(figsize=mass_conf['figsize'])
        ax = plt.axes()
        scaling.mass_density(c1, c2, **kwargs, ax=ax,
            bins1=mass_conf['catalog1']['mass_bins'],
            bins2=mass_conf['catalog2']['mass_bins'],
            ax_rotation=mass_conf['ax_rotation'],
            rotation_resolution=mass_conf['rotation_resolution'],
            **fit_kwargs, **fit_kwargs_cat['1'],
            )
        plt.savefig(f'{mass_name}_density.png', dpi=mass_conf['dpi'])
        plt.close(fig)
    if any(case in mass_conf['plot_case'] for case in ('scaling_metrics', 'all')):
        print(f"\n# Mass metrics")
        fig, axes = scaling.mass_metrics(c1, c2,
            bins1=mass_conf['catalog1']['mass_bins'],
            bins2=mass_conf['catalog2']['mass_bins'],
            **{k:mass_conf[k] for k in ('matching_type', 'log_mass')},
            fig_kwargs={'figsize': mass_conf['figsize']},
            )
        plt.savefig(f'{mass_name}_metrics.png', dpi=mass_conf['dpi'])
        plt.close(fig)
    if any(case in mass_conf['plot_case'] for case in ('density_metrics', 'all')):
        print(f"\n# Mass density metrics")
        fig, axes = scaling.mass_density_metrics(c1, c2, **kwargs,
            bins1=mass_conf['catalog1']['mass_bins'],
            bins2=mass_conf['catalog2']['mass_bins'],
            ax_rotation=mass_conf['ax_rotation'],
            rotation_resolution=mass_conf['rotation_resolution'],
            fig_kwargs={'figsize': mass_conf['figsize']},
            **fit_kwargs, **fit_kwargs_cat['1'],
            )
        plt.savefig(f'{mass_name}_density_metrics.png', dpi=mass_conf['dpi'])
        plt.close(fig)
    cats = {'1':c1, '2':c2}
    for i, j in ('12', '21'):
        mass_conf_cat = mass_conf[f'catalog{i}']
        # z Color Plot
        if any(case in mass_conf['plot_case'] for case in ('zcolor', 'all')):
            print(f"\n# Mass (catalog {i} z colors)")
            fig = plt.figure(figsize=mass_conf['figsize'])
            ax = plt.axes()
            scaling.mass_zcolor(c1, c2, **kwargs, ax=ax, color1=i=='1',
                                **fit_kwargs, **fit_kwargs_cat['1'])
            plt.savefig(f'{mass_name}_cat{i}zcolor.png', dpi=mass_conf['dpi'])
            plt.close(fig)
        # Panel density Plot
        if any(case in mass_conf['plot_case'] for case in ('density_panel', 'all')):
            print(f"\n# Mass density (catalog {i} z panel)")
            fig, axes = scaling.mass_density_zpanel(c1, c2, **kwargs, panel_cat1=i=='1',
                bins1=mass_conf['catalog1']['mass_bins'],
                bins2=mass_conf['catalog2']['mass_bins'],
                ax_rotation=mass_conf['ax_rotation'],
                rotation_resolution=mass_conf['rotation_resolution'],
                redshift_bins=mass_conf[f'catalog{i}']['redshift_bins'],
                label_fmt=mass_conf[f'catalog{i}']['redshift_num_fmt'],
                fig_kwargs={'figsize': mass_conf['figsize']},
                **fit_kwargs, **fit_kwargs_cat['1'],
                )
            plt.savefig(f'{mass_name}_density_cat{i}zpanel.png', dpi=mass_conf['dpi'])
            plt.close(fig)
        # distribution
        if any(case in mass_conf['plot_case'] for case in ('self_distribution', 'all')):
            print(f"\n# Mass density (catalog {i} m self dist)")
            fig, axes = scaling.mass_dist_self(cats[i],
                **{k:mass_conf[f'catalog{i}'][k] for k in
                    ('mass_bins', 'redshift_bins', 'mass_bins_dist')},
                log_mass=mass_conf['log_mass'],
                fig_kwargs={'figsize': mass_conf['figsize']},
                panel_label_fmt=mass_conf[f'catalog{i}']['mass_num_fmt'],
                line_label_fmt=mass_conf[f'catalog{i}']['redshift_num_fmt'],
                shape='line',
                )
            plt.savefig(f'{mass_name}_dist_self_cat{i}.png', dpi=mass_conf['dpi'])
            plt.close(fig)
        if any(case in mass_conf['plot_case'] for case in ('distribution', 'all')):
            print(f"\n# Mass density (catalog {i} m dist)")
            fig, axes = scaling.mass_dist(cats[i], cats[j],
                **{k:mass_conf[k] for k in ('matching_type', 'log_mass')},
                **{k:mass_conf[f'catalog{j}'][k] for k in
                    ('mass_bins', 'redshift_bins')},
                mass_bins_dist=mass_conf[f'catalog{i}']['mass_bins_dist'],
                fig_kwargs={'figsize': mass_conf['figsize']},
                panel_label_fmt=mass_conf[f'catalog{i}']['mass_num_fmt'],
                line_label_fmt=mass_conf[f'catalog{i}']['redshift_num_fmt'],
                shape='line',
                )
            plt.savefig(f'{mass_name}_dist_cat{i}.png', dpi=mass_conf['dpi'])
            plt.close(fig)
        # Panel density distribution
        if any(case in mass_conf['plot_case'] for case in ('density_dist', 'all')):
            print(f"\n# Mass density (catalog {i} z panel)")
            fig, axes = scaling.mass_density_dist(c1, c2, **kwargs,
                **fit_kwargs, **fit_kwargs_cat[i],
                bins1=mass_conf['catalog1']['mass_bins'],
                bins2=mass_conf['catalog2']['mass_bins'],
                ax_rotation=mass_conf['ax_rotation'],
                rotation_resolution=mass_conf['rotation_resolution'],
                redshift_bins=mass_conf[f'catalog{i}']['redshift_bins'],
                fig_kwargs={'figsize': mass_conf['figsize']},
                )
            plt.savefig(f'{mass_name}_density_cat{i}_dist.png', dpi=mass_conf['dpi'])
            plt.close(fig)
