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
    # Print metrics
    from clevar.match_metrics import recovery
    # prep configurations
    rec_conf = {
        'plot_case': 'all',
        'matching_type': 'multi_join',
        'line_type':'steps',
        'add_mass_label': True,
        'add_redshift_label': True,
        'add_colorbar': True,
        'figsize': config.get('match_metrics', {}).get('figsize', '20 20'),
        'dpi': config.get('match_metrics', {}).get('dpi', '150'),
        }
    rec_conf.update(config.get('match_metrics', {}).get('recovery', {}))
    rec_conf['figsize'] = np.array(rec_conf['figsize'].split(' '), dtype=float)/2.54
    rec_conf['dpi'] = int(rec_conf['dpi'])
    str_none = lambda x: None if str(x)=='None' else x
    for cat in ('catalog1', 'catalog2'):
        rec_conf[cat] = {
            'log_mass': True,
            'mass_num_fmt': '.2f',
            'redshift_num_fmt': '.1f',
            'recovery_label': None,
            'redshift_bins': 10,
            'mass_bins': 5,
            'mass_label': config[cat].get('labels', {}).get('mass', None),
            'redshift_label': config[cat].get('labels', {}).get('redshift', None),
            'mass_lim': None,
            'redshift_lim': None,
            'recovery_lim': None,
            }
        rec_conf[cat].update(config.get('match_metrics', {}).get('recovery', {}).get(cat, {}))
        # Format values
        rec_conf[cat]['redshift_bins'] = hf.make_bins(rec_conf[cat]['redshift_bins'])
        rec_conf[cat]['mass_bins'] = hf.make_bins(rec_conf[cat]['mass_bins'], rec_conf[cat]['log_mass'])
        rec_conf[cat] = {k: str_none(v) for k, v in rec_conf[cat].items()}
    ### Plots
    rec_name = f'{config["outpath"]}/rec_mt{rec_conf["matching_type"]}'
    for c, i in zip((c1, c2), ('1', '2')):
        rec_conf_cat = rec_conf[f'catalog{i}']
        kwargs = dict(
            matching_type=rec_conf['matching_type'],
            shape=rec_conf['line_type'],
            redshift_bins=rec_conf_cat['redshift_bins'],
            mass_bins=rec_conf_cat['mass_bins'],
            log_mass=rec_conf_cat['log_mass'],
            mass_label=rec_conf_cat['mass_label'],
            redshift_label=rec_conf_cat['redshift_label'],
            recovery_label=rec_conf_cat['recovery_label'],
            )
        # Simple plot
        if any(case in rec_conf['plot_case'] for case in ('simple', 'all')):
            # by redshift
            print(f"\n# Simple recovery catalog {i} by redshift")
            fig = plt.figure(figsize=rec_conf['figsize'])
            ax = plt.axes()
            recovery.plot(c, **kwargs, ax=ax,
                add_legend=rec_conf['add_mass_label'],
                legend_fmt=rec_conf_cat['mass_num_fmt'],
                )
            ax.set_xlim(rec_conf_cat['redshift_lim'])
            ax.set_ylim(rec_conf_cat['recovery_lim'])
            plt.savefig(f'{rec_name}_cat{i}redshift.png', dpi=rec_conf['dpi'])
            # by mass
            print(f"\n# Simple recovery catalog {i} by mass")
            plt.clf()
            ax = recovery.plot(c, **kwargs, transpose=True,
                add_legend=rec_conf['add_redshift_label'],
                legend_fmt=rec_conf_cat['redshift_num_fmt'],
                )
            ax.set_xlim(rec_conf_cat['mass_lim'])
            ax.set_ylim(rec_conf_cat['recovery_lim'])
            plt.savefig(f'{rec_name}_cat{i}mass.png', dpi=rec_conf['dpi'])
        # Panels plots
        if any(case in rec_conf['plot_case'] for case in ('panel', 'all')):
            # by redshift
            print(f"\n# Panel recovery catalog {i} by redshift")
            plt.clf()
            fig, axes = recovery.plot_panel(c, **kwargs,
                add_label=rec_conf['add_mass_label'],
                label_fmt=rec_conf_cat['mass_num_fmt'],
                fig_kwargs={'figsize': rec_conf['figsize']},
                )
            ax = axes.flatten()[0]
            ax.set_xlim(rec_conf_cat['redshift_lim'])
            ax.set_ylim(rec_conf_cat['recovery_lim'])
            plt.savefig(f'{rec_name}_cat{i}redshift_panel.png', dpi=rec_conf['dpi'])
            # by mass
            print(f"\n# Panel recovery catalog {i} by mass")
            plt.clf()
            fig, axes = recovery.plot_panel(c, **kwargs, transpose=True,
                add_label=rec_conf['add_redshift_label'],
                label_fmt=rec_conf_cat['redshift_num_fmt'],
                fig_kwargs={'figsize': rec_conf['figsize']},
                )
            ax = axes.flatten()[0]
            ax.set_xlim(rec_conf_cat['mass_lim'])
            ax.set_ylim(rec_conf_cat['recovery_lim'])
            plt.savefig(f'{rec_name}_cat{i}mass_panel.png', dpi=rec_conf['dpi'])
        # 2D plots
        kwargs.pop('shape')
        kwargs['add_cb'] = rec_conf['add_colorbar']
        if any(case in rec_conf['plot_case'] for case in ('2D', 'all')):
            # basic
            print(f"\n# 2D recovery catalog {i}")
            plt.clf()
            fig = plt.figure(figsize=rec_conf['figsize'])
            ax = plt.axes()
            ax, cb = recovery.plot2D(c, **kwargs, ax=ax)
            ax.set_xlim(rec_conf_cat['redshift_lim'])
            ax.set_ylim(rec_conf_cat['mass_lim'])
            plt.savefig(f'{rec_name}_cat{i}_2D.png', dpi=rec_conf['dpi'])
            # with number
            print(f"\n# 2D recovery catalog {i} with numbers")
            plt.clf()
            axes, cb = recovery.plot2D(c, **kwargs, add_num=True)
            ax.set_xlim(rec_conf_cat['redshift_lim'])
            ax.set_ylim(rec_conf_cat['mass_lim'])
            plt.savefig(f'{rec_name}_cat{i}_2D_num.png', dpi=rec_conf['dpi'])
