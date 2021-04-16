"""@file match_metrics_recovery_rate.py
Matching metrics - recovery rate functions for command line execution
"""
import os
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
    ftpt_qt_file1 = f"{config['outpath']}/ftpt_quantities1.fits"
    _ = c1.load_footprint_quantities(ftpt_qt_file1) if os.path.isfile(ftpt_qt_file1) else None
    print("\n# Reading Catalog 2")
    c2 = make_catalog(config['catalog2'])
    c2.load_match(f"{config['outpath']}/match2.fits")
    ftpt_qt_file2 = f"{config['outpath']}/ftpt_quantities2.fits"
    _ = c2.load_footprint_quantities(ftpt_qt_file2) if os.path.isfile(ftpt_qt_file2) else None
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config['cosmology'])
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
            'masks':{'case':'None'},
            }
        rec_conf[cat].update(config.get('match_metrics', {}).get('recovery', {}).get(cat, {}))
        # Format values
        rec_conf[cat]['redshift_bins'] = make_bins(rec_conf[cat]['redshift_bins'])
        rec_conf[cat]['mass_bins'] = make_bins(rec_conf[cat]['mass_bins'], rec_conf[cat]['log_mass'])
        rec_conf[cat] = {k: str_none(v) for k, v in rec_conf[cat].items()}
    ### Plots
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
        mask = np.zeros(c.size, dtype=bool)
        mask_case = rec_conf_cat.get('masks', {}).get('case', 'None').lower()
        if mask_case!='none':
            for mtype, mconf in rec_conf_cat['masks'].items():
                if mtype[:12]=='in_footprint' and mconf.get('use', False):
                    print(f"    # Adding footprint mask: {mconf}")
                    mask += ~c[f"ft_{mconf['name']}"]
                    print(f"      * {mask[mask].size:,} clusters masked in total")
                if mtype[:13]=='coverfraction':
                    print(f"    # Adding coverfrac: {mconf}")
                    mask += c[f"cf_{mconf['name']}"] <= float(mconf['min'])
                    print(f"      * {mask[mask].size:,} clusters masked in total")
            # Add mask to args
            kwargs[{'all': 'mask', 'unmatched': 'mask_unmatched'}[mask_case]] = mask
        rec_name = f'{config["outpath"]}/rec_mt{rec_conf["matching_type"]}'
        rec_suf = {'all':'_0mask', 'unmatched':'_0ummask', 'none':''}[mask_case]
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
            plt.savefig(f'{rec_name}_cat{i}_simple_redshift{rec_suf}.png', dpi=rec_conf['dpi'])
            # by mass
            print(f"\n# Simple recovery catalog {i} by mass")
            plt.clf()
            ax = recovery.plot(c, **kwargs, transpose=True,
                add_legend=rec_conf['add_redshift_label'],
                legend_fmt=rec_conf_cat['redshift_num_fmt'],
                )
            ax.set_xlim(rec_conf_cat['mass_lim'])
            ax.set_ylim(rec_conf_cat['recovery_lim'])
            plt.savefig(f'{rec_name}_cat{i}_simple_mass{rec_suf}.png', dpi=rec_conf['dpi'])
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
            plt.savefig(f'{rec_name}_cat{i}_panel_redshift{rec_suf}.png', dpi=rec_conf['dpi'])
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
            plt.savefig(f'{rec_name}_cat{i}_panel_mass{rec_suf}.png', dpi=rec_conf['dpi'])
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
            plt.savefig(f'{rec_name}_cat{i}_2D{rec_suf}.png', dpi=rec_conf['dpi'])
            # with number
            print(f"\n# 2D recovery catalog {i} with numbers")
            plt.clf()
            axes, cb = recovery.plot2D(c, **kwargs, add_num=True)
            ax.set_xlim(rec_conf_cat['redshift_lim'])
            ax.set_ylim(rec_conf_cat['mass_lim'])
            plt.savefig(f'{rec_name}_cat{i}_2D_num{rec_suf}.png', dpi=rec_conf['dpi'])
