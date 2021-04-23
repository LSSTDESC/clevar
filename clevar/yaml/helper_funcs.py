"""@file yaml/helper_funcs.py
Helper functions for command line execution
"""
import os, sys
import yaml
import argparse
import numpy as np

from clevar.catalog import ClData, ClCatalog
from clevar import cosmology
from clevar.utils import none_val, veclen
######################################################################
########## Monkeypatching yaml #######################################
######################################################################
def read_yaml(filename):
    """
    Read yaml file

    Parameters
    ----------
    filename: str
        Name of yaml file

    Returns
    -------
    config: dict
        Dictionary with yaml file info
    """
    f = open(filename, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config
def write_yaml(config, filename):
    """
    Write yaml file

    Parameters
    ----------
    config: dict
        Dictionary to write
    filename: str
        Name of yaml file
    """
    f = open(filename, 'w')
    yaml.dump(config, f)
    f.close()
yaml.write = write_yaml
yaml.read = read_yaml
########################################################################
def add_dicts_diff(dict1, dict2, pref='', diff_lines=[]):
    """
    Adds the differences between dictionaries to a list

    Parameters
    ----------
    dict1, dict2: dict
        Dictionaies to be compared
    pref: str
        Prefix to be added in output
    diff_lines: list
        List where differences will be appended to
    """
    for k in dict1:
        if k not in dict2:
            diff_lines.append((f'{pref}[{k}]', 'present', 'missing'))
            return
        if dict1[k]!=dict2[k]:
            if isinstance(dict1[k], dict):
                add_dicts_diff(dict1[k], dict2[k], pref=f'{pref}[{k}]',
                                diff_lines=diff_lines)
            else:
                diff_lines.append((f'{pref}[{k}]', str(dict1[k]), str(dict2[k])))
def get_dicts_diff(dict1, dict2, keys=None,
        header=['Name', 'dict1', 'dict2'],
        msg=''):
    """
    Get all the differences between dictionaries, accounting for nested dictionaries.
    If there are differences, a table with the information is printed.

    Parameters
    ----------
    dict1, dict2: dict
        Dictionaies to be compared
    keys: list, None
        List of keys to be compared. If None, all keys are compared
    header: str
        Header for differences table
    msg: str
        Message printed before the differences

    Returns
    -------
    diff_lines:
        List of dictionaries differences
    """
    diff_lines = [header]
    if keys is None:
        keys = set(list(dict1.keys())+list(dict2.keys()))
    for k in keys:
        add_dicts_diff(dict1[k], dict2[k], pref=f'[{k}]', diff_lines=diff_lines)
    if len(diff_lines)>1:
        diff_lines = np.array(diff_lines)
        max_sizes = [max(veclen(l)) for l in diff_lines.T]
        fmts = f'  %-{max_sizes[0]}s | %{max_sizes[1]}s | %{max_sizes[2]}s'
        print(msg)
        print(fmts%tuple(diff_lines[0]))
        print(f'  {"-"*max_sizes[0]}-|-{"-"*max_sizes[1]}-|-{"-"*max_sizes[2]}')
        for l in diff_lines[1:]:
            print(fmts%tuple(l))
    return diff_lines[1:]
def get_input_loop(options_msg, actions):
    """
    Get input from fixed values

    Parameters
    ----------
    options_msg: str
        Description of the possible input options
    actions: dict
        Dictionary with the actions to be made. Values must be (function, args, kwargs)
    """
    loop = True
    while loop:
        action = input(f'\n{options_msg}\n')
        loop = action not in actions
        prt = print(f'Option {action} not valid. Please choose:') if loop else None
    f, args, kwargs = actions[action]
    return f(*args, **kwargs)
def loadconf(config_file, consistency_configs=[], fail_action='ask'):
    """
    Load configuration from yaml file, creates output directory and config.log.yml

    Parameters
    ----------
    config_file: str
        Yaml configuration file
    consistency_configs: list
        List of configurations to be checked with config.log.yml
    fail_action: str
        Action to do when there is inconsistency in configs.
        Options are 'ask', 'overwrite' and 'quit'

    Returns
    -------
    dict
        Configuration for clevar
    """
    print("\n## Loading config")
    if not os.path.isfile(config_file):
        raise ValueError(f'Config file "{config_file}" not found')
    config = yaml.read(config_file)
    check_file = f'{config["outpath"]}/config.log.yml'
    if not os.path.isdir(config['outpath']):
        os.mkdir(config['outpath'])
        os.system(f'cp {config_file} {check_file}')
    else:
        check_config = yaml.read(check_file)
        diff_configs = get_dicts_diff(config, check_config, keys=consistency_configs,
                                        header=['Name', 'New', 'Saved'],
                                        msg='\nConfigurations differs from saved config:\n')
        if len(diff_configs)>0:
            actions_loop = {
                'o': (os.system, [f'cp {config_file} {check_file}'], {}),
                'q': (lambda _:0, [], {}),
                }
            f, args, kwargs = {
                'ask': (get_input_loop, ['Overwrite(o) and proceed or Quit(q)?', actions_loop], {}),
                'orverwrite': actions_loop['o'],
                'quit': actions_loop['q'],
            }[fail_action]
            f(*args, **kwargs)
    return config
def make_catalog(cat_config):
    """
    Make a clevar.ClCatalog object based on config

    Parameters
    ----------
    cat_config: dict
        ClCatalog configuration

    Returns
    -------
    clevar.ClCatalog
        ClCatalog based on input config
    """
    c0 = ClData.read(cat_config['file'])
    cat = ClCatalog(cat_config['name'],
        **{k:c0[v] for k, v in cat_config['columns'].items()})
    cat.radius_unit = cat_config.get('radius_unit', None)
    cat.labels.update(cat_config.get('labels', {}))
    return cat
def make_cosmology(cosmo_config):
    """
    Make a cosmology object based on config

    Parameters
    ----------
    cosmo_config: dict
        Cosmology configuration

    Returns
    -------
    clevar.Cosmology
        Cosmology based on the input config
    """
    if cosmo_config['backend'].lower()=='astropy':
        CosmoClass = cosmology.AstroPyCosmology
    elif cosmo_config['backend'].lower()=='ccl':
        CosmoClass = cosmology.CCLCosmology
    else:
        raise ValueError(f'Cosmology backend "{cosmo_config["backend"]}" not accepted')
    parameters = none_val(cosmo_config.get('parameters', None), {})
    return CosmoClass(**parameters)
def make_bins(input_val, log=False):
    """
    Make array for bins string input

    Parameters
    ----------
    input_val: any
        Value given by yaml config
    log: bool
        Use log scale

    Returns
    -------
    int, array
        Bins to be used
    """
    if isinstance(input_val, int):
        return input_val
    vals = input_val.split(' ')
    if len(vals)!=3:
        raise ValueError(f"Values ({input_val}) must be 1 intergers or 3 numbers (xmin, xmax, dx)")
    out = np.arange(*np.array(vals, dtype=float))
    return 10**out if log else out
