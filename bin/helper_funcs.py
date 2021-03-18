import os, sys
import yaml
import argparse
import numpy as np

import clevar
from clevar.utils import veclen
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
    action = input(f'\n{options_msg}\n')
    while action not in actions:
        action = input(f'Option {action} not valid. Please choose: {options_msg}\n')
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
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    check_file = f'{config["outpath"]}/config.log.yml'
    if not os.path.isdir(config['outpath']):
        os.mkdir(config['outpath'])
        os.system(f'cp {config_file} {check_file}')
    else:
        with open(check_file) as file:
            check_config = yaml.load(file, Loader=yaml.FullLoader)
        diff_configs = get_dicts_diff(config, check_config, keys=consistency_configs,
                                        header=['Name', 'New', 'Saved'],
                                        msg='\nConfigurations differs from saved config:\n')
        if len(diff_configs)>0:
            actions_loop = {
                'o': (os.system, [f'cp {config_file} {check_file}'], {}),
                'q': (exit, [], {}),
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
    Make a clevar.Catalog object based on config

    Parameters
    ----------
    cat_config: dict
        Catalog configuration

    Returns
    -------
    clevar.Catalog
        Catalog based on input config
    """
    c0 = clevar.ClData.read(cat_config['file'])
    cat = clevar.Catalog(cat_config['name'], 
        **{k:c0[v] for k, v in cat_config['columns'].items()})
    cat.radius_unit = cat_config['radius_unit'] if 'radius_unit' in cat_config\
                        else None
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
        CosmoClass = clevar.cosmology.AstroPyCosmology
    elif cosmo_config['backend'].lower()=='astropy':
        CosmoClass = clevar.cosmology.AstroPyCosmology
    else:
        raise ValueError(f'Cosmology backend "{cosmo_config["backend"]}" not accepted')
    parameters = cosmo_config['parameters'] if cosmo_config['parameters'] else {}
    return CosmoClass(**parameters)
