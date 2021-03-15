import os, sys
import yaml
import numpy as np
import clevar
from clevar.utils import veclen
def add_dicts_diff(dict1, dict2, pref='', diff_lines=[]):
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
    return diff_lines
def loadconf(consistency_configs=[]):
    print("\n## Loading config")
    if len(sys.argv)<2:
        raise ValueError('Config file must be provided')
    elif not os.path.isfile(sys.argv[1]):
        raise ValueError(f'Config file "{sys.argv[1]}" not found')
    config_file = sys.argv[1]
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
        if len(diff_configs)>1:
            action = input('\nOverwrite(o) and proceed or Quit(q)?\n')
            while action not in ('o', 'q'):
                action = input(f'Option {action} not valid. Please choose: Overwrite (o) or Cancel(c)\n')
            if action=='o':
                os.system(f'cp {config_file} {check_file}')
            elif action=='q':
                exit()

    return config
def make_catalog(cat_config):
    c0 = clevar.ClData.read(cat_config['file'])
    cat = clevar.Catalog(cat_config['name'], 
        **{k:c0[v] for k, v in cat_config['columns'].items()})
    cat.radius_unit = cat_config['radius_unit'] if 'radius_unit' in cat_config\
                        else None
    return cat
def make_cosmology(cosmo_config):
    print("\n# Creating cosmology")
    if cosmo_config['backend'].lower()=='astropy':
        CosmoClass = clevar.cosmology.AstroPyCosmology
    elif cosmo_config['backend'].lower()=='astropy':
        CosmoClass = clevar.cosmology.AstroPyCosmology
    else:
        raise ValueError(f'Cosmology backend "{cosmo_config["backend"]}" not accepted')
    parameters = cosmo_config['parameters'] if cosmo_config['parameters'] else {}
    return CosmoClass(**parameters)
