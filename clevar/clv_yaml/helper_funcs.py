"""@file yaml/helper_funcs.py
Helper functions for command line execution
"""
import os
import yaml
import numpy as np

from clevar.catalog import ClCatalog
from clevar import cosmology
from clevar.utils import get_dicts_diff, deep_update


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
    with open(filename, "r", encoding="UTF-8") as file_handle:
        config = yaml.load(file_handle, Loader=yaml.FullLoader)
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
    with open(filename, "w", encoding="UTF-8") as file_handle:
        yaml.dump(config, file_handle)


yaml.write = write_yaml
yaml.read = read_yaml


########################################################################
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
        action = input(f"\n{options_msg}\n")
        loop = action not in actions
        print(f"Option {action} not valid. Please choose:")
    func, args, kwargs = actions[action]
    return func(*args, **kwargs)


def loadconf(
    config_file, load_configs=None, add_new_configs=None, fail_action="ask", check_matching=False
):
    """
    Load configuration from yaml file, creates output directory and config.log.yml

    Parameters
    ----------
    config_file: str
        Yaml configuration file.
    load_configs: list, None
        List of configurations loaded (will be checked with config.log.yml).
    add_new_configs: list, None
        List of configurations that will be automatically added if not in config.log.yml.
    fail_action: str
        Action to do when there is inconsistency in configs.
        Options are 'ask', 'overwrite' and 'quit'
    check_matching: bool
        Check matching config
    Returns
    -------
    dict
        Configuration for clevar
    """
    print("\n## Loading config")
    if load_configs is None:
        load_configs = []
    if add_new_configs is None:
        add_new_configs = []
    if not os.path.isfile(config_file):
        raise ValueError(f'Config file "{config_file}" not found')
    base_cfg_file = f"{os.path.dirname(__file__)}/base_config.yml"
    config = deep_update(yaml.read(base_cfg_file), yaml.read(config_file))
    main_load_configs = ["outpath", "matching_mode"]
    if check_matching:
        main_load_configs += [f"{config['matching_mode']}_match"]
    config = {k: config[k] for k in main_load_configs + load_configs}
    # Add outpath suffix to differentiate proximity from memebership
    config["outpath"] += "_" + config["matching_mode"]
    # Checks if config is consistent with log file
    log_file = f'{config["outpath"]}/config.log.yml'
    if not os.path.isdir(config["outpath"]):
        os.mkdir(config["outpath"])
        log_config = config
    else:
        log_config = yaml.read(log_file)
        for key in add_new_configs:
            log_config[key] = log_config.get(key, config[key])
        diff_configs = get_dicts_diff(
            log_config,
            config,
            keys=load_configs,
            header=["Name", "Saved", "New"],
            msg="\nConfigurations differs from saved config:\n",
        )
        if len(diff_configs) > 0:
            actions_loop = {
                "o": (lambda: True, [], {}),
                "q": (lambda: None, [], {}),
            }
            func, args, kwargs = {
                "ask": (get_input_loop, ["Overwrite(o) and proceed or Quit(q)?", actions_loop], {}),
                "orverwrite": actions_loop["o"],
                "quit": actions_loop["q"],
            }[fail_action]
            if func(*args, **kwargs) is None:
                return None
    deep_update(log_config, config)
    yaml.write(log_config, log_file)
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
    return ClCatalog.read(
        cat_config["file"],
        name=cat_config["name"],
        tags=cat_config["columns"],
        radius_unit=cat_config.get("radius_unit", None),
        labels=cat_config.get("labels", {}),
    )


def add_mem_catalog(cat, cat_config):
    """
    Make a clevar.MemCatalog object based on config

    Parameters
    ----------
    cat_config: dict
        MemCatalog configuration

    Returns
    -------
    clevar.MemCatalog
        MemCatalog based on input config
    """
    cat.read_members(cat_config["file"], tags=cat_config["columns"])
    cat.members.labels.update(cat_config.get("labels", {}))


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
    if cosmo_config["backend"].lower() == "astropy":
        CosmoClass = cosmology.AstroPyCosmology
    elif cosmo_config["backend"].lower() == "ccl":
        CosmoClass = cosmology.CCLCosmology
    else:
        raise ValueError(f'Cosmology backend "{cosmo_config["backend"]}" not accepted')
    parameters = {
        "H0": 70.0,
        "Omega_b0": 0.05,
        "Omega_dm0": 0.25,
        "Omega_k0": 0.0,
    }
    parameters.update(cosmo_config.get("parameters", {}))
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
    vals = input_val.split(" ")
    if len(vals) != 3:
        raise ValueError(f"Values ({input_val}) must be 1 intergers or 3 numbers (xmin, xmax, dx)")
    out = np.arange(*np.array(vals, dtype=float))
    return 10**out if log else out
