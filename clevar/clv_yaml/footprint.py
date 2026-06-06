"""@file footprint.py
Footprint functions for command line execution
"""

import os

import clevar

from .helper_funcs import get_input_loop, loadconf, make_catalog, make_cosmology

_check_actions = {
    "o": (lambda: True, [], {}),
    "q": (lambda: False, [], {}),
}


def artificial(config_file, overwrite_config, overwrite_files, case):
    """Function to create footprint

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of output files
    case: str
        Run for which catalog. Options: 1, 2, both
    """
    # Create clevar objects from yml config
    config = loadconf(
        config_file,
        load_configs=["catalog1", "catalog2", "cosmology", "masks"],
        fail_action="orverwrite" if overwrite_config else "ask",
    )
    if config is None:
        return
    if case in ("1", "both"):
        print("\n# Creating footprint 1")
        save = True
        ftpt_cfg1 = config["catalog1"]["footprint"]
        if os.path.isfile(ftpt_cfg1["file"]) and not overwrite_files:
            print(f"\n*** File '{ftpt_cfg1['file']}' already exist! ***")
            save = get_input_loop("Overwrite(o) and proceed or Quit(q)?", _check_actions)
        if save:
            print("\n# Reading Catalog 1")
            cat1 = make_catalog(config["catalog1"])
            ftpt1 = clevar.footprint.create_artificial_footprint(
                cat1["ra"], cat1["dec"], nside=ftpt_cfg1["nside"], nest=ftpt_cfg1["nest"]
            )  # min_density=2, neighbor_fill=None
            ftpt1[["pixel"]].write(ftpt_cfg1["file"], overwrite=True)
    if case in ("2", "both"):
        print("\n# Creating footprint 2")
        save = True
        ftpt_cfg2 = config["catalog2"]["footprint"]
        if os.path.isfile(ftpt_cfg2["file"]) and not overwrite_files:
            print(f"\n*** File '{ftpt_cfg2['file']}' already exist! ***")
            save = get_input_loop("Overwrite(o) and proceed or Quit(q)?", _check_actions)
        if save:
            print("\n# Reading Catalog 2")
            cat2 = make_catalog(config["catalog2"])
            ftpt2 = clevar.footprint.create_artificial_footprint(
                cat2["ra"], cat2["dec"], nside=ftpt_cfg2["nside"], nest=ftpt_cfg2["nest"]
            )  # min_density=2, neighbor_fill=None
            ftpt2[["pixel"]].write(ftpt_cfg2["file"], overwrite=True)


def prep_ftpt_config(config):
    """
    Prepare footprint coming from yml file into kwargs for Footprint

    Parameters
    ----------
    config: dict
        Footprint config from yml

    Returns
    -------
    kwargs: dict
        kwargs to instanciate clevar.footprint.Footprint
    """
    kwargs = {"tags": {}}
    for key, value in config.items():
        if key == "file":
            kwargs["filename"] = value
        elif key in ("nside", "nest"):
            kwargs[key] = value
        elif value == "None":
            pass
        else:
            kwargs["tags"][key] = value
    return kwargs


def make_masks(config_file, overwrite_config, overwrite_files, case):
    """Makes footprint masks for catalogs

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    overwrite_config: bool
        Forces overwrite of config.log.yml file
    overwrite_files: bool
        Forces overwrite of output files
    case: str
        Run for which catalog. Options: 1, 2, both
    """
    # Create clevar objects from yml config
    config = loadconf(
        config_file,
        load_configs=["catalog1", "catalog2", "cosmology", "masks"],
        fail_action="orverwrite" if overwrite_config else "ask",
    )
    print("\n# Creating Cosmology")
    cosmo = make_cosmology(config["cosmology"])
    # Read footprints
    ftpt1 = (
        clevar.Footprint.read(**prep_ftpt_config(config["catalog1"]["footprint"]))
        if config["catalog1"]["footprint"]["file"] != "None"
        else None
    )
    ftpt2 = (
        clevar.Footprint.read(**prep_ftpt_config(config["catalog2"]["footprint"]))
        if config["catalog2"]["footprint"]["file"] != "None"
        else None
    )
    # Catalog 1
    if case in ("1", "both"):
        print("\n# Creating masks for catalog 1")
        print("\n# Reading Catalog 1")
        _make_mask(
            config["catalog1"],
            f"{config['outpath']}/ftpt_quantities1.fits",
            config["masks"]["catalog1"],
            ftpt1,
            ftpt2,
            cosmo,
            overwrite_files,
        )
    # Catalog 2
    if case in ("2", "both"):
        print("\n# Creating masks for catalog 2")
        print("\n# Reading Catalog 2")
        _make_mask(
            config["catalog2"],
            f"{config['outpath']}/ftpt_quantities2.fits",
            config["masks"]["catalog2"],
            ftpt2,
            ftpt1,
            cosmo,
            overwrite_files,
        )


def _make_mask(
    cat_filename,
    ftpt_quantities_filename,
    masks_config,
    ftpt_self,
    ftpt_other,
    cosmo,
    overwrite_files,
):
    """Makes footprint masks for catalog

    Parameters
    ----------
    cat_filename : str
        Name of catalog file
    ftpt_quantities_filename : str
        Name of footprint quantities file
    masks_config : dict
        Configuration for masks
    ftpt_self : clevar.Footprint object
        Footprint of this catalog
    ftpt_other : clevar.Footprint object
        Footprint of the other catalog
    cosmo : clevar.Cosmology object
        Cosmology
    overwrite_files: bool
        Forces overwrite of output files
    """
    ftpts = {"self": ftpt_self, "other": ftpt_other}
    cat = make_catalog(cat_filename)
    for cf_name, mask_cfg in masks_config.items():
        if cf_name[:12] == "in_footprint":
            print(f"\n# Adding footprint mask: {mask_cfg}")
            # pylint: disable=protected-access
            cat._add_ftpt_mask(ftpts[mask_cfg["which_footprint"]], maskname=mask_cfg["name"])
        if cf_name[:13] == "coverfraction":
            aperture, aperture_unit = clevar.utils.str2dataunit(
                mask_cfg["aperture"], clevar.geometry.units_bank
            )
            print(f"\n# Adding coverfrac: {mask_cfg}")
            cat.add_ftpt_coverfrac(
                ftpts[mask_cfg["which_footprint"]],
                aperture,
                aperture_unit,
                window=mask_cfg["window_function"],
                colname=mask_cfg["name"],
                cosmo=cosmo,
            )

    save = True
    if os.path.isfile(ftpt_quantities_filename) and not overwrite_files:
        print(f"\n*** File '{ftpt_quantities_filename}' already exist! ***")
        save = get_input_loop("Overwrite(o) and proceed or Quit(q)?", _check_actions)

    if save:
        cat.save_footprint_quantities(ftpt_quantities_filename, overwrite=True)
