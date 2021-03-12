import os, sys
import yaml
import clevar

def loadconf():
    if len(sys.argv)<2:
        raise ValueError('Config file must be provided')
    elif not os.path.isfile(sys.argv[1]):
        raise ValueError(f'Config file "{sys.argv[1]}" not found')
    with open(sys.argv[1]) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
def make_catalog(cat_config):
    c0 = clevar.ClData.read(cat_config['file'])
    cat = clevar.Catalog(cat_config['name'], 
        **{k:c0[v] for k, v in cat_config['columns'].items()})
    cat.radius_unit = cat_config['radius_unit'] if 'radius_unit' in cat_config\
                        else None
    return cat
def make_cosmology(cosmo_config):
    if cosmo_config['backend'].lower()=='astropy':
        CosmoClass = clevar.cosmology.AstroPyCosmology
    elif cosmo_config['backend'].lower()=='astropy':
        CosmoClass = clevar.cosmology.AstroPyCosmology
    else:
        raise ValueError(f'Cosmology backend "{cosmo_config["backend"]}" not accepted')
    parameters = cosmo_config['parameters'] if cosmo_config['parameters'] else {}
    return CosmoClass(**parameters)
