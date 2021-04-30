# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
import yaml
from numpy.testing import assert_raises, assert_allclose, assert_equal
from unittest import mock

from clevar import yaml as clevar_yaml

def test_yaml_helper_functions():
    from clevar.yaml import helper_funcs as hf
    hf.add_dicts_diff({'x':None}, {})
    hf.add_dicts_diff({'x':{'x':None}}, {'x':{}})
    hf.add_dicts_diff({'x':{'x':None}}, {'x':{'x':1}})
    hf.get_dicts_diff({'x':{'x':None}}, {'x':{}})
    # loadconf fail
    assert_raises(ValueError, hf.loadconf, 'None_file')
    # loadconf inputs
        # create yml and log.yml
    os.system("rm -r temp")
    yaml.write({'outpath':'temp', 'test':{'val':1}}, 'cfg.yml')
    hf.loadconf('cfg.yml')
        # change yml
    yaml.write({'outpath':'temp', 'test':{'val':2}}, 'cfg.yml')
        # test options
    hf.loadconf('cfg.yml', ['test'], 'orverwrite')
    os.system("rm cfg.yml")
    os.system("rm -r temp")
    # cosmology
    hf.make_cosmology({'backend':'astropy'})
    hf.make_cosmology({'backend':'ccl'})
    assert_raises(ValueError, hf.make_cosmology, {'backend':'unknown'})
    # make_bins
    hf.make_bins(3, log=False)
    hf.make_bins('0 1 3', log=False)
    assert_raises(ValueError, hf.make_bins, '3 3', log=False)
    # get_input_loop
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'pass'
    assert_equal(hf.get_input_loop(options_msg='test',
                                   actions={'pass':(lambda x:'ok', [0],{})}),
                 'ok')
    mock.builtins.input = original_input

def create_base_matched_files(config_file):
    os.system('ln -s demo/cat1.fits')
    os.system('ln -s demo/cat2.fits')
    os.system('rm -rf temp')
    # get demo config
    config = yaml.read(config_file)
    # Match
    clevar_yaml.match_proximity(config_file, overwrite_config=True, overwrite_files=True)
    # Footprint
    os.system(f"rm {config['catalog1']['footprint']} {config['catalog2']['footprint']}")
    clevar_yaml.artificial_footprint(config_file, True, True, case='1')
    clevar_yaml.artificial_footprint(config_file, True, True, case='2')
    # Masks
    ftpt_quantities_file1 = f"{config['outpath']}/ftpt_quantities1.fits"
    ftpt_quantities_file2 = f"{config['outpath']}/ftpt_quantities2.fits"
    os.system(f"rm {ftpt_quantities_file1} {ftpt_quantities_file2}")
    clevar_yaml.footprint_masks(config_file, True, False, case='1')
    clevar_yaml.footprint_masks(config_file, True, False, case='2')
    return config

def test_yaml_funcs():
    # Get main files
    config_file = 'demo/config.yml'
    config = create_base_matched_files(config_file)
    # Match, used diff cosmology and overwrite
    print(config.keys())
    config['proximity_match']['step1']['cosmology'] = {'backend': 'CCL'}
    yaml.write(config, 'cfg.yml')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.match_proximity('cfg.yml', overwrite_config=True, overwrite_files=False)
    mock.builtins.input = original_input
    os.system("rm cfg.yml")
    # Footprint overwrite
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.artificial_footprint(config_file, True, False, case='1')
    clevar_yaml.artificial_footprint(config_file, True, False, case='2')
    mock.builtins.input = original_input
    # Masks overwrite
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.footprint_masks(config_file, True, False, case='1')
    clevar_yaml.footprint_masks(config_file, True, False, case='2')
    mock.builtins.input = original_input
    # Write full files
    clevar_yaml.write_full_output(config_file, True, True)
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.write_full_output(config_file, True, False)
    mock.builtins.input = original_input
    # cleanup
    os.system(f"rm {config['catalog1']['footprint']} {config['catalog2']['footprint']}")
    os.system('rm cat1.fits')
    os.system('rm cat2.fits')
    os.system('rm -rf temp')

def test_yaml_plots():
    # Get main files
    config_file = 'demo/config.yml'
    config = create_base_matched_files(config_file)
    # Metrics
    clevar_yaml.match_metrics_distances(config_file)
    clevar_yaml.match_metrics_mass(config_file)
    clevar_yaml.match_metrics_recovery_rate(config_file)
    clevar_yaml.match_metrics_redshift(config_file)
    # cleanup
    os.system(f"rm {config['catalog1']['footprint']} {config['catalog2']['footprint']}")
    os.system('rm cat1.fits')
    os.system('rm cat2.fits')
    os.system('rm -rf temp')
