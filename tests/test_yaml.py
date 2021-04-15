# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
import yaml
from numpy.testing import assert_raises, assert_allclose, assert_equal
from unittest import mock

from clevar import yaml as clevar_yaml

def test_helper_functions():
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

def test_main():
    #"""
    os.system('ln -s demo/cat1.fits')
    os.system('ln -s demo/cat2.fits')
    config_file = 'demo/config.yml'
    overwrite_config, overwrite_files = True, True
    # Match
    clevar_yaml.match_proximity(config_file, overwrite_config, overwrite_files)
    # Match, used diff cosmology and overwrite
    config = yaml.read(config_file)
    print(config.keys())
    config['proximity_match'].update({'cosmology':{'backend': 'CCL'}})
    yaml.write(config, 'cfg.yml')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.match_proximity('cfg.yml', overwrite_config, overwrite_files=False)
    mock.builtins.input = original_input
    os.system("rm cfg.yml")
    # Footprint
    os.system('rm ftpt1.fits')
    os.system('rm ftpt2.fits')
    clevar_yaml.artificial_footprint(config_file, True, False, case='1')
    clevar_yaml.artificial_footprint(config_file, True, False, case='2')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.artificial_footprint(config_file, True, False, case='1')
    clevar_yaml.artificial_footprint(config_file, True, False, case='2')
    mock.builtins.input = original_input
    os.system(f"rm {config['catalog1']['footprint']} {config['catalog2']['footprint']}")
    # Metrics
    clevar_yaml.match_metrics_distances(config_file)
    clevar_yaml.match_metrics_mass(config_file)
    clevar_yaml.match_metrics_recovery_rate(config_file)
    clevar_yaml.match_metrics_redshift(config_file)
    os.system('rm cat1.fits')
    os.system('rm cat2.fits')
    os.system('rm -rf temp')
    #"""
