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
    yaml.write({'outpath':'temp', 'matching_mode':'proximity', 'test':{'val':1}}, 'cfg.yml')
    hf.loadconf('cfg.yml')
        # change yml
    yaml.write({'outpath':'temp',  'matching_mode':'proximity','test':{'val':2}}, 'cfg.yml')
        # test options
    hf.loadconf('cfg.yml', ['test'], fail_action='orverwrite')
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

def create_base_matched_files(config_file, matching_mode):
    base_matched_files_cleanup(config_file, matching_mode)
    os.system('ln -s demo/cat1.fits')
    os.system('ln -s demo/cat2.fits')
    os.system('ln -s demo/cat1_mem.fits')
    os.system('ln -s demo/cat2_mem.fits')
    # get demo config
    config = yaml.read(config_file)
    config['matching_mode'] = matching_mode
    outpath = config['outpath']+'_'+matching_mode
    config_file_temp = 'cfg_cbmf_temp.yaml'
    yaml.write(config, config_file_temp)
    # Match
    clevar_yaml.match(config_file_temp, overwrite_config=True, overwrite_files=True)
    # Footprint
    clevar_yaml.artificial_footprint(config_file_temp, True, True, case='1')
    clevar_yaml.artificial_footprint(config_file_temp, True, True, case='2')
    # Masks
    ftpt_quantities_file1 = f"{outpath}/ftpt_quantities1.fits"
    ftpt_quantities_file2 = f"{outpath}/ftpt_quantities2.fits"
    clevar_yaml.footprint_masks(config_file_temp, True, False, case='1')
    clevar_yaml.footprint_masks(config_file_temp, True, False, case='2')
    os.system(f'rm -f {config_file_temp}')
    return config

def base_matched_files_cleanup(config_file, matching_mode):
    config = yaml.read(config_file)
    outpath = config['outpath']+'_'+matching_mode
    cleanup_files = ' '.join([
        'cat1.fits',
        'cat2.fits',
        'cat1_mem.fits',
        'cat2_mem.fits',
        f'temp_{matching_mode}',
        f'{outpath}/ftpt_quantities1.fits',
        f'{outpath}/ftpt_quantities2.fits',
        'temp_mems_mt.txt',
        'temp_shared.2.p',
        'temp_shared.1.p',
        'ftpt1.fits',
        'ftpt2.fits',
        ])
    os.system(f'rm -rf {cleanup_files}')
    return

def test_yaml_funcs_prox():
    # Get main files
    in_file = 'demo/config.yml'
    config = create_base_matched_files(in_file, 'proximity')
    config_file = 'test_config.yml'
    yaml.write(config, config_file)
    # fail method for matching
    yaml.write({'outpath':'temp', 'matching_mode':'unknown'}, 'cfg.yml')
    assert_raises(ValueError, clevar_yaml.match, 'cfg.yml', overwrite_config=False, overwrite_files=False)
    # Match, used diff cosmology and overwrite
        # 1
    print("############ Fail cfg check #########")
    config['proximity_match']['step1']['cosmology'] = {'backend': 'CCL'}
    yaml.write(config, 'cfg.yml')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.match('cfg.yml', overwrite_config=False, overwrite_files=False)
        # 2
    config['cosmology'] = {'backend': 'CCL'}
    yaml.write(config, 'cfg.yml')
    mock.builtins.input = lambda _: 'o'
    clevar_yaml.match('cfg.yml', overwrite_config=False, overwrite_files=False)
    mock.builtins.input = original_input
    clevar_yaml.match(config_file, overwrite_config=True, overwrite_files=True)
    # Footprint overwrite
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.artificial_footprint(config_file, True, False, case='1')
    clevar_yaml.artificial_footprint(config_file, True, False, case='2')
    clevar_yaml.artificial_footprint('cfg.yml', False, True, case='2')
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
    base_matched_files_cleanup(config_file, 'proximity')
    os.system("rm -f test_config.yml cfg.yml")

def test_yaml_funcs_mem():
    # Get main files
    in_file = 'demo/config.yml'
    config = create_base_matched_files(in_file, 'membership')
    config_file = 'test_config.yml'
    yaml.write(config, config_file)
    # Match, used diff config and overwrite
    print(config.keys())
    config['membership_match']['match_members_kwargs'] = {'method': 'angular_distance', 'radius': '0.0001pc'}
    yaml.write(config, 'cfg.yml')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.match('cfg.yml', overwrite_config=False, overwrite_files=False)
    mock.builtins.input = lambda _: 'o'
    clevar_yaml.match('cfg.yml', overwrite_config=False, overwrite_files=False)
    mock.builtins.input = lambda _: 'o'
    clevar_yaml.match(config_file, overwrite_config=False, overwrite_files=False)
    mock.builtins.input = original_input
    #adsaf
    # Footprint overwrite
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.artificial_footprint(config_file, True, False, case='1')
    clevar_yaml.artificial_footprint(config_file, True, False, case='2')
    clevar_yaml.artificial_footprint('cfg.yml', False, True, case='2')
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
    base_matched_files_cleanup(config_file, 'membership')
    os.system("rm -f test_config.yml cfg.yml")

def test_yaml_plots():
    # Just testing for proximity match
    # Get main files
    in_file = 'demo/config.yml'
    config = create_base_matched_files(in_file, 'proximity')
    config_file = 'test_config.yml'
    yaml.write(config, config_file)
    # Metrics
    clevar_yaml.match_metrics_distances(config_file)
    clevar_yaml.match_metrics_mass(config_file)
    clevar_yaml.match_metrics_recovery_rate(config_file)
    clevar_yaml.match_metrics_redshift(config_file)
    # Create different config for skiping run
    config_diff = {}
    config_diff.update(config)
    for c in [c_ for c_ in config if c_[:10]=='mt_metrics']:
        config_diff[c]['dpi'] = 10
    yaml.write(config_diff, 'cfg.yml')
    original_input = mock.builtins.input
    mock.builtins.input = lambda _: 'q'
    clevar_yaml.match_metrics_distances('cfg.yml')
    clevar_yaml.match_metrics_mass('cfg.yml')
    clevar_yaml.match_metrics_recovery_rate('cfg.yml')
    clevar_yaml.match_metrics_redshift('cfg.yml')
    mock.builtins.input = original_input
    # cleanup
    base_matched_files_cleanup(config_file, 'proximity')
    os.system("rm -f test_config.yml cfg.yml")
