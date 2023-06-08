""" Tests for utils.py """
from clevar import utils


def test_utils_dict_functions():
    utils.add_dicts_diff({"x": None}, {})
    utils.add_dicts_diff({"x": {"x": None}}, {"x": {}})
    utils.add_dicts_diff({"x": {"x": None}}, {"x": {"x": 1}})
    utils.get_dicts_diff({"x": {"x": None}}, {"x": {}})
