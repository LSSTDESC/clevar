""" Tests for utils.py """
from clevar import utils


def test_utils_dict_functions():
    utils.add_dicts_diff({"x": None}, {})
    utils.add_dicts_diff({"x": {"x": None}}, {"x": {}})
    utils.add_dicts_diff({"x": {"x": None}}, {"x": {"x": 1}})
    utils.get_dicts_diff({"x": {"x": None}}, {"x": {}})


def test_import_safe():
    lib1 = utils.import_safe("numpy")
    assert lib1 is not None
    lib2 = utils.import_safe("NotALibrary")
    assert lib2 is None


def test_timer_class():
    utils.LPROFILER.reset_level()
    LPtime = utils.Timer("test_name")
    LPtime.title(" (some subtitle)")
    LPtime.time()
    LPtime.end()


def test_other():
    assert utils.none_val("y", "x") == "y"
    assert utils.none_val(None, "x") == "x"
    assert utils.index_list(["A", "B"], [1, 0]) == ["B", "A"]
