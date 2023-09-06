# pylint: disable=no-member, protected-access
""" Tests for match.py """
import os
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_equal

from clevar.catalog import ClCatalog, MemCatalog
from clevar.match.parent import Match
from clevar.match.spatial import SpatialMatch
from clevar.match import (
    ProximityMatch,
    MembershipMatch,
    BoxMatch,
    output_catalog_with_matching,
    output_matched_catalog,
    get_matched_pairs,
)


def test_parent():
    mt = Match()
    assert_raises(NotImplementedError, mt.prep_cat_for_match, None)
    assert_raises(NotImplementedError, mt.multiple, None, None)
    assert_raises(NotImplementedError, mt.match_from_config, None, None, None, None)
    assert_raises(ValueError, mt._get_dist_mt, None, None, "unknown match")

def test_spatial():
    mt = SpatialMatch()
    assert_raises(NotImplementedError, mt.prep_cat_for_match, None)
    assert_raises(NotImplementedError, mt.multiple, None, None)
    assert_raises(NotImplementedError, mt.match_from_config, None, None, None, None)
    assert_raises(ValueError, mt._get_dist_mt, None, None, "unknown match")


def _test_mt_results(cat, multi_self, self, cross, multi_other=None, other=None):
    multi_other = multi_self if multi_other is None else multi_other
    other = self if other is None else other
    # Check multiple match
    slists = lambda mmt: [sorted(l) for l in mmt]
    assert_equal(slists(cat["mt_multi_self"]), slists(multi_self))
    assert_equal(slists(cat["mt_multi_other"]), slists(multi_other))
    # Check unique
    assert_equal(cat["mt_self"], self)
    assert_equal(cat["mt_other"], other)
    # Check cross
    assert_equal(cat["mt_cross"], cross)


def get_test_data_prox():
    input1 = {
        "id": [f"CL{i}" for i in range(5)],
        "ra": [0.0, 0.0001, 0.00011, 25, 20],
        "dec": [0.0, 0, 0, 0, 0],
        "z": [0.2, 0.3, 0.25, 0.4, 0.35],
        "mass": [10**13.5, 10**13.4, 10**13.3, 10**13.8, 10**14],
    }
    input2 = {k: v[:4] for k, v in input1.items()}
    input2["z"][:2] = [0.3, 0.2]
    input2["mass"][:3] = input2["mass"][:3][::-1]
    return input1, input2


def test_proximity(CosmoClass):
    input1, input2 = get_test_data_prox()
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    print(cat1.data)
    print(cat2.data)
    cosmo = CosmoClass()
    mt = ProximityMatch()
    # test missing data
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = "xx"
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = None
    # init match
    mt_config1 = {"delta_z": 0.2, "match_radius": "1 mpc", "cosmo": cosmo}
    mt_config2 = {"delta_z": 0.2, "match_radius": "1 arcsec"}
    mt.prep_cat_for_match(cat1, **mt_config1)
    mt.prep_cat_for_match(cat2, **mt_config2)
    # Check prep cat
    assert_allclose(cat2.mt_input["ang"], np.ones(cat2.size) / 3600.0)
    # Check multiple match
    mmt = [
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL3"],
        [],
    ]
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "angular_proximity")
    mt.unique(cat2, cat1, "angular_proximity")
    cat1.show_mt_hist()
    cat2.show_mt_hist()
    cat1.cross_match()
    cat2.cross_match()
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    # Check unique with mass preference
    for col in ("mt_self", "mt_other"):
        cat1[col] = None
        cat2[col] = None
    mt.unique(cat1, cat2, "more_massive")
    mt.unique(cat2, cat1, "more_massive")
    mt.cross_match(cat1)
    mt.cross_match(cat2)
    smt = ["CL2", "CL1", "CL0", "CL3", None]
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    # Check unique with z preference
    for col in ("mt_self", "mt_other"):
        cat1[col] = None
        cat2[col] = None
    cat2["mt_other"][0] = "CL3"  # to force a replacement
    mt.unique(cat1, cat2, "redshift_proximity")
    mt.unique(cat2, cat1, "redshift_proximity")
    mt.cross_match(cat1)
    mt.cross_match(cat2)
    smt = ["CL1", "CL0", "CL2", None, None]
    omt = ["CL1", "CL0", "CL2", "CL3", None]
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt, other=omt)
    smt = ["CL1", "CL0", "CL2", "CL3", None]
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    # Error for unkown preference
    assert_raises(ValueError, mt.unique, cat1, cat2, "unknown")
    # Check save and load matching
    mt.save_matches(cat1, cat2, out_dir="temp", overwrite=True)
    cat1_v2 = ClCatalog("Cat1", **input1)
    cat2_v2 = ClCatalog("Cat2", **input2)
    mt.load_matches(cat1_v2, cat2_v2, out_dir="temp")
    for col in ("mt_self", "mt_other", "mt_multi_self", "mt_multi_other"):
        assert_equal(cat1[col], cat1_v2[col])
        assert_equal(cat2[col], cat2_v2[col])
    os.system("rm -rf temp")
    # Other config of prep for matching
    # No redshift use
    mt.prep_cat_for_match(cat1, delta_z=None, match_radius="1 mpc", cosmo=cosmo)
    assert all(cat1.mt_input["zmin"] < cat1["z"].min())
    assert all(cat1.mt_input["zmax"] > cat1["z"].max())
    # missing all zmin/zmax info in catalog
    assert_raises(
        ValueError, mt.prep_cat_for_match, cat1, delta_z="cat", match_radius="1 mpc", cosmo=cosmo
    )
    # zmin/zmax in catalog
    cat1["zmin"] = cat1["z"] - 0.2
    cat1["zmax"] = cat1["z"] + 0.2
    cat1["z_err"] = 0.1
    mt.prep_cat_for_match(cat1, delta_z="cat", match_radius="1 mpc", cosmo=cosmo)
    assert_allclose(cat1.mt_input["zmin"], cat1["zmin"])
    assert_allclose(cat1.mt_input["zmax"], cat1["zmax"])
    # z_err in catalog
    del cat1["zmin"], cat1["zmax"]
    mt.prep_cat_for_match(cat1, delta_z="cat", match_radius="1 mpc", cosmo=cosmo)
    assert_allclose(cat1.mt_input["zmin"], cat1["z"] - cat1["z_err"])
    assert_allclose(cat1.mt_input["zmax"], cat1["z"] + cat1["z_err"])
    # zmin/zmax from aux file
    zv = np.linspace(0, 5, 10)
    np.savetxt("zvals.dat", [zv, zv - 0.22, zv + 0.33])
    mt.prep_cat_for_match(cat1, delta_z="zvals.dat", match_radius="1 mpc", cosmo=cosmo)
    assert_allclose(cat1.mt_input["zmin"], cat1["z"] - 0.22)
    assert_allclose(cat1.mt_input["zmax"], cat1["z"] + 0.33)
    os.system("rm -rf zvals.dat")
    # radus in catalog
    cat1["radius"] = 1
    cat1.radius_unit = "Mpc"
    mt.prep_cat_for_match(cat1, delta_z="cat", match_radius="cat", cosmo=cosmo)
    # radus in catalog - mass units
    cat1["rad"] = 1e14
    cat1.radius_unit = "M200c"
    mt.prep_cat_for_match(cat1, delta_z="cat", match_radius="cat", cosmo=cosmo)
    cat1.radius_unit = "M200"
    assert_raises(
        ValueError, mt.prep_cat_for_match, cat1, delta_z="cat", match_radius="cat", cosmo=cosmo
    )
    cat1.radius_unit = "MXXX"
    assert_raises(
        ValueError, mt.prep_cat_for_match, cat1, delta_z="cat", match_radius="cat", cosmo=cosmo
    )
    # radus in unknown unit
    assert_raises(
        ValueError,
        mt.prep_cat_for_match,
        cat1,
        delta_z="cat",
        match_radius="1 unknown",
        cosmo=cosmo,
    )
    # Other multiple match configs
    mt.prep_cat_for_match(cat1, **mt_config1)
    mt.multiple(cat1, cat2, radius_selection="self")
    mt.multiple(cat1, cat2, radius_selection="other")
    mt.multiple(cat1, cat2, radius_selection="min")


def test_proximity_cfg(CosmoClass):
    input1, input2 = get_test_data_prox()
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    print(cat1.data)
    print(cat2.data)
    # init match
    cosmo = CosmoClass()
    mt = ProximityMatch()
    # test wrong matching config
    assert_raises(ValueError, mt.match_from_config, cat1, cat2, {"type": "unknown"}, cosmo=cosmo)
    ### test 0 ###
    mt_config = {
        "which_radius": "max",
        "type": "cross",
        "preference": "angular_proximity",
        "catalog1": {"delta_z": 0.2, "match_radius": "1 mpc"},
        "catalog2": {"delta_z": 0.2, "match_radius": "1 arcsec"},
    }
    # Check multiple match
    mmt = [
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL3"],
        [],
    ]
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    ### test 0 ###
    mt.match_from_config(cat1, cat2, mt_config, cosmo=cosmo)
    # Check prep cat
    assert_allclose(cat2.mt_input["ang"], np.ones(cat2.size) / 3600.0)
    # Check match
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    ### test 1 ###
    mt_config["which_radius"] = "cat1"
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    mt.match_from_config(cat1, cat2, mt_config, cosmo=cosmo)
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    ### test 2 ###
    mt_config["which_radius"] = "cat2"
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    mt.match_from_config(cat1, cat2, mt_config, cosmo=cosmo)
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])


def get_test_data_mem():
    ncl = 5
    input1 = {
        "id": [f"CL{i}" for i in range(ncl)],
        "mass": [30 + i for i in range(ncl)],
    }
    input2 = {k: v[:-1] for k, v in input1.items()}
    # members
    mem_dat = [
        (f"MEM{imem}", f"CL{icl}")
        for imem, icl in enumerate([i for i in range(ncl) for j in range(i, ncl)])
    ]
    input1_mem = {"id_cluster": [f"CL{i}" for i in range(ncl) for j in range(i, ncl)]}
    input2_mem = {"id_cluster": [f"CL{i}" for i in range(ncl) for j in range(i, ncl)][:-1]}
    input1_mem["id"] = [f"MEM{i}" for i in range(len(input1_mem["id_cluster"]))]
    input2_mem["id"] = [f"MEM{i}" for i in range(len(input2_mem["id_cluster"]))]
    input1_mem["ra"] = np.arange(len(input1_mem["id_cluster"]))
    input2_mem["ra"] = np.arange(len(input2_mem["id_cluster"]))
    input1_mem["dec"] = np.zeros(len(input1_mem["id_cluster"]))
    input2_mem["dec"] = np.zeros(len(input2_mem["id_cluster"]))
    input2_mem["id_cluster"][0] = f"CL{ncl-2}"
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    cat1.add_members(**input1_mem)
    cat2.add_members(**input2_mem)
    return cat1, cat2


def test_membership():
    cat1, cat2 = get_test_data_mem()
    print(cat1.data)
    print(cat2.data)
    # init match
    mt = MembershipMatch()
    # test missing data
    assert_raises(AttributeError, mt.match_members, cat1.members, None)
    assert_raises(AttributeError, mt.match_members, None, cat2.members)
    assert_raises(AttributeError, mt.save_matched_members, "xxx")
    assert_raises(AttributeError, mt.save_shared_members, cat1, cat2, "xxx")
    assert_raises(AttributeError, mt.fill_shared_members, cat1, cat2)
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = "xx"
    assert_raises(AttributeError, mt.save_shared_members, cat1, cat2, "xxx")
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = None
    # Check both methods
    mt.match_members(cat1.members, cat2.members, method="id")
    mt2 = MembershipMatch()
    mt2.match_members(cat1.members, cat2.members, method="angular_distance", radius="1arcsec")
    assert_equal(mt.matched_mems, mt2.matched_mems)
    # Save and load matched members
    mt.save_matched_members("temp_mem.txt")
    mt2.load_matched_members("temp_mem.txt")
    assert_equal(mt.matched_mems, mt2.matched_mems)
    os.system("rm temp_mem.txt")
    # Fill shared members
    mt.fill_shared_members(cat1, cat2)
    # Save and load shared members
    mt.save_shared_members(cat1, cat2, "temp")
    cat1_test, cat2_test = ClCatalog("test1"), ClCatalog("test2")
    mt.load_shared_members(cat1_test, cat2_test, "temp")
    for c in ("nmem", "share_mems"):
        assert_equal(cat1.mt_input[c], cat1_test.mt_input[c])
        assert_equal(cat2.mt_input[c], cat2_test.mt_input[c])
    os.system("rm temp.1.p temp.2.p")
    # Check multiple match
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "shared_member_fraction")
    mt.unique(cat2, cat1, "shared_member_fraction")
    cat1.cross_match()
    cat2.cross_match()
    print(cat1)
    print(cat2)
    print(cat1["mt_multi_self", "mt_multi_other"])
    print(cat2["mt_multi_self", "mt_multi_other"])
    mmt1 = [["CL0", "CL3"], ["CL1"], ["CL2"], ["CL3"], []]
    mmt2 = [
        ["CL0"],
        ["CL1"],
        ["CL2"],
        ["CL0", "CL3"],
    ]
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt2, self=smt[:-1], cross=smt[:-1])
    # Check with minimum_share_fraction
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "shared_member_fraction", minimum_share_fraction=0.7)
    mt.unique(cat2, cat1, "shared_member_fraction", minimum_share_fraction=0.7)
    cat1.cross_match()
    cat2.cross_match()
    print("########")
    print(cat1)
    print(cat2)
    print(cat1["mt_multi_self", "mt_multi_other"])
    print(cat2["mt_multi_self", "mt_multi_other"])
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    cmt = ["CL0", "CL1", "CL2", None, None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=cmt, other=cmt)
    _test_mt_results(cat2, multi_self=mmt2, self=cmt[:-1], cross=cmt[:-1], other=smt[:-1])
    # Check with minimum_share_fraction
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "shared_member_fraction", minimum_share_fraction=0.9)
    mt.unique(cat2, cat1, "shared_member_fraction", minimum_share_fraction=0.9)
    cat1.cross_match()
    cat2.cross_match()
    print("########")
    print(cat1)
    print(cat2)
    print(cat1["mt_multi_self", "mt_multi_other"])
    print(cat2["mt_multi_self", "mt_multi_other"])
    smt = [None, "CL1", "CL2", "CL3", None]
    omt = ["CL0", "CL1", "CL2", None, None]
    cmt = [None, "CL1", "CL2", None, None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=cmt, other=omt)
    _test_mt_results(cat2, multi_self=mmt2, self=omt[:-1], cross=cmt[:-1], other=smt[:-1])
    # Test in_mt_sample col
    mt1, mt2 = get_matched_pairs(cat1, cat2, "cross")
    assert_equal(mt1.members["in_mt_sample"].all(), True)
    assert_equal(mt2.members["in_mt_sample"].all(), True)
    # Check save and load matching
    mt.save_matches(cat1, cat2, out_dir="temp", overwrite=True)
    cat1_test, cat2_test = get_test_data_mem()[:2]
    mt.load_matches(cat1_test, cat2_test, out_dir="temp")
    cat1_test._set_mt_hist([{}])
    mt.load_matches(cat1_test, cat2_test, out_dir="temp")
    for col in cat1.data.colnames:
        if col[:3] == "mt_":
            assert_equal(cat1[col], cat1_test[col])
            assert_equal(cat2[col], cat2_test[col])
    os.system("rm -rf temp")

    # Test with replacement
    cat1, cat2 = get_test_data_mem()
    mt.match_members(cat1.members, cat2.members, method="id")
    mt.fill_shared_members(cat1, cat2)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    cat1["mt_other"][0] = "CL3"
    mt.unique(cat2, cat1, "shared_member_fraction")
    mt.unique(cat1, cat2, "shared_member_fraction")
    cat1.cross_match()
    cat2.cross_match()
    smt = ["CL0", "CL1", "CL2", None, None]
    omt = ["CL0", "CL1", "CL2", "CL3", None]
    _test_mt_results(cat2, multi_self=mmt2, self=smt[:-1], cross=smt[:-1], other=omt[:-1])

    # Test with no match
    cat1["mt_self"] = None
    cat1["mt_other"] = None
    cat2["mt_self"] = None
    cat2["mt_other"] = None
    cat2["mass"][-1] = 1.0
    del cat2.mt_input["share_mems"][-1]["CL3"]
    mt.unique(cat2, cat1, "shared_member_fraction")

    # Test without replacement
    cat1, cat2 = get_test_data_mem()
    mt.match_members(cat1.members, cat2.members, method="id")
    mt.fill_shared_members(cat1, cat2)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    cat2["mt_self"][0] = "CL0"
    cat1["mt_other"][0] = "CL0"
    mt.unique(cat2, cat1, "shared_member_fraction")
    mt.unique(cat1, cat2, "shared_member_fraction")
    cat1.cross_match()
    cat2.cross_match()
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    _test_mt_results(cat2, multi_self=mmt2, self=smt[:-1], cross=smt[:-1], other=smt[:-1])
    print(cat1.members)
    print(cat2.members)


def test_membership_cfg(CosmoClass):
    cat1, cat2 = get_test_data_mem()
    print(cat1.data)
    print(cat2.data)
    # init match
    cosmo = CosmoClass()
    mt = MembershipMatch()
    # test wrong matching config
    assert_raises(ValueError, mt.match_from_config, cat1, cat2, {"type": "unknown"})
    ### test 0 ###
    match_config = {
        "type": "cross",
        "preference": "shared_member_fraction",
        "minimum_share_fraction1_unique": 0,
        "minimum_share_fraction2_unique": 0,
        "match_members_kwargs": {"method": "id"},
        "match_members_save": True,
        "shared_members_save": True,
        "match_members_file": "temp_mem.txt",
        "shared_members_file": "temp",
    }
    mt.match_from_config(cat1, cat2, match_config)
    # Test with loaded match members file
    cat1_test, cat2_test = get_test_data_mem()[:2]
    match_config_test = {}
    match_config_test.update(match_config)
    match_config_test["match_members_load"] = True
    mt.match_from_config(cat1_test, cat2_test, match_config_test)
    for col in cat1.data.colnames:
        if col[:3] == "mt_":
            assert_equal(cat1[col], cat1_test[col])
            assert_equal(cat2[col], cat2_test[col])
    # Test with loaded shared members files
    cat1_test, cat2_test = get_test_data_mem()[:2]
    match_config_test = {}
    match_config_test.update(match_config)
    match_config_test["shared_members_load"] = True
    mt.match_from_config(cat1_test, cat2_test, match_config_test)
    for col in cat1.data.colnames:
        if col[:3] == "mt_":
            assert_equal(cat1[col], cat1_test[col])
            assert_equal(cat2[col], cat2_test[col])
    # Test with ang mem match
    mmt1 = [["CL0", "CL3"], ["CL1"], ["CL2"], ["CL3"], []]
    mmt2 = [
        ["CL0"],
        ["CL1"],
        ["CL2"],
        ["CL0", "CL3"],
    ]
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt2, self=smt[:-1], cross=smt[:-1])
    # Check with minimum_share_fraction_unique
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    match_config_test["minimum_share_fraction1_unique"] = 0.7
    match_config_test["minimum_share_fraction2_unique"] = 0.7
    mt.match_from_config(cat1, cat2, match_config_test)
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    cmt = ["CL0", "CL1", "CL2", None, None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=cmt, other=cmt)
    _test_mt_results(cat2, multi_self=mmt2, self=cmt[:-1], cross=cmt[:-1], other=smt[:-1])
    # Check with minimum_share_fraction_unique
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    match_config_test["minimum_share_fraction1_unique"] = 0.9
    match_config_test["minimum_share_fraction2_unique"] = 0.9
    mt.match_from_config(cat1, cat2, match_config_test)
    smt = [None, "CL1", "CL2", "CL3", None]
    omt = ["CL0", "CL1", "CL2", None, None]
    cmt = [None, "CL1", "CL2", None, None]
    _test_mt_results(cat1, multi_self=mmt1, self=smt, cross=cmt, other=omt)
    _test_mt_results(cat2, multi_self=mmt2, self=omt[:-1], cross=cmt[:-1], other=smt[:-1])

def get_test_data_box():
    input1 = {
        "id": [f"CL{i}" for i in range(5)],
        "ra": np.array([0.0, 0.0001, 0.00011, 25, 20]),
        "dec": np.array([0.0, 0, 0, 0, 0]),
        "z": [0.2, 0.3, 0.25, 0.4, 0.35],
        "mass": [10**13.5, 10**13.4, 10**13.3, 10**13.8, 10**14],
    }
    input1["ra_min"] = input1["ra"]-1/60.
    input1["ra_max"] = input1["ra"]+1/60.
    input1["dec_min"] = input1["dec"]-1/60.
    input1["dec_max"] = input1["dec"]+1/60.
    input2 = {k: v[:4] for k, v in input1.items()}
    input2["z"][:2] = [0.3, 0.2]
    input2["mass"][:3] = input2["mass"][:3][::-1]
    return input1, input2

def _validate_unique_matching(
    mt, cat1, cat2, match_preference,
    mmt1, smt1, omt1,
    mmt2, smt2, omt2,
):
    for col in ("mt_self", "mt_other"):
        cat1[col] = None
        cat2[col] = None
    mt.unique(cat1, cat2, match_preference)
    mt.unique(cat2, cat1, match_preference)
    cat1.show_mt_hist()
    cat2.show_mt_hist()
    cat1.cross_match()
    cat2.cross_match()
    _test_mt_results(cat1, multi_self=mmt1, self=smt1, cross=smt1, other=omt1)
    _test_mt_results(cat2, multi_self=mmt2, self=omt2, cross=smt2, other=omt2)


def test_box(CosmoClass):
    input1, input2 = get_test_data_box()
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    print(cat1.data)
    print(cat2.data)
    cosmo = CosmoClass()
    mt = BoxMatch()
    # test mask intersection
    res = np.zeros(2, dtype=bool)
    for i, add in enumerate([1, -1, 1, -1]):
        args = np.zeros((8, 2))
        args[i] += add
        print(args)
        assert_equal(mt.mask_intersection(*args), res)
    # test missing data
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = "xx"
    assert_raises(AttributeError, mt.multiple, cat1, cat2)
    cat1.mt_input = None
    # init match
    mt_config1 = {"delta_z": 0.2}
    mt_config2 = {"delta_z": 0.2}
    mt.prep_cat_for_match(cat1, **mt_config1)
    mt.prep_cat_for_match(cat2, **mt_config2)
    # Check multiple match
    assert_raises(ValueError, mt.multiple, cat1, cat2, metric="unknown")
    mmt = [
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL3"],
        [],
    ]
    for metric in ("GIoU", "IoAmin", "IoAmax", "IoAself", "IoAother"):
        mt.multiple(cat1, cat2, metric=metric)
        mt.multiple(cat2, cat1, metric=metric)
    # Check unique with different preferences
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    for pref in ("angular_proximity", "GIoU", "IoAmin", "IoAmax", "IoAself", "IoAother"):
        print(pref)
        _validate_unique_matching(
            mt, cat1, cat2, pref,
            mmt, smt, smt,
            mmt[:-1], smt[:-1], smt[:-1],
        )
    # Check unique with mass preference
    smt = ["CL2", "CL1", "CL0", "CL3", None]
    _validate_unique_matching(
        mt, cat1, cat2, "more_massive",
        mmt, smt, smt,
        mmt[:-1], smt[:-1], smt[:-1],
    )
    # Check unique with z preference
    cat2["mt_other"][0] = "CL3"  # to force a replacement
    smt = ["CL1", "CL0", "CL2", "CL3", None]
    omt = ["CL1", "CL0", "CL2", "CL3", None]
    _validate_unique_matching(
        mt, cat1, cat2, "redshift_proximity",
        mmt, smt, omt,
        mmt[:-1], omt[:-1], smt[:-1],
    )
    # Error for unkown preference
    assert_raises(ValueError, mt.unique, cat1, cat2, "unknown")
    # Check save and load matching
    mt.save_matches(cat1, cat2, out_dir="temp", overwrite=True)
    cat1_v2 = ClCatalog("Cat1", **input1)
    cat2_v2 = ClCatalog("Cat2", **input2)
    mt.load_matches(cat1_v2, cat2_v2, out_dir="temp")
    for col in ("mt_self", "mt_other", "mt_multi_self", "mt_multi_other"):
        assert_equal(cat1[col], cat1_v2[col])
        assert_equal(cat2[col], cat2_v2[col])
    os.system("rm -rf temp")
    # Other config of prep for matching
    # No redshift use
    mt.prep_cat_for_match(cat1, delta_z=None)
    assert all(cat1.mt_input["zmin"] < cat1["z"].min())
    assert all(cat1.mt_input["zmax"] > cat1["z"].max())
    # missing all zmin/zmax info in catalog
    # zmin/zmax in catalog
    cat1["zmin"] = cat1["z"] - 0.2
    cat1["zmax"] = cat1["z"] + 0.2
    cat1["z_err"] = 0.1
    mt.prep_cat_for_match(cat1, delta_z="cat")
    assert_allclose(cat1.mt_input["zmin"], cat1["zmin"])
    assert_allclose(cat1.mt_input["zmax"], cat1["zmax"])
    # z_err in catalog
    del cat1["zmin"], cat1["zmax"]
    mt.prep_cat_for_match(cat1, delta_z="cat")
    assert_allclose(cat1.mt_input["zmin"], cat1["z"] - cat1["z_err"])
    assert_allclose(cat1.mt_input["zmax"], cat1["z"] + cat1["z_err"])
    # zmin/zmax from aux file
    zv = np.linspace(0, 5, 10)
    np.savetxt("zvals.dat", [zv, zv - 0.22, zv + 0.33])
    mt.prep_cat_for_match(cat1, delta_z="zvals.dat")
    assert_allclose(cat1.mt_input["zmin"], cat1["z"] - 0.22)
    assert_allclose(cat1.mt_input["zmax"], cat1["z"] + 0.33)
    os.system("rm -rf zvals.dat")


def test_box_cfg(CosmoClass):
    input1, input2 = get_test_data_box()
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    print(cat1.data)
    print(cat2.data)
    # init match
    mt = BoxMatch()
    # test wrong matching config
    assert_raises(ValueError, mt.match_from_config, cat1, cat2, {"type": "unknown"})
    ### test 0 ###
    mt_config = {
        "type": "cross",
        "preference": "GIoU",
        "catalog1": {"delta_z": 0.2,},
        "catalog2": {"delta_z": 0.2,},
    }
    # Check multiple match
    mmt = [
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL0", "CL1", "CL2"],
        ["CL3"],
        [],
    ]
    smt = ["CL0", "CL1", "CL2", "CL3", None]
    ### test 0 ###
    mt.match_from_config(cat1, cat2, mt_config)
    # Check match
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])
    ### test 1 ###
    cat1._init_match_vals(overwrite=True)
    cat2._init_match_vals(overwrite=True)
    mt.match_from_config(cat1, cat2, mt_config)
    _test_mt_results(cat1, multi_self=mmt, self=smt, cross=smt)
    _test_mt_results(cat2, multi_self=mmt[:-1], self=smt[:-1], cross=smt[:-1])

def test_output_catalog_with_matching():
    # input data
    cat1, cat2 = get_test_data_mem()
    mt = MembershipMatch()
    match_config = {
        "type": "cross",
        "preference": "shared_member_fraction",
        "minimum_share_fraction1_unique": 0,
        "minimum_share_fraction2_unique": 0,
        "match_members_kwargs": {"method": "id"},
        "match_members_save": False,
        "shared_members_save": False,
    }
    mt.match_from_config(cat1, cat2, match_config)
    # test
    file_in, file_out = "temp_init_cat.fits", "temp_out_cat.fits"
    cat1.data["id", "mass"].write(file_in)
    # diff size files
    assert_raises(ValueError, output_catalog_with_matching, file_in, file_out, cat1[:-1])
    # normal functioning
    output_catalog_with_matching(file_in, file_out, cat1)
    os.system(f"rm -f {file_in} {file_out}")


def test_output_matched_catalog():
    # input data
    cat1, cat2 = get_test_data_mem()
    mt = MembershipMatch()
    match_config = {
        "type": "cross",
        "preference": "shared_member_fraction",
        "minimum_share_fraction1_unique": 0,
        "minimum_share_fraction2_unique": 0,
        "match_members_kwargs": {"method": "id"},
        "match_members_save": False,
        "shared_members_save": False,
    }
    mt.match_from_config(cat1, cat2, match_config)
    # test
    file_in1, file_in2 = "temp_init1_cat.fits", "temp_init2_cat.fits"
    file_out = "temp_out_cat.fits"
    cat1.data["id", "mass"].write(file_in1)
    cat2.data["id", "mass"].write(file_in2)
    # diff size files
    assert_raises(
        ValueError, output_matched_catalog, file_in1, file_in2, file_out, cat1[:-1], cat2, "cross"
    )
    assert_raises(
        ValueError, output_matched_catalog, file_in1, file_in2, file_out, cat1, cat2[:-1], "cross"
    )
    # normal functioning
    output_matched_catalog(file_in1, file_in2, file_out, cat1, cat2, "cross")
    os.system(f"rm -f {file_in1} {file_in2} {file_out}")
