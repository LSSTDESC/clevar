"""Tests for clevar/match_metrics/recovery"""
import numpy as np
from clevar.catalog import ClCatalog
from clevar.cosmology import AstroPyCosmology as CosmoClass
from clevar.utils import gaussian
from clevar.match import ProximityMatch
from clevar.match_metrics import recovery as rc
from numpy.testing import assert_raises


##############################
#### Input data ##############
##############################
class _test_data:
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
    cat1 = ClCatalog("Cat1", **input1)
    cat2 = ClCatalog("Cat2", **input2)
    cosmo = CosmoClass()
    mt = ProximityMatch()
    mt_config1 = {"delta_z": 0.2, "match_radius": "1 mpc", "cosmo": cosmo}
    mt_config2 = {"delta_z": 0.2, "match_radius": "1 arcsec"}
    mt.prep_cat_for_match(cat1, **mt_config1)
    mt.prep_cat_for_match(cat2, **mt_config2)
    mt.multiple(cat1, cat2)
    mt.multiple(cat2, cat1)
    mt.unique(cat1, cat2, "angular_proximity")
    mt.unique(cat2, cat1, "angular_proximity")


##############################
### Test #####################
##############################
def test_plot():
    cat = _test_data.cat1
    matching_type = "cat1"
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e16]
    rc.plot(cat, matching_type, redshift_bins, mass_bins)
    rc.plot(cat, matching_type, redshift_bins, mass_bins, shape="line")
    rc.plot(cat, matching_type, redshift_bins, mass_bins, add_legend=True)
    assert_raises(
        ValueError, rc.plot, cat, matching_type, redshift_bins, mass_bins, shape="unknown"
    )
    rc.plot(
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        add_legend=True,
        p_m1_m2=lambda m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1),
    )
    rc.plot(
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        add_legend=True,
        transpose=True,
        p_m1_m2=lambda m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1),
    )
    rc.plot(cat, matching_type, redshift_bins, mass_bins, plt_kwargs={"n_increase": 3})
    rc.plot(
        cat, matching_type, redshift_bins, mass_bins, shape="line", plt_kwargs={"n_increase": 3}
    )
    rc.plot(
        cat, matching_type, redshift_bins, mass_bins, plt_kwargs={"n_increase": 3, "scheme": [1, 1]}
    )


def test_plot_panel():
    cat = _test_data.cat1
    matching_type = "cat1"
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e14, 1e15, 1e16]
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins)
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins, add_label=True, label_fmt=".2f")
    rc.plot_panel(cat, matching_type, redshift_bins, mass_bins, transpose=True)
    rc.plot_panel(
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        p_m1_m2=lambda m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1),
    )
    rc.plot_panel(
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        transpose=True,
        p_m1_m2=lambda m1, m2: gaussian(np.log10(m2), np.log10(m1), 0.1),
    )


def test_plot2D():
    cat = _test_data.cat1
    matching_type = "cat1"
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e14, 1e15, 1e16]
    rc.plot2D(
        cat,
        matching_type,
        redshift_bins,
        mass_bins,
        transpose=False,
        log_mass=True,
        redshift_label=None,
        mass_label=None,
    )
    rc.plot2D(cat, matching_type, redshift_bins, mass_bins, add_num=True)


def test_skyplot():
    cat = _test_data.cat1
    matching_type = "cat1"
    rc.skyplot(cat, matching_type)
    # for monocromatic map
    rc.skyplot(
        cat[cat.get_matching_mask("self")],
        matching_type,
        recovery_label=None,
    )


def test_fscore():
    matching_type = "cat1"
    redshift_bins = [0, 0.5, 1]
    mass_bins = [1e13, 1e14, 1e15, 1e16]
    args = (
        _test_data.cat1,
        redshift_bins,
        mass_bins,
        _test_data.cat2,
        redshift_bins,
        mass_bins,
        matching_type,
    )
    rc.plot_fscore(
        *args,
        beta=1,
        pref="cat1",
        par_order=(0, 1, 2, 3),
        cat1_mask=None,
        cat1_mask_unmatched=None,
        cat2_mask=None,
        cat2_mask_unmatched=None,
        log_mass=True,
        fscore_label="fscore",
        xlabel="x",
    )
    rc.plot_fscore(
        *args,
        beta=1,
        pref="cat2",
        par_order=(0, 1, 2, 3),
        cat1_mask=None,
        cat1_mask_unmatched=None,
        cat2_mask=None,
        cat2_mask_unmatched=None,
        log_mass=True,
        fscore_label=None,
    )
    assert_raises(ValueError, rc.plot_fscore, *args, pref="unknown")
