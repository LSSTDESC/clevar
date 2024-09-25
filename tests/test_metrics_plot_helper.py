"""Tests for clevar/match_metrics/plot_helper"""
import pylab as plt
from numpy.testing import assert_raises
from clevar.match_metrics import plot_helper as ph


def test_add_panel_bin_label():
    fig, axes = plt.subplots(2, 2)
    edges = range(5)
    edges_lower, edges_higher = edges[:-1], edges[1:]
    for position in ("top", "bottom", "left", "right"):
        ph.add_panel_bin_label(axes, edges_lower, edges_higher, position=position)
    assert_raises(
        ValueError, ph.add_panel_bin_label, axes, edges_lower, edges_higher, position="unknown"
    )
