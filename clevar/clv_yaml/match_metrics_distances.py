"""@file match_metrics_distances.py
Matching metrics - distances functions for command line execution
"""
import numpy as np
import pylab as plt

from clevar.match_metrics import distances
from .helper_funcs import make_bins, dict_with_none, make_cosmology
from .match_metrics_parent import MetricYamlFuncs


class DistancesYamlFuncs(MetricYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", "mt_metrics_distances"],
            add_new_configs=["mt_metrics_distances"],
            pref_save="dist_cent",
        )

    def _set_individual_conf(self, general_conf):
        # prep general_confurations
        self.conf.update(general_conf["mt_metrics_distances"])
        # Format values
        self.conf["figsize"] = np.array(self.conf["figsize"].split(" "), dtype=float) / 2.54
        self.conf["dpi"] = int(self.conf["dpi"])
        for cat in ("catalog1", "catalog2"):
            self.conf[cat]["redshift_bins"] = make_bins(self.conf[cat]["redshift_bins"])
            self.conf[cat]["mass_bins"] = make_bins(
                self.conf[cat]["mass_bins"], self.conf[cat]["log_mass"]
            )
            self.conf[cat] = dict_with_none(self.conf[cat])
        ### Plots
        # Central distances
        self.kwargs = {
            "matching_type": self.conf["matching_type"],
            "shape": self.conf["line_type"],
            "radial_bins": self.conf["radial_bins"],
            "radial_bin_units": self.conf["radial_bin_units"],
            "cosmo": make_cosmology(general_conf["cosmology"]),
        }
        self.pref_save_cen = f'{general_conf["outpath"]}/dist_cent_{self.conf["radial_bin_units"]}'
        self.pref_save_z = f'{general_conf["outpath"]}/dist_z'

    def _main(self):
        # Central distances
        self._central_distace_plot_no_bins()
        for ind_i, ind_j in ("12", "21"):
            self._central_distace_plot_catalog_mass_bins(ind_i, ind_j)
            self._central_distace_plot_catalog_redshift_bins(ind_i, ind_j)

        # Redshift distances
        self.kwargs.pop("radial_bins")
        self.kwargs.pop("radial_bin_units")
        self.kwargs.pop("cosmo")
        self.kwargs["redshift_bins"] = self.conf["delta_redshift_bins"]

        self._redshift_distance_plot_no_bins()
        for ind_i, ind_j in ("12", "21"):
            self._redshift_distance_plot_catalog_mass_bins(ind_i, ind_j)
            self._redshift_distance_plot_catalog_redshift_bins(ind_i, ind_j)

    def _central_distace_plot_no_bins(self):
        print("\n# Central distace plot (no bins)")
        plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.central_position(self.cats["1"], self.cats["2"], **self.kwargs, ax=ax)
        plt.savefig(f"{self.pref_save_cen}.png", dpi=self.conf["dpi"])

    def _central_distace_plot_catalog_mass_bins(self, ind_i, ind_j):
        print(f"\n# Central distace plot (catalog {ind_i} mass bins)")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.central_position(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            ax=ax,
            quantity_bins="mass",
            bins=self.conf[f"catalog{ind_i}"]["mass_bins"],
            log_quantity=self.conf[f"catalog{ind_i}"]["log_mass"],
            add_legend=self.conf["add_mass_label"],
            legend_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
        )
        plt.savefig(f"{self.pref_save_cen}_cat{ind_i}mass.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _central_distace_plot_catalog_redshift_bins(self, ind_i, ind_j):
        print(f"\n# Central distace plot (catalog {ind_i} redshift bins)")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.central_position(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            ax=ax,
            quantity_bins="z",
            bins=self.conf[f"catalog{ind_i}"]["redshift_bins"],
            add_legend=self.conf["add_redshift_label"],
            legend_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
        )
        plt.savefig(f"{self.pref_save_cen}_cat{ind_i}redshift.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _redshift_distance_plot_no_bins(self):
        print("\n# Redshift distance plot (no bins)")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.redshift(self.cats["1"], self.cats["2"], ax=ax, **self.kwargs)
        plt.savefig(f"{self.pref_save_z}.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _redshift_distance_plot_catalog_mass_bins(self, ind_i, ind_j):
        print(f"\n# Redshift distance plot (catalog {ind_i} mass bins)")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.redshift(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            ax=ax,
            quantity_bins="mass",
            bins=self.conf[f"catalog{ind_i}"]["mass_bins"],
            log_quantity=self.conf[f"catalog{ind_i}"]["log_mass"],
            add_legend=self.conf["add_mass_label"],
            legend_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
        )
        plt.savefig(f"{self.pref_save_z}_cat{ind_i}mass.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _redshift_distance_plot_catalog_redshift_bins(self, ind_i, ind_j):
        print(f"\n# Redshift distance plot (catalog {ind_i} redshift bins)")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        distances.redshift(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            ax=ax,
            quantity_bins="z",
            bins=self.conf[f"catalog{ind_i}"]["redshift_bins"],
            add_legend=self.conf["add_redshift_label"],
            legend_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
        )
        plt.savefig(f"{self.pref_save_z}_cat{ind_i}redshift.png", dpi=self.conf["dpi"])
        plt.close(fig)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    DistancesYamlFuncs(config_file)()
