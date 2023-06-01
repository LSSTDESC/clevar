"""@file match_metrics_mass.py
Matching metrics - mass functions for command line execution
"""
import numpy as np
import pylab as plt

from clevar.match_metrics import scaling
from .helper_funcs import make_bins, dict_with_none
from .match_metrics_parent import MetricYamlFuncs


class MassYamlFuncs(MetricYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", "mt_metrics_mass"],
            add_new_configs=["mt_metrics_mass"],
            pref_save="mass",
        )

    def _set_individual_conf(self, general_conf):
        # prep configurations
        self.conf.update(general_conf["mt_metrics_mass"])
        # Format values
        self.conf["figsize"] = np.array(self.conf["figsize"].split(" "), dtype=float) / 2.54
        self.conf["dpi"] = int(self.conf["dpi"])
        for cat in ("catalog1", "catalog2"):
            self.conf[cat]["redshift_bins"] = make_bins(self.conf[cat]["redshift_bins"])
            self.conf[cat]["mass_bins"] = make_bins(
                self.conf[cat]["mass_bins"], self.conf["log_mass"]
            )
            self.conf[cat]["fit_mass_bins"] = make_bins(
                self.conf[cat]["fit_mass_bins"], self.conf["log_mass"]
            )
            self.conf[cat]["fit_mass_bins_dist"] = make_bins(
                self.conf[cat]["fit_mass_bins_dist"], self.conf["log_mass"]
            )
            self.conf[cat] = dict_with_none(self.conf[cat])
        ### Plots
        self.kwargs = {k: self.conf[k] for k in ("matching_type", "log_mass", "add_err", "add_cb")}
        self.fit_kwargs = {
            k: self.conf[k] for k in ("add_bindata", "add_fit", "add_fit_err", "fit_statistics")
        }
        self.fit_kwargs_cat = {
            i: {
                "fit_bins1": self.conf[f"catalog{i}"]["fit_mass_bins"],
                "fit_bins2": self.conf[f"catalog{i}"]["fit_mass_bins_dist"],
            }
            for i in "12"
        }

    def _mass_density_colors(self):
        print("\n# Mass density colors")
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        scaling.mass_density(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            bins1=self.conf["catalog1"]["mass_bins"],
            bins2=self.conf["catalog2"]["mass_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_metrics(self):
        print("\n# Mass metrics")
        conf = scaling.mass_metrics(
            self.cats["1"],
            self.cats["2"],
            bins1=self.conf["catalog1"]["mass_bins"],
            bins2=self.conf["catalog2"]["mass_bins"],
            **{k: self.conf[k] for k in ("matching_type", "log_mass")},
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        plt.savefig(f"{self.pref_save}_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_density_metrics(self):
        print("\n# Mass density metrics")
        conf = scaling.mass_density_metrics(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            bins1=self.conf["catalog1"]["mass_bins"],
            bins2=self.conf["catalog2"]["mass_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_catalog_z_colors(self, ind_i):
        print(f"\n# Mass (catalog {ind_i} z colors)")
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        scaling.mass_zcolor(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            color1=ind_i == "1",
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_cat{ind_i}zcolor.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_density_catalog_z_panel(self, ind_i):
        print(f"\n# Mass density (catalog {ind_i} z panel)")
        conf = scaling.mass_density_zpanel(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            panel_cat1=ind_i == "1",
            bins1=self.conf["catalog1"]["mass_bins"],
            bins2=self.conf["catalog2"]["mass_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            redshift_bins=self.conf[f"catalog{ind_i}"]["redshift_bins"],
            label_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density_cat{ind_i}zpanel.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_density_catalog_m_self_dist(self, ind_i):
        print(f"\n# Mass density (catalog {ind_i} m self dist)")
        conf = scaling.mass_dist_self(
            self.cats[ind_i],
            **{
                k: self.conf[f"catalog{ind_i}"][k]
                for k in ("mass_bins", "redshift_bins", "mass_bins_dist")
            },
            log_mass=self.conf["log_mass"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_self_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_density_catalog_m_dist(self, ind_i, ind_j):
        print(f"\n# Mass density (catalog {ind_i} m dist)")
        conf = scaling.mass_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **{k: self.conf[k] for k in ("matching_type", "log_mass")},
            **{k: self.conf[f"catalog{ind_j}"][k] for k in ("mass_bins", "redshift_bins")},
            mass_bins_dist=self.conf[f"catalog{ind_i}"]["mass_bins_dist"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _mass_density_catalog_m_panel(self, ind_i, ind_j):
        print(f"\n# Mass density (catalog {ind_i} z panel)")
        conf = scaling.mass_density_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat[ind_i],
            bins1=self.conf[f"catalog{ind_i}"]["mass_bins"],
            bins2=self.conf[f"catalog{ind_j}"]["mass_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        plt.savefig(f"{self.pref_save}_density_cat{ind_i}_dist.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _main(self):
        if any(case in self.conf["plot_case"] for case in ("density", "all")):
            self._mass_density_colors()
        if any(case in self.conf["plot_case"] for case in ("scaling_metrics", "all")):
            self._mass_metrics()
        if any(case in self.conf["plot_case"] for case in ("density_metrics", "all")):
            self._mass_density_metrics()
        for ind_i, ind_j in ("12", "21"):
            if any(case in self.conf["plot_case"] for case in ("zcolor", "all")):
                self._mass_catalog_z_colors(ind_i)
            if any(case in self.conf["plot_case"] for case in ("density_panel", "all")):
                self._mass_density_catalog_z_panel(ind_i)
            if any(case in self.conf["plot_case"] for case in ("self_distribution", "all")):
                self._mass_density_catalog_m_self_dist(ind_i)
            if any(case in self.conf["plot_case"] for case in ("distribution", "all")):
                self._mass_density_catalog_m_dist(ind_i, ind_j)
            if any(case in self.conf["plot_case"] for case in ("density_dist", "all")):
                self._mass_density_catalog_m_panel(ind_i, ind_j)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    MassYamlFuncs(config_file)()
