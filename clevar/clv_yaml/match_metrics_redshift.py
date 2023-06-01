"""@file match_metrics_redshift.py
Matching metrics - redshift rate functions for command line execution
"""
import numpy as np
import pylab as plt

from clevar.match_metrics import scaling
from .helper_funcs import make_bins, dict_with_none
from .match_metrics_parent import MetricYamlFuncs


class RedshiftYamlFuncs(MetricYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", "mt_metrics_redshift"],
            add_new_configs=["mt_metrics_redshift"],
            pref_save="redshift",
        )

    def _set_individual_conf(self, general_conf):
        # prep configurations
        self.conf.update(general_conf["mt_metrics_redshift"])
        # Format values
        self.conf["figsize"] = np.array(self.conf["figsize"].split(" "), dtype=float) / 2.54
        self.conf["dpi"] = int(self.conf["dpi"])
        for cat in ("catalog1", "catalog2"):
            self.conf[cat]["redshift_bins"] = make_bins(self.conf[cat]["redshift_bins"])
            self.conf[cat]["mass_bins"] = make_bins(
                self.conf[cat]["mass_bins"], self.conf["log_mass"]
            )
            self.conf[cat]["fit_redshift_bins"] = make_bins(self.conf[cat]["fit_redshift_bins"])
            self.conf[cat]["fit_redshift_bins_dist"] = make_bins(
                self.conf[cat]["fit_redshift_bins_dist"]
            )
            self.conf[cat] = dict_with_none(self.conf[cat])
        ### Plots
        self.kwargs = {k: self.conf[k] for k in ("matching_type", "add_err", "add_cb")}
        self.fit_kwargs = {
            k: self.conf[k] for k in ("add_bindata", "add_fit", "add_fit_err", "fit_statistics")
        }
        self.fit_kwargs_cat = {
            i: {
                "fit_bins1": self.conf[f"catalog{i}"]["fit_redshift_bins"],
                "fit_bins2": self.conf[f"catalog{i}"]["fit_redshift_bins_dist"],
            }
            for i in "12"
        }

    def _redshift_density_colors(self):
        print("\n# Redshift density colors")
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        scaling.redshift_density(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            bins1=self.conf["catalog1"]["redshift_bins"],
            bins2=self.conf["catalog2"]["redshift_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_metrics(self):
        print("\n# Redshift metrics")
        conf = scaling.redshift_metrics(
            self.cats["1"],
            self.cats["2"],
            bins1=self.conf["catalog1"]["redshift_bins"],
            bins2=self.conf["catalog2"]["redshift_bins"],
            matching_type=self.conf["matching_type"],
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        plt.savefig(f"{self.pref_save}_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_metrics(self):
        print("\n# Redshift density metrics")
        conf = scaling.redshift_density_metrics(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            bins1=self.conf["catalog1"]["redshift_bins"],
            bins2=self.conf["catalog2"]["redshift_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_z_colors(self, ind_i):
        print(f"\n# Redshift (catalog {ind_i} z colors)")
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        scaling.redshift_masscolor(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            color1=ind_i == "1",
            log_mass=self.conf["log_mass"],
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_cat{ind_i}zcolor.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_panel(self, ind_i):
        print(f"\n# Redshift density (catalog {ind_i} mass panel)")
        conf = scaling.redshift_density_masspanel(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            panel_cat1=ind_i == "1",
            bins1=self.conf["catalog1"]["redshift_bins"],
            bins2=self.conf["catalog2"]["redshift_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            mass_bins=self.conf[f"catalog{ind_i}"]["mass_bins"],
            label_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density_cat{ind_i}masspanel.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_self_dist(self, ind_i):
        print(f"\n# Redshift density (catalog {ind_i} m self dist)")
        conf = scaling.redshift_dist_self(
            self.cats[ind_i],
            **{
                k: self.conf[f"catalog{ind_i}"][k]
                for k in ("redshift_bins", "mass_bins", "redshift_bins_dist")
            },
            log_mass=self.conf["log_mass"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_self_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_dist(self, ind_i, ind_j):
        print(f"\n# Redshift density (catalog {ind_i} m dist)")
        conf = scaling.redshift_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **{k: self.conf[k] for k in ("matching_type", "log_mass")},
            **{k: self.conf[f"catalog{ind_j}"][k] for k in ("redshift_bins", "mass_bins")},
            redshift_bins_dist=self.conf[f"catalog{ind_i}"]["redshift_bins_dist"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"]["mass_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_z_panel(self, ind_i, ind_j):
        print(f"\n# Redshift density (catalog {ind_i} z panel)")
        conf = scaling.redshift_density_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat[ind_i],
            bins1=self.conf[f"catalog{ind_i}"]["redshift_bins"],
            bins2=self.conf[f"catalog{ind_j}"]["redshift_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        plt.savefig(f"{self.pref_save}_density_cat{ind_i}_dist.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _main(self):
        # Density Plot
        if any(case in self.conf["plot_case"] for case in ("density", "all")):
            self._redshift_density_colors()
        if any(case in self.conf["plot_case"] for case in ("scaling_metrics", "all")):
            self._redshift_metrics()
        if any(case in self.conf["plot_case"] for case in ("density_metrics", "all")):
            self._redshift_density_metrics()
        for ind_i, ind_j in ("12", "21"):
            # z Color Plot
            if any(case in self.conf["plot_case"] for case in ("masscolor", "all")):
                self._redshift_z_colors(ind_i)
            # Panel density Plot
            if any(case in self.conf["plot_case"] for case in ("density_panel", "all")):
                self._redshift_density_m_panel(ind_i)
            # distribution
            if any(case in self.conf["plot_case"] for case in ("self_distribution", "all")):
                self._redshift_density_m_self_dist(ind_i)
            if any(case in self.conf["plot_case"] for case in ("distribution", "all")):
                self._redshift_density_m_dist(ind_i, ind_j)
            # Panel density distribution
            if any(case in self.conf["plot_case"] for case in ("density_dist", "all")):
                self._redshift_density_z_panel(ind_i, ind_j)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    RedshiftYamlFuncs(config_file)()
