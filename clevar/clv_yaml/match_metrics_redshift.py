"""@file match_metrics_redshift.py
Matching metrics - redshift rate functions for command line execution
"""
import numpy as np
import pylab as plt

from clevar.match_metrics import scaling
from .helper_funcs import loadconf, make_catalog, make_bins


class RunFuncs:
    """Main plot class"""
    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        config = loadconf(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", "mt_metrics_redshift"],
            add_new_configs=["mt_metrics_redshift"],
            check_matching=True,
        )
        self.skip = config is None
        if self.skip:
            return
        self.z_name = f'{config["outpath"]}/redshift'

        self.cats = {}
        print("\n# Reading Catalog 1")
        self.cats["1"] = make_catalog(config["catalog1"])
        self.cats["1"].load_match(f"{config['outpath']}/match1.fits")
        print("\n# Reading Catalog 2")
        self.cats["2"] = make_catalog(config["catalog2"])
        self.cats["2"].load_match(f"{config['outpath']}/match2.fits")

        # prep configurations
        self.z_conf = {}
        self.z_conf.update(config["mt_metrics_redshift"])
        # Format values
        self.z_conf["figsize"] = np.array(self.z_conf["figsize"].split(" "), dtype=float) / 2.54
        self.z_conf["dpi"] = int(self.z_conf["dpi"])
        for cat in ("catalog1", "catalog2"):
            self.z_conf[cat]["redshift_bins"] = make_bins(self.z_conf[cat]["redshift_bins"])
            self.z_conf[cat]["mass_bins"] = make_bins(
                self.z_conf[cat]["mass_bins"], self.z_conf["log_mass"]
            )
            self.z_conf[cat]["fit_redshift_bins"] = make_bins(self.z_conf[cat]["fit_redshift_bins"])
            self.z_conf[cat]["fit_redshift_bins_dist"] = make_bins(
                self.z_conf[cat]["fit_redshift_bins_dist"]
            )
            self.z_conf[cat] = {
                k: (None if str(v) == "None" else v) for k, v in self.z_conf[cat].items()
            }
        ### Plots
        self.kwargs = {k: self.z_conf[k] for k in ("matching_type", "add_err", "add_cb")}
        self.fit_kwargs = {
            k: self.z_conf[k] for k in ("add_bindata", "add_fit", "add_fit_err", "fit_statistics")
        }
        self.fit_kwargs_cat = {
            i: {
                "fit_bins1": self.z_conf[f"catalog{i}"]["fit_redshift_bins"],
                "fit_bins2": self.z_conf[f"catalog{i}"]["fit_redshift_bins_dist"],
            }
            for i in "12"
        }

    def _redshift_density_colors(self):
        print("\n# Redshift density colors")
        conf = {"fig": plt.figure(figsize=self.z_conf["figsize"])}
        ax = plt.axes()
        scaling.redshift_density(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            bins1=self.z_conf["catalog1"]["redshift_bins"],
            bins2=self.z_conf["catalog2"]["redshift_bins"],
            ax_rotation=self.z_conf["ax_rotation"],
            rotation_resolution=self.z_conf["rotation_resolution"],
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.z_name}_density.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_metrics(self):
        print("\n# Redshift metrics")
        conf = scaling.redshift_metrics(
            self.cats["1"],
            self.cats["2"],
            bins1=self.z_conf["catalog1"]["redshift_bins"],
            bins2=self.z_conf["catalog2"]["redshift_bins"],
            matching_type=self.z_conf["matching_type"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
        )
        plt.savefig(f"{self.z_name}_metrics.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_metrics(self):
        print("\n# Redshift density metrics")
        conf = scaling.redshift_density_metrics(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            bins1=self.z_conf["catalog1"]["redshift_bins"],
            bins2=self.z_conf["catalog2"]["redshift_bins"],
            ax_rotation=self.z_conf["ax_rotation"],
            rotation_resolution=self.z_conf["rotation_resolution"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.z_name}_density_metrics.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_z_colors(self, ind_i):
        print(f"\n# Redshift (catalog {ind_i} z colors)")
        conf = {"fig": plt.figure(figsize=self.z_conf["figsize"])}
        ax = plt.axes()
        scaling.redshift_masscolor(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            ax=ax,
            color1=ind_i == "1",
            log_mass=self.z_conf["log_mass"],
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.z_name}_cat{ind_i}zcolor.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_panel(self, ind_i):
        print(f"\n# Redshift density (catalog {ind_i} mass panel)")
        conf = scaling.redshift_density_masspanel(
            self.cats["1"],
            self.cats["2"],
            **self.kwargs,
            panel_cat1=ind_i == "1",
            bins1=self.z_conf["catalog1"]["redshift_bins"],
            bins2=self.z_conf["catalog2"]["redshift_bins"],
            ax_rotation=self.z_conf["ax_rotation"],
            rotation_resolution=self.z_conf["rotation_resolution"],
            mass_bins=self.z_conf[f"catalog{ind_i}"]["mass_bins"],
            label_fmt=self.z_conf[f"catalog{ind_i}"]["mass_num_fmt"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.z_name}_density_cat{ind_i}masspanel.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_self_dist(self, ind_i):
        print(f"\n# Redshift density (catalog {ind_i} m self dist)")
        conf = scaling.redshift_dist_self(
            self.cats[ind_i],
            **{
                k: self.z_conf[f"catalog{ind_i}"][k]
                for k in ("redshift_bins", "mass_bins", "redshift_bins_dist")
            },
            log_mass=self.z_conf["log_mass"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
            panel_label_fmt=self.z_conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            line_label_fmt=self.z_conf[f"catalog{ind_i}"]["mass_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.z_name}_dist_self_cat{ind_i}.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_m_dist(self, ind_i, ind_j):
        print(f"\n# Redshift density (catalog {ind_i} m dist)")
        conf = scaling.redshift_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **{k: self.z_conf[k] for k in ("matching_type", "log_mass")},
            **{k: self.z_conf[f"catalog{ind_j}"][k] for k in ("redshift_bins", "mass_bins")},
            redshift_bins_dist=self.z_conf[f"catalog{ind_i}"]["redshift_bins_dist"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
            panel_label_fmt=self.z_conf[f"catalog{ind_i}"]["redshift_num_fmt"],
            line_label_fmt=self.z_conf[f"catalog{ind_i}"]["mass_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.z_name}_dist_cat{ind_i}.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def _redshift_density_z_panel(self, ind_i, ind_j):
        print(f"\n# Redshift density (catalog {ind_i} z panel)")
        conf = scaling.redshift_density_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat[ind_i],
            bins1=self.z_conf[f"catalog{ind_i}"]["redshift_bins"],
            bins2=self.z_conf[f"catalog{ind_j}"]["redshift_bins"],
            ax_rotation=self.z_conf["ax_rotation"],
            rotation_resolution=self.z_conf["rotation_resolution"],
            fig_kwargs={"figsize": self.z_conf["figsize"]},
        )
        plt.savefig(f"{self.z_name}_density_cat{ind_i}_dist.png", dpi=self.z_conf["dpi"])
        plt.close(conf["fig"])

    def __call__(self):
        if self.skip:
            return
        # Density Plot
        if any(case in self.z_conf["plot_case"] for case in ("density", "all")):
            self._redshift_density_colors()
        if any(case in self.z_conf["plot_case"] for case in ("scaling_metrics", "all")):
            self._redshift_metrics()
        if any(case in self.z_conf["plot_case"] for case in ("density_metrics", "all")):
            self._redshift_density_metrics()
        for ind_i, ind_j in ("12", "21"):
            # z Color Plot
            if any(case in self.z_conf["plot_case"] for case in ("masscolor", "all")):
                self._redshift_z_colors(ind_i)
            # Panel density Plot
            if any(case in self.z_conf["plot_case"] for case in ("density_panel", "all")):
                self._redshift_density_m_panel(ind_i)
            # distribution
            if any(case in self.z_conf["plot_case"] for case in ("self_distribution", "all")):
                self._redshift_density_m_self_dist(ind_i)
            if any(case in self.z_conf["plot_case"] for case in ("distribution", "all")):
                self._redshift_density_m_dist(ind_i, ind_j)
            # Panel density distribution
            if any(case in self.z_conf["plot_case"] for case in ("density_dist", "all")):
                self._redshift_density_z_panel(ind_i, ind_j)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    RunFuncs(config_file)()
