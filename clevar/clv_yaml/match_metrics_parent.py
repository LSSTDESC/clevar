"""@file match_metrics_parent.py
Matching metrics parent functions for command line execution
"""
import numpy as np
import pylab as plt

from .helper_funcs import loadconf, make_catalog, make_bins, dict_with_none


class MetricYamlFuncs:
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file, load_configs, add_new_configs, pref_save):
        # Create clevar objects from yml config
        general_conf = loadconf(
            config_file,
            load_configs=load_configs,
            add_new_configs=add_new_configs,
            check_matching=True,
        )
        self.skip = general_conf is None
        if self.skip:
            return
        self.pref_save = f'{general_conf["outpath"]}/{pref_save}'

        # prep cats
        self.cats = {}
        print("\n# Reading Catalog 1")
        self.cats["1"] = make_catalog(general_conf["catalog1"])
        self.cats["1"].load_match(f"{general_conf['outpath']}/match1.fits")
        print("\n# Reading Catalog 2")
        self.cats["2"] = make_catalog(general_conf["catalog2"])
        self.cats["2"].load_match(f"{general_conf['outpath']}/match2.fits")

        # prep configs
        self.conf = {}
        self._set_individual_conf(general_conf)

    def _set_individual_conf(self, general_conf):
        raise NotImplementedError("Not Implemented")

    def __call__(self):
        if self.skip:
            return
        self._main()

    def _main(self):
        raise NotImplementedError("Not Implemented")


class ScalingYamlFuncs(MetricYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config_file, load_configs, add_new_configs, self_name, other_name):
        self.self_name = self_name
        self.other_name = other_name
        self.other_name_short = {"mass": "mass", "redshift": "z"}[self_name]
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", f"mt_metrics_{self.self_name}"],
            add_new_configs=[f"mt_metrics_{self.self_name}"],
            pref_save=self.self_name,
        )

        # prep configurations

    def _set_individual_conf(self, general_conf):
        self.conf = {**general_conf[f"mt_metrics_{self.self_name}"]}
        # Format values
        self.conf["figsize"] = np.array(self.conf["figsize"].split(" "), dtype=float) / 2.54
        self.conf["dpi"] = int(self.conf["dpi"])
        # plot kwargs config
        self.sc_kwargs = {k: self.conf[k] for k in ("add_err", "add_cb")}
        if self.self_name == "mass":
            log_bins = self.conf["log_mass"]
            self.conf_kwargs = {k: self.conf[k] for k in ("matching_type", "log_mass")}
            self.add_kwargs = {}
        else:
            log_bins = False
            self.conf_kwargs = {"matching_type": self.conf["matching_type"]}
            self.add_kwargs = {"log_mass": self.conf["log_mass"]}
        # prep bins
        for cat in ("catalog1", "catalog2"):
            self.conf[cat]["redshift_bins"] = make_bins(self.conf[cat]["redshift_bins"])
            self.conf[cat]["mass_bins"] = make_bins(
                self.conf[cat]["mass_bins"], self.conf["log_mass"]
            )
            self.conf[cat][f"fit_{self.self_name}_bins"] = make_bins(
                self.conf[cat][f"fit_{self.self_name}_bins"], log_bins
            )
            self.conf[cat][f"fit_{self.self_name}_bins_dist"] = make_bins(
                self.conf[cat][f"fit_{self.self_name}_bins_dist"], log_bins
            )
            self.conf[cat] = dict_with_none(self.conf[cat])
        # fit config
        self.fit_kwargs = {
            k: self.conf[k] for k in ("add_bindata", "add_fit", "add_fit_err", "fit_statistics")
        }
        self.fit_kwargs_cat = {
            i: {
                "fit_bins1": self.conf[f"catalog{i}"][f"fit_{self.self_name}_bins"],
                "fit_bins2": self.conf[f"catalog{i}"][f"fit_{self.self_name}_bins_dist"],
            }
            for i in "12"
        }

    def _main(self):
        if any(case in self.conf["plot_case"] for case in ("density", "all")):
            self._plot_density_colors()
        if any(case in self.conf["plot_case"] for case in ("scaling_metrics", "all")):
            self._plot_metrics()
        if any(case in self.conf["plot_case"] for case in ("density_metrics", "all")):
            self._plot_density_metrics()
        for ind_i, ind_j in ("12", "21"):
            if any(
                case in self.conf["plot_case"] for case in (f"{self.other_name_short}color", "all")
            ):
                self._plot_other_colors(ind_i)
            if any(case in self.conf["plot_case"] for case in ("density_panel", "all")):
                self._plot_density_other_panel(ind_i)
            if any(case in self.conf["plot_case"] for case in ("self_distribution", "all")):
                self._plot_dist_self(ind_i)
            if any(case in self.conf["plot_case"] for case in ("distribution", "all")):
                self._plot_dist(ind_i, ind_j)
            if any(case in self.conf["plot_case"] for case in ("density_dist", "all")):
                self._plot_density_dist(ind_i, ind_j)

    def _plot_density_colors(self):
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        self._core_density(
            self.cats["1"],
            self.cats["2"],
            ax=ax,
            bins1=self.conf["catalog1"][f"{self.self_name}_bins"],
            bins2=self.conf["catalog2"][f"{self.self_name}_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            **self.conf_kwargs,
            **self.sc_kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _plot_metrics(self):
        conf = self._core_metrics(
            self.cats["1"],
            self.cats["2"],
            bins1=self.conf["catalog1"][f"{self.self_name}_bins"],
            bins2=self.conf["catalog2"][f"{self.self_name}_bins"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.conf_kwargs,
        )
        plt.savefig(f"{self.pref_save}_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _plot_density_metrics(self):
        print("\n# Mass density metrics")
        conf = self._core_density_metrics(
            self.cats["1"],
            self.cats["2"],
            bins1=self.conf["catalog1"][f"{self.self_name}_bins"],
            bins2=self.conf["catalog2"][f"{self.self_name}_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.conf_kwargs,
            **self.sc_kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(f"{self.pref_save}_density_metrics.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _plot_other_colors(self, ind_i):
        print(f"\n# Mass (catalog {ind_i} {self.other_name} colors)")
        conf = {"fig": plt.figure(figsize=self.conf["figsize"])}
        ax = plt.axes()
        self._core_other_color(
            self.cats["1"],
            self.cats["2"],
            ax=ax,
            color1=ind_i == "1",
            **self.conf_kwargs,
            **self.sc_kwargs,
            **self.add_kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
        )
        plt.savefig(
            f"{self.pref_save}_cat{ind_i}{self.other_name_short}color.png", dpi=self.conf["dpi"]
        )
        plt.close(conf["fig"])

    def _plot_density_other_panel(self, ind_i):
        print(f"\n# Mass density (catalog {ind_i} {self.other_name} panel)")
        conf = self._core_density_other_panel(
            self.cats["1"],
            self.cats["2"],
            bins1=self.conf["catalog1"][f"{self.self_name}_bins"],
            bins2=self.conf["catalog2"][f"{self.self_name}_bins"],
            panel_cat1=ind_i == "1",
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            label_fmt=self.conf[f"catalog{ind_i}"][f"{self.other_name}_num_fmt"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.conf_kwargs,
            **self.sc_kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat["1"],
            **{f"{self.other_name}_bins": self.conf[f"catalog{ind_i}"][f"{self.other_name}_bins"]},
        )
        plt.savefig(
            f"{self.pref_save}_density_cat{ind_i}{self.other_name_short}panel.png",
            dpi=self.conf["dpi"],
        )
        plt.close(conf["fig"])

    def _plot_dist_self(self, ind_i):
        print(f"\n# Mass density (catalog {ind_i} self dist)")
        conf = self._core_dist_self(
            self.cats[ind_i],
            **{
                k: self.conf[f"catalog{ind_i}"][k]
                for k in (
                    f"{self.self_name}_bins",
                    f"{self.other_name}_bins",
                    f"{self.self_name}_bins_dist",
                )
            },
            log_mass=self.conf["log_mass"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"][f"{self.self_name}_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"][f"{self.other_name}_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_self_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _plot_dist(self, ind_i, ind_j):
        print(f"\n# Mass density (catalog {ind_i}-{ind_j} dist)")
        conf = self._core_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            **self.conf_kwargs,
            **self.add_kwargs,
            **{
                k: self.conf[f"catalog{ind_j}"][k]
                for k in (f"{self.self_name}_bins", f"{self.other_name}_bins")
            },
            **{
                f"{self.self_name}_bins_dist": self.conf[f"catalog{ind_i}"][
                    f"{self.self_name}_bins_dist"
                ]
            },
            fig_kwargs={"figsize": self.conf["figsize"]},
            panel_label_fmt=self.conf[f"catalog{ind_i}"][f"{self.self_name}_num_fmt"],
            line_label_fmt=self.conf[f"catalog{ind_i}"][f"{self.other_name}_num_fmt"],
            shape="line",
        )
        plt.savefig(f"{self.pref_save}_dist_cat{ind_i}.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _plot_density_dist(self, ind_i, ind_j):
        print(f"\n# Mass density (catalog {ind_i}-{ind_j} {self.other_name} panel)")
        conf = self._core_density_dist(
            self.cats[ind_i],
            self.cats[ind_j],
            bins1=self.conf[f"catalog{ind_i}"][f"{self.self_name}_bins"],
            bins2=self.conf[f"catalog{ind_j}"][f"{self.self_name}_bins"],
            ax_rotation=self.conf["ax_rotation"],
            rotation_resolution=self.conf["rotation_resolution"],
            fig_kwargs={"figsize": self.conf["figsize"]},
            **self.conf_kwargs,
            **self.sc_kwargs,
            **self.add_kwargs,
            **self.fit_kwargs,
            **self.fit_kwargs_cat[ind_i],
        )
        plt.savefig(f"{self.pref_save}_density_cat{ind_i}_dist.png", dpi=self.conf["dpi"])
        plt.close(conf["fig"])

    def _core_density(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_metrics(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_density_metrics(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_other_color(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_density_other_panel(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_dist_self(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_dist(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")

    def _core_density_dist(self, *args, **kwargs):
        raise NotImplementedError("Not Implemented")
