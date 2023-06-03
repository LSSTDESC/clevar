"""@file match_metrics_recovery_rate.py
Matching metrics - recovery rate functions for command line execution
"""
import os
import numpy as np
import pylab as plt

from clevar.match_metrics import recovery
from .match_metrics_parent import MetricYamlFuncs


class RecoveryYamlFuncs(MetricYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "masks", "mt_metrics_recovery"],
            add_new_configs=["mt_metrics_recovery"],
            pref_save="rec_mt",
        )
        self.rec_name = ""
        self.rec_suf = ""

    def _set_individual_conf(self, general_conf):
        print("\n# Add Catalog 1 footprint quantities")
        ftpt_qt_file1 = f"{general_conf['outpath']}/ftpt_quantities1.fits"
        _ = (
            self.cats["1"].load_footprint_quantities(ftpt_qt_file1)
            if os.path.isfile(ftpt_qt_file1)
            else None
        )
        print("\n# Add Catalog 2 footprint quantities")
        ftpt_qt_file2 = f"{general_conf['outpath']}/ftpt_quantities2.fits"
        _ = (
            self.cats["2"].load_footprint_quantities(ftpt_qt_file2)
            if os.path.isfile(ftpt_qt_file2)
            else None
        )
        # prep configurations
        self.conf.update(general_conf["mt_metrics_recovery"])
        self._set_basic_conf()
        ### Plots
        self.kwargs = {
            "matching_type": self.conf["matching_type"],
        }

    def _main(self):
        ### Plots
        for ind in "12":
            self._update_kwargs(ind)
            # Simple plot
            if any(case in self.conf["plot_case"] for case in ("simple", "all")):
                self._simple_recovery_catalog_by_redshift(ind)
                self._simple_recovery_catalog_by_mass(ind)
            # Panels plots
            if any(case in self.conf["plot_case"] for case in ("panel", "all")):
                self._panel_recovery_catalog_by_redshift(ind)
                self._panel_recovery_catalog_by_mass(ind)
            # 2D plots
            if any(case in self.conf["plot_case"] for case in ("2D", "all")):
                self.kwargs.pop("shape", None)
                self.kwargs.pop("recovery_label", None)
                self.kwargs["add_cb"] = self.conf["add_cb"]
                self._2d_recovery_catalog(ind)
                self._2d_recovery_catalog_with_numbers(ind)
                self.kwargs.pop("add_cb", None)

    def _update_kwargs(self, ind):
        self.kwargs.update(
            {
                "redshift_bins": self.conf[f"catalog{ind}"]["redshift_bins"],
                "mass_bins": self.conf[f"catalog{ind}"]["mass_bins"],
                "log_mass": self.conf[f"catalog{ind}"]["log_mass"],
                "recovery_label": self.conf[f"catalog{ind}"]["recovery_label"],
                "shape": self.conf["line_type"],
            }
        )
        mask = np.zeros(self.cats[ind].size, dtype=bool)
        mask_case = self.conf[f"catalog{ind}"]["masks"]["case"].lower()
        if mask_case is not None:
            for mtype, mconf in self.conf[f"catalog{ind}"]["masks"].items():
                if mtype[:12] == "in_footprint" and mconf.get("use", False):
                    print(f"    # Adding footprint mask: {mconf}")
                    mask += ~self.cats[ind][f"ft_{mconf['name']}"]
                    print(f"      * {mask[mask].size:,} clusters masked in total")
                if mtype[:13] == "coverfraction":
                    print(f"    # Adding coverfrac: {mconf}")
                    mask += self.cats[ind][f"cf_{mconf['name']}"] <= float(mconf["min"])
                    print(f"      * {mask[mask].size:,} clusters masked in total")
            # Add mask to args
            self.kwargs[{"all": "mask", "unmatched": "mask_unmatched"}[mask_case]] = mask
        self.rec_name = f'{self.pref_save}{self.conf["matching_type"]}'
        self.rec_suf = {"all": "_0mask", "unmatched": "_0ummask", "none": ""}[mask_case]

    def _simple_recovery_catalog_by_redshift(self, ind):
        print(f"\n# Simple recovery catalog {ind} by redshift")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        recovery.plot(
            self.cats[ind],
            **self.kwargs,
            ax=ax,
            add_legend=self.conf["add_mass_label"],
            legend_fmt=self.conf[f"catalog{ind}"]["mass_num_fmt"],
        )
        ax.set_xlim(self.conf[f"catalog{ind}"]["redshift_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["recovery_lim"])
        plt.savefig(
            f"{self.rec_name}_cat{ind}_simple_redshift{self.rec_suf}.png", dpi=self.conf["dpi"]
        )
        plt.close(fig)

    def _simple_recovery_catalog_by_mass(self, ind):
        print(f"\n# Simple recovery catalog {ind} by mass")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        recovery.plot(
            self.cats[ind],
            **self.kwargs,
            transpose=True,
            ax=ax,
            add_legend=self.conf["add_redshift_label"],
            legend_fmt=self.conf[f"catalog{ind}"]["redshift_num_fmt"],
        )
        ax.set_xlim(self.conf[f"catalog{ind}"]["mass_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["recovery_lim"])
        plt.savefig(f"{self.rec_name}_cat{ind}_simple_mass{self.rec_suf}.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _panel_recovery_catalog_by_redshift(self, ind):
        print(f"\n# Panel recovery catalog {ind} by redshift")
        info = recovery.plot_panel(
            self.cats[ind],
            **self.kwargs,
            add_label=self.conf["add_mass_label"],
            label_fmt=self.conf[f"catalog{ind}"]["mass_num_fmt"],
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        ax = info["axes"].flatten()[0]
        ax.set_xlim(self.conf[f"catalog{ind}"]["redshift_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["recovery_lim"])
        plt.savefig(
            f"{self.rec_name}_cat{ind}_panel_redshift{self.rec_suf}.png", dpi=self.conf["dpi"]
        )
        plt.close(info["fig"])

    def _panel_recovery_catalog_by_mass(self, ind):
        print(f"\n# Panel recovery catalog {ind} by mass")
        info = recovery.plot_panel(
            self.cats[ind],
            **self.kwargs,
            transpose=True,
            add_label=self.conf["add_redshift_label"],
            label_fmt=self.conf[f"catalog{ind}"]["redshift_num_fmt"],
            fig_kwargs={"figsize": self.conf["figsize"]},
        )
        ax = info["axes"].flatten()[0]
        ax.set_xlim(self.conf[f"catalog{ind}"]["mass_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["recovery_lim"])
        plt.savefig(f"{self.rec_name}_cat{ind}_panel_mass{self.rec_suf}.png", dpi=self.conf["dpi"])
        plt.close(info["fig"])

    def _2d_recovery_catalog(self, ind):
        print(f"\n# 2D recovery catalog {ind}")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        recovery.plot2D(self.cats[ind], **self.kwargs, ax=ax)
        ax.set_xlim(self.conf[f"catalog{ind}"]["redshift_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["mass_lim"])
        plt.savefig(f"{self.rec_name}_cat{ind}_2D{self.rec_suf}.png", dpi=self.conf["dpi"])
        plt.close(fig)

    def _2d_recovery_catalog_with_numbers(self, ind):
        print(f"\n# 2D recovery catalog {ind} with numbers")
        fig = plt.figure(figsize=self.conf["figsize"])
        ax = plt.axes()
        recovery.plot2D(self.cats[ind], **self.kwargs, ax=ax, add_num=True)
        ax.set_xlim(self.conf[f"catalog{ind}"]["redshift_lim"])
        ax.set_ylim(self.conf[f"catalog{ind}"]["mass_lim"])
        plt.savefig(f"{self.rec_name}_cat{ind}_2D_num{self.rec_suf}.png", dpi=self.conf["dpi"])
        plt.close(fig)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    RecoveryYamlFuncs(config_file)()
