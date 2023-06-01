"""@file match_metrics_redshift.py
Matching metrics - redshift rate functions for command line execution
"""
from .helper_funcs import loadconf, make_catalog


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
