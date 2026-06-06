"""@file match_metrics_redshift.py
Matching metrics - redshift functions for command line execution
"""

from clevar.match_metrics import scaling

from .match_metrics_parent import ScalingYamlFuncs


class RedshiftYamlFuncs(ScalingYamlFuncs):
    """Main plot class"""

    # pylint: disable=too-few-public-methods

    def __init__(self, config_file):
        # Create clevar objects from yml config
        super().__init__(
            config_file,
            load_configs=["catalog1", "catalog2", "cosmology", "mt_metrics_redshift"],
            add_new_configs=["mt_metrics_redshift"],
            self_name="redshift",
            other_name="mass",
        )

    def _core_density(self, *args, **kwargs):
        return scaling.redshift_density(*args, **kwargs)

    def _core_metrics(self, *args, **kwargs):
        return scaling.redshift_metrics(*args, **kwargs)

    def _core_density_metrics(self, *args, **kwargs):
        return scaling.redshift_density_metrics(*args, **kwargs)

    def _core_other_color(self, *args, **kwargs):
        return scaling.redshift_masscolor(*args, **kwargs)

    def _core_density_other_panel(self, *args, **kwargs):
        return scaling.redshift_density_masspanel(*args, **kwargs)

    def _core_dist_self(self, *args, **kwargs):
        return scaling.redshift_dist_self(*args, **kwargs)

    def _core_dist(self, *args, **kwargs):
        return scaling.redshift_dist(*args, **kwargs)

    def _core_density_dist(self, *args, **kwargs):
        return scaling.redshift_density_dist(*args, **kwargs)


def run(config_file):
    """Main plot function

    Parameters
    ----------
    config_file: str
        Yaml file with configuration to run
    """
    RedshiftYamlFuncs(config_file)()
