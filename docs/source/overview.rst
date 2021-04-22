******************
Rapid overview
******************
ClEvaR provides a framework for easy matching of catalogs,
and computation of the main metrics and scaling relations.
This library has been developed in the LSST context,
but can also be used with any set of catalogs.

ClEvaR can be used as a python library for flexible use in scritps and notebooks,
or as a command line executable.
The executable mode is configured by an input `yaml` file and does not require a knowledge of the inner structure of the code.
Details for this use can be found `here <https://github.com/LSSTDESC/clevar/blob/main/CLEVAR_EXE.md>`_.

The code was developed with a object oriented design for easy integration with other libraries.
It has specific objects for the catalogs, footprints, catalog matching, cosmology and functions for metrics and scaling relations.
The cosmology object has a backend that can use either `astropy` or `CCL <https://github.com/LSSTDESC/CCL>`_,
with a possibility of adding others.
A set of notebooks showing the different application of ClEvaR can be found in the
`examples <https://github.com/LSSTDESC/clevar/blob/main/examples/>`_
folder of the project.

The `ClCatalog` object
======================

The `ClCatalog` object is the main data structure for cluster catalogs, holding the input and matching data,
with some inbuilt functionality to facilitate the matching and footprint related processes.
It contains:

* The catalog name
* A table with the input information (id, ra, dec, z, mass, radius), where matching information will be added.
* A internal data with matching input information.
* A dictionary for easy lookup of cluster by id in the main table.
* Unit of the cluster radius (optional).
* Customized labels of the catalog columns to be used in plots (optional).

The `Matching` objects
======================

There are several `Matching` objects (currently only `ProximityMatching`) constructed with
a unified api for a more intuitive use.

The first step of the matching is done using the `prep_cat_for_match` function where the necessary
matching preparations are added
to each catalogs. Then the catalogs are matched using `multiple`,
where all candidates that satisfy the matching criteria are stored in the catalogs.
The `unique` function makes the selection for the best candidates
and makes sure each matching is unique.
Finally `cross_match` checks is the same pairs are found in both directions.

Another option is to use `match_from_config`, where all steps above are made according to a input
configutaion dictionary.

These objects also have internal functions to save and load the matching information:
`save_matches`, `load_matches`.

The `Cosmology` objects
=======================

There are several `Cosmology` objects (`AstropyCosmology`, `CCLCosmology`) constructed with
a unified api for a more intuitive use. These objects have the following functions used by `ClEvaR`:

* `get_Omega_m` - Gets the value of the dimensionless matter density.
* `get_E2` - Gets hubble parameter squared.
* `eval_da_z1z2` - Computes the angular diameter distance between z1 and z2.
* `eval_da` - Computes the angular diameter distance between 0.0 and z.
* `rad2mpc` - Convert between radians and Mpc using the small angle approximation.
* `mpc2rad` - Convert between Mpc and radians using the small angle approximation.
* `eval_mass2radius` - Computes the radius from M_Delta.

Metrics and scaling relation of matched catalogs
================================================

Once the catalogs have been matched, it is possible to plot some metrics of the matching
and scaling relations of the catalogs quantities using the `match_metrics` package.
This package has three main modules `recovery`, `distances` and `scaling`.

The `recovery` module is used to compute the recovery rates (completeness, purity) of each catalog.
The main functions of this module take `ClCatalogs` as inputs with pre-fixed columns:

* `get_recovery_rate` - Get recovery rate binned in 2 components.
* `plot` - Plot recovery rate as lines, with each line binned by redshift inside a mass bin.
* `plot_panel` - Plot recovery rate as lines in panels, with each line binned by redshift and each panel is based on the data inside a mass bin.
* `plot2D` - Plot recovery rate as in 2D (redshift, mass) bins.

The `distances` module is used to compute the recovery rates (completeness, purity) of each catalog.
The main functions of this module take `ClCatalogs` as inputs with pre-fixed columns:

* `central_position` - Plot recovery rate as lines, with each line binned by redshift inside a mass bin.
* `redshift` - Plot recovery rate as lines, with each line binned by redshift inside a mass bin.

The `scaling` module is used to compute scaling relations of matched clusters, with
focus on mass (or mass proxy) and redshifts.
The main functions of this module take `ClCatalogs` as inputs with pre-fixed columns:

* `redshift` - Scatter plot with errorbars and color based on input.
* `redshift_density` - Scatter plot with errorbars and color based on point density.
* `redshift_masscolor` - Scatter plot with errorbars and color based on input.
* `redshift_masspanel` - Scatter plot with errorbars and color based on input with panels.
* `redshift_density_masspanel` - Scatter plot with errorbars and color based on point density with panels.
* `redshift_metrics` - Plot metrics.
* `redshift_density_metrics` - Scatter plot with errorbars and color based on point density with scatter and bias panels.
* `mass` - Scatter plot with errorbars and color based on input.
* `mass_zcolor` - Scatter plot with errorbars and color based on input.
* `mass_density` - Scatter plot with errorbars and color based on point density.
* `mass_zpanel` - Scatter plot with errorbars and color based on input with panels.
* `mass_density_zpanel` - Scatter plot with errorbars and color based on point density with panels.
* `mass_metrics` - Plot metrics.
* `mass_density_metrics` - Scatter plot with errorbars and color based on point density with scatter and bias panels.

Each of these modules have internal classes named `ArrayFunc` and `CatalogFuncs` that makes the plots
based on arrays and `ClCatalogs` (with generic columns) respectively.
These classes can be used for a more flexible application of these functions.
