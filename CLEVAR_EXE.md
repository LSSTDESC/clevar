# Using `ClEvaR` as an executable

`ClEvaR` can be used directly from the command line with `yml` configuration files.
Some examples of config files can be found in the [demo](https://github.com/LSSTDESC/clevar/tree/main/demo) directory.

* [Main readme](README.md)

## Table of contents
1. [Loading `ClEvaR` environment](#environment)
2. [Executing `ClEvaR` operations](#executing)
 1. [Matching catalogs](#matching)
 2. [Footprint application](#footprint)
 3. [Metrics of matching](#metrics)
3. [Configuration file](#config)
 1. [cosmology](#config_cosmology)
 2. [catalog1](and catalog2)](#config_cat)
 3. [proximity_match](#config_proximity_match)
 4. [masks](#config_mask)
 5. [match_metrics](#config_match_metrics)
  1. [recovery](#config_metics_recovery)
  2. [distances](#config_metics_distances)
  3. [Mass](#config_metics_mass)
  4. [redshift](#config_metics_redshift)

## Loading `ClEvaR` environment <a name="environment"/a>

The first step is to load the clevar functions into your environment using the `SOURCE_ME` file in the main directory of `ClEvaR`:

```
  source SOURCE_ME
```

## Executing `ClEvaR` operations <a name="executing"/a>

Once you sourced `ClEvaR` environment, you will be able to run its functions. All `ClEvaR` commands have a `clevar_` prefix and require a `.yml` configuration file.
The examples below will assume you are using a confiration file named `config.yml`.

```
  clevar_<<function>> source SOURCE_ME
```

### Matching catalogs <a name="matching"/a>

Currently, only proximity matching is implemented in `ClEvaR`. To run this operation:

```
  clevar_match_proximity config.yml
```

### Footprint application <a name="footprint"/a>

`ClEvaR` has inbuilt functionality to compute mask for the catalogs according to footprint criteria.
For this operation, use:

```
  clevar_add_masks config.yml
```

It can also create a footprint based on cluster positions. This can be done using:

```
  clevar_artificial_footprint config.yml
```

### Metrics of matching <a name="metrics"/a>

To plot each of the possible metrics of the matching, use the following commands:

```
  clevar_match_metrics_recovery_rate config.yml
  clevar_match_metrics_distances config.yml
  clevar_match_metrics_mass config.yml
  clevar_match_metrics_redshift config.yml
```

## Configuration file <a name="config"/a>

All operations with command line require a `yml` configuration file.
Each different section of this file is described here.
The main sections of this file are:

* `outpath` - Path to save output.
* `cosmology` - Configuration for cosmology.
* `catalog1` - Configuration for catalog 1.
* `catalog2` - Configuration for catalog 2.
* `proximity_match` -Configuration for proximity matching.
* `masks` - Configuration to be used for mask creation and in recovery rate.
* `match_metrics` - Configuration for metrics of matching.

Each configuration is detailed below.

### cosmology <a name="config_cosmology"/a>

Configuration for comology package and parameters:

* `backend` - Library for cosmology. Options are Astropy, CCL.
* `parameters` - cosmological parameters.

### catalog1 (and catalog2) <a name="config_cat"/a>

Configuration of input catalogs:

* `file` - file of catalog
* `name` - name of catalog (for plots)
* `columns` - Section for column names.
  * `ra` - Ra
  * `dec` - Dec
  * `z` - redshift
  * `mass` - Mass or proxy
  * `radius` - Radius of cluster
* `radius_unit` - Units of the radius. Options are: `radians`, `degrees`, `arcmin`, `arcsec`, `pc`, `kpc`, `Mpc`. A mass can also be used and converted to radius with the cosmology, in this case, must be in the format `M{delta}{TYPE}` , ex: `M200b` (background), `M500c`(critical).
* `labels` - Labels for plots. If not availble, `column_{name}` used. Ex:
  * `mass` - Mass1
* `footprint` - Footprint information. Used to create artificial footprint and on recovery rate.
  * `file` - file for foorptint
  * `nside` - healpix NSIDE.
  * `nest` - healpix nested ordering. if false use ring.
  * `pixel_name` - Name of pixel column of footprint.
  * `detfrac_name` - Name of detfrac column of footprint. Use `None` if not existing.
  * `zmax_name` - Name of zmax column of footprint. Use `None` if not existing.

### proximity_match <a name="config_proximity_match"/a>

Configuration for proximity matching.

* `which_radius` - Case of radius to be used, can be: `cat1`, `cat2`, `min`, `max`.
* `type` - Options are cross, cat1, cat2.
* `preference` - Preference for multiple candidadtes. Options are `more_massive`, `angular_proximity` or `redshift_proximity`.
* `catalog1` - Options for catalog 1
  * `delta_z` - Defines the zmin, zmax for matching. If `cat` uses redshift properties of the catalog, if `spline.filename` interpolates data in `filename` (z, zmin, zmax) fmt, if `float` uses `delta_z*(1+z)`, if `None` does not use z.
  * `match_radius` - Radius to be used in the matching. If `cat` uses the radius in the catalog, else must be in format `value unit` (ex: `1 arcsec`, `1 Mpc`).
* `catalog2` - Options for catalog 2
  * `delta_z` - Defines the zmin, zmax for matching. If `cat` uses redshift properties of the catalog, if `spline.filename` interpolates data in `filename` (z, zmin, zmax) fmt, if `float` uses `delta_z*(1+z)`, if `None` does not use z.
  * `match_radius` - Radius to be used in the matching. If `cat` uses the radius in the catalog, else must be in format `value unit` (ex: `1 arcsec`, `1 Mpc`).

### masks <a name="config_mask"/a>

Configuration to make masks for recovery rate computations.
These configurations must be inside a section for the corresponding catalog (`catalog1` or `catalog2`):

* `in_footprint` - Flag if cluster is inside footprint. Add other sections with the prefix `in_footprint` for more masks (\*).
  * `which_footprint` - Which footprint to use for computaions. Options: `self`, `other`.
  * `name` - Name of this mask.
* `coverfraction` - Coverfraction configuration. Add other sections with the prefix `coverfraction` for more computation (\*).
  * `name` - Name for coverfraction column. It will receive a `cf_` prefix.
  * `which_footprint` - Which footprint to use for computaions. Options: `self`, `other`.
  * `aperture` - Size of aperture with units (ex: `1 arcmin`, `1 mpc`).
  * `window_function` - Window to weight the coverfraction. Options are `flat`, `nfw2D`.

(\*) Each section name must be different or they will be overwritten.

### match_metrics <a name="config_match_metrics"/a>

Configuration for the metric plots.
There are two main parameters:

* `figsize` - Figure size of figures in cm, must be 2 numbers.
* `dpi` - Resolution in dots per inch.

Each matching metrics operation are configured by subsections below.

#### recovery <a name="config_metics_recovery"/a>

Configuration for recovery rate plots, main parameters are:

* `figsize` - Figure size of figures in cm, must be 2 numbers. Overwrites match_metrics figsize.
* `dpi` - Resolution in dots per inch. Overwrites match_metrics dpi.
* `plot_case` - Types of plots to be done. Options are `simple`, `panel`, `2D` or `all`.
* `matching_type` - Options are `cross`, `cat1`, `cat2`, `multi_cat1`, `multi_cat2`, `multi_join`.
* `line_type` - Type of line. Options are: `line`, `steps`.

There are also configurations relative to each catalog that must be inside the corresponding section (`catalog1` or `catalog2`):

* `log_mass` - Use mass in log scale.
* `mass_num_fmt` - Format the values of mass binedges (ex: '.2f') in label.
* `redshift_num_fmt` - Format the values of redshift binedges (ex: '.2f') in label.
* `recovery_label` - Labey for recovery rate.
* `mass_bins` - Mass bins. Can be number of bins, or `xmin, xmax, dx`. If log_mass provide log of values.
* `redshift_bins` - Redshift bins. Can be number of bins, or `xmin, xmax, dx`.
* `mass_lim` - Mass limits in plots. Must be 2 numbers (min, max) or `None`.
* `redshift_lim` - Redshift limits in plots. Must be 2 numbers (min, max) or `None`.
* `recovery_lim` - Recovery rate limits in plots. Must be 2 numbers (min, max) or `None`.
* `add_mass_label` - Add labels and legends of mass bins.
* `add_redshift_label` - Add labels and legends of redshift bins.
* `masks` - Mask objects used in the computation of recovery rates. Names must correspond to those in the main `masks` section.
  * `case` - Which clusters to mask on recovery rate computations. Options: `None`, `All`, `Unmatched`.
  * `in_footprint` - Footprin mask. Add other sections with the prefix `in_footprint` for more masks (\*).
    * `name` - Name for mask.
    * `use` - Use this mask.
  * `coverfraction` - Coverfraction mask. Add other sections with the prefix `coverfraction` for more computation (\*).
    * `name` - Name for coverfraction column. It will receive a `cf_` prefix.
    * `min` - Minimum value of cover fraction to be considered.

(\*) Each section name must be different or they will be overwritten.

#### distances <a name="config_metics_distances"/a>

Configuration for distances of matched clusters, main parameters are:

* `figsize` - Figure size of figures in cm, must be 2 numbers. Overwrites match_metrics figsize.
* `dpi` - Resolution in dots per inch. Overwrites match_metrics dpi.
* `plot_case` - Types of plots to be done. Options are `simple`, `panel`, `2D or all`.
* `matching_type` - Options are cross, cat1, cat2.
* `line_type` - Type of line. Options are: `line`, `steps`.
* `radial_bins` - Bins for radial distances. Can be number of bins, or `xmin, xmax, dx`.

* `radial_bin_units` - arcmin # units of radial bins.
* `delta_redshift_bins` - 20 # bins for redshift distances.

There are also configurations relative to each catalog that must be inside the corresponding section (`catalog1` or `catalog2`):

* `log_mass` - Use mass in log scale.
* `mass_num_fmt` - Format the values of mass binedges (ex: `.2f`) in label.
* `redshift_num_fmt` - Format the values of redshift binedges (ex: `.2f`) in label.
* `mass_bins` - Mass bins. Can be number of bins, or `xmin, xmax, dx`. If log_mass provide log of values.
* `redshift_bins` - Redshift bins. Can be number of bins, or `xmin, xmax, dx`.
* `add_mass_label` - Add labels and legends of mass bins.
* `add_redshift_label` - Add labels and legends of redshift bins.

#### mass <a name="config_metics_mass"/a>

Configuration for mass scaling relation, main parameters are:

* `figsize` - Figure size of figures in cm, must be 2 numbers. Overwrites match_metrics figsize.
* `plot_case` - Types of plots to be done. Options are zcolors, density, density_panel.
* `matching_type` - Options are cross, cat1, cat2.
* `add_redshift_label` - Add redshift label in panels.
* `add_err` - plot errorbars when available.
* `add_cb` - add color bar in color plots.
* `log_mass` - mass in log scale.
* `ax_rotation` - for density plots. angle (in degrees) for rotation of axis of binning. overwrites use of mass_bins in catalogs.
* `rotation_resolution` - for density plots. number of bins to be used when ax_rotation!=0.

There are also configurations relative to each catalog that must be inside the corresponding section (`catalog1` or `catalog2`):

* `redshift_bins` - Redshift bins for panels. Can be number of bins, or `xmin, xmax, dx`.
* `redshift_num_fmt` - Format the values of redshift binedges (ex: `.2f`) in label.
* `mass_bins` - Mass bins for density colors. Can be number of bins, or `xmin, xmax, dx`. If log_mass provide log of values.

#### redshift - # Scaling relation <a name="config_metics_redshift"/a>

Configuration for mass scaling relation, main parameters are:

* `figsize` - Figure size of figures in cm, must be 2 numbers. Overwrites match_metrics figsize.
* `plot_case` - Types of plots to be done. Options are zcolors, density, density_panel.
* `matching_type` - Options are cross, cat1, cat2.
* `add_mass_label` - True # Add mass label in panels.
* `add_err` - plot errorbars when available.
* `add_cb` - add color bar in color plots.
* `log_mass` - mass in log scale.
* `ax_rotation` - for density plots. angle (in degrees) for rotation of axis of binning. overwrites use of mass_bins in catalogs.
* `rotation_resolution` - for density plots. number of bins to be used when ax_rotation!=0.

There are also configurations relative to each catalog that must be inside the corresponding section (`catalog1` or `catalog2`):

* `redshift_bins` - Redshift bins for panels. Can be number of bins, or `xmin, xmax, dx`.
* `redshift_num_fmt` - Format the values of redshift binedges (ex: `.2f`) in label.
* `mass_bins` - Mass bins for density colors. Can be number of bins, or `xmin, xmax, dx`. If log_mass provide log of values.
