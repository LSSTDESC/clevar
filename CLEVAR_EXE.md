# Using `ClEvaR` as an executable

`ClEvaR` can be used directly from the command line with `yml` configuration files.
Some examples of config files can be found in the [demo](https://github.com/LSSTDESC/clevar/tree/main/demo) directory.

* [Main readme](README.md)

## Loading `ClEvaR` environment

The first step is to load the clevar functions into your environment using the `SOURCE_ME` file in the main directory of `ClEvaR`:

```
  source SOURCE_ME
```

## Executing `ClEvaR` operations

Once you sourced `ClEvaR` environment, you will be able to run its functions. All `ClEvaR` commands have a `clevar_` prefix and require a `.yml` configuration file.
The examples below will assume you are using a confiration file named `config.yml`.

```
  clevar_<<function>> source SOURCE_ME
```

### Matching catalogs

Currently, only proximity matching is implemented in `ClEvaR`. To run this operation:

```
  clevar_match_proximity config.yml
```
