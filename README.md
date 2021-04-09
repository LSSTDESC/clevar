# Cluster Evalutaion Resources (ClEvaR)
[![Build Status](https://travis-ci.org/LSSTDESC/clevar.svg?branch=master)](https://travis-ci.org/LSSTDESC/clevar)
[![Build and Check](https://github.com/LSSTDESC/clevar/workflows/Build%20and%20Check/badge.svg)](https://github.com/LSSTDESC/clevar/actions?query=workflow%3A%22Build+and+Check%22)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/clevar/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/clevar?branch=master)
Library to validate cluster detection

# Contriutors
* [Michel Aguena](https://github.com/m-aguena) (LAPP / LIneA)

## Table of contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Running ClEvaR](#running)
4. [Contributing](#contributing)
5. [Contact](#contact)

## Requirements <a name="requirements"></a>

ClEvaR requires Python version 3.6 or later.  ClEvaR has the following dependencies:

- [NumPy](http://www.numpy.org/) (1.17 or later)
- [SciPy](http://www.numpy.org/) (1.3 or later)
- [Astropy](https://www.astropy.org/) (3.x or later for units and cosmology dependence)
- [Matplotlib](https://matplotlib.org/) (for plotting and going through tutorials)
- [Healpy](https://healpy.readthedocs.io/en/latest/) (1.14 or later for footprint computations)

```
  pip install numpy scipy astropy matplotlib healpy
```

(See the [INSTALL documentation](INSTALL.md) for more detailed installation instructions.)

For developers, you will also need to install:

- [pytest](https://docs.pytest.org/en/latest/) (3.x or later for testing)
- [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) (for documentation)

These are also pip installable:
```
  pip install pytest sphinx sphinx_rtd_theme
```
Note, the last item, `sphinx_rtd_theme` is to make the docs.

## Installation <a name="installation"></a>

To install ClEvaR you currently need to build it from source:

```
  git clone https://github.com/LSSTDESC/clevar.git
  cd clevar
  python setup.py install --user   # Add --user flag to install it locally
```
See the [INSTALL documentation](INSTALL.md) for more detailed installation instructions.

To run the tests you can do:

  `pytest`

## Running ClEvaR <a name="running"></a>

`ClEvaR` can be imported as a python library or as an executable based on configuration `yaml` files.
Check detailed description of each usage below:

- [`ClEvaR` library](RUN_LIB.md)
- [`ClEvaR` executables](RUN_EXE.md)

## Contributing <a name="contributing"></a>

Contributing documentation can be found [here](CONTRIBUTING.md)

## Contact <a name="contact"></a>
If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/clevar/issues).
