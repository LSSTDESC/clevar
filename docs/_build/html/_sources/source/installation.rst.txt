**************
Installation
**************

To install ClEvaR you currently need to build it from source::
  
  git clone https://github.com/DESC/ClEvaR.git
  cd ClEvaR
  python setup.py install

To run the tests you can do::

  pytest
  
Requirements
============
ClEvaR requires Python version 3.6 or later.  ClEvaR has the following dependencies:

- `numpy <http://www.numpy.org/>`_: 1.17 or later
- `scipy <http://www.scipy.org/>`_: 1.3 or later
- `astropy <https://www.astropy.org/>`_: 3.x or later
- `healpy <https://healpy.readthedocs.io/en/latest/>`_: 1.14 or later (for footprint computations)
- `matplotlib <https://matplotlib.org/>`_

These are pip installable::

  pip install numpy scipy astropy matplotlib


For comological computations, ClEvaR can use `astropy` or one of the following libraries (that must be installed):

- `CCL <https://ccl.readthedocs.io/en/v2.0.0/>`_

See the `INSTALL documentation <https://github.com/LSSTDESC/ClEvaR/blob/master/INSTALL.md>`_ for more detailed installation instructions.

For developers, you will also need to install:

- `pytest <https://docs.pytest.org/en/latest/>`_ (3.x or later for testing)
- `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_ (for documentation)

These are also pip installable::

  pip install pytest sphinx sphinx_rtd_theme
