
# Installation instructions

* [Main readme](README.md)

## Table of contents
1. [Basic installation](#basic_install)
2. [Access to the proper environment on cori.nersc.gov](#access_to_the_proper_environment_on_cori)
3. [An alternative installation at NERSC or at CC-IN2P3 for DESC members](#from_desc_conda_env)
4. [Making a local copy of `ClEvaR`](#making_a_local_copy_of_clevar)


## Basic procedure <a name="basic_install"></a>

Here we provide a quick guide for a basic instalation, this will install all the packages in your current environment.
To create a specific conda environment for `ClEvaR`, we recommend you to check the begining of section
[Access to the proper environment on cori.nersc.gov](#access_to_the_proper_environment_on_cori).

### `ClEvaR` and dependency installation

Now, you can install `ClEvaR` and its dependencies as

```bash
    pip install numpy scipy astropy matplotlib
    pip install pytest sphinx sphinx_rtd_theme
    pip install jupyter  # need to have jupyter notebook tied to this environment, you can then see the environment in jupyter.nersc.gov
    git clone https://github.com/LSSTDESC/clevar.git  # If you'd like to contribute but don't have edit permissions to the `ClEvaR` repo, see below how to fork the repo instead.
    cd clevar
    python setup.py install     # build from source
```

## Access to the proper environment on cori.nersc.gov <a name="access_to_the_proper_environment_on_cori"></a>

If you have access to NERSC, this will likely be the easiest to make sure you have the appropriate environment.  After logging into cori.nersc.gov, you will need to execute the following.  We recommend executing line-by-line to avoid errors:

```bash
    module load python  # Also loads anaconda
    conda create --name clevarenv  # Create an anaconda environment for clevar
    source activate clevarenv  # switch to your newly created environment
    conda install pip  # need pip to install everything else necessary for clevar
    conda install ipython # need to have the ipython tied to this environment
    conda install -c conda-forge firefox  # Need a browser to view jupyter notebooks
```

Note, for regular contributions and use, we recommend adding `module load python` to your `~/.bashrc` so you have anaconda installed every time you log in.  You will subseqeuntly also want to be in the correct environment whenever working with `clevar`, which means running `source activate clevarenv` at the start of each session.

Once in your `ClEvaR` conda env, you may follow the [basic procedure](#basic_install) to install `ClEvaR` and its dependencies.

The above allows you to develop at NERSC and run pytest.  Your workflow as a developer would be to make your changes, do a `python setup.py install` then `pytest` to make sure your changes did not break any tests.

If you are a DESC member you may also add to your `ClEvaR` environment the GCR and GCRCatalog packages to access the DC2 datasets at NERSC. To run the DC2 example notebooks provided in `ClEvaR`, the following need to be installed in your `ClEvaR` environment at NERSC. Once in your `ClEvaR` environment (`source activate clevarenv`), run

```bash
    pip install pandas
    pip install pyarrow
    pip install healpy
    pip install h5py
    pip install GCR
    pip install https://github.com/LSSTDESC/gcr-catalogs/archive/master.zip
    pip install FoFCatalogMatching
```

To open up a notebook from NERSC in your browser, you will need to go to the [nersc jupyter portal](https://jupyter.nersc.gov) and sign in. You will need to make this conda environment available in the kernel list:

```bash
    python -m ipykernel install --user --name=conda-clevarenv
```

Clicking on the upper right corner of the notebook will provide options for your kernel.  Choose the kernel `conda-clevarenv` that you just created.

## Making a local copy of `ClEvaR` <a name="making_a_local_copy_of_clevar"></a>

As a newcomer, you likely will not have edit access to the main `ClEvaR` repository.
Without edit privileges, you won't be able to create or push changes to branches in the base repository. You can get around this by creating a [fork](https://help.github.com/articles/fork-a-repo/), a linked copy of the `ClEvaR` repository under your Github username. You can then push code changes to your fork which can later be merged with the base repository.

To create a fork, navigate to the [`ClEvaR` home page](https://github.com/LSSTDESC/clevar) and click 'Fork' in the upper right hand corner. The fork has been created under your username on Github's remote server and can now be cloned to your local repository with

```bash
    git clone git@github.com:YOUR-USERNAME/clevar.git
    cd clevar
    git remote add base git@github.com:LSSTDESC/clevar.git
```
If you do have edit privileges to `ClEvaR`, it may be easier to simply clone the base `ClEvaR` repository.
``` bash
    git clone git@github.com:LSSTDESC/clevar.git
```
