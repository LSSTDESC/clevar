# Contributing to `ClEvaR`

This is a brief guide to contributing to `ClEvaR`, including information about identifiying code issues and submitting code changes or documentation.

* [Main readme](README.md)

## Table of contents
1. [Identifying Issues](#identifying_issues)
2. [Making and submitting changes](#making_and_submitting_changes)
3. [Adding documentation](#adding_documentation)
4. [Reviewing an open pull request](#reviewing_an_open_pull_request)
5. [Steps to merging a pull request](#steps_to_merging_a_pull_request)
6. [Updating Public Documentation on lsstdesc.org](#updating_public_docs)
7. [Additional resources](#additional_resources)
8. [Contact](#contact)

## Identifying Issues <a name="identifying_issues"></a>

Action items for `ClEvaR` code improvements are listed as [GitHub Issues](https://github.com/LSSTDESC/clevar/issues).
Issues marked with the label `good first issue` are well-suited for new contributors.


## Making and submitting changes <a name="making_and_submitting_changes"></a>
Once you've [created a local copy of `ClEvaR`](INSTALL.md) on your machine, you can begin making changes to the code and submitting them for review.
To do this, follow the following steps from within your local copy of `ClEvaR` (forked or base).

1. Checkout a new branch to contain your code changes independently from the `main` repository.
    [Branches](https://help.github.com/articles/about-branches/) allow you to isolate temporary development work without permanently affecting code in the repository.
    ```bash
    git checkout -b branchname
    ```
    Your `branchname` should be descriptive of your code changes.
    If you are addressing a particular issue #`xx`, then `branchname` should be formatted as `issue/xx/summary` where `summary` is a description of your changes.
2. Make your changes in the files stored in your local directory.
3. Commit and push your changes to the `branchname` branch of the remote repository.
    ```bash
    git add NAMES-OF-CHANGED-FILES
    git commit -m "Insert a descriptive commit message here"
    git pull origin main
    git push origin branchname
    ```
4. You can continue to edit your code and push changes to the `branchname` remote branch.
    Once you are satisfied with your changes, you can submit a [pull request](https://help.github.com/articles/about-pull-requests/) to merge your changes from `branchname` into the `main` branch.
    Navigate to the [`ClEvaR` Pull Requests](https://github.com/LSSTDESC/clevar/pulls) and click 'New pull request.'
    Select `branchname`, fill out a title and description for the pull request, and, optionally, request review by a `ClEvaR` team member.
    Once the pull request is approved, it will be merged into the `ClEvaR` main branch.

NOTE: Code is not complete without unit tests and documentation. Please ensure that unit tests (both new and old) all pass and that docs compile successfully.

To test this, first install the code by running `python setup.py install --user` (required after any change whatsoever to the `.py` files in `clmm/` directory). To run all of the unit tests, run `pytest` in the root package directory. To test the docs, in the root package directory after installing, run `./update_docs`. This script both deletes the old compiled documentation files and rebuilds them. You can view the compiled docs by running `open docs/_build/html/index.html`.

## Adding documentation <a name="adding_documentation"></a>

If you are adding documentation either in the form of example jupyter notebooks or new python modules, your documentation will need to compile for our online documentation hosted by the LSST-DESC website: http://lsstdesc.org/clevar/

We have done most of the hard work for you. Simply edit the configuration file, `docs/doc-config.ini`. If you are looking at add a module, put the module name under the `APIDOC` heading. If you are adding a demo notebook to demonstrate how to use the code, place the path from the `docs/` directory to the notebook under the `DEMO` heading. If you are adding an example notebook that shows off how to use `ClEvaR` to do science, place the path from the `docs/` directory to the notebook under the `EXAMPLE` heading.

Once it has been added to the config file, simply run `./update_docs` from the top level directory of the repository and your documentation should compile and be linked in the correct places!


## Reviewing an open pull request <a name="reviewing_an_open_pull_request"></a>

To review an open pull request submitted by another developer, there are several steps that you should take.

1. For each new or changed file, ensure that the changes are correct, well-documented, and easy to follow. If you notice anything that can be improved, leave an inline comment (click the line of code in the review interface on Github).
2. For any new or changed code, ensure that there are new tests in the appropriate directory. Try to come up with additional tests that either do or can break the code. If you can think of any such tests, suggest adding them in your review.
3. Double check any relevant documentation. For any new or changed code, ensure that the documentation is accurate (i.e. references, equations, parameter descriptions).
4. Next, checkout the branch to a location that you can run `ClEvaR`. From the top level package directory (the directory that has `setup.py`) install the code via `python setup.py install --user`. Then, run `pytest` to run the full testing suite.
5. Now that tests are passing, the code likely works (assuming we have sufficient tests!) so we want to finalize the new code. We can do this by running a linter on any new or changed files `pylint {filename}`. This will take a look at the file and identify any style problems. If there are only a couple, feel free to resolve them yourself, otherwise leave a comment in your review that the author should perform this step.
6. We can now check that the documentation looks as it should. We provide a convenient bash script to compile the documentation. To completely rebuild the documentation, AFTER INSTALLING (if you made any changes, even to docstrings), run `./update_docs`. This will delete any compiled documentation that may already exist and rebuild it. If this runs without error, you can then take a look by `open docs/_build/html/index.html`. Make sure any docstrings that were changed compile correctly.
7. Finally, install (`python setup.py install --user`), run tests (`pytest`), and compile documentation (`./update_docs`) one file time and if everything passes, accept the review!

NOTE: We have had several branches that have exploded in commit number. If you are merging a branch and it has more than ~20 commits, strongly recommend using the "Squash and Merge" option for merging a branch.

## Steps to merging a pull request <a name="steps_to_merging_a_pull_request"></a>

To ensure consistency between our code and documentation, we need to take care of a couple of more things after accepting a review on a PR into `main`.

1. In the branch of the pull request, change the version number of the code located in `clmm/__init__.py`, commit and push. If you are unsure of how you should change the version number, don't hesitate to ask!

We use [semantic versioning](https://semver.org/), X.Y.Z. If the PR makes a small change, such as a bug fix, documentation updates, style changes, etc., increment Z. If the PR adds a new feature, such as adding support for a new profile, increment Y (and reset Z to 0). If a PR adds a feature or makes a change that breaks the old API, increment X (and reset Y and Z to 0). After the first tagged release of `ClEvaR`, anything that is a candidate to increment X should be extensively discussed beforehand.

2. "Squash and Merge" the pull request into `main`. It asks for a squashed commit message. This should be descriptive of the feature or bug resolved in the PR and should be pre-prended by a [conventional commit scope](https://www.conventionalcommits.org/).

Please choose from `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`. If this commit breaks the previous API, add an explanation mark (for example, `fix!:`). Definitions of each scope can be found at the above link.

Note: `fix:` should correspond to version changes to Y. The rest of the scopes above should be version changes to Z.

3. Tag and push this new version of the code. In the `main` branch use the following commands:

    ```bash
    git tag X.Y.Z
    git push --tag
    ```

of course replacing `X.Y.Z` by the new version.

## Updating Public Documentation on lsstdesc.org <a name="updating_public_docs"></a>

This is easy! Once you have merged all approved changes into `main`, you will want to update the public documentation.
All these steps should be done on the `publish-docs` branch (just `git checkout publish-docs` on your local computer):
1. Merge all of the latest changes from main `git merge main`.
2. If you have figures in notebooks that you would like rendered on the website, you will want to execute all cells of demo notebooks.
3. From the main `ClEvaR` directory (the one that contains `setup.py`) run `./publish_docs` (note, this is different from `./update_docs` that you did in your development branch) and it does all of the work for you (including automatically pushing changes to Github)!

## Additional resources <a name="additional_resources"></a>

Here's a list of additional resources which you may find helpful in navigating git for the first time.
* The DESC Confluence page on [Getting Started with Git and Github](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Git+and+GitHub)
* [Phil Marshall's Getting Started repository and FAQ](https://github.com/drphilmarshall/GettingStarted#forks)
* [Phil Marshall's Git tutorial video lesson](https://www.youtube.com/watch?v=2g9lsbJBPEs)
* [The Github Help Pages](https://help.github.com/)
