# pydiverse.transform

[![tests](https://github.com/pydiverse/pydiverse.transform/actions/workflows/tests.yml/badge.svg)](https://github.com/pydiverse/pydiverse.transform/actions/workflows/tests.yml)

Pipe based dataframe manipulation library that can also transform data on SQL databases

## Installation

To install the package locally in development mode, you first have to install
[Poetry](https://python-poetry.org/docs/#installation).
After that, install pydiverse transform like this:

```bash
git clone https://github.com/pydiverse/pydiverse.transform.git
cd pydiverse.transform

# Create the environment, activate it and install the pre-commit hooks
poetry install --with=dev
poetry shell
pre-commit install
```

## Testing

After installation, you should be able to run:

```bash
poetry run pytest
```

## Packaging

For publishing with poetry to pypi, see:
https://www.digitalocean.com/community/tutorials/how-to-publish-python-packages-to-pypi-using-poetry-on-ubuntu-22-04

Packages are first released on test.pypi.org:

- see https://stackoverflow.com/questions/68882603/using-python-poetry-to-publish-to-test-pypi-org
- `poetry version prerelease` or `poetry version patch`
- push increased version number to `main` branch
- `poetry build`
- `poetry publish -r test-pypi`
- verify with https://test.pypi.org/search/?q=pydiverse.transform

Finally, they are published via:

- `git tag `\<version>
- `git push --tags`
- `poetry publish`

Conda-forge packages are updated via:

- https://github.com/conda-forge/pydiverse-transform-feedstock#updating-pydiverse-transform-feedstock
- update `recipe/meta.yaml`
- test meta.yaml in transform repo: `conda-build build ../pydiverse-transform-feedstock/recipe/meta.yaml`
- commit `recipe/meta.yaml` to branch of fork and submit PR
