# pydiverse.transform

[![tests](https://github.com/pydiverse/pydiverse.transform/actions/workflows/tests.yml/badge.svg)](https://github.com/pydiverse/pydiverse.transform/actions/workflows/tests.yml)

Pipe based dataframe manipulation library that can also transform data on SQL databases

## Installation

To install the package locally in development mode, you will need to install
[pixi](https://pixi.sh/latest/). For those who haven't used pixi before, it is a
poetry style dependency management tool based on conda/micromamba/conda-forge package
ecosystem. The conda-forge repository has well maintained packages for Linux, macOS,
and Windows supporting both ARM and X86 architectures. Especially, installing
psycopg2 in a portable way got much easier with pixi. In addition, pixi is really
strong in creating lock files for reproducible environments (including system libraries)
with many essential features missing in alternative tools like poetry (see [pixi.toml](pixi.toml)).

To start developing, you can run the following commands:

```bash
git clone https://github.com/pydiverse/pydiverse.transform.git
cd pydiverse.transform

# Create the environment, activate it and install the pre-commit hooks
pixi install
pixi run pre-commit install
```

## Testing

After installation, you should be able to run:

```bash
pixi run pytest
```

## Testing more database backends

To facilitate easy testing, we provide a Docker Compose file to start all required servers.
Just run `docker compose up` in the root directory of the project to start everything, and then run `pytest` in a new tab.

Afterwards you can run:

```bash
pixi run pytest --postgres --mssql
```

## Testing db2 functionality

For running @pytest.mark.ibm_db2 tests, you need to spin up a docker container without `docker compose` since it needs
the `--priviledged` option which `docker compose` does not offer.

```bash
docker run -h db2server --name db2server --restart=always --detach --privileged=true -p 50000:50000 --env-file docker_db2.env_list -v /Docker:/database ibmcom/db2
```

Then check `docker logs db2server | grep -i completed` until you see `(*) Setup has completed.`.

Afterwards you can run:

```bash
pixi run -e py312ibm pytest --ibm_db2
```

## Packaging and publishing to pypi and conda-forge using github actions

- bump version number in [pyproject.toml](pyproject.toml)
- set correct release date in [changelog.md](docs/source/changelog.md)
- push increased version number to `main` branch
- tag commit with `git tag <version>`, e.g. `git tag 0.7.0`
- `git push --tags`

The package should appear on https://pypi.org/project/pydiverse-transform/ in a timely manner. It is normal that it takes
a few hours until the new package version is available on https://conda-forge.org/packages/.

### Packaging and publishing to Pypi manually

Packages are first released on test.pypi.org:

- bump version number in [pyproject.toml](pyproject.toml) (check consistency with [changelog.md](docs/source/changelog.md))
- push increased version number to `main` branch
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository testpypi dist/*`
- verify with https://test.pypi.org/search/?q=pydiverse.transform

Finally, they are published via:

- `git tag <version>`
- `git push --tags`
- Attention: Please, only continue here, if automatic publishing fails for some reason!
- `pixi run -e release hatch build`
- `pixi run -e release twine upload --repository pypi dist/*`

### Publishing package on conda-forge manually

Conda-forge packages are updated via:

- Attention: Please, only continue here, if automatic conda-forge publishing fails for longer than 24h!
- https://github.com/conda-forge/pydiverse-transform-feedstock#updating-pydiverse-transform-feedstock
- update `recipe/meta.yaml`
- test meta.yaml in transform repo: `conda-build build ../pydiverse-transform-feedstock/recipe/meta.yaml`
- commit `recipe/meta.yaml` to branch of fork and submit PR
