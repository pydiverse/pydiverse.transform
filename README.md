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
pixi run postinstall
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

### IBM DB2 development

The `ibm_db` package is only available on the following platforms: linux-64, osx-arm64, win-64.

> [!NOTE]
> Because of this, the IBM DB2 drivers are only available in the `py312ibm` and `py310ibm`
> environments.
> You can run tests using `pixi run -e py312ibm pytest --ibm_db2 -m ibm_db2`.

## Troubleshooting

### IBM DB2 container not yet up and running

The IBM DB2 container takes a long time to start. You can find out the name of the container with `docker ps`
(see column `NAMES`):
```
CONTAINER ID   IMAGE                                        COMMAND                  CREATED         STATUS         PORTS                                                                                 NAMES
8578e0e471ff   icr.io/db2_community/db2                     "/var/db2_setup/lib/…"   3 minutes ago   Up 3 minutes   22/tcp, 55000/tcp, 60006-60007/tcp, 0.0.0.0:50000->50000/tcp, [::]:50000->50000/tcp   pydiversepipedag-ibm_db2-1
```

If it is `pydiversepipedag-ibm_db2-1`, then you can look for `Setup has completed` in the log with
`docker logs pydiversepipedag-ibm_db2-1`.

### Installing mssql odbc driver for macOS and Linux

Install via Microsoft's
instructions for [Linux](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)
or [macOS](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/install-microsoft-odbc-driver-sql-server-macos).

In one Linux installation case, `odbcinst -j` revealed that it installed the configuration in `/etc/unixODBC/*`.
But conda installed pyodbc brings its own `odbcinst` executable and that shows odbc config files are expected in
`/etc/*`. Symlinks were enough to fix the problem. Try `pixi run python -c 'import pyodbc;print(pyodbc.drivers())'`
and see whether you get more than an empty list.

Same happened for MacOS. The driver was installed in `/opt/homebrew/etc/odbcinst.ini` but pyodbc expected it in
`/etc/odbcinst.ini`. This can also be solved by `sudo ln -s /opt/homebrew/etc/odbcinst.ini /etc/odbcinst.ini`.

Furthermore, make sure you use 127.0.0.1 instead of localhost. It seems that /etc/hosts is ignored.

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
