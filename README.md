# pydiverse.transform

[![CI](https://github.com/Quantco/pydiverse.transform/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/pydiverse.transform/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?style=plastic)](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/pydiverse.transform/latest/index.html)

Pipe based dataframe manipulation library that can also transform data on SQL databases

## Installation

You can install the package in development mode using:

```bash
git clone https://github.com/pydiverse/pydiverse.transform.git
cd pydiverse.transform

# create and activate a fresh environment named pydiverse.transform
# see environment.yml for details
mamba env create
conda activate pydiverse.transform

pre-commit install
pip install --no-build-isolation -e .
```
