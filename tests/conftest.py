# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import logging

import pytest

from pydiverse.common.util.structlog import setup_logging

# Setup


supported_options = [
    "duckdb_parquet",
    "sqlite",
    "postgres",
    "mssql",
    "ibm_db2",
]


def pytest_addoption(parser):
    for opt in supported_options:
        parser.addoption(
            "--" + opt,
            action="store_true",
            default=False,
            help=f"run test that require {opt}",
        )


def pytest_collection_modifyitems(config: pytest.Config, items):
    for opt in supported_options:
        if not config.getoption("--" + opt):
            skip = pytest.mark.skip(reason=f"{opt} not selected")
            for item in items:
                if opt in item.keywords or any(
                    kw.startswith(f"{opt}-") or kw.endswith(f"-{opt}") or f"-{opt}-" in kw for kw in item.keywords
                ):
                    item.add_marker(skip)


setup_logging(log_level=logging.INFO)
