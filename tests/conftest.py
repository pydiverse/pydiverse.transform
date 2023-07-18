from __future__ import annotations

import pytest

# Setup


supported_options = [
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
                if opt in item.keywords:
                    item.add_marker(skip)
