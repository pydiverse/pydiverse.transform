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

    parser.addoption(
        "--fuzzing", action="store_true", default=False, help="enable fuzz tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items):
    if not config.getoption("--fuzzing"):
        skip = pytest.mark.skip(reason="need --fuzzing to run")
        for item in items:
            if "test_fuzzing" in item.module.__name__:
                item.add_marker(skip)

    for opt in supported_options:
        if not config.getoption("--" + opt):
            skip = pytest.mark.skip(reason=f"{opt} not selected")
            for item in items:
                if opt in item.keywords:
                    item.add_marker(skip)
