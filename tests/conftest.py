from __future__ import annotations

import itertools
import logging
import os
from collections import defaultdict

import pytest

from tests.fixtures.instances import INSTANCE_MARKS

pytest_plugins = ["tests.parallelize.plugin"]


# Setup

log_level = logging.INFO if not os.environ.get("DEBUG", "") else logging.DEBUG
logging.basicConfig(level=log_level)


def pytest_addoption(parser):
    for opt in INSTANCE_MARKS.keys():
        parser.addoption(
            "--" + opt,
            action="store_true",
            default=False,
            help=f"run test that require {opt}",
        )


def pytest_collection_modifyitems(config: pytest.Config, items):
    for opt in INSTANCE_MARKS.keys():
        if not config.getoption("--" + opt):
            skip = pytest.mark.skip(reason=f"{opt} not selected")
            for item in items:
                if opt in item.keywords:
                    item.add_marker(skip)


@pytest.hookimpl
def pytest_parallelize_group_items(config, items):
    groups = defaultdict(list)
    auto_group_iter = itertools.cycle([f"auto_{i}" for i in range(os.cpu_count() or 4)])
    for item in items:
        group = "DEFAULT"

        if hasattr(item, "get_closest_marker"):
            if marker := item.get_closest_marker("parallelize"):
                if marker.args:
                    group = marker.args[0]
                else:
                    group = next(auto_group_iter)

        groups[group].append(item)
    return groups
