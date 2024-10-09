from __future__ import annotations

from itertools import chain

import pytest

# Pytest markers associated with specific backend
BACKEND_MARKS = {
    "postgres": pytest.mark.postgres,
    "mssql": pytest.mark.mssql,
    "ibm_db2": pytest.mark.ibm_db2,
}


def with_backends(*backends):
    """Decorator to run a test with a specific set of backends

    :param backends: Names of the backends to use.
    """
    return pytest.mark.backends(*flatten(backends))


def skip_backends(*backends):
    """Decorator to skip running a test with a specific set of backends"""
    return pytest.mark.skip_backends(*flatten(backends))


def flatten(it):
    """Flatten an iterable"""
    if isinstance(it, list | tuple):
        yield from chain(*map(flatten, it))
    else:
        yield it
