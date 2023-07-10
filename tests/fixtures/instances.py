from __future__ import annotations

import pytest

__all__ = [
    "INSTANCE_MARKS",
]


# Pytest markers associated with specific instance name
INSTANCE_MARKS = {
    "postgres": pytest.mark.postgres,
    "mssql": pytest.mark.mssql,
    "ibm_db2": pytest.mark.ibm_db2,
}
