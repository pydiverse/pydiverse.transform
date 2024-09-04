from __future__ import annotations

from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.backend.targets import DuckDb, Target


class DuckDbImpl(SqlImpl):
    @staticmethod
    def backend_marker() -> Target:
        return DuckDb()
