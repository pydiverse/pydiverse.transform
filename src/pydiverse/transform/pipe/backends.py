# This module defines the config classes provided to the user to configure
# the backend on import / export.


# TODO: better name for this? (the user sees this)
from __future__ import annotations


class Backend: ...


class Polars(Backend):
    def __init__(self, *, lazy: bool = True) -> None:
        self.lazy = lazy


class DuckDB(Backend): ...


class SqlAlchemy(Backend): ...
