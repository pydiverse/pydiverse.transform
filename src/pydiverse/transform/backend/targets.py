# This module defines the config classes provided to the user to configure
# the backend on import / export.


from __future__ import annotations

import sqlalchemy as sqa


# TODO: better name for this? (the user sees this)
class Target: ...


class Polars(Target):
    def __init__(self, *, lazy: bool = True) -> None:
        self.lazy = lazy


class DuckDb(Target): ...


class SqlAlchemy(Target):
    def __init__(self, engine: sqa.Engine, *, schema: str | None = None):
        self.engine = engine
        self.schema = schema
