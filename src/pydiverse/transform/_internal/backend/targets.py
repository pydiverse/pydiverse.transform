# This module defines the config classes provided to the user to configure
# the backend on import / export.


from __future__ import annotations

import sqlalchemy as sqa


class Target: ...


class Polars(Target):
    def __init__(self, *, lazy: bool = False) -> None:
        self.lazy = lazy


class Pandas(Target): ...


class DuckDb(Target): ...


class SqlAlchemy(Target):
    def __init__(self, engine: sqa.Engine, *, schema: str | None = None):
        self.engine = engine
        self.schema = schema


class Scalar(Target): ...
