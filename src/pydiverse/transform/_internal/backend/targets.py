# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

# This module defines the config classes provided to the user to configure
# the backend on import / export.


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


class Dict(Target): ...


class DictOfLists(Target): ...


class ListOfDicts(Target): ...
