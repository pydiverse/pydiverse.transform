from __future__ import annotations

from typing import Any
from uuid import UUID

import duckdb
import duckdb_engine
import polars as pl
import sqlalchemy as sqa

from pydiverse.common import Dtype
from pydiverse.transform._internal.backend.duckdb import DuckDbImpl
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.tree.ast import AstNode


# TODO: we should move the engine of SqlImpl in the subclasses and let this thing
# inherit from SqlImpl in order to make the usage of SqlImpl.compile_ast more clean.
# Currently it works only since this class also has a table object, but it should be
# enforced by inheritance.
class DuckDbPolarsImpl(TableImpl):
    backend_name = "polars"

    def __init__(self, name: str, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()

        super().__init__(
            name,
            {
                name: Dtype.from_polars(dtype)
                for name, dtype in df.collect_schema().items()
            },
        )

        self.table = sqa.Table(
            name,
            sqa.MetaData(),
            *(
                sqa.Column(col.name, DuckDbImpl.sqa_type(col.dtype()))
                for col in self.cols.values()
            ),
        )

    @staticmethod
    def build_query(nd: AstNode) -> str | None:
        return DuckDbImpl.build_query(nd, dialect=duckdb_engine.Dialect())

    @staticmethod
    def export(
        nd: AstNode,
        target: Target,
        *,
        schema_overrides: dict[UUID, Any],  # TODO: use this
    ) -> pl.DataFrame:
        if isinstance(target, Polars):
            sel = DuckDbImpl.build_select(nd)
            query_str = str(
                sel.compile(
                    dialect=duckdb_engine.Dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )

            # tell duckdb which table names in the SQL query correspond to which
            # data frames
            for desc in nd.iter_subtree_postorder():
                if isinstance(desc, DuckDbPolarsImpl):
                    duckdb.register(desc.table.name, desc.df)

            return duckdb.sql(query_str).pl()

        raise AssertionError

    def _clone(self) -> tuple[AstNode, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned = DuckDbPolarsImpl(self.name, self.df)
        return (
            cloned,
            {self: cloned},
            {
                self.cols[name]._uuid: cloned.cols[name]._uuid
                for name in self.cols.keys()
            },
        )
