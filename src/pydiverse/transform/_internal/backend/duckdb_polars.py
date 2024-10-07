from __future__ import annotations

from uuid import UUID

import duckdb
import duckdb_engine
import polars as pl
import sqlalchemy as sqa

from pydiverse.transform._internal.backend.duckdb import DuckDbImpl
from pydiverse.transform._internal.backend.polars import polars_type_to_pdt
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col


# TODO: we should move the engine of SqlImpl in the subclasses and let this thing
# inherit from SqlImpl in order to make the usage of SqlImpl.compile_ast more clean.
# Currently it works only since this class also has a table object, but it should be
# enforced by inheritance.
class DuckDbPolarsImpl(TableImpl):
    def __init__(self, name: str, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()

        super().__init__(
            name,
            {
                name: polars_type_to_pdt(dtype)
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
    def build_query(nd: AstNode, final_select: list[Col]) -> str | None:
        return DuckDbImpl.build_query(nd, final_select)

    @staticmethod
    def export(nd: AstNode, target: Target, final_select: list[Col]) -> pl.DataFrame:
        if isinstance(target, Polars):
            sel = DuckDbImpl.build_select(nd, final_select)
            query_str = str(
                sel.compile(
                    dialect=duckdb_engine.Dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )

            # tell duckdb which table names in the SQL query correspond to which
            # data frames
            for desc in nd.iter_subtree():
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
