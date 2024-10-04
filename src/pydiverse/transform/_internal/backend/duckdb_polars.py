from __future__ import annotations

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

    @classmethod
    def export(nd: AstNode, target: Target, final_select: list[Col]) -> pl.DataFrame:
        if isinstance(target, Polars):
            sel = DuckDbImpl.build_select(nd, final_select)
            query_str = str(
                sel.compile(
                    dialect=duckdb_engine.Dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )

            # put all the source dfs in global scope
            for desc in nd.iter_subtree():
                if isinstance(desc, DuckDbPolarsImpl):
                    globals()[desc.table.name] = desc.df

            return duckdb.sql(query_str).pl()

        raise AssertionError
