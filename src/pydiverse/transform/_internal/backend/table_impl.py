from __future__ import annotations

import functools
import uuid
from collections.abc import Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

import polars as pl
import sqlalchemy as sqa

from pydiverse.transform._internal.backend.impl_store import ImplStore
from pydiverse.transform._internal.backend.targets import Target
from pydiverse.transform._internal.errors import NotSupportedError
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColFn, LiteralCol
from pydiverse.transform._internal.tree.types import Dtype

try:
    import pandas as pd
except ImportError:
    pd = None

if TYPE_CHECKING:
    from pydiverse.transform._internal.ops.ops import Operator


class TableImpl(AstNode):
    backend_name: str
    impl_store = ImplStore()

    def __init__(self, name: str, schema: dict[str, Dtype]):
        self.name = name
        self.cols = {
            name: Col(name, self, uuid.uuid1(), dtype, Ftype.ELEMENT_WISE)
            for name, dtype in schema.items()
        }

    def __init_subclass__(cls) -> None:
        cls.impl_store = ImplStore()

    @staticmethod
    def from_resource(
        resource: Any,
        backend: Target | None = None,
        *,
        name: str | None = None,
        uuids: dict[str, UUID] | None = None,
    ) -> TableImpl:
        from pydiverse.transform._internal.backend.targets import (
            DuckDb,
            Polars,
            SqlAlchemy,
        )

        if isinstance(resource, TableImpl):
            res = resource

        elif isinstance(resource, dict):
            return TableImpl.from_resource(
                pl.DataFrame(resource), backend, name=name, uuids=uuids
            )

        elif pd is not None and isinstance(resource, pd.DataFrame):
            # copy pandas dataframe to polars
            # TODO: try zero-copy for arrow backed pandas
            return TableImpl.from_resource(
                pl.DataFrame(resource), backend, name=name, uuids=uuids
            )

        elif isinstance(resource, pl.DataFrame | pl.LazyFrame):
            if name is None:
                # If the data frame has be previously exported by transform, a
                # name attribute was added.
                if hasattr(resource, "name"):
                    name = resource.name
                else:
                    name = "<unnamed>"
            if backend is None or isinstance(backend, Polars):
                from pydiverse.transform._internal.backend.polars import PolarsImpl

                res = PolarsImpl(name, resource)
            elif isinstance(backend, DuckDb):
                from pydiverse.transform._internal.backend.duckdb_polars import (
                    DuckDbPolarsImpl,
                )

                res = DuckDbPolarsImpl(name, resource)

        elif isinstance(resource, str | sqa.Table):
            if not isinstance(backend, SqlAlchemy):
                raise TypeError(
                    "If `resource` is a string or a SQLAlchemy table, `backend` must "
                    "have type `SqlALchemy` and contain an engine."
                )

            from pydiverse.transform._internal.backend.sql import SqlImpl

            res = SqlImpl(resource, backend, name)

        else:
            raise AssertionError

        if uuids is not None:
            for name, col in res.cols.items():
                col._uuid = uuids[name]

        return res

    def iter_subtree_postorder(self) -> Iterable[AstNode]:
        yield self

    def iter_subtree_preorder(self) -> Generator[AstNode, bool | None, None]:
        yield self

    @classmethod
    def build_query(cls, nd: AstNode) -> str | None: ...

    @classmethod
    def export(
        cls,
        nd: AstNode,
        target: Target,
        *,
        schema_overrides: dict[UUID, Any],
    ) -> Any: ...

    @classmethod
    def get_impl(cls, op: Operator, sig: Sequence[Dtype]) -> Any:
        if (impl := cls.impl_store.get_impl(op, sig)) is not None:
            return impl

        if cls is TableImpl:
            raise NotSupportedError

        try:
            return cls.__bases__[0].get_impl(op, sig)
        except NotSupportedError as err:
            raise NotSupportedError(
                f"operation `{op.name}` is not supported by the backend "
                f"`{cls.__name__.lower()[:-4]}`"
            ) from err


def get_backend(nd: AstNode) -> type[TableImpl]:
    from pydiverse.transform._internal.tree.verbs import Verb

    if isinstance(nd, Verb):
        return get_backend(nd.child)
    assert isinstance(nd, TableImpl) and nd is not TableImpl
    return nd.__class__


def split_join_cond(expr: ColFn) -> list[ColFn]:
    assert isinstance(expr, ColFn | LiteralCol)
    if isinstance(expr, LiteralCol):
        return []
    elif expr.op == ops.bool_and:
        return split_join_cond(expr.args[0]) + split_join_cond(expr.args[1])
    else:
        return [expr]


with TableImpl.impl_store.impl_manager as impl:

    @impl(ops.add)
    def _add(lhs, rhs):
        return lhs + rhs

    @impl(ops.sub)
    def _sub(lhs, rhs):
        return lhs - rhs

    @impl(ops.mul)
    def _mul(lhs, rhs):
        return lhs * rhs

    @impl(ops.truediv)
    def _truediv(lhs, rhs):
        return lhs / rhs

    @impl(ops.floordiv)
    def _floordiv(lhs, rhs):
        return lhs // rhs

    @impl(ops.pow)
    def _pow(lhs, rhs):
        return lhs**rhs

    @impl(ops.mod)
    def _mod(lhs, rhs):
        return lhs % rhs

    @impl(ops.neg)
    def _neg(x):
        return -x

    @impl(ops.pos)
    def _pos(x):
        return +x

    @impl(ops.abs)
    def _abs(x):
        return abs(x)

    @impl(ops.bool_and)
    def _and(lhs, rhs):
        return lhs & rhs

    @impl(ops.bool_or)
    def _or(lhs, rhs):
        return lhs | rhs

    @impl(ops.bool_xor)
    def _xor(lhs, rhs):
        return lhs ^ rhs

    @impl(ops.bool_invert)
    def _invert(x):
        return ~x

    @impl(ops.equal)
    def _eq(lhs, rhs):
        return lhs == rhs

    @impl(ops.not_equal)
    def _ne(lhs, rhs):
        return lhs != rhs

    @impl(ops.less_than)
    def _lt(lhs, rhs):
        return lhs < rhs

    @impl(ops.less_equal)
    def _le(lhs, rhs):
        return lhs <= rhs

    @impl(ops.greater_than)
    def _gt(lhs, rhs):
        return lhs > rhs

    @impl(ops.greater_equal)
    def _ge(lhs, rhs):
        return lhs >= rhs

    @impl(ops.horizontal_all)
    def _horizontal_all(*args):
        return functools.reduce(_and, args)

    @impl(ops.horizontal_any)
    def _horizontal_any(*args):
        return functools.reduce(_or, args)

    @impl(ops.horizontal_sum)
    def _horizontal_sum(*args):
        return functools.reduce(_add, args)
