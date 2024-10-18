from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
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
from pydiverse.transform._internal.tree.col_expr import Col
from pydiverse.transform._internal.tree.types import Dtype

if TYPE_CHECKING:
    from pydiverse.transform._internal.ops.ops import Operator


class TableImpl(AstNode):
    impl_store: ImplStore

    def __init__(self, name: str, schema: dict[str, Dtype]):
        self.name = name
        self.cols = {
            name: Col(name, self, uuid.uuid1(), dtype, Ftype.ELEMENT_WISE)
            for name, dtype in schema.items()
        }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Add new `registry` class variable to subclass.
        # We define the super registry by walking up the MRO. This allows us
        # to check for potential operation definitions in the parent classes.
        super_reg = None
        for super_cls in cls.__mro__:
            if hasattr(super_cls, "registry"):
                super_reg = super_cls.registry
                break
        cls.impl_store = ImplStore(cls, super_reg)

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

        elif isinstance(resource, pl.DataFrame | pl.LazyFrame):
            if name is None:
                # If the data frame has be previously exported by transform, a
                # name attribute was added.
                if hasattr(resource, "name"):
                    name = resource.name
                else:
                    name = "?"
            if backend is None or isinstance(backend, Polars):
                from pydiverse.transform._internal.backend.polars import PolarsImpl

                res = PolarsImpl(name, resource)
            elif isinstance(backend, DuckDb):
                from pydiverse.transform._internal.backend.duckdb_polars import (
                    DuckDbPolarsImpl,
                )

                res = DuckDbPolarsImpl(name, resource)

        elif isinstance(resource, str | sqa.Table):
            if isinstance(backend, SqlAlchemy):
                from pydiverse.transform._internal.backend.sql import SqlImpl

                res = SqlImpl(resource, backend, name)

        else:
            raise AssertionError

        if uuids is not None:
            for name, col in res.cols.items():
                col._uuid = uuids[name]

        return res

    def iter_subtree(self) -> Iterable[AstNode]:
        yield self

    @classmethod
    def build_query(cls, nd: AstNode, final_select: list[Col]) -> str | None: ...

    @classmethod
    def export(cls, nd: AstNode, target: Target, final_select: list[Col]) -> Any: ...

    @classmethod
    def get_impl(cls, op: Operator, sig: Sequence[Dtype]) -> Any:
        if (impl := cls.impl_store.get_impl(op, sig)) is not None:
            return impl
        if cls is TableImpl:
            raise Exception

        try:
            super().get_impl(op, sig)
        except Exception as err:
            raise NotSupportedError(
                f"operation `{op.name}` is not supported by the backend "
                f"`{cls.__name__.lower()[:-4]}`"
            ) from err


with TableImpl.impl_store.impl_manager as cm:

    @cm(ops.add)
    def _nulls_first(_):
        raise AssertionError


with TableImpl.op(ops.NullsLast()) as op:

    @op.auto
    def _nulls_last(_):
        raise AssertionError


with TableImpl.op(ops.Ascending()) as op:

    @op.auto
    def _ascending(_):
        raise AssertionError


with TableImpl.op(ops.Descending()) as op:

    @op.auto
    def _descending(_):
        raise AssertionError


with TableImpl.op(ops.Add()) as op:

    @op.auto
    def _add(lhs, rhs):
        return lhs + rhs

    @op.extension(ops.StrAdd)
    def _str_add(lhs, rhs):
        return lhs + rhs


with TableImpl.op(ops.Sub()) as op:

    @op.auto
    def _sub(lhs, rhs):
        return lhs - rhs


with TableImpl.op(ops.Mul()) as op:

    @op.auto
    def _mul(lhs, rhs):
        return lhs * rhs


with TableImpl.op(ops.TrueDiv()) as op:

    @op.auto
    def _truediv(lhs, rhs):
        return lhs / rhs


with TableImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return lhs // rhs


with TableImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        return lhs**rhs


with TableImpl.op(ops.Mod()) as op:

    @op.auto
    def _mod(lhs, rhs):
        return lhs % rhs


with TableImpl.op(ops.Neg()) as op:

    @op.auto
    def _neg(x):
        return -x


with TableImpl.op(ops.Pos()) as op:

    @op.auto
    def _pos(x):
        return +x


with TableImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return abs(x)


with TableImpl.op(ops.And()) as op:

    @op.auto
    def _and(lhs, rhs):
        return lhs & rhs


with TableImpl.op(ops.Or()) as op:

    @op.auto
    def _or(lhs, rhs):
        return lhs | rhs


with TableImpl.op(ops.Xor()) as op:

    @op.auto
    def _xor(lhs, rhs):
        return lhs ^ rhs


with TableImpl.op(ops.Invert()) as op:

    @op.auto
    def _invert(x):
        return ~x


with TableImpl.op(ops.Equal()) as op:

    @op.auto
    def _eq(lhs, rhs):
        return lhs == rhs


with TableImpl.op(ops.NotEqual()) as op:

    @op.auto
    def _ne(lhs, rhs):
        return lhs != rhs


with TableImpl.op(ops.Less()) as op:

    @op.auto
    def _lt(lhs, rhs):
        return lhs < rhs


with TableImpl.op(ops.LessEqual()) as op:

    @op.auto
    def _le(lhs, rhs):
        return lhs <= rhs


with TableImpl.op(ops.Greater()) as op:

    @op.auto
    def _gt(lhs, rhs):
        return lhs > rhs


with TableImpl.op(ops.GreaterEqual()) as op:

    @op.auto
    def _ge(lhs, rhs):
        return lhs >= rhs
