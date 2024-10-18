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


with TableImpl.impl_store.impl_manager as impl:

    @impl(ops.nulls_first)
    def _nulls_first(_):
        raise AssertionError

    @impl(ops.nulls_last)
    def _nulls_last(_):
        raise AssertionError

    @impl(ops.ascending)
    def _ascending(_):
        raise AssertionError

    @impl(ops.descending)
    def _descending(_):
        raise AssertionError

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

    @impl(ops.lt)
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
