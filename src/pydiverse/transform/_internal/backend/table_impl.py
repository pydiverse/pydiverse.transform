# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import functools
import operator
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

    def __init__(self, name: str | None, schema: dict[str, Dtype]):
        self.name = name
        self.cols = {name: Col(name, self, uuid.uuid1(), dtype, Ftype.ELEMENT_WISE) for name, dtype in schema.items()}

    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map) -> str:
        return self._ast_node_repr(expr_depth, display_name_map)

    def _ast_node_repr(self, expr_depth, display_name_map):
        return display_name_map[self]

    def _table_def_repr(self) -> str:
        raise NotImplementedError()

    def short_name(self):
        return ("?" if self.name is None else self.name) + f" (source table, backend: '{self.backend_name}')"

    def __init_subclass__(cls) -> None:
        cls.impl_store = ImplStore()

    @staticmethod
    def from_resource(
        resource: Any,
        backend: Target | None = None,
        *,
        name: str | None = None,
        uuids: dict[str, UUID] | None = None,
    ) -> "TableImpl":
        from pydiverse.transform._internal.backend.targets import (
            DuckDb,
            Polars,
            SqlAlchemy,
        )

        if isinstance(resource, AstNode):
            res = resource

        elif isinstance(resource, dict):
            return TableImpl.from_resource(pl.DataFrame(resource), backend, name=name, uuids=uuids)

        elif pd is not None and isinstance(resource, pd.DataFrame):
            # copy pandas dataframe to polars
            # TODO: try zero-copy for arrow backed pandas
            return TableImpl.from_resource(pl.DataFrame(resource), backend, name=name, uuids=uuids)

        elif isinstance(resource, pl.DataFrame | pl.LazyFrame):
            if name is None:
                # If the data frame has be previously exported by transform, a
                # name attribute was added.
                if hasattr(resource, "name"):
                    name = resource.name
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

    def iter_subtree_preorder(self) -> Iterable[AstNode]:
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
    def get_impl(cls, op: "Operator", sig: Sequence[Dtype]) -> Any:
        if (impl := cls.impl_store.get_impl(op, sig)) is not None:
            return impl

        if cls is TableImpl:
            raise NotSupportedError

        try:
            return cls.__bases__[0].get_impl(op, sig)
        except NotSupportedError as err:
            raise NotSupportedError(
                f"operation `{op.name}` is not supported by the backend `{cls.__name__.lower()[:-4]}`"
            ) from err


def split_join_cond(on: ColFn) -> list[ColFn]:
    if isinstance(on, LiteralCol):
        return []
    elif on.op == ops.bool_and:
        return split_join_cond(on.args[0]) + split_join_cond(on.args[1])
    elif on.op == ops.horizontal_all:
        return functools.reduce(operator.add, (split_join_cond(arg) for arg in on.args))
    else:
        return [on]


# Returns the left and right columns of a list of equality predicates.
def get_left_right_on(
    eq_predicates: list[ColFn], left_uuids: set[UUID], right_uuids: set[UUID]
) -> tuple[list[Col], list[Col]] | None:
    left_on = []
    right_on = []
    for pred in eq_predicates:
        left_on.append(pred.args[0])
        right_on.append(pred.args[1])

        must_swap_cols = None
        for e in pred.args[0].iter_subtree_postorder():
            if isinstance(e, Col):
                must_swap_cols = e._uuid in right_uuids
                assert must_swap_cols or e._uuid in left_uuids
                break

        assert must_swap_cols is not None

        if must_swap_cols:
            left_on[-1], right_on[-1] = right_on[-1], left_on[-1]

    return (left_on, right_on)


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
