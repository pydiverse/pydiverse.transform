from __future__ import annotations

import dataclasses
import functools
import itertools
import operator
from collections.abc import Iterable
from typing import Any, Generic

from pydiverse.transform._typing import ImplT
from pydiverse.transform.ops.core import OpType
from pydiverse.transform.tree import dtypes
from pydiverse.transform.tree.dtypes import DType, python_type_to_pdt
from pydiverse.transform.tree.registry import OperatorRegistry
from pydiverse.transform.tree.table_expr import TableExpr
from pydiverse.transform.util import Map2d


def expr_repr(it: Any):
    if isinstance(it, ColExpr):
        return it._expr_repr()
    if isinstance(it, (list, tuple)):
        return f"[{ ', '.join([expr_repr(e) for e in it]) }]"
    return repr(it)


_dunder_expr_repr = {
    "__add__": lambda lhs, rhs: f"({lhs} + {rhs})",
    "__radd__": lambda rhs, lhs: f"({lhs} + {rhs})",
    "__sub__": lambda lhs, rhs: f"({lhs} - {rhs})",
    "__rsub__": lambda rhs, lhs: f"({lhs} - {rhs})",
    "__mul__": lambda lhs, rhs: f"({lhs} * {rhs})",
    "__rmul__": lambda rhs, lhs: f"({lhs} * {rhs})",
    "__truediv__": lambda lhs, rhs: f"({lhs} / {rhs})",
    "__rtruediv__": lambda rhs, lhs: f"({lhs} / {rhs})",
    "__floordiv__": lambda lhs, rhs: f"({lhs} // {rhs})",
    "__rfloordiv__": lambda rhs, lhs: f"({lhs} // {rhs})",
    "__pow__": lambda lhs, rhs: f"({lhs} ** {rhs})",
    "__rpow__": lambda rhs, lhs: f"({lhs} ** {rhs})",
    "__mod__": lambda lhs, rhs: f"({lhs} % {rhs})",
    "__rmod__": lambda rhs, lhs: f"({lhs} % {rhs})",
    "__round__": lambda x, y=None: f"round({x}, {y})" if y else f"round({x})",
    "__pos__": lambda x: f"(+{x})",
    "__neg__": lambda x: f"(-{x})",
    "__abs__": lambda x: f"abs({x})",
    "__and__": lambda lhs, rhs: f"({lhs} & {rhs})",
    "__rand__": lambda rhs, lhs: f"({lhs} & {rhs})",
    "__or__": lambda lhs, rhs: f"({lhs} | {rhs})",
    "__ror__": lambda rhs, lhs: f"({lhs} | {rhs})",
    "__xor__": lambda lhs, rhs: f"({lhs} ^ {rhs})",
    "__rxor__": lambda rhs, lhs: f"({lhs} ^ {rhs})",
    "__invert__": lambda x: f"(~{x})",
    "__lt__": lambda lhs, rhs: f"({lhs} < {rhs})",
    "__le__": lambda lhs, rhs: f"({lhs} <= {rhs})",
    "__eq__": lambda lhs, rhs: f"({lhs} == {rhs})",
    "__ne__": lambda lhs, rhs: f"({lhs} != {rhs})",
    "__gt__": lambda lhs, rhs: f"({lhs} > {rhs})",
    "__ge__": lambda lhs, rhs: f"({lhs} >= {rhs})",
}


class ColExpr:
    __slots__ = ["dtype"]

    __contains__ = None
    __iter__ = None

    def __init__(self, dtype: DType | None = None):
        self.dtype = dtype

    def _expr_repr(self) -> str:
        """String repr that, when executed, returns the same expression"""
        raise NotImplementedError

    def __getattr__(self, name: str) -> FnAttr:
        if name.startswith("_") and name.endswith("_"):
            # that hasattr works correctly
            raise AttributeError(f"`ColExpr` has no attribute `{name}`")
        return FnAttr(name, self)

    def __bool__(self):
        raise TypeError(
            "cannot call __bool__() on a ColExpr. hint: A ColExpr cannot be "
            "converted to a boolean or used with the and, or, not keywords"
        )


class Col(ColExpr, Generic[ImplT]):
    def __init__(self, name: str, table: TableExpr, dtype: DType | None = None) -> Col:
        self.name = name
        self.table = table
        super().__init__(dtype)

    def __repr__(self):
        return f"<{self.table.name}.{self.name}>"

    def _expr_repr(self) -> str:
        return f"{self.table.name}.{self.name}"


class ColName(ColExpr):
    def __init__(self, name: str, dtype: DType | None = None):
        self.name = name
        super().__init__(dtype)

    def __repr__(self):
        return f"<C.{self.name}>"

    def _expr_repr(self) -> str:
        return f"C.{self.name}"


class LiteralCol(ColExpr):
    def __init__(self, val: Any):
        self.val = val
        super().__init__(python_type_to_pdt(type(val)))

    def __repr__(self):
        return f"<Lit: {self.expr} ({self.typed_value.dtype})>"

    def _expr_repr(self) -> str:
        return repr(self)


class ColFn(ColExpr):
    def __init__(self, name: str, *args: ColExpr, **kwargs: list[ColExpr]):
        self.name = name
        self.args = list(args)
        self.context_kwargs = {
            key: val for key, val in kwargs.items() if val is not None
        }
        super().__init__()

    def __repr__(self):
        args = [repr(e) for e in self.args] + [
            f"{k}={repr(v)}" for k, v in self.context_kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def _expr_repr(self) -> str:
        args = [expr_repr(e) for e in self.args] + [
            f"{k}={expr_repr(v)}" for k, v in self.context_kwargs.items()
        ]

        if self.name in _dunder_expr_repr:
            return _dunder_expr_repr[self.name](*args)

        if len(self.args) == 0:
            args_str = ", ".join(args)
            return f"f.{self.name}({args_str})"
        else:
            args_str = ", ".join(args[1:])
            return f"{args[0]}.{self.name}({args_str})"


class WhenClause:
    def __init__(self, cases: list[tuple[ColExpr, ColExpr]], cond: ColExpr):
        self.cases = cases
        self.cond = cond

    def then(self, value: ColExpr) -> CaseExpr:
        return CaseExpr((*self.cases, (self.cond, value)))


class CaseExpr(ColExpr):
    def __init__(
        self,
        cases: Iterable[tuple[ColExpr, ColExpr]],
        default_val: ColExpr | None = None,
    ):
        self.cases = list(cases)
        self.default_val = default_val

    def __repr__(self):
        return (
            "case("
            + functools.reduce(
                operator.add, (f"{cond} -> {val}, " for cond, val in self.cases), ""
            )
            + f"otherwise={self.default_val})"
        )

    def _expr_repr(self) -> str:
        prefix = "f"
        if self.switching_on:
            prefix = expr_repr(self.switching_on)

        args = [expr_repr(case) for case in self.cases]
        args.append(f"default={expr_repr(self.default)}")
        return f"{prefix}.case({', '.join(args)})"

    def when(self, condition: ColExpr) -> WhenClause:
        if self.default_val is not None:
            raise TypeError("cannot call `when` on a case expression after `otherwise`")
        return WhenClause(self.cases, condition)

    def otherwise(self, value: ColExpr) -> CaseExpr:
        if self.default_val is not None:
            raise TypeError("cannot call `otherwise` twice on a case expression")
        return CaseExpr(self.cases, value)


class Cast(ColExpr):
    def __init__(self, value: ColExpr, dtype: DType):
        self.value = value
        super().__init__(dtype)


@dataclasses.dataclass
class FnAttr:
    name: str
    arg: ColExpr

    def __getattr__(self, name) -> FnAttr:
        return FnAttr(f"{self.name}.{name}", self.arg)

    def __call__(self, *args, **kwargs) -> ColExpr:
        return ColFn(self.name, self.arg, *args, **kwargs)


def rename_overwritten_cols(expr: ColExpr, name_map: dict[str, str]):
    if isinstance(expr, ColName):
        if expr.name in name_map:
            expr.name = name_map[expr.name]

    elif isinstance(expr, ColFn):
        for arg in expr.args:
            rename_overwritten_cols(arg, name_map)
        for val in itertools.chain.from_iterable(expr.context_kwargs.values()):
            rename_overwritten_cols(val, name_map)

    elif isinstance(expr, CaseExpr):
        rename_overwritten_cols(expr.default_val, name_map)
        for cond, val in expr.cases:
            rename_overwritten_cols(cond, name_map)
            rename_overwritten_cols(val, name_map)


def update_partition_by_kwarg(expr: ColExpr, group_by: list[Col | ColName]) -> None:
    if isinstance(expr, ColFn):
        # TODO: backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        impl = PolarsImpl.operator_registry.get_operator(expr.name)
        # TODO: what exactly are WINDOW / AGGREGATE fns? for the user? for the backend?
        if (
            impl.ftype in (OpType.WINDOW, OpType.AGGREGATE)
            and "partition_by" not in expr.context_kwargs
        ):
            expr.context_kwargs["partition_by"] = group_by

        for arg in expr.args:
            update_partition_by_kwarg(arg, group_by)
        for val in itertools.chain.from_iterable(expr.context_kwargs.values()):
            if isinstance(val, Order):
                update_partition_by_kwarg(val.order_by, group_by)
            else:
                update_partition_by_kwarg(val, group_by)

    elif isinstance(expr, CaseExpr):
        update_partition_by_kwarg(expr.default_val, group_by)
        for cond, val in expr.cases:
            update_partition_by_kwarg(cond, group_by)
            update_partition_by_kwarg(val, group_by)

    else:
        assert isinstance(expr, (Col, ColName, LiteralCol))


def get_needed_cols(expr: ColExpr | Order) -> Map2d[TableExpr, set[str]]:
    if isinstance(expr, Order):
        return get_needed_cols(expr.order_by)

    if isinstance(expr, Col):
        return Map2d({expr.table: {expr.name}})

    elif isinstance(expr, ColFn):
        needed_cols = Map2d()
        for v in itertools.chain(expr.args, expr.context_kwargs.values()):
            needed_cols.inner_update(get_needed_cols(v))
        return needed_cols

    elif isinstance(expr, CaseExpr):
        needed_cols = get_needed_cols(expr.default_val)
        for cond, val in expr.cases:
            needed_cols.inner_update(get_needed_cols(cond))
            needed_cols.inner_update(get_needed_cols(val))
        return needed_cols

    elif isinstance(expr, LiteralCol):
        return Map2d()

    return Map2d()


def propagate_names(
    expr: ColExpr | Order, col_to_name: Map2d[TableExpr, dict[str, str]]
) -> ColExpr | Order:
    if isinstance(expr, Order):
        return Order(
            propagate_names(expr.order_by, col_to_name),
            expr.descending,
            expr.nulls_last,
        )

    if isinstance(expr, Col):
        return ColName(col_to_name[expr.table][expr.name])

    elif isinstance(expr, ColFn):
        return ColFn(
            expr.name,
            *[propagate_names(arg, col_to_name) for arg in expr.args],
            **{
                key: [propagate_names(v, col_to_name) for v in arr]
                for key, arr in expr.context_kwargs.items()
            },
        )

    elif isinstance(expr, CaseExpr):
        return CaseExpr(
            [
                (propagate_names(cond, col_to_name), propagate_names(val, col_to_name))
                for cond, val in expr.cases
            ],
            propagate_names(expr.default_val, col_to_name),
        )

    return expr


def propagate_types(expr: ColExpr, col_types: dict[str, DType]) -> ColExpr:
    assert not isinstance(expr, Col)
    if isinstance(expr, Order):
        return Order(
            propagate_types(expr.order_by, col_types), expr.descending, expr.nulls_last
        )

    elif isinstance(expr, ColName):
        return ColName(expr.name, col_types[expr.name])

    elif isinstance(expr, ColFn):
        typed_fn = ColFn(
            expr.name,
            *(propagate_types(arg, col_types) for arg in expr.args),
            **{
                key: [propagate_types(val, col_types) for val in arr]
                for key, arr in expr.context_kwargs.items()
            },
        )

        # TODO: create a backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        typed_fn.dtype = PolarsImpl.operator_registry.get_implementation(
            expr.name, [arg.dtype for arg in typed_fn.args]
        ).return_type
        return typed_fn

    elif isinstance(expr, CaseExpr):
        typed_cases: list[tuple[ColExpr, ColExpr]] = []
        for cond, val in expr.cases:
            typed_cases.append(
                (propagate_types(cond, col_types), propagate_types(val, col_types))
            )
            # TODO: error message, check that the value types of all cases and the
            # default match
            assert isinstance(typed_cases[-1][0].dtype, dtypes.Bool)
        return CaseExpr(typed_cases, propagate_types(expr.default_val, col_types))

    elif isinstance(expr, LiteralCol):
        return expr  # TODO: can literal columns even occur here?

    else:  # TODO: add type checking. check if it is one of the supported builtins
        return LiteralCol(expr)


def clone(expr: ColExpr, table_map: dict[TableExpr, TableExpr]) -> ColExpr:
    if isinstance(expr, Order):
        return Order(clone(expr.order_by, table_map), expr.descending, expr.nulls_last)

    if isinstance(expr, Col):
        return Col(expr.name, table_map[expr.table], expr.dtype)

    elif isinstance(expr, ColName):
        return ColName(expr.name, expr.dtype)

    elif isinstance(expr, LiteralCol):
        return LiteralCol(expr.val)

    elif isinstance(expr, ColFn):
        return ColFn(
            expr.name,
            *(clone(arg, table_map) for arg in expr.args),
            **{
                kwarg: [clone(val, table_map) for val in arr]
                for kwarg, arr in expr.context_kwargs.items()
            },
        )

    elif isinstance(expr, CaseExpr):
        return CaseExpr(
            [
                (clone(cond, table_map), clone(val, table_map))
                for cond, val in expr.cases
            ],
            clone(expr.default_val, table_map),
        )

    else:
        return expr


@dataclasses.dataclass
class Order:
    order_by: ColExpr
    descending: bool
    nulls_last: bool

    # the given `expr` may contain nulls_last markers or `-` (descending markers). the
    # order_by of the Order does not contain these special functions and can thus be
    # translated normally.
    @staticmethod
    def from_col_expr(expr: ColExpr) -> Order:
        descending = False
        nulls_last = None
        while isinstance(expr, ColFn):
            if expr.name == "__neg__":
                descending = not descending
            elif nulls_last is None:
                if expr.name == "nulls_last":
                    nulls_last = True
                elif expr.name == "nulls_first":
                    nulls_last = False
            if expr.name in ("__neg__", "__pos__", "nulls_last", "nulls_first"):
                assert len(expr.args) == 1
                assert len(expr.context_kwargs) == 0
                expr = expr.args[0]
            else:
                break
        if nulls_last is None:
            nulls_last = False
        return Order(expr, descending, nulls_last)


# Add all supported dunder methods to `ColExpr`. This has to be done, because Python
# doesn't call __getattr__ for dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return ColFn(op, *args, **kwargs)

    return impl


for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(ColExpr, dunder, create_operator(dunder))
del create_operator
