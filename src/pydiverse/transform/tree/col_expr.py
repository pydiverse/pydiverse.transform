from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable
from typing import Any, Generic

from pydiverse.transform._typing import ImplT
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
    dtype: DType | None = None

    def _expr_repr(self) -> str:
        """String repr that, when executed, returns the same expression"""
        raise NotImplementedError

    def __getattr__(self, item) -> FnAttr:
        return FnAttr(item, self)

    __contains__ = None
    __iter__ = None

    def __bool__(self):
        raise TypeError(
            "cannot call __bool__() on a ColExpr. hint: A ColExpr cannot be "
            "converted to a boolean or used with the and, or, not keywords"
        )


class Col(ColExpr, Generic[ImplT]):
    def __init__(self, name: str, table: TableExpr, dtype: DType | None = None) -> Col:
        self.name = name
        self.table = table
        self.dtype = dtype

    def __repr__(self):
        return f"<{self.table.name}.{self.name}>"

    def _expr_repr(self) -> str:
        return f"{self.table.name}.{self.name}"


class ColName(ColExpr):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<C.{self.name}>"

    def _expr_repr(self) -> str:
        return f"C.{self.name}"


class LiteralCol(ColExpr):
    def __init__(self, val: Any):
        self.val = val
        self.dtype = python_type_to_pdt(type(val))

    def __repr__(self):
        return f"<Lit: {self.expr} ({self.typed_value.dtype})>"

    def _expr_repr(self) -> str:
        return repr(self)


class ColFn(ColExpr):
    def __init__(self, name: str, *args: ColExpr, **kwargs: ColExpr):
        self.name = name
        self.args = args
        self.context_kwargs = kwargs

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


class CaseExpr(ColExpr):
    def __init__(
        self, switching_on: Any | None, cases: Iterable[tuple[Any, Any]], default: Any
    ):
        self.switching_on = switching_on
        self.cases = list(cases)
        self.default = default

    def __repr__(self):
        if self.switching_on:
            return f"case({self.switching_on}, {self.cases}, default={self.default})"
        else:
            return f"case({self.cases}, default={self.default})"

    def _expr_repr(self) -> str:
        prefix = "f"
        if self.switching_on:
            prefix = expr_repr(self.switching_on)

        args = [expr_repr(case) for case in self.cases]
        args.append(f"default={expr_repr(self.default)}")
        return f"{prefix}.case({', '.join(args)})"

    def iter_children(self):
        if self.switching_on:
            yield self.switching_on

        for k, v in self.cases:
            yield k
            yield v

        yield self.default


@dataclasses.dataclass
class FnAttr:
    name: str
    arg: ColExpr

    def __getattr__(self, name) -> FnAttr:
        return FnAttr(f"{self.name}.{name}", self.arg)

    def __call__(self) -> ColExpr:
        return ColFn(self.name, self.arg)


def get_needed_cols(expr: ColExpr) -> Map2d[TableExpr, set[str]]:
    if isinstance(expr, Col):
        return Map2d({expr.table: {expr.name}})
    elif isinstance(expr, ColFn):
        needed_cols = Map2d()
        for v in itertools.chain(expr.args, expr.context_kwargs.values()):
            needed_cols.inner_update(get_needed_cols(v))
        return needed_cols
    elif isinstance(expr, CaseExpr):
        raise NotImplementedError
    elif isinstance(expr, LiteralCol):
        return Map2d()
    return Map2d()


def propagate_names(
    expr: ColExpr, col_to_name: Map2d[TableExpr, dict[str, str]]
) -> ColExpr:
    if isinstance(expr, Col):
        return ColName(col_to_name[expr.table][expr.name])
    elif isinstance(expr, ColFn):
        expr.args = [propagate_names(arg, col_to_name) for arg in expr.args]
        expr.context_kwargs = {
            key: [propagate_names(v, col_to_name) for v in arr]
            for key, arr in expr.context_kwargs
        }
    elif isinstance(expr, CaseExpr):
        raise NotImplementedError

    return expr


def propagate_types(expr: ColExpr, col_types: dict[str, DType]) -> ColExpr:
    assert not isinstance(expr, Col)
    if isinstance(expr, ColName):
        expr.dtype = col_types[expr.name]
        return expr
    elif isinstance(expr, ColFn):
        expr.args = [propagate_types(arg, col_types) for arg in expr.args]
        expr.context_kwargs = {
            key: [propagate_types(v, col_types) for v in arr]
            for key, arr in expr.context_kwargs
        }
        # TODO: create a backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        expr.dtype = PolarsImpl.operator_registry.get_implementation(
            expr.name, [arg.dtype for arg in expr.args]
        ).return_type
        return expr
    elif isinstance(expr, LiteralCol):
        expr.dtype = python_type_to_pdt(type(expr))
        return expr
    else:
        return LiteralCol(expr)


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
