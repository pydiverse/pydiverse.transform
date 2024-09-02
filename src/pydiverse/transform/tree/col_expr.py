from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any, Generic

from pydiverse.transform._typing import ImplT, T
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.registry import OperatorRegistry
from pydiverse.transform.tree.table_expr import TableExpr


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

    def __getattr__(self, item) -> ColExpr:
        if item in ("str", "dt"):
            return FnNamespace(item, self)
        return ColFn(item, self)

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
        return f"<{self.table._impl.name}.{self.name}>"

    def _expr_repr(self) -> str:
        return f"{self.table.name}.{self.name}"

    def __eq__(self, other):
        return self.table == other.table & self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.name), hash(self.table)))


class ColName(ColExpr):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<C.{self.name}>"

    def _expr_repr(self) -> str:
        return f"C.{self.name}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("C", self.name))


class LiteralCol(ColExpr, Generic[T]):
    __slots__ = ("typed_value", "expr", "backend")

    def __init__(
        self,
        expr: Any,
    ):
        self.expr = expr

    def __repr__(self):
        return f"<Lit: {self.expr} ({self.typed_value.dtype})>"

    def _expr_repr(self) -> str:
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.typed_value == other.typed_value
            and self.expr == other.expr
            and self.backend == other.backend
        )

    def __ne__(self, other):
        return not self.__eq__(other)


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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.args, tuple(self.context_kwargs.items())))

    def iter_children(self):
        yield from self.args


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
class FnNamespace:
    name: str
    arg: ColExpr

    def __getattr__(self, name) -> ColExpr:
        return ColFn(self.name + name, self.arg)


def get_needed_tables(expr: ColExpr) -> set[TableExpr]:
    if isinstance(expr, Col):
        return set(expr.table)
    elif isinstance(expr, ColFn):
        needed_tables = set()
        for v in expr.args:
            needed_tables |= get_needed_tables(v)
        for v in expr.context_kwargs.values():
            needed_tables |= get_needed_tables(v)
        return needed_tables
    elif isinstance(expr, CaseExpr):
        raise NotImplementedError
    elif isinstance(expr, LiteralCol):
        raise NotImplementedError
    return set()


def propagate_col_names(expr: ColExpr, col_to_name: dict[Col, ColName]) -> ColExpr:
    if isinstance(expr, Col):
        col_name = col_to_name.get(expr)
        return col_name if col_name is not None else expr
    elif isinstance(expr, ColFn):
        expr.args = [propagate_col_names(arg, col_to_name) for arg in expr.args]
        expr.context_kwargs = {
            key: [propagate_col_names(v, col_to_name) for v in arr]
            for key, arr in expr.context_kwargs
        }
    elif isinstance(expr, CaseExpr):
        raise NotImplementedError

    return expr


def propagate_types(expr: ColExpr, col_types: dict[ColName, DType]) -> ColExpr:
    if isinstance(expr, ColName):
        expr._type = col_types[expr]
        return expr
    elif isinstance(expr, ColFn):
        expr.args = [propagate_types(arg, col_types) for arg in expr.args]
        expr.context_kwargs = {
            key: [propagate_types(v, col_types) for v in arr]
            for key, arr in expr.context_kwargs
        }
        # TODO: create a backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        expr._type = PolarsImpl.operator_registry.get_implementation(
            expr.name, [arg._type for arg in expr.args]
        ).return_type
        return expr

    raise NotImplementedError


# Add all supported dunder methods to `ColExpr`. This has to be done, because Python
# doesn't call __getattr__ for dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return ColFn(op, *args, **kwargs)

    return impl


for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(ColExpr, dunder, create_operator(dunder))
del create_operator


@dataclasses.dataclass
class Order:
    order_by: ColExpr
    descending: bool
    nulls_last: bool

    # the given `expr` may contain nulls_last markers or `-` (descending markers). the
    # order_by of the Order does not contain these special functions and can thus be
    # translated normally.
    @classmethod
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
        return Order(expr, descending, nulls_last)