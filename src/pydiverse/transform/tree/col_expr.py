from __future__ import annotations

import copy
import dataclasses
import functools
import html
import itertools
import operator
from collections.abc import Callable, Iterable
from typing import Any

from pydiverse.transform.errors import DataTypeError, FunctionTypeError
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree import dtypes
from pydiverse.transform.tree.dtypes import Bool, Dtype, python_type_to_pdt
from pydiverse.transform.tree.registry import OperatorRegistry
from pydiverse.transform.tree.table_expr import TableExpr


class ColExpr:
    __slots__ = ["_dtype", "_ftype"]

    __contains__ = None
    __iter__ = None

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

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(repr(self))}</pre>"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def dtype(self) -> Dtype:
        return self._dtype

    def ftype(self, *, agg_is_window: bool) -> Ftype:
        return self._ftype

    def map(
        self, mapping: dict[tuple | ColExpr, ColExpr], *, default: ColExpr = None
    ) -> CaseExpr:
        return CaseExpr(
            (
                (
                    self.isin(
                        *wrap_literal(key if isinstance(key, Iterable) else (key,))
                    ),
                    wrap_literal(val),
                )
                for key, val in mapping.items()
            ),
            default,
        )

    # yields all ColExpr`s appearing in the subtree of `self`. Python builtin types
    # and `Order` expressions are not yielded.
    def iter_nodes(self) -> Iterable[ColExpr]:
        yield self

    def map_nodes(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        return g(self)


class Col(ColExpr):
    def __init__(
        self,
        name: str,
        table: TableExpr,
    ):
        self.name = name
        self.table = table
        if (dftype := table._schema.get(name)) is None:
            raise ValueError(f"column `{name}` does not exist in table `{table.name}`")
        self._dtype, self._ftype = dftype

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.table.name}.{self.name}"
            f"({self.dtype()})>"
        )

    def __str__(self) -> str:
        try:
            from pydiverse.transform.backend.targets import Polars
            from pydiverse.transform.pipe.verbs import export, select

            df = self.table >> select(self) >> export(Polars())
            return str(df)
        except Exception as e:
            return (
                repr(self)
                + f"\ncould evaluate {repr(self)} due to"
                + f"{e.__class__.__name__}: {str(e)}"
            )


class ColName(ColExpr):
    def __init__(
        self, name: str, dtype: Dtype | None = None, ftype: Ftype | None = None
    ):
        self.name = name
        self._dtype = dtype
        self._ftype = ftype

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} C.{self.name}"
            f"{f" ({self.dtype()})" if self.dtype() else ""}>"
        )


class LiteralCol(ColExpr):
    def __init__(self, val: Any):
        self.val = val
        self._dtype = python_type_to_pdt(type(val))
        self._dtype.const = True
        self._ftype = Ftype.EWISE

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.val} ({self.dtype()})>"


class ColFn(ColExpr):
    def __init__(self, name: str, *args: ColExpr, **kwargs: list[ColExpr | Order]):
        self.name = name
        self.args = list(args)
        self.context_kwargs = kwargs
        if arrange := self.context_kwargs.get("arrange"):
            self.context_kwargs["arrange"] = [
                Order.from_col_expr(expr) if isinstance(expr, ColExpr) else expr
                for expr in arrange
            ]

        self._dtype = None
        self._ftype = None

    def __repr__(self) -> str:
        args = [repr(e) for e in self.args] + [
            f"{key}={repr(val)}" for key, val in self.context_kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def iter_nodes(self) -> Iterable[ColExpr]:
        for val in itertools.chain(self.args, *self.context_kwargs.values()):
            yield from val.iter_nodes()
        yield self

    def map_nodes(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        new_fn = copy.copy(self)
        new_fn.args = [arg.map_nodes(g) for arg in self.args]
        new_fn.context_kwargs = {
            key: [val.map_nodes(g) for val in arr]
            for key, arr in self.context_kwargs.items()
        }
        return g(new_fn)

    def dtype(self) -> Dtype:
        if self._dtype is not None:
            return self._dtype

        # TODO: create a backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        self._dtype = PolarsImpl.registry.get_impl(
            self.name, [arg.dtype() for arg in self.args]
        ).return_type

        return self._dtype

    def ftype(self, *, agg_is_window: bool):
        """
        Determine the ftype based on a function implementation and the arguments.

            e(e) -> e       a(e) -> a       w(e) -> w
            e(a) -> a       a(a) -> Err     w(a) -> w
            e(w) -> w       a(w) -> Err     w(w) -> Err

        If the implementation ftype is incompatible with the arguments, this
        function raises an Exception.
        """

        # TODO: This causes wrong results if ftype is called once with
        # agg_is_window=True and then with agg_is_window=False.
        if self._ftype is not None:
            return self._ftype

        from pydiverse.transform.backend.polars import PolarsImpl

        op = PolarsImpl.registry.get_op(self.name)

        ftypes = [arg.ftype(agg_is_window=agg_is_window) for arg in self.args]
        if op.ftype == Ftype.AGGREGATE and agg_is_window:
            op_ftype = Ftype.WINDOW
        else:
            op_ftype = op.ftype

        if op_ftype == Ftype.EWISE:
            if Ftype.WINDOW in ftypes:
                self._ftype = Ftype.WINDOW
            elif Ftype.AGGREGATE in ftypes:
                self._ftype = Ftype.AGGREGATE
            else:
                self._ftype = op_ftype

        elif op_ftype == Ftype.AGGREGATE:
            if Ftype.WINDOW in ftypes:
                raise FunctionTypeError(
                    "cannot nest a window function inside an aggregate function"
                    f" ({op.name})."
                )

            if Ftype.AGGREGATE in ftypes:
                raise FunctionTypeError(
                    "cannot nest an aggregate function inside an aggregate function"
                    f" ({op.name})."
                )
            self._ftype = op_ftype

        else:
            if Ftype.WINDOW in ftypes:
                raise FunctionTypeError(
                    "cannot nest a window function inside a window function"
                    f" ({op.name})."
                )
            self._ftype = op_ftype

        return self._ftype


@dataclasses.dataclass
class FnAttr:
    name: str
    arg: ColExpr

    def __getattr__(self, name) -> FnAttr:
        return FnAttr(f"{self.name}.{name}", self.arg)

    def __call__(self, *args, **kwargs) -> ColExpr:
        return ColFn(
            self.name,
            wrap_literal(self.arg),
            *wrap_literal(args),
            **wrap_literal(kwargs),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}({self.arg})>"


class WhenClause:
    def __init__(self, cases: list[tuple[ColExpr, ColExpr]], cond: ColExpr):
        self.cases = cases
        self.cond = cond

    def then(self, value: ColExpr) -> CaseExpr:
        return CaseExpr((*self.cases, (self.cond, wrap_literal(value))))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.cond}>"


class CaseExpr(ColExpr):
    def __init__(
        self,
        cases: Iterable[tuple[ColExpr, ColExpr]],
        default_val: ColExpr | None = None,
    ):
        self.cases = list(cases)

        # We distinguish `None` and `LiteralCol(None)` as a `default_val`. The first one
        # signals that the user has not yet set a default value, the second one
        # indicates that the user set `None` as a default value.
        self.default_val = default_val
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            + functools.reduce(
                operator.add, (f"{cond} -> {val}, " for cond, val in self.cases), ""
            )
            + f"default={self.default_val}>"
        )

    def iter_nodes(self) -> Iterable[ColExpr]:
        for expr in itertools.chain.from_iterable(self.cases):
            yield from expr.iter_nodes()
        if self.default_val is not None:
            yield from self.default_val.iter_nodes()
        yield self

    def map_nodes(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        new_case_expr = copy.copy(self)
        new_case_expr.cases = [
            (cond.map_nodes(g), val.map_nodes(g)) for cond, val in self.cases
        ]
        new_case_expr.default_val = (
            self.default_val.map_nodes(g) if self.default_val is not None else None
        )
        return g(new_case_expr)

    def dtype(self):
        if self._dtype is not None:
            return self._dtype

        try:
            self._dtype = dtypes.promote_dtypes(
                [
                    self.default_val.dtype().without_modifiers(),
                    *(val.dtype().without_modifiers() for _, val in self.cases),
                ]
            )
        except Exception as e:
            raise DataTypeError(f"invalid case expression: {e}") from e

        for cond, _ in self.cases:
            if not isinstance(cond.dtype(), Bool):
                raise DataTypeError(
                    "invalid case expression: condition {cond} has type "
                    f"{cond.dtype()} but all conditions must be boolean"
                )

    def ftype(self, *, agg_is_window: bool):
        if self._ftype is not None:
            return self._ftype

        val_ftypes = set()
        if self.default_val is not None and not self.default_val.dtype().const:
            val_ftypes.add(self.default_val._ftype)

        for _, val in self.cases:
            if not val.dtype().const:
                val_ftypes.add(val.ftype(agg_is_window=agg_is_window))

        if len(val_ftypes) == 0:
            self._ftype = Ftype.EWISE
        elif len(val_ftypes) == 1:
            (self._ftype,) = val_ftypes
        elif Ftype.WINDOW in val_ftypes:
            self._ftype = Ftype.WINDOW
        else:
            # AGGREGATE and EWISE are incompatible
            raise FunctionTypeError(
                "Incompatible function types found in case statement: " ", ".join(
                    val_ftypes
                )
            )

        return self._ftype

    def when(self, condition: ColExpr) -> WhenClause:
        if self.default_val is not None:
            raise TypeError("cannot call `when` on a case expression after `otherwise`")
        return WhenClause(self.cases, wrap_literal(condition))

    def otherwise(self, value: ColExpr) -> CaseExpr:
        if self.default_val is not None:
            raise TypeError("cannot call `otherwise` twice on a case expression")
        return CaseExpr(self.cases, wrap_literal(value))


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

    def iter_nodes(self) -> Iterable[ColExpr]:
        yield from self.order_by.iter_nodes()

    def map_nodes(self, g: Callable[[ColExpr], ColExpr]) -> Order:
        return Order(self.order_by.map_nodes(g), self.descending, self.nulls_last)


# Add all supported dunder methods to `ColExpr`. This has to be done, because Python
# doesn't call __getattr__ for dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return ColFn(op, *wrap_literal(args), **wrap_literal(kwargs))

    return impl


for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(ColExpr, dunder, create_operator(dunder))
del create_operator


def wrap_literal(expr: Any) -> Any:
    if isinstance(expr, ColExpr | Order):
        return expr
    elif isinstance(expr, dict):
        return {key: wrap_literal(val) for key, val in expr.items()}
    elif isinstance(expr, (list, tuple)):
        return expr.__class__(wrap_literal(elem) for elem in expr)
    else:
        return LiteralCol(expr)
