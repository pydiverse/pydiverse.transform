from __future__ import annotations

import copy
import dataclasses
import functools
import html
import itertools
import operator
from collections.abc import Callable, Generator, Iterable
from typing import Any
from uuid import UUID

from pydiverse.transform._internal.errors import FunctionTypeError
from pydiverse.transform._internal.ops.core import Ftype, Operator
from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.dtypes import Dtype, python_type_to_pdt
from pydiverse.transform._internal.tree.registry import OperatorRegistry


class ColExpr:
    __slots__ = ["_dtype", "_ftype"]

    __contains__ = None
    __iter__ = None

    def __init__(self, _dtype: Dtype | None = None, _ftype: Ftype | None = None):
        self._dtype = _dtype
        self._ftype = _ftype

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

    def __setstate__(self, d):  # to avoid very annoying AttributeErrors
        for slot, val in d[1].items():
            setattr(self, slot, val)

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(repr(self))}</pre>"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def dtype(self) -> Dtype:
        return self._dtype

    def ftype(self, *, agg_is_window: bool) -> Ftype:
        return self._ftype

    def map(
        self, mapping: dict[tuple | ColExpr, ColExpr], *, default: ColExpr | None = None
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
            wrap_literal(default) if default is not None else self,
        )

    def iter_children(self) -> Iterable[ColExpr]:
        return iter(())

    # yields all ColExpr`s appearing in the subtree of `self`. Python builtin types
    # and `Order` expressions are not yielded.
    def iter_subtree(self) -> Iterable[ColExpr]:
        for node in self.iter_children():
            yield from node.iter_subtree()
        yield self

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        return g(self)


class Col(ColExpr):
    __slots__ = ["name", "_ast", "_uuid"]

    def __init__(
        self, name: str, _ast: AstNode, _uuid: UUID, _dtype: Dtype, _ftype: Ftype
    ):
        self.name = name
        self._ast = _ast
        self._uuid = _uuid
        super().__init__(_dtype, _ftype)

    def __repr__(self) -> str:
        dtype_str = f" ({self.dtype()})" if self.dtype() is not None else ""
        return f"{self._ast.name}.{self.name}{dtype_str}"

    def __str__(self) -> str:
        try:
            from pydiverse.transform._internal.backend.polars import PolarsImpl
            from pydiverse.transform._internal.backend.targets import Polars

            df = PolarsImpl.export(self._ast, Polars(lazy=False), [self])
            return str(df.get_column(df.columns[0]))
        except Exception as e:
            return (
                repr(self)
                + f"\ncould evaluate {repr(self)} due to"
                + f"{e.__class__.__name__}: {str(e)}"
            )

    def __hash__(self) -> int:
        return hash(self._uuid)


class ColName(ColExpr):
    __slots__ = ["name"]

    def __init__(
        self, name: str, dtype: Dtype | None = None, ftype: Ftype | None = None
    ):
        self.name = name
        super().__init__(dtype, ftype)

    def __repr__(self) -> str:
        dtype_str = f" ({self.dtype()})" if self.dtype() is not None else ""
        return f"C.{self.name}{dtype_str}"


class LiteralCol(ColExpr):
    __slots__ = ["val"]

    def __init__(self, val: Any, dtype: dtypes.Dtype | None = None):
        self.val = val
        if dtype is None:
            dtype = python_type_to_pdt(type(val))
        dtype.const = True
        super().__init__(dtype, Ftype.EWISE)

    def __repr__(self):
        return f"lit({self.val}, {self.dtype()})"


class ColFn(ColExpr):
    __slots__ = ["name", "args", "context_kwargs"]

    def __init__(self, name: str, *args: ColExpr, **kwargs: list[ColExpr | Order]):
        self.name = name
        self.args = list(args)
        self.context_kwargs = kwargs

        if filters := self.context_kwargs.get("filter"):
            if len(self.args) == 0:
                assert self.name == "count"
                self.args = [LiteralCol(0)]

            # TODO: check that this is an aggregation

            assert len(self.args) == 1
            self.args[0] = CaseExpr(
                [
                    (
                        functools.reduce(operator.and_, (cond for cond in filters)),
                        self.args[0],
                    )
                ]
            )
            del self.context_kwargs["filter"]

        super().__init__()
        # try to eagerly resolve the types to get a nicer stack trace on type errors
        self.dtype()

    def __repr__(self) -> str:
        args = [repr(e) for e in self.args] + [
            f"{key}={repr(val)}" for key, val in self.context_kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def op(self) -> Operator:
        # TODO: backend agnostic registry, make this an attribute?
        from pydiverse.transform._internal.backend.polars import PolarsImpl

        return PolarsImpl.registry.get_op(self.name)

    def iter_children(self) -> Iterable[ColExpr]:
        yield from itertools.chain(self.args, *self.context_kwargs.values())

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        new_fn = copy.copy(self)
        new_fn.args = [arg.map_subtree(g) for arg in self.args]

        new_fn.context_kwargs = {
            key: [val.map_subtree(g) for val in arr]
            for key, arr in self.context_kwargs.items()
        }
        return g(new_fn)

    def dtype(self) -> Dtype:
        if self._dtype is not None:
            return self._dtype

        arg_dtypes = [arg.dtype() for arg in self.args]
        if None in arg_dtypes:
            return None

        from pydiverse.transform._internal.backend.polars import PolarsImpl

        self._dtype = PolarsImpl.registry.get_impl(self.name, arg_dtypes).return_type
        return self._dtype

    def ftype(self, *, agg_is_window: bool):
        """
        Determine the ftype based on the arguments.

            e(e) -> e       a(e) -> a       w(e) -> w
            e(a) -> a       a(a) -> Err     w(a) -> w
            e(w) -> w       a(w) -> Err     w(w) -> Err

        If the operator ftype is incompatible with the arguments, this function raises
        an Exception.
        """

        # TODO: This causes wrong results if ftype is called once with
        # agg_is_window=True and then with agg_is_window=False.
        if self._ftype is not None:
            return self._ftype

        ftypes = [arg.ftype(agg_is_window=agg_is_window) for arg in self.args]
        if None in ftypes:
            return None

        op = self.op()

        actual_ftype = (
            Ftype.WINDOW if op.ftype == Ftype.AGGREGATE and agg_is_window else op.ftype
        )

        if actual_ftype == Ftype.EWISE:
            # this assert is ok since window functions in `summarize` are already kicked
            # out by the `summarize` constructor.
            assert not (Ftype.WINDOW in ftypes and Ftype.AGGREGATE in ftypes)

            if Ftype.WINDOW in ftypes:
                self._ftype = Ftype.WINDOW
            elif Ftype.AGGREGATE in ftypes:
                self._ftype = Ftype.AGGREGATE
            else:
                self._ftype = actual_ftype

        else:
            self._ftype = actual_ftype

            # kick out nested window / aggregation functions
            for node in self.iter_subtree():
                if (
                    node is not self
                    and isinstance(node, ColFn)
                    and (
                        (desc_ftype := node.op().ftype)
                        in (Ftype.AGGREGATE, Ftype.WINDOW)
                    )
                ):
                    assert isinstance(self, ColFn)
                    ftype_string = {
                        Ftype.AGGREGATE: "aggregation",
                        Ftype.WINDOW: "window",
                    }
                    raise FunctionTypeError(
                        f"{ftype_string[desc_ftype]} function `{node.name}` nested "
                        f"inside {ftype_string[self._ftype]} function `{self.name}`.\n"
                        "hint: There may be at most one window / aggregation function "
                        "in a column expression on any path from the root to a leaf."
                    )

        return self._ftype


@dataclasses.dataclass(slots=True)
class FnAttr:
    name: str
    arg: ColExpr

    def __getattr__(self, name) -> FnAttr:
        return FnAttr(f"{self.name}.{name}", self.arg)

    def __call__(self, *args, **kwargs) -> ColExpr:
        if self.name == "cast":
            if len(kwargs) > 0:
                raise ValueError("`cast` does not take any keyword arguments")
            return Cast(self.arg, *args)

        return ColFn(
            self.name,
            wrap_literal(self.arg)
            # TODO: this is very hacky, but once we do code gen we don't have this
            # problem anymore
            if not isinstance(self.arg, ColFn) or self.arg.name not in MARKERS
            else self.arg,
            *wrap_literal(args),
            **clean_kwargs(**kwargs),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.arg})"


@dataclasses.dataclass(slots=True)
class WhenClause:
    cases: list[tuple[ColExpr, ColExpr]]
    cond: ColExpr

    def then(self, value: ColExpr) -> CaseExpr:
        return CaseExpr((*self.cases, (self.cond, wrap_literal(value))))

    def __repr__(self) -> str:
        return f"when_clause({self.cond})"


class CaseExpr(ColExpr):
    __slots__ = ["cases", "default_val"]

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
        self.dtype()

    def __repr__(self) -> str:
        return (
            "case_when( "
            + functools.reduce(
                operator.add, (f"{cond} -> {val}, " for cond, val in self.cases), ""
            )
            + f"default={self.default_val})"
        )

    def iter_children(self) -> Iterable[ColExpr]:
        yield from itertools.chain.from_iterable(self.cases)
        if self.default_val is not None:
            yield self.default_val

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        new_case_expr = copy.copy(self)
        new_case_expr.cases = [
            (cond.map_subtree(g), val.map_subtree(g)) for cond, val in self.cases
        ]
        new_case_expr.default_val = (
            self.default_val.map_subtree(g) if self.default_val is not None else None
        )
        return g(new_case_expr)

    def dtype(self):
        if self._dtype is not None:
            return self._dtype

        try:
            val_types = [val.dtype() for _, val in self.cases]
            if self.default_val is not None:
                val_types.append(self.default_val.dtype())

            if None in val_types:
                return None

            self._dtype = dtypes.promote_dtypes(
                [dtype.without_modifiers() for dtype in val_types]
            )
        except Exception as e:
            raise TypeError(f"invalid case expression: {e}") from e

        for cond, _ in self.cases:
            if cond.dtype() is not None and cond.dtype() != dtypes.Bool:
                raise TypeError(
                    f"argument `{cond}` for `when` must be of boolean type, but has "
                    f"type `{cond.dtype()}`"
                )

        return self._dtype

    def ftype(self, *, agg_is_window: bool):
        if self._ftype is not None:
            return self._ftype

        val_ftypes = set()
        if self.default_val is not None and not self.default_val.dtype().const:
            val_ftypes.add(self.default_val.ftype(agg_is_window=agg_is_window))

        for _, val in self.cases:
            if val.dtype() is not None and not val.dtype().const:
                val_ftypes.add(val.ftype(agg_is_window=agg_is_window))

        if None in val_ftypes:
            return None

        if len(val_ftypes) == 0:
            self._ftype = Ftype.EWISE
        elif len(val_ftypes) == 1:
            (self._ftype,) = val_ftypes
        elif Ftype.WINDOW in val_ftypes:
            self._ftype = Ftype.WINDOW
        else:
            # AGGREGATE and EWISE are incompatible
            raise FunctionTypeError(
                "incompatible function types found in case statement: " ", ".join(
                    val_ftypes
                )
            )

        return self._ftype

    def when(self, condition: ColExpr) -> WhenClause:
        if self.default_val is not None:
            raise TypeError("cannot call `when` on a closed case expression after")

        condition = wrap_literal(condition)
        if condition.dtype() is not None and not isinstance(
            condition.dtype(), dtypes.Bool
        ):
            raise TypeError(
                "argument for `when` must be of boolean type, but has type "
                f"`{condition.dtype()}`"
            )

        return WhenClause(self.cases, wrap_literal(condition))

    def otherwise(self, value: ColExpr) -> CaseExpr:
        if self.default_val is not None:
            raise TypeError("default value is already set on this case expression")
        return CaseExpr(self.cases, wrap_literal(value))


class Cast(ColExpr):
    __slots__ = ["val", "target_type"]

    def __init__(self, val: ColExpr, target_type: Dtype):
        self.val = val
        self.target_type = target_type
        super().__init__(target_type)
        self.dtype()

    def dtype(self) -> Dtype:
        # Since `ColExpr.dtype` is also responsible for type checking, we may not set
        # `_dtype` until we are able to retrieve the type of `val`.
        if self.val.dtype() is None:
            return None

        if not self.val.dtype().can_promote_to(self.target_type):
            valid_casts = {
                (dtypes.String, dtypes.Int64),
                (dtypes.String, dtypes.Float64),
                (dtypes.Float64, dtypes.Int64),
                (dtypes.DateTime, dtypes.Date),
                (dtypes.Int64, dtypes.String),
                (dtypes.Float64, dtypes.String),
                (dtypes.DateTime, dtypes.String),
                (dtypes.Date, dtypes.String),
            }

            if (
                self.val.dtype().__class__,
                self.target_type.__class__,
            ) not in valid_casts:
                hint = ""
                if self.val.dtype() == dtypes.String and self.target_type in (
                    dtypes.DateTime,
                    dtypes.Date,
                ):
                    hint = (
                        "\nhint: to convert a str to datetime, call "
                        f"`.str.to_{self.target_type.name}()` on the expression."
                    )

                raise TypeError(
                    f"cannot cast type {self.val.dtype()} to {self.target_type}."
                    f"{hint}"
                )

        return self._dtype

    def ftype(self, *, agg_is_window: bool) -> Ftype:
        return self.val.ftype(agg_is_window=agg_is_window)

    def iter_children(self) -> Iterable[ColExpr]:
        yield self.val

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        return g(Cast(self.val.map_subtree(g), self.target_type))


MARKERS = (
    "ascending",
    "descending",
    "nulls_first",
    "nulls_last",
)


@dataclasses.dataclass(slots=True)
class Order:
    order_by: ColExpr
    descending: bool = False
    nulls_last: bool | None = None

    # The given `expr` may contain nulls_last markers or descending markers. The
    # order_by of the Order does not contain these special functions and can thus be
    # translated normally.
    @staticmethod
    def from_col_expr(expr: ColExpr) -> Order:
        descending = None
        nulls_last = None
        while isinstance(expr, ColFn):
            if descending is None:
                if expr.name == "descending":
                    descending = True
                elif expr.name == "ascending":
                    descending = False

            if nulls_last is None:
                if expr.name == "nulls_last":
                    nulls_last = True
                elif expr.name == "nulls_first":
                    nulls_last = False

            if expr.name in MARKERS:
                assert len(expr.args) == 1
                assert len(expr.context_kwargs) == 0
                expr = expr.args[0]
            else:
                break

        if descending is None:
            descending = False

        return Order(expr, descending, nulls_last)

    def iter_subtree(self) -> Iterable[ColExpr]:
        yield from self.order_by.iter_subtree()

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> Order:
        return Order(self.order_by.map_subtree(g), self.descending, self.nulls_last)


# Add all supported dunder methods to `ColExpr`. This has to be done, because Python
# doesn't call __getattr__ for dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return ColFn(op, *wrap_literal(args), **clean_kwargs(**kwargs))

    return impl


for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(ColExpr, dunder, create_operator(dunder))
del create_operator


def wrap_literal(expr: Any) -> Any:
    if isinstance(expr, ColExpr | Order):
        if (
            isinstance(expr, ColFn)
            and expr.name in MARKERS
            and (not isinstance(expr.args[0], ColFn) or expr.args[0].name in MARKERS)
        ):
            raise TypeError(
                f"invalid usage of `.{expr.name}()` in a column expression.\n"
                "note: This marker function can only be used in arguments to the "
                "`arrange` verb or the `arrange=` keyword argument to window "
                "functions. Furthermore, all markers have to be at the top of the "
                "expression tree (i.e. cannot be nested inside a column function)."
            )
        return expr
    elif isinstance(expr, dict):
        return {key: wrap_literal(val) for key, val in expr.items()}
    elif isinstance(expr, list | tuple):
        return expr.__class__(wrap_literal(elem) for elem in expr)
    elif isinstance(expr, Generator):
        return (wrap_literal(elem) for elem in expr)
    elif isinstance(expr, FnAttr):
        raise TypeError(
            "invalid usage of a column function as an expression.\n"
            "hint: Maybe you forgot to put parentheses `()` after the function?"
        )
    else:
        return LiteralCol(expr)


def clean_kwargs(**kwargs) -> dict[str, list[ColExpr]]:
    kwargs = {
        key: [val] if not isinstance(val, Iterable) else val
        for key, val in kwargs.items()
        if val is not None
    }
    return {
        key: wrap_literal(
            val if key != "arrange" else [Order.from_col_expr(ord) for ord in val]
        )
        for key, val in kwargs.items()
    }
