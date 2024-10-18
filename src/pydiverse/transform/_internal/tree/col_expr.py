# ruff: noqa: A002

from __future__ import annotations

import copy
import dataclasses
import functools
import html
import itertools
import operator
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, overload
from uuid import UUID

from pydiverse.transform._internal.errors import FunctionTypeError
from pydiverse.transform._internal.ops import ops, signature
from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.ops.markers import Marker
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.types import (
    FLOAT_SUBTYPES,
    IMPLICIT_CONVS,
    INT_SUBTYPES,
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Duration,
    Float,
    Int,
    String,
    python_type_to_pdt,
)

T = TypeVar("T")


class ColExpr(Generic[T]):
    __slots__ = ["_dtype", "_ftype"]

    __contains__ = None
    __iter__ = None

    def __init__(self, _dtype: Dtype | None = None, _ftype: Ftype | None = None):
        self._dtype = _dtype
        self._ftype = _ftype

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
                        *(
                            (wrap_literal(elem) for elem in key)
                            if isinstance(key, Iterable)
                            else (wrap_literal(key),)
                        ),
                    ),
                    wrap_literal(val),
                )
                for key, val in mapping.items()
            ),
            wrap_literal(default) if default is not None else self,
        )

    def cast(self, target_type: Dtype) -> Cast:
        return Cast(self, target_type)

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

    @overload
    def abs(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def abs(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def abs(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def abs(self: ColExpr) -> ColExpr:
        return ColFn(ops.abs, self)

    @overload
    def __add__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __add__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __add__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __add__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[String]: ...

    def __add__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.add, self, rhs)

    @overload
    def __radd__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __radd__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __radd__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __radd__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[String]: ...

    def __radd__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.add, rhs, self)

    def all(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]:
        return ColFn(ops.all, self, partition_by=partition_by, filter=filter)

    def any(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]:
        return ColFn(ops.any, self, partition_by=partition_by, filter=filter)

    def ascending(self: ColExpr) -> ColExpr:
        return ColFn(ops.ascending, self)

    def __and__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_and, self, rhs)

    def __rand__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_and, rhs, self)

    def __invert__(self: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_invert, self)

    def __or__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_or, self, rhs)

    def __ror__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_or, rhs, self)

    def __xor__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_xor, self, rhs)

    def __rxor__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        return ColFn(ops.bool_xor, rhs, self)

    @overload
    def ceil(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def ceil(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def ceil(self: ColExpr) -> ColExpr:
        return ColFn(ops.ceil, self)

    def count(
        self: ColExpr,
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]:
        return ColFn(ops.count, self, partition_by=partition_by, filter=filter)

    def descending(self: ColExpr) -> ColExpr:
        return ColFn(ops.descending, self)

    def __eq__(self: ColExpr) -> ColExpr:
        return ColFn(ops.equal, self)

    def exp(self: ColExpr[Float]) -> ColExpr[Float]:
        return ColFn(ops.exp, self)

    def fill_null(self: ColExpr) -> ColExpr:
        return ColFn(ops.fill_null, self)

    @overload
    def floor(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def floor(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def floor(self: ColExpr) -> ColExpr:
        return ColFn(ops.floor, self)

    def __floordiv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        return ColFn(ops.floordiv, self, rhs)

    def __rfloordiv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        return ColFn(ops.floordiv, rhs, self)

    @overload
    def __ge__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __ge__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __ge__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __ge__(self: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __ge__(self: ColExpr[Datetime]) -> ColExpr[Datetime]: ...

    @overload
    def __ge__(self: ColExpr[Date]) -> ColExpr[Date]: ...

    def __ge__(self: ColExpr) -> ColExpr:
        return ColFn(ops.greater_equal, self)

    @overload
    def __gt__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __gt__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __gt__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __gt__(self: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __gt__(self: ColExpr[Datetime]) -> ColExpr[Datetime]: ...

    @overload
    def __gt__(self: ColExpr[Date]) -> ColExpr[Date]: ...

    def __gt__(self: ColExpr) -> ColExpr:
        return ColFn(ops.greater_than, self)

    def is_in(self: ColExpr, *rhs: ColExpr) -> ColExpr:
        return ColFn(ops.is_in, self, *rhs)

    def is_inf(self: ColExpr[Float]) -> ColExpr[Bool]:
        return ColFn(ops.is_inf, self)

    def is_nan(self: ColExpr[Float]) -> ColExpr[Bool]:
        return ColFn(ops.is_nan, self)

    def is_not_inf(self: ColExpr[Float]) -> ColExpr[Bool]:
        return ColFn(ops.is_not_inf, self)

    def is_not_nan(self: ColExpr[Float]) -> ColExpr[Bool]:
        return ColFn(ops.is_not_nan, self)

    def is_not_null(self: ColExpr) -> ColExpr:
        return ColFn(ops.is_not_null, self)

    def is_null(self: ColExpr) -> ColExpr:
        return ColFn(ops.is_null, self)

    @overload
    def __le__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __le__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __le__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __le__(self: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __le__(self: ColExpr[Datetime]) -> ColExpr[Datetime]: ...

    @overload
    def __le__(self: ColExpr[Date]) -> ColExpr[Date]: ...

    def __le__(self: ColExpr) -> ColExpr:
        return ColFn(ops.less_equal, self)

    @overload
    def __lt__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __lt__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __lt__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __lt__(self: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __lt__(self: ColExpr[Datetime]) -> ColExpr[Datetime]: ...

    @overload
    def __lt__(self: ColExpr[Date]) -> ColExpr[Date]: ...

    def __lt__(self: ColExpr) -> ColExpr:
        return ColFn(ops.less_than, self)

    def log(self: ColExpr[Float]) -> ColExpr[Float]:
        return ColFn(ops.log, self)

    @overload
    def max(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def max(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def max(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def max(
        self: ColExpr[String],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[String]: ...

    @overload
    def max(
        self: ColExpr[Datetime],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Datetime]: ...

    @overload
    def max(
        self: ColExpr[Date],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Date]: ...

    def max(
        self: ColExpr,
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        return ColFn(ops.max, self, partition_by=partition_by, filter=filter)

    @overload
    def mean(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def mean(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def mean(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    def mean(
        self: ColExpr,
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        return ColFn(ops.mean, self, partition_by=partition_by, filter=filter)

    @overload
    def min(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def min(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def min(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def min(
        self: ColExpr[String],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[String]: ...

    @overload
    def min(
        self: ColExpr[Datetime],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Datetime]: ...

    @overload
    def min(
        self: ColExpr[Date],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Date]: ...

    def min(
        self: ColExpr,
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        return ColFn(ops.min, self, partition_by=partition_by, filter=filter)

    def __mod__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        return ColFn(ops.mod, self, rhs)

    def __rmod__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        return ColFn(ops.mod, rhs, self)

    @overload
    def __mul__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __mul__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __mul__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __mul__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.mul, self, rhs)

    @overload
    def __rmul__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __rmul__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rmul__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __rmul__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.mul, rhs, self)

    @overload
    def __neg__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __neg__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __neg__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __neg__(self: ColExpr) -> ColExpr:
        return ColFn(ops.neg, self)

    def __ne__(self: ColExpr) -> ColExpr:
        return ColFn(ops.not_equal, self)

    def nulls_first(self: ColExpr) -> ColExpr:
        return ColFn(ops.nulls_first, self)

    def nulls_last(self: ColExpr) -> ColExpr:
        return ColFn(ops.nulls_last, self)

    @overload
    def __pos__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __pos__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __pos__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __pos__(self: ColExpr) -> ColExpr:
        return ColFn(ops.pos, self)

    @overload
    def pow(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Float]: ...

    @overload
    def pow(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def pow(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def pow(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.pow, self, rhs)

    @overload
    def round(self: ColExpr[Int], decimals: int = 0) -> ColExpr[Int]: ...

    @overload
    def round(self: ColExpr[Float], decimals: int = 0) -> ColExpr[Float]: ...

    @overload
    def round(self: ColExpr[Decimal], decimals: int = 0) -> ColExpr[Decimal]: ...

    def round(self: ColExpr, decimals: int = 0) -> ColExpr:
        return ColFn(ops.round, self, decimals)

    @overload
    def __sub__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __sub__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __sub__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __sub__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.sub, self, rhs)

    @overload
    def __rsub__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __rsub__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rsub__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __rsub__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.sub, rhs, self)

    @overload
    def sum(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def sum(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def sum(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    def sum(
        self: ColExpr,
        *,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        return ColFn(ops.sum, self, partition_by=partition_by, filter=filter)

    @overload
    def __truediv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Float]: ...

    @overload
    def __truediv__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __truediv__(
        self: ColExpr[Decimal], rhs: ColExpr[Decimal]
    ) -> ColExpr[Decimal]: ...

    def __truediv__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.truediv, self, rhs)

    @overload
    def __rtruediv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Float]: ...

    @overload
    def __rtruediv__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rtruediv__(
        self: ColExpr[Decimal], rhs: ColExpr[Decimal]
    ) -> ColExpr[Decimal]: ...

    def __rtruediv__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        return ColFn(ops.truediv, rhs, self)

    @property
    def str(self):
        return StrNamespace(self)

    @property
    def dt(self):
        return DtNamespace(self)


@dataclasses.dataclass(slots=True)
class FnNamespace:
    arg: ColExpr


@dataclasses.dataclass(slots=True)
class StrNamespace(FnNamespace):
    def contains(self: ColExpr[String], substr: str) -> ColExpr[Bool]:
        return ColFn(ops.str_contains, self.arg, substr)

    def ends_with(self: ColExpr[String], suffix: str) -> ColExpr[Bool]:
        return ColFn(ops.str_ends_with, self.arg, suffix)

    def len(self: ColExpr[String]) -> ColExpr[Int]:
        return ColFn(ops.str_len, self.arg)

    def lower(self: ColExpr[String]) -> ColExpr[String]:
        return ColFn(ops.str_lower, self.arg)

    def replace_all(
        self: ColExpr[String], substr: str, replacement: str
    ) -> ColExpr[String]:
        return ColFn(ops.str_replace_all, self.arg, substr, replacement)

    def slice(
        self: ColExpr[String], offset: ColExpr[Int], n: ColExpr[Int]
    ) -> ColExpr[String]:
        return ColFn(ops.str_slice, self.arg, offset, n)

    def starts_with(self: ColExpr[String], prefix: str) -> ColExpr[Bool]:
        return ColFn(ops.str_starts_with, self.arg, prefix)

    def strip(self: ColExpr[String]) -> ColExpr[String]:
        return ColFn(ops.str_strip, self.arg)

    def to_date(self: ColExpr[String]) -> ColExpr[Date]:
        return ColFn(ops.str_to_date, self.arg)

    def to_datetime(self: ColExpr[String]) -> ColExpr[Datetime]:
        return ColFn(ops.str_to_datetime, self.arg)

    def upper(self: ColExpr[String]) -> ColExpr[String]:
        return ColFn(ops.str_upper, self.arg)


@dataclasses.dataclass(slots=True)
class DtNamespace(FnNamespace):
    @overload
    def day(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day(self: ColExpr) -> ColExpr:
        return ColFn(ops.dt_day, self.arg)

    @overload
    def day_of_week(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day_of_week(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day_of_week(self: ColExpr) -> ColExpr:
        return ColFn(ops.dt_day_of_week, self.arg)

    @overload
    def day_of_year(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day_of_year(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day_of_year(self: ColExpr) -> ColExpr:
        return ColFn(ops.dt_day_of_year, self.arg)

    def days(self: ColExpr[Duration]) -> ColExpr[Int]:
        return ColFn(ops.dt_days, self.arg)

    def hour(self: ColExpr[Datetime]) -> ColExpr[Int]:
        return ColFn(ops.dt_hour, self.arg)

    def hours(self: ColExpr[Duration]) -> ColExpr[Int]:
        return ColFn(ops.dt_hours, self.arg)

    def microsecond(self: ColExpr[Datetime]) -> ColExpr[Int]:
        return ColFn(ops.dt_microsecond, self.arg)

    def millisecond(self: ColExpr[Datetime]) -> ColExpr[Int]:
        return ColFn(ops.dt_millisecond, self.arg)

    def milliseconds(self: ColExpr[Duration]) -> ColExpr[Int]:
        return ColFn(ops.dt_milliseconds, self.arg)

    def minute(self: ColExpr[Datetime]) -> ColExpr[Int]:
        return ColFn(ops.dt_minute, self.arg)

    def minutes(self: ColExpr[Duration]) -> ColExpr[Int]:
        return ColFn(ops.dt_minutes, self.arg)

    @overload
    def month(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def month(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def month(self: ColExpr) -> ColExpr:
        return ColFn(ops.dt_month, self.arg)

    def second(self: ColExpr[Datetime]) -> ColExpr[Int]:
        return ColFn(ops.dt_second, self.arg)

    def seconds(self: ColExpr[Duration]) -> ColExpr[Int]:
        return ColFn(ops.dt_seconds, self.arg)

    @overload
    def year(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def year(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def year(self: ColExpr) -> ColExpr:
        return ColFn(ops.dt_year, self.arg)


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

    def __init__(self, val: Any, dtype: types.Dtype | None = None):
        self.val = val
        if dtype is None:
            dtype = python_type_to_pdt(type(val))
        dtype.const = True
        super().__init__(dtype, Ftype.ELEMENT_WISE)

    def __repr__(self):
        return f"lit({self.val}, {self.dtype()})"


class ColFn(ColExpr):
    __slots__ = ("op", "args", "context_kwargs")

    def __init__(self, op: Operator, *args: ColExpr, **kwargs: list[ColExpr | Order]):
        self.op = op
        # While building the expression tree, we have to allow markers.
        self.args: list[ColExpr] = [
            wrap_literal(arg, allow_markers=True) for arg in args
        ]
        self.context_kwargs = clean_kwargs(**kwargs)

        if filters := self.context_kwargs.get("filter"):
            if len(self.args) == 0:
                assert self.op.name == "len"
            else:
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
        return f'{self.op.name}({", ".join(args)})'

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

        self._dtype = self.op.return_type(arg_dtypes)
        self._dtype.const = all(argt.const for argt in arg_dtypes)

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

        actual_ftype = (
            Ftype.WINDOW
            if self.op.ftype == Ftype.AGGREGATE and agg_is_window
            else self.op.ftype
        )

        if actual_ftype == Ftype.ELEMENT_WISE:
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
                        (desc_ftype := node.op.ftype) in (Ftype.AGGREGATE, Ftype.WINDOW)
                    )
                ):
                    assert isinstance(self, ColFn)
                    ftype_string = {
                        Ftype.AGGREGATE: "aggregation",
                        Ftype.WINDOW: "window",
                    }
                    raise FunctionTypeError(
                        f"{ftype_string[desc_ftype]} function `{node.op.name}` nested "
                        f"inside {ftype_string[self._ftype]} function `{self.op.name}`"
                        ".\nhint: There may be at most one window / aggregation "
                        "function in a column expression on any path from the root to "
                        "a leaf."
                    )

        return self._ftype


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

        for cond, _ in self.cases:
            if cond.dtype() is not None and not cond.dtype() <= types.Bool():
                raise TypeError(
                    f"argument `{cond}` for `when` must be of boolean type, but has "
                    f"type `{cond.dtype()}`"
                )

        val_types = [val.dtype().without_const() for _, val in self.cases]
        if self.default_val is not None:
            val_types.append(self.default_val.dtype().without_const())

        if None in val_types:
            return None

        if (
            common_ancestors := functools.reduce(
                operator.and_,
                (set(IMPLICIT_CONVS[t].keys()) for t in val_types[1:]),
                IMPLICIT_CONVS[val_types[0]].keys(),
            )
        ) is None:
            raise TypeError(
                f"incompatible types `{", ".join(val_types)}` in case expression."
            )

        common_ancestors: list[Dtype] = list(common_ancestors)
        self._dtype = common_ancestors[
            signature.best_signature_match(
                val_types, [[anc] * len(val_types) for anc in common_ancestors]
            )
        ]
        self._dtype.const = all(
            *(cond.dtype().const and val.dtype().const for cond, val in self.cases),
            self.default_val.dtype().const,
        )

        return self._dtype

    def ftype(self, *, agg_is_window: bool):
        if self._ftype is not None:
            return self._ftype

        val_ftypes = set()
        # TODO: does it actually matter if we add stuff that is const? it should be
        # elemwise anyway...
        if self.default_val is not None and not self.default_val.dtype().const:
            val_ftypes.add(self.default_val.ftype(agg_is_window=agg_is_window))

        for _, val in self.cases:
            if val.dtype() is not None and not val.dtype().const:
                val_ftypes.add(val.ftype(agg_is_window=agg_is_window))

        if None in val_ftypes:
            return None

        if len(val_ftypes) == 0:
            self._ftype = Ftype.ELEMENT_WISE
        elif len(val_ftypes) == 1:
            (self._ftype,) = val_ftypes
        elif Ftype.WINDOW in val_ftypes:
            self._ftype = Ftype.WINDOW
        else:
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
            condition.dtype(), types.Bool
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
        if target_type.const:
            raise TypeError("cannot cast to `const` type")

        self.val = val
        self.target_type = target_type
        super().__init__(target_type)
        self.dtype()

    def dtype(self) -> Dtype:
        # Since `ColExpr.dtype` is also responsible for type checking, we may not set
        # `_dtype` until we are able to retrieve the type of `val`.
        if self.val.dtype() is None:
            return None

        if not self.val.dtype().converts_to(self.target_type):
            valid_casts = {
                *((String(), t) for t in (*INT_SUBTYPES, *FLOAT_SUBTYPES)),
                *(
                    (ft, it)
                    for ft, it in itertools.product(FLOAT_SUBTYPES, INT_SUBTYPES)
                ),
                (Datetime, Date),
                *(
                    (t, String())
                    for t in (
                        Int(),
                        *INT_SUBTYPES,
                        Float() * FLOAT_SUBTYPES,
                        Datetime(),
                        Date(),
                    )
                ),
            }

            if (
                self.val.dtype().without_const(),
                self.target_type,
            ) not in valid_casts:
                hint = ""
                if self.val.dtype() == String() and self.target_type in (
                    Datetime(),
                    Date(),
                ):
                    hint = (
                        "\nhint: to convert a str to datetime, call "
                        f"`.str.to_{self.target_type.__class__.__name__.lower()}()` on "
                        "the expression."
                    )

                raise TypeError(
                    f"cannot cast type {self.val.dtype()} to {self.target_type}."
                    f"{hint}"
                )

        self._dtype.const = self.val.dtype().const
        return self._dtype

    def ftype(self, *, agg_is_window: bool) -> Ftype:
        return self.val.ftype(agg_is_window=agg_is_window)

    def iter_children(self) -> Iterable[ColExpr]:
        yield self.val

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> ColExpr:
        return g(Cast(self.val.map_subtree(g), self.target_type))


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
                if expr.op == ops.descending:
                    descending = True
                elif expr.op == ops.ascending:
                    descending = False

            if nulls_last is None:
                if expr.op == ops.nulls_last:
                    nulls_last = True
                elif expr.op == ops.nulls_first:
                    nulls_last = False

            if isinstance(expr.op, Marker):
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


def wrap_literal(expr: Any, *, allow_markers=False) -> Any:
    if isinstance(expr, ColExpr | Order):
        if isinstance(expr, ColFn) and (
            (isinstance(expr.op, Marker) and not allow_markers)
            or (
                # markers can only be at the top of an expression tree
                not isinstance(expr.op, Marker)
                and any(
                    isinstance(arg, ColFn) and isinstance(arg.op, Marker)
                    for arg in expr.args
                )
            )
        ):
            raise TypeError(
                f"invalid usage of `{expr.op.name}` in a column expression.\n"
                "note: This marker function can only be used in arguments to the "
                "`arrange` verb or the `arrange=` keyword argument to window "
                "functions. Furthermore, all markers have to be at the top of the "
                "expression tree (i.e. cannot be nested inside a column function)."
            )
        return expr

    else:
        return LiteralCol(expr)


def clean_kwargs(**kwargs) -> dict[str, list[ColExpr]]:
    kwargs = {
        key: [val] if not isinstance(val, Iterable) else val
        for key, val in kwargs.items()
        if val is not None
    }
    if (arrange := kwargs.get("arrange")) is not None:
        kwargs["arrange"] = [Order.from_col_expr(ord) for ord in arrange]
    return {key: [wrap_literal(val) for val in arr] for key, arr in kwargs.items()}
