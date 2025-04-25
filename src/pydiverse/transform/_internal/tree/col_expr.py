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

import pandas as pd
import polars as pl

from pydiverse.common import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Duration,
    Float,
    Int,
    List,
    String,
)
from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.targets import Pandas, Polars, Target
from pydiverse.transform._internal.errors import FunctionTypeError
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.ops.markers import Marker
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.types import (
    FLOAT_SUBTYPES,
    INT_SUBTYPES,
    Const,
)

T = TypeVar("T")


# for proper documentation of .str
class Accessor:
    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor

        accessor_obj = self._accessor(obj)
        setattr(obj, self._name, accessor_obj)

        return accessor_obj


def register_accessor(name):
    def func(accessor):
        setattr(ColExpr, name, Accessor(name, accessor))

        return accessor

    return func


class ColExpr(Generic[T]):
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

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(repr(self))}</pre>"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

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

    def dtype(self) -> Dtype:
        """
        Returns the data type of the expression.
        """
        return self._dtype

    def ftype(self, *, agg_is_window: bool | None = None) -> Ftype:
        return self._ftype

    def map(
        self, mapping: dict[tuple | ColExpr, ColExpr], *, default: ColExpr | None = None
    ) -> CaseExpr:
        """
        Replaces given values by other expressions.

        :param mapping:
            A dictionary of expressions / tuples of expressions to expressions. The
            input is compared against key of the dictionary, and if it matches, the
            corresponding value of the key is inserted. If the key is a tuple, the input
            is compared against each element of the tuple and required to equal at least
            one of them.

        :param default:
            The value to insert if the input matches none of the keys of `mapping`.

        Note
        ----
        If there are multiple columns in the key which have the same value at some row,
        any of the corresponding values may be inserted (i.e. ensuring uniqueness of the
        keys is your responsibility).

        Example
        -------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [4, 3, -35, 24, 105],
        ...         "b": [4, 4, 0, -23, 42],
        ...     }
        ... )
        >>> t >> mutate(c=t.a.is_in(t.b, 24)) >> show()
        Table <unnamed>, backend: PolarsImpl
        shape: (5, 3)
        ┌─────┬─────┬───────┐
        │ a   ┆ b   ┆ c     │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 4   ┆ 4   ┆ true  │
        │ 3   ┆ 4   ┆ false │
        │ -35 ┆ 0   ┆ false │
        │ 24  ┆ -23 ┆ true  │
        │ 105 ┆ 42  ┆ false │
        └─────┴─────┴───────┘
        """
        return CaseExpr(
            (
                (
                    self.is_in(
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

    def cast(self, target_type: Dtype | type) -> Cast:
        """
        Cast to a different data type.

        :param target_type:
            The type to cast to.

        The following casts are possible:

        .. list-table::
            :header-rows: 1

            * - Input type
              - Target type
              - Note
            * - Float
              - Int8, Int16, Int32, Int64
              - Extracts the integer part (i.e. rounds towards 0).
            * - String
              - Int8, Int16, Int32, Int64
              - Parses the string as an integer.
            * - String
              - Float32, Float64
              - Parses the string as a floating point number.
            * - Int
              - String
              - Writes the integer in base 10 as a string.
            * - Float
              - String
              - Writes the floating point number in decimal notation in base 10.
            * - Int
              - Int8, Int16, Int32, Int64
              - Casts to an integer with a specified number of bits. Behavior is
                backend-dependent.
            * - Float
              - Float32, Float64
              - Casts to a floating point number with a specified number of bits.
                Behavior is backend-dependent.
            * - Datetime
              - Date
              - Removes the time component of the Datetime.
            * - Datetime
              - String
              - Writes the datetime in the format YYYY-MM-DD HH:MM:SS.SSSSSS.
                Seconds are printed up to microsecond resolution.
            * - Date
              - String
              - Writes the date in the format YYYY-MM-DD.


        In addition to these casts, there are implicit conversion of integers to
        floating point numbers and dates to datetimes. They happens automatically and
        do not require an explicit cast.

        Note
        ----
        In casts from strings, neither leading nor trailing whitespace is allowed.

        Examples
        --------
        >>> t = pdt.Table({"a": [3.5, 10.3, -434.4, -0.2]}, name="T")
        >>> t >> mutate(b=t.a.cast(pdt.Int32())) >> show()
        Table T, backend: PolarsImpl
        shape: (4, 2)
        ┌────────┬──────┐
        │ a      ┆ b    │
        │ ---    ┆ ---  │
        │ f64    ┆ i32  │
        ╞════════╪══════╡
        │ 3.5    ┆ 3    │
        │ 10.3   ┆ 10   │
        │ -434.4 ┆ -434 │
        │ -0.2   ┆ 0    │
        └────────┴──────┘
        """

        errors.check_arg_type(Dtype | type, "ColExpr.cast", "target_type", target_type)
        if type(target_type) is type and not issubclass(target_type, Dtype):
            TypeError(
                "argument for parameter `target_type` of `ColExpr.cast` must be an"
                "instance or subclass of pdt.Dtype"
            )
        return Cast(self, target_type)

    def __rshift__(self, rhs):
        from pydiverse.transform._internal.pipe.pipeable import Pipeable

        if not isinstance(rhs, Pipeable):
            raise TypeError(
                "the right shift operator `>>` can only be applied to a column "
                "expression if a verb is on the right"
            )

        return rhs(self)

    def rank(
        self: ColExpr,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    ) -> ColExpr[Int]:
        """
        Alias for :doc:`/reference/operators/_generated/pydiverse.transform.rank`.
        """
        return ColFn(ops.rank, partition_by=partition_by, arrange=self)

    def dense_rank(
        self: ColExpr,
        partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    ) -> ColExpr[Int]:
        """
        Alias for :doc:`/reference/operators/_generated/pydiverse.transform.dense_rank`.
        """
        return ColFn(ops.dense_rank, partition_by=partition_by, arrange=self)

    # --- generated code starts here, do not delete this comment ---

    @overload
    def abs(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def abs(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def abs(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def abs(self: ColExpr) -> ColExpr:
        """Computes the absolute value."""

        return ColFn(ops.abs, self)

    @overload
    def __add__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __add__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __add__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __add__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __add__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Int]: ...

    @overload
    def __add__(
        self: ColExpr[Duration], rhs: ColExpr[Duration]
    ) -> ColExpr[Duration]: ...

    @overload
    def __add__(
        self: ColExpr[Datetime], rhs: ColExpr[Duration]
    ) -> ColExpr[Datetime]: ...

    @overload
    def __add__(
        self: ColExpr[Duration], rhs: ColExpr[Datetime]
    ) -> ColExpr[Datetime]: ...

    def __add__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Addition +"""

        return ColFn(ops.add, self, rhs)

    @overload
    def __radd__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __radd__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __radd__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __radd__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[String]: ...

    @overload
    def __radd__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Int]: ...

    @overload
    def __radd__(
        self: ColExpr[Duration], rhs: ColExpr[Duration]
    ) -> ColExpr[Duration]: ...

    @overload
    def __radd__(
        self: ColExpr[Datetime], rhs: ColExpr[Duration]
    ) -> ColExpr[Datetime]: ...

    @overload
    def __radd__(
        self: ColExpr[Duration], rhs: ColExpr[Datetime]
    ) -> ColExpr[Datetime]: ...

    def __radd__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Addition +"""

        return ColFn(ops.add, rhs, self)

    def all(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]:
        """Indicates whether every non-null value in a group is True."""

        return ColFn(ops.all, self, partition_by=partition_by, filter=filter)

    def any(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]:
        """Indicates whether at least one value in a group is True."""

        return ColFn(ops.any, self, partition_by=partition_by, filter=filter)

    def ascending(self: ColExpr) -> ColExpr:
        """
        The default ordering.

        Can only be used in expressions given to the `arrange` verb or as as an
        `arrange` keyword argument.
        """

        return ColFn(ops.ascending, self)

    def __and__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean AND (__and__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a & t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ true  │
        │ true  ┆ false ┆ false │
        │ true  ┆ null  ┆ null  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ false │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_and, self, rhs)

    def __rand__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean AND (__and__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a & t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ true  │
        │ true  ┆ false ┆ false │
        │ true  ┆ null  ┆ null  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ false │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_and, rhs, self)

    def __invert__(self: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean inversion (__invert__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=~t.a) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ false │
        │ true  ┆ false ┆ false │
        │ true  ┆ null  ┆ false │
        │ false ┆ false ┆ true  │
        │ false ┆ null  ┆ true  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_invert, self)

    def __or__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean OR (__or__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a | t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ true  │
        │ true  ┆ false ┆ true  │
        │ true  ┆ null  ┆ true  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_or, self, rhs)

    def __ror__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean OR (__or__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a | t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ true  │
        │ true  ┆ false ┆ true  │
        │ true  ┆ null  ┆ true  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_or, rhs, self)

    def __xor__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean XOR (__xor__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a ^ t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ false │
        │ true  ┆ false ┆ true  │
        │ true  ┆ null  ┆ null  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_xor, self, rhs)

    def __rxor__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]:
        """
        Boolean XOR (__xor__)

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [True, True, True, False, False, None],
        ...         "b": [True, False, None, False, None, None],
        ...     },
        ...     name="bool table",
        ... )
        >>> t >> mutate(x=t.a ^ t.b) >> show()
        Table bool table, backend: PolarsImpl
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ x     │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ false │
        │ true  ┆ false ┆ true  │
        │ true  ┆ null  ┆ null  │
        │ false ┆ false ┆ false │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
        """

        return ColFn(ops.bool_xor, rhs, self)

    @overload
    def ceil(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def ceil(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def ceil(self: ColExpr) -> ColExpr:
        """Returns the smallest integer greater than or equal to the input."""

        return ColFn(ops.ceil, self)

    def count(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]:
        """
        Counts the number of non-null elements in the column.
        """

        return ColFn(ops.count, self, partition_by=partition_by, filter=filter)

    def descending(self: ColExpr) -> ColExpr:
        """
        Reverses the default ordering.

        Can only be used in expressions given to the `arrange` verb or as as an
        `arrange` keyword argument.
        """

        return ColFn(ops.descending, self)

    def __eq__(self: ColExpr, rhs: ColExpr) -> ColExpr[Bool]:
        """Equality comparison =="""

        return ColFn(ops.equal, self, rhs)

    def exp(self: ColExpr[Float]) -> ColExpr[Float]:
        """Computes the exponential function."""

        return ColFn(ops.exp, self)

    def fill_null(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Replaces every null by the given value."""

        return ColFn(ops.fill_null, self, rhs)

    @overload
    def floor(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def floor(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def floor(self: ColExpr) -> ColExpr:
        """Returns the largest integer less than or equal to the input."""

        return ColFn(ops.floor, self)

    def __floordiv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        """
        Integer division //

        Warning
        -------
        The behavior of this operator differs from polars and python. Polars and python
        always round towards negative infinity, whereas pydiverse.transform always
        rounds towards zero, regardless of the sign. This behavior matches the one of C,
        C++ and all currently supported SQL backends.

        See also
        --------
        __mod__

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [65, -65, 65, -65],
        ...         "b": [7, 7, -7, -7],
        ...     }
        ... )
        >>> t >> mutate(r=t.a // t.b) >> show()
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ r   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 65  ┆ 7   ┆ 9   │
        │ -65 ┆ 7   ┆ -9  │
        │ 65  ┆ -7  ┆ -9  │
        │ -65 ┆ -7  ┆ 9   │
        └─────┴─────┴─────┘
        """

        return ColFn(ops.floordiv, self, rhs)

    def __rfloordiv__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        """
        Integer division //

        Warning
        -------
        The behavior of this operator differs from polars and python. Polars and python
        always round towards negative infinity, whereas pydiverse.transform always
        rounds towards zero, regardless of the sign. This behavior matches the one of C,
        C++ and all currently supported SQL backends.

        See also
        --------
        __mod__

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [65, -65, 65, -65],
        ...         "b": [7, 7, -7, -7],
        ...     }
        ... )
        >>> t >> mutate(r=t.a // t.b) >> show()
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ r   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 65  ┆ 7   ┆ 9   │
        │ -65 ┆ 7   ┆ -9  │
        │ 65  ┆ -7  ┆ -9  │
        │ -65 ┆ -7  ┆ 9   │
        └─────┴─────┴─────┘
        """

        return ColFn(ops.floordiv, rhs, self)

    @overload
    def __ge__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[Datetime], rhs: ColExpr[Datetime]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Bool]: ...

    @overload
    def __ge__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]: ...

    def __ge__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Greater than or equal to comparison >="""

        return ColFn(ops.greater_equal, self, rhs)

    @overload
    def __gt__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[Datetime], rhs: ColExpr[Datetime]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Bool]: ...

    @overload
    def __gt__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]: ...

    def __gt__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Greater than comparison >"""

        return ColFn(ops.greater_than, self, rhs)

    def is_in(self: ColExpr, *rhs: ColExpr) -> ColExpr[Bool]:
        """
        Whether the value equals one of the given.

        Note
        ----
        The expression ``t.c.is_in(a1, a2, ...)`` is equivalent to
        ``(t.c == a1) | (t.c == a2) | ...``, so passing null to ``is_in`` will result in
        null. To compare for equality with null, use
        :doc:`pydiverse.transform.ColExpr.is_null`.
        """

        return ColFn(ops.is_in, self, *rhs)

    def is_inf(self: ColExpr[Float]) -> ColExpr[Bool]:
        """
        Whether the number is infinite.

        Note
        ----
        This is currently only useful for backends supporting IEEE 754-floats. On
        other backends it always returns False.
        """

        return ColFn(ops.is_inf, self)

    def is_nan(self: ColExpr[Float]) -> ColExpr[Bool]:
        """"""

        return ColFn(ops.is_nan, self)

    def is_not_inf(self: ColExpr[Float]) -> ColExpr[Bool]:
        """"""

        return ColFn(ops.is_not_inf, self)

    def is_not_nan(self: ColExpr[Float]) -> ColExpr[Bool]:
        """"""

        return ColFn(ops.is_not_nan, self)

    def is_not_null(self: ColExpr) -> ColExpr[Bool]:
        """Indicates whether the value is not null."""

        return ColFn(ops.is_not_null, self)

    def is_null(self: ColExpr) -> ColExpr[Bool]:
        """Indicates whether the value is null."""

        return ColFn(ops.is_null, self)

    @overload
    def __le__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[Datetime], rhs: ColExpr[Datetime]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Bool]: ...

    @overload
    def __le__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]: ...

    def __le__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Less than or equal to comparison <="""

        return ColFn(ops.less_equal, self, rhs)

    @overload
    def __lt__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[String], rhs: ColExpr[String]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[Datetime], rhs: ColExpr[Datetime]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Bool]: ...

    @overload
    def __lt__(self: ColExpr[Bool], rhs: ColExpr[Bool]) -> ColExpr[Bool]: ...

    def __lt__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Less than comparison <"""

        return ColFn(ops.less_than, self, rhs)

    def log(self: ColExpr[Float]) -> ColExpr[Float]:
        """Computes the natural logarithm."""

        return ColFn(ops.log, self)

    @overload
    def max(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def max(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def max(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def max(
        self: ColExpr[String],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[String]: ...

    @overload
    def max(
        self: ColExpr[Datetime],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Datetime]: ...

    @overload
    def max(
        self: ColExpr[Date],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Date]: ...

    @overload
    def max(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]: ...

    def max(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        """Computes the maximum value in each group."""

        return ColFn(ops.max, self, partition_by=partition_by, filter=filter)

    @overload
    def mean(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def mean(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def mean(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    def mean(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        """Computes the average value in each group."""

        return ColFn(ops.mean, self, partition_by=partition_by, filter=filter)

    @overload
    def min(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def min(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def min(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def min(
        self: ColExpr[String],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[String]: ...

    @overload
    def min(
        self: ColExpr[Datetime],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Datetime]: ...

    @overload
    def min(
        self: ColExpr[Date],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Date]: ...

    @overload
    def min(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Bool]: ...

    def min(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        """Computes the minimum value in each group."""

        return ColFn(ops.min, self, partition_by=partition_by, filter=filter)

    def __mod__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        """
        The remainder of integer division %

        Warning
        -------
        This operator behaves differently than in polars. There are at least two
        conventions how `%` and :doc:`// <pydiverse.transform.ColExpr.__floordiv__>`
        should behave  for negative inputs. We follow the one that C, C++ and all
        currently supported SQL backends follow. This means that the output has the same
        sign as the left hand side of the input, regardless of the right hand side.

        See also
        --------
        __floordiv__

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [65, -65, 65, -65],
        ...         "b": [7, 7, -7, -7],
        ...     }
        ... )
        >>> t >> mutate(r=t.a % t.b) >> show()
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ r   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 65  ┆ 7   ┆ 2   │
        │ -65 ┆ 7   ┆ -2  │
        │ 65  ┆ -7  ┆ 2   │
        │ -65 ┆ -7  ┆ -2  │
        └─────┴─────┴─────┘
        """

        return ColFn(ops.mod, self, rhs)

    def __rmod__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]:
        """
        The remainder of integer division %

        Warning
        -------
        This operator behaves differently than in polars. There are at least two
        conventions how `%` and :doc:`// <pydiverse.transform.ColExpr.__floordiv__>`
        should behave  for negative inputs. We follow the one that C, C++ and all
        currently supported SQL backends follow. This means that the output has the same
        sign as the left hand side of the input, regardless of the right hand side.

        See also
        --------
        __floordiv__

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [65, -65, 65, -65],
        ...         "b": [7, 7, -7, -7],
        ...     }
        ... )
        >>> t >> mutate(r=t.a % t.b) >> show()
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ r   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 65  ┆ 7   ┆ 2   │
        │ -65 ┆ 7   ┆ -2  │
        │ 65  ┆ -7  ┆ 2   │
        │ -65 ┆ -7  ┆ -2  │
        └─────┴─────┴─────┘
        """

        return ColFn(ops.mod, rhs, self)

    @overload
    def __mul__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __mul__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __mul__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __mul__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Multiplication *"""

        return ColFn(ops.mul, self, rhs)

    @overload
    def __rmul__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __rmul__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rmul__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __rmul__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Multiplication *"""

        return ColFn(ops.mul, rhs, self)

    @overload
    def __neg__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __neg__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __neg__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __neg__(self: ColExpr) -> ColExpr:
        """The unary - (negation) operator (__neg__)"""

        return ColFn(ops.neg, self)

    def __ne__(self: ColExpr, rhs: ColExpr) -> ColExpr[Bool]:
        """Non-equality comparison !="""

        return ColFn(ops.not_equal, self, rhs)

    def nulls_first(self: ColExpr) -> ColExpr:
        """
        Specifies that nulls are placed at the beginning of the ordering.

        This does not mean that nulls are considered to be `less` than any other
        element. I.e. if both `nulls_first` and `descending` are given, nulls will still
        be placed at the beginning.

        If neither `nulls_first` nor `nulls_last` is specified, the position of nulls is
        backend-dependent.

        Can only be used in expressions given to the `arrange` verb or as as an
        `arrange` keyword argument.
        """

        return ColFn(ops.nulls_first, self)

    def nulls_last(self: ColExpr) -> ColExpr:
        """
        Specifies that nulls are placed at the end of the ordering.

        This does not mean that nulls are considered to be `greater` than any other
        element. I.e. if both `nulls_last` and `descending` are given, nulls will still
        be placed at the end.

        If neither `nulls_first` nor `nulls_last` is specified, the position of nulls is
        backend-dependent.

        Can only be used in expressions given to the `arrange` verb or as as an
        `arrange` keyword argument.
        """

        return ColFn(ops.nulls_last, self)

    @overload
    def __pos__(self: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __pos__(self: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __pos__(self: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __pos__(self: ColExpr) -> ColExpr:
        """The unary + operator (__pos__)"""

        return ColFn(ops.pos, self)

    @overload
    def __pow__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Float]: ...

    @overload
    def __pow__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __pow__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __pow__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """
        Computes the power x ** y.

        Note
        ----
        Polars throws on negative exponents in the integer case. A polars error like
        `failed to convert X to u32` may be due to negative inputs to this function.
        """

        return ColFn(ops.pow, self, rhs)

    @overload
    def __rpow__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Float]: ...

    @overload
    def __rpow__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rpow__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    def __rpow__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """
        Computes the power x ** y.

        Note
        ----
        Polars throws on negative exponents in the integer case. A polars error like
        `failed to convert X to u32` may be due to negative inputs to this function.
        """

        return ColFn(ops.pow, rhs, self)

    @overload
    def prefix_sum(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def prefix_sum(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def prefix_sum(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr[Decimal]: ...

    def prefix_sum(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr:
        """
        The sum of all preceding elements and the current element.
        """

        return ColFn(ops.prefix_sum, self, partition_by=partition_by, arrange=arrange)

    @overload
    def round(self: ColExpr[Int], decimals: int = 0) -> ColExpr[Int]: ...

    @overload
    def round(self: ColExpr[Float], decimals: int = 0) -> ColExpr[Float]: ...

    @overload
    def round(self: ColExpr[Decimal], decimals: int = 0) -> ColExpr[Decimal]: ...

    def round(self: ColExpr, decimals: int = 0) -> ColExpr:
        """
        Rounds to a given number of decimals.

        :param decimals:
            The number of decimals to round by.
        """

        return ColFn(ops.round, self, decimals)

    def shift(
        self: ColExpr,
        n: int,
        fill_value: ColExpr = None,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr:
        """
        Shifts values in the column by an offset.

        :param n:
            The number of places to shift by. May be negative.

        :param fill_value:
            The value to write to the empty spaces created by the shift. Defaults to
            null.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": [5, -1, 435, -34, 8, None, 0],
        ...         "b": ["r", "True", "??", ".  .", "-1/12", "abc", "#"],
        ...     }
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         x=t.a.shift(2, -40),
        ...         y=t.b.shift(1, arrange=t.a.nulls_last()),
        ...     )
        ...     >> show()
        ... )
        Table <unnamed>, backend: PolarsImpl
        shape: (7, 4)
        ┌──────┬───────┬─────┬───────┐
        │ a    ┆ b     ┆ x   ┆ y     │
        │ ---  ┆ ---   ┆ --- ┆ ---   │
        │ i64  ┆ str   ┆ i64 ┆ str   │
        ╞══════╪═══════╪═════╪═══════╡
        │ 5    ┆ r     ┆ -40 ┆ #     │
        │ -1   ┆ True  ┆ -40 ┆ .  .  │
        │ 435  ┆ ??    ┆ 5   ┆ -1/12 │
        │ -34  ┆ .  .  ┆ -1  ┆ null  │
        │ 8    ┆ -1/12 ┆ 435 ┆ r     │
        │ null ┆ abc   ┆ -34 ┆ ??    │
        │ 0    ┆ #     ┆ 8   ┆ True  │
        └──────┴───────┴─────┴───────┘
        """

        return ColFn(
            ops.shift, self, n, fill_value, partition_by=partition_by, arrange=arrange
        )

    @overload
    def __sub__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __sub__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __sub__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __sub__(
        self: ColExpr[Datetime], rhs: ColExpr[Datetime]
    ) -> ColExpr[Duration]: ...

    @overload
    def __sub__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Duration]: ...

    def __sub__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Subtraction -"""

        return ColFn(ops.sub, self, rhs)

    @overload
    def __rsub__(self: ColExpr[Int], rhs: ColExpr[Int]) -> ColExpr[Int]: ...

    @overload
    def __rsub__(self: ColExpr[Float], rhs: ColExpr[Float]) -> ColExpr[Float]: ...

    @overload
    def __rsub__(self: ColExpr[Decimal], rhs: ColExpr[Decimal]) -> ColExpr[Decimal]: ...

    @overload
    def __rsub__(
        self: ColExpr[Datetime], rhs: ColExpr[Datetime]
    ) -> ColExpr[Duration]: ...

    @overload
    def __rsub__(self: ColExpr[Date], rhs: ColExpr[Date]) -> ColExpr[Duration]: ...

    def __rsub__(self: ColExpr, rhs: ColExpr) -> ColExpr:
        """Subtraction -"""

        return ColFn(ops.sub, rhs, self)

    @overload
    def sum(
        self: ColExpr[Int],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    @overload
    def sum(
        self: ColExpr[Float],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Float]: ...

    @overload
    def sum(
        self: ColExpr[Decimal],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Decimal]: ...

    @overload
    def sum(
        self: ColExpr[Bool],
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[Int]: ...

    def sum(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr:
        """Computes the sum of values in each group."""

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
        """True division /"""

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
        """True division /"""

        return ColFn(ops.truediv, rhs, self)

    str: StrNamespace
    dt: DtNamespace
    dur: DurNamespace
    list: ListNamespace


@dataclasses.dataclass(slots=True)
class FnNamespace:
    arg: ColExpr


@register_accessor("str")
@dataclasses.dataclass(slots=True)
class StrNamespace(FnNamespace):
    def contains(self: ColExpr[String], substr: str) -> ColExpr[Bool]:
        """
        Whether the string contains a given substring.

        :param substr:
            The substring to look for.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
        ...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
        ...     },
        ...     name="string table",
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         j=t.a.str.contains(" "),
        ...         k=t.b.str.contains("a"),
        ...         l=t.b.str.contains(""),
        ...     )
        ...     >> show()
        ... )
        Table string table, backend: PolarsImpl
        shape: (5, 5)
        ┌────────┬────────────┬───────┬───────┬──────┐
        │ a      ┆ b          ┆ j     ┆ k     ┆ l    │
        │ ---    ┆ ---        ┆ ---   ┆ ---   ┆ ---  │
        │ str    ┆ str        ┆ bool  ┆ bool  ┆ bool │
        ╞════════╪════════════╪═══════╪═══════╪══════╡
        │   BCD  ┆ 12431      ┆ true  ┆ false ┆ true │
        │ -- 00  ┆ transform  ┆ true  ┆ true  ┆ true │
        │  A^^u  ┆ 12__*m     ┆ true  ┆ false ┆ true │
        │ -O2    ┆            ┆ false ┆ false ┆ true │
        │        ┆ abbabbabba ┆ false ┆ true  ┆ true │
        └────────┴────────────┴───────┴───────┴──────┘
        """

        return ColFn(ops.str_contains, self.arg, substr)

    def ends_with(self: ColExpr[String], suffix: str) -> ColExpr[Bool]:
        """
        Whether the string ends with a given suffix.

        :param suffix:
            The suffix to check.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
        ...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
        ...     },
        ...     name="string table",
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         j=t.a.str.ends_with(""),
        ...         k=t.b.str.ends_with("m"),
        ...         l=t.a.str.ends_with("^u"),
        ...     )
        ...     >> show()
        ... )
        Table string table, backend: PolarsImpl
        shape: (5, 5)
        ┌────────┬────────────┬──────┬───────┬───────┐
        │ a      ┆ b          ┆ j    ┆ k     ┆ l     │
        │ ---    ┆ ---        ┆ ---  ┆ ---   ┆ ---   │
        │ str    ┆ str        ┆ bool ┆ bool  ┆ bool  │
        ╞════════╪════════════╪══════╪═══════╪═══════╡
        │   BCD  ┆ 12431      ┆ true ┆ false ┆ false │
        │ -- 00  ┆ transform  ┆ true ┆ true  ┆ false │
        │  A^^u  ┆ 12__*m     ┆ true ┆ true  ┆ true  │
        │ -O2    ┆            ┆ true ┆ false ┆ false │
        │        ┆ abbabbabba ┆ true ┆ false ┆ false │
        └────────┴────────────┴──────┴───────┴───────┘
        """

        return ColFn(ops.str_ends_with, self.arg, suffix)

    def join(
        self: ColExpr[String],
        delimiter: str = "",
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
    ) -> ColExpr[String]:
        """
        Concatenates all strings in a group to a single string.

        :param delimiter:
            The string to insert between the elements."""

        return ColFn(
            ops.str_join,
            self.arg,
            delimiter,
            partition_by=partition_by,
            filter=filter,
            arrange=arrange,
        )

    def len(self: ColExpr[String]) -> ColExpr[Int]:
        """
        Computes the length of the string.

        Leading and trailing whitespace is included in the length.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
        ...         "b": ["12431", "transform", "12__*m", "   "],
        ...     },
        ...     name="string table",
        ... )
        >>> t >> mutate(j=t.a.str.len(), k=t.b.str.len()) >> show()
        Table string table, backend: PolarsImpl
        shape: (4, 4)
        ┌────────┬───────────┬─────┬─────┐
        │ a      ┆ b         ┆ j   ┆ k   │
        │ ---    ┆ ---       ┆ --- ┆ --- │
        │ str    ┆ str       ┆ i64 ┆ i64 │
        ╞════════╪═══════════╪═════╪═════╡
        │   BCD  ┆ 12431     ┆ 6   ┆ 5   │
        │ -- 00  ┆ transform ┆ 5   ┆ 9   │
        │  A^^u  ┆ 12__*m    ┆ 5   ┆ 6   │
        │ -O2    ┆           ┆ 3   ┆ 3   │
        └────────┴───────────┴─────┴─────┘
        """

        return ColFn(ops.str_len, self.arg)

    def lower(self: ColExpr[String]) -> ColExpr[String]:
        """
        Converts all alphabet letters to lower case.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
        ...         "b": ["12431", "transform", "12__*m", "   "],
        ...     },
        ...     name="string table",
        ... )
        >>> t >> mutate(j=t.a.str.lower(), k=t.b.str.lower()) >> show()
        Table string table, backend: PolarsImpl
        shape: (4, 4)
        ┌────────┬───────────┬────────┬───────────┐
        │ a      ┆ b         ┆ j      ┆ k         │
        │ ---    ┆ ---       ┆ ---    ┆ ---       │
        │ str    ┆ str       ┆ str    ┆ str       │
        ╞════════╪═══════════╪════════╪═══════════╡
        │   BCD  ┆ 12431     ┆   bcd  ┆ 12431     │
        │ -- 00  ┆ transform ┆ -- 00  ┆ transform │
        │  A^^u  ┆ 12__*m    ┆  a^^u  ┆ 12__*m    │
        │ -O2    ┆           ┆ -o2    ┆           │
        └────────┴───────────┴────────┴───────────┘
        """

        return ColFn(ops.str_lower, self.arg)

    def replace_all(
        self: ColExpr[String], substr: str, replacement: str
    ) -> ColExpr[String]:
        """
        Replaces all occurrences of a given substring by a different string.

        :param substr:
            The string to replace.

        :param replacement:
            The replacement string.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
        ...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
        ...     },
        ...     name="string table",
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         r=t.a.str.replace_all("-", "?"),
        ...         s=t.b.str.replace_all("ansf", "[---]"),
        ...         u=t.b.str.replace_all("abba", "#"),
        ...     )
        ...     >> show()
        ... )
        Table string table, backend: PolarsImpl
        shape: (5, 5)
        ┌────────┬────────────┬────────┬────────────┬───────────┐
        │ a      ┆ b          ┆ r      ┆ s          ┆ u         │
        │ ---    ┆ ---        ┆ ---    ┆ ---        ┆ ---       │
        │ str    ┆ str        ┆ str    ┆ str        ┆ str       │
        ╞════════╪════════════╪════════╪════════════╪═══════════╡
        │   BCD  ┆ 12431      ┆   BCD  ┆ 12431      ┆ 12431     │
        │ -- 00  ┆ transform  ┆ ?? 00  ┆ tr[---]orm ┆ transform │
        │  A^^u  ┆ 12__*m     ┆  A^^u  ┆ 12__*m     ┆ 12__*m    │
        │ -O2    ┆            ┆ ?O2    ┆            ┆           │
        │        ┆ abbabbabba ┆        ┆ abbabbabba ┆ #bb#      │
        └────────┴────────────┴────────┴────────────┴───────────┘
        """

        return ColFn(ops.str_replace_all, self.arg, substr, replacement)

    def slice(
        self: ColExpr[String], offset: ColExpr[Int], n: ColExpr[Int]
    ) -> ColExpr[String]:
        """
        Returns a substring of the input string.

        :param offset:
            The 0-based index of the first character included in the result.

        :param n:
            The number of characters to include. If the string is shorter than *offset*
            + *n*, the result only includes as many characters as there are.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
        ...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
        ...     },
        ...     name="string table",
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         j=t.a.str.slice(0, 2),
        ...         k=t.b.str.slice(4, 10),
        ...     )
        ...     >> show()
        ... )
        Table string table, backend: PolarsImpl
        shape: (5, 4)
        ┌────────┬────────────┬─────┬────────┐
        │ a      ┆ b          ┆ j   ┆ k      │
        │ ---    ┆ ---        ┆ --- ┆ ---    │
        │ str    ┆ str        ┆ str ┆ str    │
        ╞════════╪════════════╪═════╪════════╡
        │   BCD  ┆ 12431      ┆     ┆ 1      │
        │ -- 00  ┆ transform  ┆ --  ┆ sform  │
        │  A^^u  ┆ 12__*m     ┆  A  ┆ *m     │
        │ -O2    ┆            ┆ -O  ┆        │
        │        ┆ abbabbabba ┆     ┆ bbabba │
        └────────┴────────────┴─────┴────────┘
        """

        return ColFn(ops.str_slice, self.arg, offset, n)

    def starts_with(self: ColExpr[String], prefix: str) -> ColExpr[Bool]:
        """
        Whether the string starts with a given prefix.

        :param prefix:
            The prefix to check.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
        ...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
        ...     },
        ...     name="string table",
        ... )
        >>> (
        ...     t
        ...     >> mutate(
        ...         j=t.a.str.starts_with("-"),
        ...         k=t.b.str.starts_with("12"),
        ...     )
        ...     >> show()
        ... )
        Table string table, backend: PolarsImpl
        shape: (5, 4)
        ┌────────┬────────────┬───────┬───────┐
        │ a      ┆ b          ┆ j     ┆ k     │
        │ ---    ┆ ---        ┆ ---   ┆ ---   │
        │ str    ┆ str        ┆ bool  ┆ bool  │
        ╞════════╪════════════╪═══════╪═══════╡
        │   BCD  ┆ 12431      ┆ false ┆ true  │
        │ -- 00  ┆ transform  ┆ true  ┆ false │
        │  A^^u  ┆ 12__*m     ┆ false ┆ true  │
        │ -O2    ┆            ┆ true  ┆ false │
        │        ┆ abbabbabba ┆ false ┆ false │
        └────────┴────────────┴───────┴───────┘
        """

        return ColFn(ops.str_starts_with, self.arg, prefix)

    def strip(self: ColExpr[String]) -> ColExpr[String]:
        """
        Removes leading and trailing whitespace.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
        ...         "b": ["12431", "transform", "12__*m", "   "],
        ...     },
        ...     name="string table",
        ... )
        >>> t >> mutate(j=t.a.str.strip(), k=t.b.str.strip()) >> show()
        Table string table, backend: PolarsImpl
        shape: (4, 4)
        ┌────────┬───────────┬───────┬───────────┐
        │ a      ┆ b         ┆ j     ┆ k         │
        │ ---    ┆ ---       ┆ ---   ┆ ---       │
        │ str    ┆ str       ┆ str   ┆ str       │
        ╞════════╪═══════════╪═══════╪═══════════╡
        │   BCD  ┆ 12431     ┆ BCD   ┆ 12431     │
        │ -- 00  ┆ transform ┆ -- 00 ┆ transform │
        │  A^^u  ┆ 12__*m    ┆ A^^u  ┆ 12__*m    │
        │ -O2    ┆           ┆ -O2   ┆           │
        └────────┴───────────┴───────┴───────────┘
        """

        return ColFn(ops.str_strip, self.arg)

    def to_date(self: ColExpr[String]) -> ColExpr[Date]:
        """"""

        return ColFn(ops.str_to_date, self.arg)

    def to_datetime(self: ColExpr[String]) -> ColExpr[Datetime]:
        """"""

        return ColFn(ops.str_to_datetime, self.arg)

    def upper(self: ColExpr[String]) -> ColExpr[String]:
        """
        Converts all alphabet letters to upper case.

        Examples
        --------
        >>> t = pdt.Table(
        ...     {
        ...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
        ...         "b": ["12431", "transform", "12__*m", "   "],
        ...     },
        ...     name="string table",
        ... )
        >>> t >> mutate(j=t.a.str.upper(), k=t.b.str.upper()) >> show()
        Table string table, backend: PolarsImpl
        shape: (4, 4)
        ┌────────┬───────────┬────────┬───────────┐
        │ a      ┆ b         ┆ j      ┆ k         │
        │ ---    ┆ ---       ┆ ---    ┆ ---       │
        │ str    ┆ str       ┆ str    ┆ str       │
        ╞════════╪═══════════╪════════╪═══════════╡
        │   BCD  ┆ 12431     ┆   BCD  ┆ 12431     │
        │ -- 00  ┆ transform ┆ -- 00  ┆ TRANSFORM │
        │  A^^u  ┆ 12__*m    ┆  A^^U  ┆ 12__*M    │
        │ -O2    ┆           ┆ -O2    ┆           │
        └────────┴───────────┴────────┴───────────┘
        """

        return ColFn(ops.str_upper, self.arg)


@register_accessor("dt")
@dataclasses.dataclass(slots=True)
class DtNamespace(FnNamespace):
    @overload
    def day(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day(self: ColExpr) -> ColExpr:
        """Extracts the day component."""

        return ColFn(ops.dt_day, self.arg)

    @overload
    def day_of_week(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day_of_week(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day_of_week(self: ColExpr) -> ColExpr:
        """
        The number of the current weekday.

        This is one-based, so Monday is 1 and Sunday is 7.
        """

        return ColFn(ops.dt_day_of_week, self.arg)

    @overload
    def day_of_year(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def day_of_year(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def day_of_year(self: ColExpr) -> ColExpr:
        """
        The number of days since the beginning of the year.

        This is one-based, so it returns 1 for the 1st of January.
        """

        return ColFn(ops.dt_day_of_year, self.arg)

    def hour(self: ColExpr[Datetime]) -> ColExpr[Int]:
        """Extracts the hour component."""

        return ColFn(ops.dt_hour, self.arg)

    def microsecond(self: ColExpr[Datetime]) -> ColExpr[Int]:
        """Extracts the microsecond component."""

        return ColFn(ops.dt_microsecond, self.arg)

    def millisecond(self: ColExpr[Datetime]) -> ColExpr[Int]:
        """Extracts the millisecond component."""

        return ColFn(ops.dt_millisecond, self.arg)

    def minute(self: ColExpr[Datetime]) -> ColExpr[Int]:
        """Extracts the minute component."""

        return ColFn(ops.dt_minute, self.arg)

    @overload
    def month(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def month(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def month(self: ColExpr) -> ColExpr:
        """Extracts the month component."""

        return ColFn(ops.dt_month, self.arg)

    def second(self: ColExpr[Datetime]) -> ColExpr[Int]:
        """Extracts the second component."""

        return ColFn(ops.dt_second, self.arg)

    @overload
    def year(self: ColExpr[Date]) -> ColExpr[Int]: ...

    @overload
    def year(self: ColExpr[Datetime]) -> ColExpr[Int]: ...

    def year(self: ColExpr) -> ColExpr:
        """Extracts the year component."""

        return ColFn(ops.dt_year, self.arg)


@register_accessor("dur")
@dataclasses.dataclass(slots=True)
class DurNamespace(FnNamespace):
    def days(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_days, self.arg)

    def hours(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_hours, self.arg)

    def microseconds(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_microseconds, self.arg)

    def milliseconds(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_milliseconds, self.arg)

    def minutes(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_minutes, self.arg)

    def seconds(self: ColExpr[Duration]) -> ColExpr[Int]:
        """"""

        return ColFn(ops.dur_seconds, self.arg)


@register_accessor("list")
@dataclasses.dataclass(slots=True)
class ListNamespace(FnNamespace):
    def agg(
        self: ColExpr,
        *,
        partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
        arrange: ColExpr | Iterable[ColExpr] | None = None,
        filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
    ) -> ColExpr[List]:
        """
        Collect the elements of each group in a list.
        """

        return ColFn(
            ops.list_agg,
            self.arg,
            partition_by=partition_by,
            arrange=arrange,
            filter=filter,
        )


# --- generated code ends here, do not delete this comment ---


class Col(ColExpr):
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
            return str(self.export(Polars(lazy=False)))
        except Exception as e:
            return (
                f"could not evaluate {repr(self)} due to "
                f"{e.__class__.__name__}: {str(e)}"
            )

    def __hash__(self) -> int:
        return hash(self._uuid)

    def export(self, target: Target) -> Any:
        """
        Exports a column to a polars or pandas Series.

        :param target:
            The data frame library to export to. Can be a ``Polars`` or ``Pandas``
            object. The ``lazy`` kwarg for polars is ignored.

        :return:
            A polars or pandas Series.

        Examples
        --------
        >>> t1 = pdt.Table({"h": [2.465, 0.22, -4.477, 10.8, -81.2, 0.0]})
        >>> t1.h.show()
        shape: (6,)
        Series: 'h' [f64]
        [
                2.465
                0.22
                -4.477
                10.8
                -81.2
                0.0
        ]
        >>> t1.h.export(Pandas())
        0    2.465
        1     0.22
        2   -4.477
        3     10.8
        4    -81.2
        5      0.0
        Name: h, dtype: double[pyarrow]
        """

        from pydiverse.transform._internal.backend.table_impl import get_backend
        from pydiverse.transform._internal.tree.verbs import Select

        ast = Select(self._ast, [self])
        df = get_backend(self._ast).export(ast, target, schema_overrides={})
        if isinstance(target, Polars):
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            return df.get_column(self.name)
        else:
            assert isinstance(target, Pandas)
            return pd.Series(df[self.name])


class ColName(ColExpr):
    def __init__(
        self, name: str, dtype: Dtype | None = None, ftype: Ftype | None = None
    ):
        self.name = name
        super().__init__(dtype, ftype)

    def __repr__(self) -> str:
        dtype_str = f" ({self.dtype()})" if self.dtype() is not None else ""
        return f"C.{self.name}{dtype_str}"


class LiteralCol(ColExpr):
    def __init__(self, val: Any, dtype: Dtype | None = None):
        self.val = val
        if dtype is None:
            try:
                dtype = types.from_python(val)
            except KeyError as ke:
                raise TypeError(
                    f"invalid type `{type(val).__name__}` found in column expression. "
                    "Objects used in a column expression must have type `ColExpr` or a "
                    "suitable python builtin type"
                ) from ke
        dtype = types.with_const(dtype)
        super().__init__(dtype, Ftype.ELEMENT_WISE)

    def __repr__(self):
        return f"lit({repr(self.val)}, {self.dtype()})"


class ColFn(ColExpr):
    def __init__(self, op: Operator, *args: ColExpr, **kwargs: list[ColExpr | Order]):
        self.op = op
        # While building the expression tree, we have to allow markers.
        self.args: list[ColExpr] = [
            wrap_literal(arg, allow_markers=True) for arg in args
        ]
        self.context_kwargs = clean_kwargs(**kwargs)

        # TODO: probably it is faster and produces nicer SQL code if we put this in
        # WITHIN GROUP for aggregation functions. On polars use col.filter().
        if filters := self.context_kwargs.get("filter"):
            if len(self.args) == 0:
                assert self.op == ops.count_star
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
        context_kwarg_dtypes = [
            elem.dtype() for elem in itertools.chain(*self.context_kwargs.values())
        ]

        # we don't need the context_kwargs' types but we need to make sure type checks
        # are run on them
        if None in arg_dtypes or None in context_kwarg_dtypes:
            return None

        self._dtype = self.op.return_type(arg_dtypes)
        if self.op.ftype == Ftype.ELEMENT_WISE and all(
            types.is_const(argt)
            for argt in itertools.chain(arg_dtypes, context_kwarg_dtypes)
        ):
            self._dtype = Const(self._dtype)

        return self._dtype

    def ftype(self, *, agg_is_window: bool | None = None):
        """
        Determine the ftype based on the arguments.

            e(e) -> e       a(e) -> a       w(e) -> w
            e(a) -> a       a(a) -> Err     w(a) -> w
            e(w) -> w       a(w) -> Err     w(w) -> Err

        If the operator ftype is incompatible with the arguments, this function raises
        an Exception.
        """

        if self._ftype is not None:
            return self._ftype
        if self.op.ftype == Ftype.AGGREGATE and agg_is_window is None:
            return None

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
            if (
                cond.dtype() is not None
                and not types.without_const(cond.dtype()) == types.Bool()
            ):
                raise TypeError(
                    f"argument `{cond}` for `when` must be of boolean type, but has "
                    f"type `{cond.dtype()}`"
                )

        val_types = [val.dtype() for _, val in self.cases]
        if self.default_val is not None:
            val_types.append(self.default_val.dtype())
        if None in val_types:
            return None
        val_types = [types.without_const(t) for t in val_types]

        if any(cond.dtype() is None for cond, _ in self.cases):
            return None

        self._dtype = types.lca_type(val_types)

        if all(
            (
                *(
                    types.is_const(cond.dtype()) and types.is_const(val.dtype())
                    for cond, val in self.cases
                ),
                self.default_val is None or types.is_const(self.default_val.dtype()),
            )
        ):
            self._dtype = types.with_const(self._dtype)

        return self._dtype

    def ftype(self, *, agg_is_window: bool | None = None):
        if self._ftype is not None:
            return self._ftype

        val_ftypes = set()
        # TODO: does it actually matter if we add stuff that is const? it should be
        # elemwise anyway...
        if self.default_val is not None and not types.is_const(
            self.default_val.dtype()
        ):
            val_ftypes.add(self.default_val.ftype(agg_is_window=agg_is_window))

        for cond, val in self.cases:
            cond.ftype(agg_is_window=agg_is_window)
            if val.dtype() is not None and not types.is_const(val.dtype()):
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
    def __init__(self, val: ColExpr, target_type: Dtype):
        if type(target_type) is type:
            target_type = target_type()
        if types.is_const(target_type):
            raise TypeError("cannot cast to `const` type")

        self.val = val
        self.target_type = copy.copy(target_type)
        super().__init__(copy.copy(target_type))
        self.dtype()

    def dtype(self) -> Dtype:
        # Since `ColExpr.dtype` is also responsible for type checking, we may not set
        # `_dtype` until we are able to retrieve the type of `val`.
        if self.val.dtype() is None:
            return None

        assert not types.is_const(self.target_type)

        if not types.converts_to(self.val.dtype(), self.target_type):
            valid_casts = {
                *((String(), t) for t in (*INT_SUBTYPES, *FLOAT_SUBTYPES)),
                *(
                    (ft, it)
                    for ft, it in itertools.product(FLOAT_SUBTYPES, INT_SUBTYPES)
                ),
                (Datetime(), Date()),
                (Date(), Datetime()),
                *(
                    (t, String())
                    for t in (
                        Int(),
                        *INT_SUBTYPES,
                        Float(),
                        *FLOAT_SUBTYPES,
                        Datetime(),
                        Date(),
                    )
                ),
                *(
                    (t, u)
                    for t, u in itertools.chain(
                        itertools.product(
                            (Int(), *INT_SUBTYPES), (*FLOAT_SUBTYPES, *INT_SUBTYPES)
                        ),
                        itertools.product(
                            (Float(), *FLOAT_SUBTYPES), (*FLOAT_SUBTYPES, *INT_SUBTYPES)
                        ),
                    )
                ),
                *((Bool(), t) for t in itertools.chain(FLOAT_SUBTYPES, INT_SUBTYPES)),
            }

            if (
                types.without_const(self.val.dtype()),
                self.target_type,
            ) not in valid_casts:
                hint = ""
                if types.without_const(
                    self.val.dtype()
                ) == String() and self.target_type in (
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

        if types.is_const(self.val.dtype()):
            self._dtype = types.with_const(self._dtype)
        return self._dtype

    def ftype(self, *, agg_is_window: bool | None = None) -> Ftype:
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
        if not isinstance(expr, ColExpr):
            raise TypeError(
                f"argument to `arrange` must be a `ColExpr`, found `{type(expr)}` "
                "instead.\n"
                "hint: maybe you forgot to close parentheses `()`?"
            )
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

    def dtype(self) -> Dtype:
        return self.order_by.dtype()

    def ftype(self, *, agg_is_window: bool | None = None) -> Ftype:
        return self.order_by.ftype(agg_is_window=agg_is_window)

    def iter_subtree(self) -> Iterable[ColExpr]:
        yield from self.order_by.iter_subtree()

    def iter_children(self) -> Iterable[ColExpr]:
        yield from self.order_by.iter_children()

    def map_subtree(self, g: Callable[[ColExpr], ColExpr]) -> Order:
        return Order(self.order_by.map_subtree(g), self.descending, self.nulls_last)


def wrap_literal(expr: Any, *, allow_markers=False) -> Any:
    if isinstance(expr, ColExpr | Order):
        if isinstance(expr, ColFn) and (
            (isinstance(expr.op, Marker) and not allow_markers)
            or (
                # markers can only be at the top of an expression tree
                not isinstance(expr.op, Marker)
                and (
                    marker_args := [
                        arg
                        for arg in expr.args
                        if isinstance(arg, ColFn) and isinstance(arg.op, Marker)
                    ]
                )
            )
        ):
            marker = expr.op if isinstance(expr.op, Marker) else marker_args[0].op
            raise TypeError(
                f"invalid usage of `{marker.name}` in a column expression.\n"
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
        key: [val]
        if not isinstance(val, Iterable) or isinstance(val, str)
        else list(val)
        for key, val in kwargs.items()
        if val is not None
    }
    if (partition_by := kwargs.get("partition_by")) is not None:
        kwargs["partition_by"] = [
            ColName(col) if isinstance(col, str) else col for col in partition_by
        ]
    if (arrange := kwargs.get("arrange")) is not None:
        kwargs["arrange"] = [
            Order.from_col_expr(ColName(ord) if isinstance(ord, str) else ord)
            for ord in arrange
        ]
    return {key: [wrap_literal(val) for val in arr] for key, arr in kwargs.items()}
