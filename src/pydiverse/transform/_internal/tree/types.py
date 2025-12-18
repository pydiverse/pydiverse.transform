# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
import datetime
import functools
import operator
from typing import Any

from pydiverse.common import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Duration,
    Enum,
    Float,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    NullType,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from pydiverse.transform._internal import errors
from pydiverse.transform._internal.errors import DataTypeError
from pydiverse.transform._internal.ops import signature

NoneType = type(None)


class Const(Dtype):
    __slots__ = ("base",)

    def __init__(self, base: Dtype):
        if isinstance(base, Const):
            raise TypeError("the base type of a const type may not be const")
        self.base = base

    def __repr__(self):
        return "const " + repr(self.base)

    def __hash__(self):
        return hash((Const, self.base))

    def is_int(self):
        return self.base.is_int()

    def is_float(self):
        return self.base.is_float()

    def to_polars(self):
        return self.base.to_polars()

    def to_sql(self):
        return self.base.to_sql()

    def is_subtype(self, rhs):
        if is_const(rhs):
            return self.base.is_subtype(rhs.base)
        return self.base.is_subtype(rhs)


def is_const(dtype: Dtype):
    return isinstance(dtype, Const)


def without_const(dtype: Dtype):
    """
    Removes a `const` modifier from the data type (if present).
    """
    errors.check_arg_type(Dtype, "without_const", "dtype", dtype)
    if isinstance(dtype, Const):
        return dtype.base
    return dtype


def with_const(dtype: Dtype) -> Dtype:
    """
    Adds a `const` modifier from the data type.
    """
    errors.check_arg_type(Dtype, "with_const", "dtype", dtype)
    if isinstance(dtype, Const):
        return dtype
    return Const(dtype)


def converts_to(source: Dtype, target: Dtype) -> bool:
    if is_const(target):
        return is_const(source) and converts_to(without_const(source), without_const(target))
    source = without_const(source)
    if isinstance(source, List):
        return isinstance(target, List) and converts_to(source.inner, target.inner)
    if isinstance(source, Enum | String):
        return (
            target == source
            or target == String()
            or (type(target) is String and source.max_length is not None and target.max_length > source.max_length)
        )
    if isinstance(source, Decimal):
        return (
            target == source
            or target in FLOAT_SUBTYPES
            or target == Float()
            or target == Decimal()
            or (
                isinstance(target, Decimal)
                and target.scale >= source.scale
                and (target.precision - target.scale >= source.precision - source.scale)
            )
        )
    return target in IMPLICIT_CONVS[source]


def to_python(dtype: Dtype):
    if isinstance(dtype, Const):
        return to_python(dtype.base)
    if dtype.is_int():
        return int
    elif dtype.is_float():
        return float
    elif isinstance(dtype, List):
        return list
    elif isinstance(dtype, Enum | String):
        return str

    return {
        String(): str,
        Bool(): bool,
        Datetime(): datetime.datetime,
        Time(): datetime.time,
        Date(): datetime.date,
        Duration(): datetime.timedelta,
        NullType(): NoneType,
    }[dtype]


def from_python(value: Any):
    assert not isinstance(value, type)

    if isinstance(value, list):
        if len(value) == 0:
            return List(NullType())
        return List(lca_type([from_python(elem) for elem in value]))

    return {
        int: Int64(),
        float: Float64(),
        bool: Bool(),
        str: String(),
        datetime.datetime: Datetime(),
        datetime.date: Date(),
        datetime.time: Time(),
        datetime.timedelta: Duration(),
        NoneType: NullType(),
    }[type(value)]


class Tyvar(Dtype):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def __eq__(self, rhs: Dtype) -> bool:
        return isinstance(rhs, Tyvar) and rhs.name == self.name

    def __hash__(self):
        return hash((Tyvar, self.name))

    def __repr__(self):
        return f'Tyvar "{self.name}"'


S = Tyvar("S")


def lca_type(dtypes: list[Dtype]) -> Dtype:
    dtypes = [without_const(dtype) for dtype in dtypes if not isinstance(dtype, NullType)]
    if len(dtypes) == 0:
        return NullType()

    # reduce to simple types
    if isinstance(dtypes[0], List):
        if diff := next((dtype for dtype in dtypes if not isinstance(dtype, List)), None):
            raise DataTypeError(f"type `{diff.__name__}` is not compatible with `List` type")

        return List(lca_type([dtype.inner for dtype in dtypes]))

    if any(isinstance(dtype, Enum | String) for dtype in dtypes):
        if all(dtype == dtypes[0] for dtype in dtypes):
            return copy.copy(dtypes[0])
        if all(isinstance(dtype, Enum | String) for dtype in dtypes):
            return String()
        raise DataTypeError(f"incompatible types `{', '.join(str(d) for d in dtypes)}`")

    if any(isinstance(dtype, Decimal) for dtype in dtypes):
        if all(dtype == dtypes[0] for dtype in dtypes):
            return copy.copy(dtypes[0])
        if all(isinstance(dtype, Decimal) for dtype in dtypes):
            precision_diff = max(dtype.precision - dtype.scale for dtype in dtypes)
            scale = max(dtype.scale for dtype in dtypes)
            precision = precision_diff + scale
            return Decimal(precision, scale)
        raise DataTypeError(f"incompatible types `{', '.join(str(d) for d in dtypes)}`")

    if not (
        common_ancestors := functools.reduce(
            operator.and_,
            (set(IMPLICIT_CONVS[t].keys()) for t in dtypes[1:]),
            IMPLICIT_CONVS[dtypes[0]].keys(),
        )
    ):
        raise DataTypeError(f"incompatible types `{', '.join(str(d) for d in dtypes)}`")

    common_ancestors: list[Dtype] = list(common_ancestors)
    return copy.copy(
        common_ancestors[
            signature.best_signature_match(
                dtypes,
                [[ancestor] * len(dtypes) for ancestor in common_ancestors],
            )
        ]
    )


INT_SUBTYPES = (
    UInt8(),
    UInt16(),
    UInt32(),
    UInt64(),
    Int8(),
    Int16(),
    Int32(),
    Int64(),
)
FLOAT_SUBTYPES = (Float32(), Float64(), Decimal())
SIMPLE_TYPES = (
    *INT_SUBTYPES,
    *FLOAT_SUBTYPES,
    Int(),
    Float(),
    Decimal(),
    String(),
    Date(),
    Datetime(),
    Time(),
    Bool(),
    NullType(),
    Duration(),
)


def is_subtype(dtype: Dtype) -> bool:
    if isinstance(dtype, List):
        return is_subtype(dtype.inner)
    if isinstance(dtype, Const):
        return is_subtype(dtype.base)
    return type(dtype) is not Int and type(dtype) is not Float


# all types the given type can implicitly convert to
def implicit_conversions(dtype: Dtype) -> list[Dtype]:
    if isinstance(dtype, List):
        return [List(inner) for inner in implicit_conversions(dtype.inner)]
    if isinstance(dtype, Enum | String):
        return [String()] + ([dtype] if dtype.max_length is not None else [])
    if isinstance(dtype, Decimal):
        return list(FLOAT_SUBTYPES) + [Float()] + ([dtype] if dtype != Decimal() else [])
    return list(IMPLICIT_CONVS[dtype].keys())


IMPLICIT_CONVS: dict[Dtype, dict[Dtype, tuple[int, int]]] = {
    Int(): {Float(): (1, 0), Decimal(): (2, 0), Int(): (0, 0)},
    **{int_subtype: {Int(): (0, 1), int_subtype: (0, 0)} for int_subtype in INT_SUBTYPES},
    **{float_subtype: {Float(): (0, 1), float_subtype: (0, 0)} for float_subtype in FLOAT_SUBTYPES},
    Float(): {Float(): (0, 0)},
    String(): {String(): (0, 0)},
    Decimal(): {Decimal(): (0, 0), Float(): (0, 1)},
    Datetime(): {Datetime(): (0, 0)},
    Time(): {Time(): (0, 0)},
    Date(): {Date(): (0, 0)},
    Bool(): {Bool(): (0, 0)},
    NullType(): {
        NullType(): (0, 0),
        **{t: (1, 0) for t in SIMPLE_TYPES if t != NullType()},
    },
    Duration(): {Duration(): (0, 0)},
}

# compute transitive closure of cost graph
for start_type in (*INT_SUBTYPES, *FLOAT_SUBTYPES):
    added_edges = {}
    for intermediate_type, cost1 in IMPLICIT_CONVS[start_type].items():
        if intermediate_type in IMPLICIT_CONVS:
            for target_type, cost2 in IMPLICIT_CONVS[intermediate_type].items():
                added_edges[target_type] = tuple(sum(z) for z in zip(cost1, cost2, strict=True))
    if start_type not in IMPLICIT_CONVS:
        IMPLICIT_CONVS[start_type] = added_edges
    IMPLICIT_CONVS[start_type] |= added_edges


def conversion_cost(dtype: Dtype, target: Dtype) -> tuple[int, int]:
    if is_const(target):
        assert is_const(dtype)
        return conversion_cost(without_const(dtype), without_const(target))
    dtype = without_const(dtype)
    if isinstance(dtype, List):
        return conversion_cost(dtype.inner, target.inner)
    if isinstance(dtype, Enum | String | Decimal):
        return (0, 0) if dtype == target else (0, 1) if type(dtype) is type(target) else (0, 2)
    return IMPLICIT_CONVS[dtype][target]


NUMERIC = (Int(), Float())
COMPARABLE = (
    Int(),
    Float(),
    String(),
    Datetime(),
    Time(),
    Duration(),
    Date(),
    Bool(),
)
