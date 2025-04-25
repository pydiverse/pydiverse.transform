from __future__ import annotations

import copy
import datetime
import functools
import operator
from types import NoneType
from typing import Any

from pydiverse.common import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Duration,
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
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from pydiverse.transform._internal import errors
from pydiverse.transform._internal.ops import signature


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
    if isinstance(source, List):
        return isinstance(target, List) and converts_to(source.inner, target.inner)
    return (not is_const(target) or is_const(source)) and (
        without_const(target) in IMPLICIT_CONVS[without_const(source)]
    )


def to_python(dtype: Dtype):
    if isinstance(dtype, Const):
        return to_python(dtype.base)
    if dtype.is_int():
        return int
    elif dtype.is_float():
        return float
    elif isinstance(dtype, List):
        return list

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
    if len(dtypes) == 0:
        return NullType()

    # reduce to simple types
    if isinstance(dtypes[0], List):
        if diff := next(
            (dtype for dtype in dtypes if not isinstance(dtype, List)), None
        ):
            raise TypeError(
                f"type `{diff.__name__}` is not compatible with `List` type"
            )

        return List(lca_type([dtype.inner for dtype in dtypes]))

    if not (
        common_ancestors := functools.reduce(
            operator.and_,
            (set(IMPLICIT_CONVS[t].keys()) for t in dtypes[1:]),
            IMPLICIT_CONVS[dtypes[0]].keys(),
        )
    ):
        raise TypeError(f'incompatible types `{", ".join(dtypes)}`')

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
    Uint8(),
    Uint16(),
    Uint32(),
    Uint64(),
    Int8(),
    Int16(),
    Int32(),
    Int64(),
)
FLOAT_SUBTYPES = (Float32(), Float64())
SIMPLE_TYPES = (
    *INT_SUBTYPES,
    *FLOAT_SUBTYPES,
    Int(),
    Float(),
    Decimal(),
    String(),
    Date(),
    Datetime(),
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
    return list(IMPLICIT_CONVS[dtype].keys())


IMPLICIT_CONVS: dict[Dtype, dict[Dtype, tuple[int, int]]] = {
    Int(): {Float(): (1, 0), Decimal(): (2, 0), Int(): (0, 0)},
    **{
        int_subtype: {Int(): (0, 1), int_subtype: (0, 0)}
        for int_subtype in INT_SUBTYPES
    },
    **{
        float_subtype: {Float(): (0, 1), float_subtype: (0, 0)}
        for float_subtype in FLOAT_SUBTYPES
    },
    Float(): {Float(): (0, 0)},
    String(): {String(): (0, 0)},
    Decimal(): {Decimal(): (0, 0)},
    Datetime(): {Datetime(): (0, 0)},
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
                added_edges[target_type] = tuple(
                    sum(z) for z in zip(cost1, cost2, strict=True)
                )
    if start_type not in IMPLICIT_CONVS:
        IMPLICIT_CONVS[start_type] = added_edges
    IMPLICIT_CONVS[start_type] |= added_edges


def conversion_cost(dtype: Dtype, target: Dtype) -> tuple[int, int]:
    if isinstance(dtype, List):
        return conversion_cost(dtype.inner, target.inner)
    return IMPLICIT_CONVS[without_const(dtype)][without_const(target)]


NUMERIC = (Int(), Float(), Decimal())
COMPARABLE = (
    Int(),
    Float(),
    Decimal(),
    String(),
    Datetime(),
    Date(),
    Bool(),
)
