from __future__ import annotations

import pydiverse.common as pdc
from pydiverse.transform._internal import errors


class Const(pdc.Dtype):
    __slots__ = ("base",)

    def __init__(self, base: pdc.Dtype):
        self.base = base

    def __repr__(self):
        return "const " + repr(self.base)


def without_const(dtype: pdc.Dtype):
    """
    Removes a `const` modifier from the data type (if present).
    """
    errors.check_arg_type(pdc.Dtype, "without_const", "dtype", dtype)
    if isinstance(dtype, Const):
        return dtype.base
    return dtype


def with_const(dtype: pdc.Dtype) -> pdc.Dtype:
    """
    Adds a `const` modifier from the data type.
    """
    errors.check_arg_type(pdc.Dtype, "with_const", "dtype", dtype)
    if isinstance(dtype, Const):
        return dtype
    return Const(dtype)


def converts_to(self, target: pdc.Dtype) -> bool:
    return (
        not target.const or self.const
    ) and target.without_const() in IMPLICIT_CONVS[self.without_const()]


class Tvar(pdc.Dtype):
    __slots__ = ("name",)

    def __init__(self, name: str, *, const: bool = False):
        self.name = name
        super().__init__(const=const)

    def __eq__(self, rhs: pdc.Dtype) -> bool:
        if rhs is None:
            return False
        if not isinstance(rhs, pdc.Dtype):
            raise TypeError(f"cannot compare type `Dtype` with type `{type(rhs)}`")
        return (
            self.const == rhs.const and isinstance(rhs, Tvar) and rhs.name == self.name
        )

    def __hash__(self):
        return hash((Tvar, self.const, self.name))

    def with_const(self) -> pdc.Dtype:
        return Tvar(self.name, const=True)

    def without_const(self) -> pdc.Dtype:
        return Tvar(self.name)


D = Tvar("T")


# def python_type_to_pdt(t: type) -> Dtype:
#     if t is int:
#         return Int64()
#     elif t is float:
#         return Float64()
#     elif t is bool:
#         return Bool()
#     elif t is str:
#         return String()
#     elif t is datetime.datetime:
#         return Datetime()
#     elif t is datetime.date:
#         return Date()
#     elif t is datetime.timedelta:
#         return Duration()
#     elif t is list:
#         return List()
#     elif t is type(None):
#         return NullType()

#     raise TypeError(
#         "objects used in a column expression must have type `ColExpr` or "
#         f"a suitable python builtin type, found `{t.__name__}` instead"
#     )


# def pdt_type_to_python(t: Dtype) -> type:
#     if t <= Int():
#         return int
#     elif t <= Float():
#         return float
#     elif t <= Bool():
#         return bool
#     elif t <= String():
#         return str
#     elif t <= Datetime():
#         return datetime.datetime
#     elif t <= Date():
#         return datetime.date
#     elif t <= Duration():
#         return datetime.timedelta
#     elif t <= List():
#         return list
#     elif t <= NullType():
#         return type(None)

#     raise AssertionError


def promote_dtypes(dtypes: list[pdc.Dtype]) -> pdc.Dtype:
    if len(dtypes) == 0:
        raise ValueError("expected non empty list of dtypes")

    promoted = dtypes[0]
    for dtype in dtypes[1:]:
        if isinstance(dtype, pdc.NullType):
            continue
        if isinstance(promoted, pdc.NullType):
            promoted = dtype
            continue

        if dtype.converts_to(promoted):
            continue
        if promoted.converts_to(dtype):
            promoted = dtype
            continue

        raise TypeError(f"incompatible types {dtype} and {promoted}")

    return promoted


INT_SUBTYPES = (
    pdc.Uint8(),
    pdc.Uint16(),
    pdc.Uint32(),
    pdc.Uint64(),
    pdc.Int8(),
    pdc.Int16(),
    pdc.Int32(),
    pdc.Int64(),
)
FLOAT_SUBTYPES = (pdc.Float32(), pdc.Float64())
SIMPLE_TYPES = (
    *INT_SUBTYPES,
    *FLOAT_SUBTYPES,
    pdc.Int(),
    pdc.Float(),
    pdc.Decimal(),
    pdc.String(),
    pdc.Date(),
    pdc.Datetime(),
    pdc.Bool(),
    pdc.NullType(),
    pdc.Duration(),
)


def is_supertype(dtype: pdc.Dtype) -> bool:
    return not any(isinstance(dtype, type(t)) for t in (*INT_SUBTYPES, *FLOAT_SUBTYPES))


def is_subtype(dtype: pdc.Dtype) -> bool:
    return type(dtype) is not pdc.Int and type(dtype) is not pdc.Float


IMPLICIT_CONVS: dict[pdc.Dtype, dict[pdc.Dtype, tuple[int, int]]] = {
    pdc.Int(): {pdc.Float(): (1, 0), pdc.Decimal(): (2, 0), pdc.Int(): (0, 0)},
    **{
        int_subtype: {pdc.Int(): (0, 1), int_subtype: (0, 0)}
        for int_subtype in INT_SUBTYPES
    },
    **{
        float_subtype: {pdc.Float(): (0, 1), float_subtype: (0, 0)}
        for float_subtype in FLOAT_SUBTYPES
    },
    pdc.Float(): {pdc.Float(): (0, 0)},
    pdc.String(): {pdc.String(): (0, 0)},
    pdc.Decimal(): {pdc.Decimal(): (0, 0)},
    pdc.Datetime(): {pdc.Datetime(): (0, 0)},
    pdc.Date(): {pdc.Date(): (0, 0)},
    pdc.Bool(): {pdc.Bool(): (0, 0)},
    pdc.NullType(): {
        pdc.NullType(): (0, 0),
        **{t: (1, 0) for t in SIMPLE_TYPES if t != NullType()},
    },
    pdc.Duration(): {pdc.Duration(): (0, 0)},
    **{List(t): {List(t): (0, 0)} for t in SIMPLE_TYPES},
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


def conversion_cost(dtype: pdc.Dtype, target: pdc.Dtype) -> tuple[int, int]:
    return IMPLICIT_CONVS[dtype.without_const()][target.without_const()]


NUMERIC = (pdc.Int(), pdc.Float(), pdc.Decimal())
COMPARABLE = (
    pdc.Int(),
    pdc.Float(),
    pdc.Decimal(),
    pdc.String(),
    pdc.Datetime(),
    pdc.Date(),
)
