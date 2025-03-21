from __future__ import annotations

import datetime
import inspect


class Dtype:
    """
    Base class for all data types.
    """

    __slots__ = ("const",)

    def __init__(self, *, const: bool = False):
        self.const = const

    def __eq__(self, rhs: Dtype | type[Dtype] | None) -> bool:
        if rhs is None:
            return False
        if inspect.isclass(rhs) and issubclass(rhs, Dtype):
            rhs = rhs()

        if isinstance(rhs, Dtype):
            return self.const == rhs.const and type(self) is type(rhs)
        elif inspect.isclass(rhs) and issubclass(rhs, Dtype):
            return not self.const and type(self) is rhs
        raise TypeError(f"cannot compare type `Dtype` with type `{type(rhs)}`")

    def __le__(self, rhs: Dtype):
        if rhs.const and not self.const:
            return False
        return isinstance(self, type(rhs))

    def __ne__(self, rhs: object) -> bool:
        return not self.__eq__(rhs)

    def __hash__(self):
        return hash((type(self), self.const))

    def __repr__(self) -> str:
        return ("const " if self.const else "") + self.__class__.__name__

    def with_const(self) -> Dtype:
        """
        Adds a `const` modifier from the data type.
        """
        return type(self)(const=True)

    def without_const(self) -> Dtype:
        """
        Removes a `const` modifier from the data type (if present).
        """
        return type(self)()

    def converts_to(self, target: Dtype) -> bool:
        return (
            not target.const or self.const
        ) and target.without_const() in IMPLICIT_CONVS[self.without_const()]


class Float(Dtype): ...


class Float64(Float): ...


class Float32(Float): ...


class Decimal(Dtype): ...


class Int(Dtype): ...


class Int64(Int): ...


class Int32(Int): ...


class Int16(Int): ...


class Int8(Int): ...


class Uint64(Int): ...


class Uint32(Int): ...


class Uint16(Int): ...


class Uint8(Int): ...


class String(Dtype): ...


class Bool(Dtype): ...


class Datetime(Dtype): ...


class Date(Dtype): ...


class Duration(Dtype): ...


class List(Dtype): ...


class NullType(Dtype): ...


class Tvar(Dtype):
    __slots__ = ("name",)

    def __init__(self, name: str, *, const: bool = False):
        self.name = name
        super().__init__(const=const)

    def __eq__(self, rhs: Dtype) -> bool:
        if rhs is None:
            return False
        if not isinstance(rhs, Dtype):
            raise TypeError(f"cannot compare type `Dtype` with type `{type(rhs)}`")
        return (
            self.const == rhs.const and isinstance(rhs, Tvar) and rhs.name == self.name
        )

    def __hash__(self):
        return hash((Tvar, self.const, self.name))

    def with_const(self) -> Dtype:
        return Tvar(self.name, const=True)

    def without_const(self) -> Dtype:
        return Tvar(self.name)


D = Tvar("T")


def python_type_to_pdt(t: type) -> Dtype:
    if t is int:
        return Int64()
    elif t is float:
        return Float64()
    elif t is bool:
        return Bool()
    elif t is str:
        return String()
    elif t is datetime.datetime:
        return Datetime()
    elif t is datetime.date:
        return Date()
    elif t is datetime.timedelta:
        return Duration()
    elif t is list:
        return List()
    elif t is type(None):
        return NullType()

    raise TypeError(
        "objects used in a column expression must have type `ColExpr` or "
        f"a suitable python builtin type, found `{t.__name__}` instead"
    )


def pdt_type_to_python(t: Dtype) -> type:
    if t <= Int():
        return int
    elif t <= Float():
        return float
    elif t <= Bool():
        return bool
    elif t <= String():
        return str
    elif t <= Datetime():
        return datetime.datetime
    elif t <= Date():
        return datetime.date
    elif t <= Duration():
        return datetime.timedelta
    elif t <= List():
        return list
    elif t <= NullType():
        return type(None)

    raise AssertionError


def promote_dtypes(dtypes: list[Dtype]) -> Dtype:
    if len(dtypes) == 0:
        raise ValueError("expected non empty list of dtypes")

    promoted = dtypes[0]
    for dtype in dtypes[1:]:
        if isinstance(dtype, NullType):
            continue
        if isinstance(promoted, NullType):
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
ALL_TYPES = (
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
    List(),
)


def is_supertype(dtype: Dtype) -> bool:
    return not any(isinstance(dtype, type(t)) for t in (*INT_SUBTYPES, *FLOAT_SUBTYPES))


def is_subtype(dtype: Dtype) -> bool:
    return type(dtype) is not Int and type(dtype) is not Float


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
        **{t: (1, 0) for t in ALL_TYPES if t != NullType()},
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
    return IMPLICIT_CONVS[dtype.without_const()][target.without_const()]


NUMERIC = (Int(), Float(), Decimal())
COMPARABLE = (Int(), Float(), Decimal(), String(), Datetime(), Date())
