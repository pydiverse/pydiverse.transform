from __future__ import annotations

import datetime


class Dtype:
    __slots__ = ("const",)

    def __init__(self, *, const: bool = False):
        self.const = const

    def __eq__(self, rhs: Dtype) -> bool:
        if not isinstance(rhs, Dtype):
            raise TypeError(f"cannot compare type `Dtype` with type `{type(rhs)}`")
        return self.const == rhs.const and type(self) is type(rhs)

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

    def without_const(self) -> Dtype:
        return type(self)()

    def can_convert_to(self, target: Dtype) -> bool:
        return (not target.const or self.const) and (
            self.without_const(),
            target.without_const(),
        ) in implicit_conversions


class Float(Dtype):
    name = "float"


class Float64(Float): ...


class Float32(Float): ...


class Decimal(Dtype):
    name = "decimal"


class Int(Dtype):
    name = "int"


class Int64(Int): ...


class Int32(Int): ...


class Int16(Int): ...


class Int8(Int): ...


class Uint64(Int): ...


class Uint32(Int): ...


class Uint16(Int): ...


class Uint8(Int): ...


class String(Dtype):
    name = "str"


class Bool(Dtype):
    name = "bool"


class Datetime(Dtype):
    name = "datetime"


class Date(Dtype):
    name = "date"


class Duration(Dtype):
    name = "duration"


# TODO: this shouldn't be a type. create a parameter class, pack vararg in there
# and allow parameters of type template
class Template(Dtype):
    name = None

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def without_modifiers(self: Dtype) -> Dtype:
        return type(self)(self.name)

    def same_kind(self, other: Dtype) -> bool:
        if not super().same_kind(other):
            return False

        return self.name == other.name

    def modifiers_compatible(self, other: Dtype) -> bool:
        """
        Check if another dtype object is compatible with the modifiers of the template.
        """
        if self.const and not other.const:
            return False
        return True


class NullType(Dtype): ...


class Tvar(Dtype):
    __slots__ = ("name",)

    def __init__(self, name: str, *, const: bool = True):
        self.name = name
        super().__init__(const=const)


D = Tvar("T")


def python_type_to_pdt(t: type) -> Dtype:
    if t is int:
        return Int()
    elif t is float:
        return Float()
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
    elif t is type(None):
        return NullType()

    raise TypeError(f"invalid usage of type {t} in a column expression")


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
    elif t <= NullType():
        return type(None)

    raise AssertionError


def dtype_from_string(t: str) -> Dtype:
    parts = [part for part in t.split(" ") if part]

    is_const = False
    is_vararg = False

    # Handle vararg
    if parts[-1] == "...":
        del parts[-1]
        is_vararg = True
    elif parts[-1].endswith("..."):
        parts[-1] = parts[-1][:-3]
        is_vararg = True

    *modifiers, base_type = parts

    # Handle modifiers
    for modifier in modifiers:
        if modifier == "const":
            is_const = True
        else:
            raise ValueError(f"Unknown type modifier '{modifier}'.")

    # Handle type
    is_template = len(base_type) == 1 and 65 <= ord(base_type) <= 90

    if is_template:
        return Template(base_type, const=is_const, vararg=is_vararg)

    if base_type == "int":
        return Int(const=is_const, vararg=is_vararg)
    if base_type == "float":
        return Float(const=is_const, vararg=is_vararg)
    if base_type == "decimal":
        return Decimal(const=is_const, vararg=is_vararg)
    if base_type == "str":
        return String(const=is_const, vararg=is_vararg)
    if base_type == "bool":
        return Bool(const=is_const, vararg=is_vararg)
    if base_type == "date":
        return Date(const=is_const, vararg=is_vararg)
    if base_type == "datetime":
        return Datetime(const=is_const, vararg=is_vararg)
    if base_type == "duration":
        return Duration(const=is_const, vararg=is_vararg)
    if base_type == "null":
        return NullType(const=is_const, vararg=is_vararg)

    raise ValueError(f"Unknown type '{base_type}'")


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

        if dtype.can_convert_to(promoted):
            continue
        if promoted.can_convert_to(dtype):
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


def is_abstract(dtype: Dtype) -> bool:
    return dtype.without_const() not in (*INT_SUBTYPES, *FLOAT_SUBTYPES)


def is_concrete(dtype: Dtype) -> bool:
    return dtype.without_const() not in (Int(), Float())


implicit_conversions = {
    (Int(), Float()): (1, 0),
    (Int(), Decimal()): (2, 0),
    **{(int_subtype, Int()): (0, 1) for int_subtype in INT_SUBTYPES},
    **{(float_subtype, Float()): (0, 1) for float_subtype in FLOAT_SUBTYPES},
}

# compute transitive closure of cost graph
for dtype in (*INT_SUBTYPES, *FLOAT_SUBTYPES):
    added_edges = {}
    for (u, v), c in implicit_conversions.items():
        if u == dtype:
            for (s, t), d in implicit_conversions.items():
                if s == v:
                    added_edges[(u, t)] = (sum(z) for z in zip(c, d, strict=True))
    implicit_conversions |= added_edges


def conversion_cost(dtype: Dtype, target: Dtype) -> tuple[int, int]:
    return implicit_conversions[(dtype, target)]


NUMERIC = (Int(), Float(), Decimal())
COMPARABLE = (Int(), Float(), Decimal(), String(), Datetime(), Date())
