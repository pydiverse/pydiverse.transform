from __future__ import annotations

import datetime
from abc import ABC, abstractmethod


class Dtype(ABC):
    def __init__(self, *, const: bool = False, vararg: bool = False):
        self.const = const
        self.vararg = vararg

    def __eq__(self, rhs):
        if type(self) is rhs:
            return True
        if type(self) is not type(rhs):
            return False
        if self.const != rhs.const:
            return False
        if self.vararg != rhs.vararg:
            return False
        if self.name != rhs.name:
            return False
        return True

    def __ne__(self, rhs: object) -> bool:
        return not self.__eq__(rhs)

    def __hash__(self):
        return hash((self.name, self.const, self.vararg, type(self).__qualname__))

    def __repr__(self):
        dtype_str = ""
        if self.const:
            dtype_str += "const "
        dtype_str += self.name
        if self.vararg:
            dtype_str += "..."

        return dtype_str

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def without_modifiers(self: Dtype) -> Dtype:
        """Returns a copy of `self` with all modifiers removed"""
        return type(self)()

    def same_kind(self, other: Dtype) -> bool:
        """Check if `other` is of the same type as self.

        More specifically, `other` must be a stricter subtype of `self`.
        """
        if not isinstance(other, type(self)):
            return False

        # self.const -> other.const
        if self.const and not other.const:
            return False

        # other.vararg -> self.vararg
        if other.vararg and not self.vararg:
            return False

        return True

    def can_promote_to(self, other: Dtype) -> bool:
        return other.same_kind(self)


class Int64(Dtype):
    name = "int64"

    MIN = -(1 << 63)
    MAX = (1 << 63) - 1

    def can_promote_to(self, other: Dtype) -> bool:
        if super().can_promote_to(other):
            return True

        # int64 can be promoted to float64
        if Float64().same_kind(other):
            if other.const and not self.const:
                return False

            return True

        return False


class Float64(Dtype):
    name = "float64"


class Decimal(Dtype):
    name = "decimal"


class String(Dtype):
    name = "str"


class Bool(Dtype):
    name = "bool"


class DateTime(Dtype):
    name = "datetime"


class Date(Dtype):
    name = "date"


class Duration(Dtype):
    name = "duration"


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


class NoneDtype(Dtype):
    """DType used to represent the `None` value."""

    name = "none"


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
        return DateTime()
    elif t is datetime.date:
        return Date()
    elif t is datetime.timedelta:
        return Duration()
    elif t is type(None):
        return NoneDtype()

    raise TypeError(f"invalid usage of type {t} in a column expression")


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

    if base_type == "int64":
        return Int64(const=is_const, vararg=is_vararg)
    if base_type == "float64":
        return Float64(const=is_const, vararg=is_vararg)
    if base_type == "decimal":
        return Decimal(const=is_const, vararg=is_vararg)
    if base_type == "str":
        return String(const=is_const, vararg=is_vararg)
    if base_type == "bool":
        return Bool(const=is_const, vararg=is_vararg)
    if base_type == "date":
        return Date(const=is_const, vararg=is_vararg)
    if base_type == "datetime":
        return DateTime(const=is_const, vararg=is_vararg)
    if base_type == "duration":
        return Duration(const=is_const, vararg=is_vararg)
    if base_type == "none":
        return NoneDtype(const=is_const, vararg=is_vararg)

    raise ValueError(f"Unknown type '{base_type}'")


def promote_dtypes(dtypes: list[Dtype]) -> Dtype:
    if len(dtypes) == 0:
        raise ValueError("expected non empty list of dtypes")

    promoted = dtypes[0]
    for dtype in dtypes[1:]:
        if isinstance(dtype, NoneDtype):
            continue
        if isinstance(promoted, NoneDtype):
            promoted = dtype
            continue

        if dtype.can_promote_to(promoted):
            continue
        if promoted.can_promote_to(dtype):
            promoted = dtype
            continue

        raise TypeError(f"incompatible types {dtype} and {promoted}")

    return promoted
