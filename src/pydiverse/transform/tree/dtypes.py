from __future__ import annotations

from abc import ABC, abstractmethod

from pydiverse.transform._typing import T
from pydiverse.transform.errors import ExpressionTypeError


class DType(ABC):
    def __init__(self, *, const: bool = False, vararg: bool = False):
        self.const = const
        self.vararg = vararg

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if self.const != other.const:
            return False
        if self.vararg != other.vararg:
            return False
        if self.name != other.name:
            return False
        return True

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

    def without_modifiers(self: T) -> T:
        """Returns a copy of `self` with all modifiers removed"""
        return type(self)()

    def same_kind(self, other: DType) -> bool:
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

    def can_promote_to(self, other: DType) -> bool:
        return other.same_kind(self)


class Int(DType):
    name = "int"

    def can_promote_to(self, other: DType) -> bool:
        if super().can_promote_to(other):
            return True

        # int can be promoted to float
        if Float().same_kind(other):
            if other.const and not self.const:
                return False

            return True

        return False


class Float(DType):
    name = "float"


class String(DType):
    name = "str"


class Bool(DType):
    name = "bool"


class DateTime(DType):
    name = "datetime"


class Date(DType):
    name = "date"


class Duration(DType):
    name = "duration"


class Template(DType):
    name = None

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def without_modifiers(self: T) -> T:
        return type(self)(self.name)

    def same_kind(self, other: DType) -> bool:
        if not super().same_kind(other):
            return False

        return self.name == other.name

    def modifiers_compatible(self, other: DType) -> bool:
        """
        Check if another dtype object is compatible with the modifiers of the template.
        """
        if self.const and not other.const:
            return False
        return True


class NoneDType(DType):
    """DType used to represent the `None` value."""

    name = "none"


def dtype_from_string(t: str) -> DType:
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
        return NoneDType(const=is_const, vararg=is_vararg)

    raise ValueError(f"Unknown type '{base_type}'")


def promote_dtypes(dtypes: list[DType]) -> DType:
    if len(dtypes) == 0:
        raise ValueError("Expected non empty list of dtypes")

    promoted = dtypes[0]
    for dtype in dtypes[1:]:
        if isinstance(dtype, NoneDType):
            continue
        if isinstance(promoted, NoneDType):
            promoted = dtype
            continue

        if dtype.can_promote_to(promoted):
            continue
        if promoted.can_promote_to(dtype):
            promoted = dtype
            continue

        raise ExpressionTypeError(f"Incompatible types {dtype} and {promoted}.")

    return promoted
