from __future__ import annotations

import typing
from typing import Any


class FunctionTypeError(Exception):
    """
    Exception related to function type
    """


class NotSupportedError(Exception):
    """
    Signals operations that are not supported by a backend.
    """


class SubqueryError(Exception):
    """
    Raised for subqueries that would be required on SQL but were not marked explicitly
    by an `>> alias()`.
    """


class NonStandardWarning(UserWarning):
    """
    Category for when a specific backend deviates from
    the expected standard behaviour.
    """


# Our error message format: The first line is in lowercase letters, without a dot at
# the end. More detail is given in the following lines in normal english sentences.
# To give advice to to the user, we write `hint: ...`.


def check_arg_type(
    expected_type: type,
    fn: str,
    param_name: str,
    arg: Any,
):
    if not isinstance(arg, expected_type):
        type_args = typing.get_args(expected_type)
        expected_type_str = (
            expected_type.__name__
            if not type_args
            else " | ".join(t.__name__ for t in type_args)
        )
        raise TypeError(
            f"argument for parameter `{param_name}` of `{fn}` must have type "
            f"`{expected_type_str}`, found `{type(arg).__name__}` instead"
        )


def check_vararg_type(expected_type: type, fn: str, *args: Any):
    for arg in args:
        if not isinstance(arg, expected_type):
            type_args = typing.get_args(expected_type)
            expected_type_str = (
                expected_type.__name__
                if not type_args
                else " | ".join(t.__name__ for t in type_args)
            )
            raise TypeError(
                f"varargs to `{fn}` must have type `{expected_type_str}`, found "
                f"`{type(arg).__name__}` instead"
            )


def check_literal_type(allowed_vals: list[Any], fn: str, param_name: str, arg: Any):
    if arg not in allowed_vals:
        raise TypeError(
            f"argument `{arg}` not allowed for parameter `{param_name}` of `{fn}`, "
            "must be one of " + ", ".join(repr(val) for val in allowed_vals)
        )
