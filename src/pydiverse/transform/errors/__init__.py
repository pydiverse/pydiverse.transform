from __future__ import annotations


class DataTypeError(Exception):
    """
    Exception related to invalid types in an expression
    """


class FunctionTypeError(Exception):
    """
    Exception related to function type
    """


class NotSupportedError(Exception):
    """
    Signals operations that are not supported by a backend.
    """


class NonStandardWarning(UserWarning):
    """
    Category for when a specific backend deviates from
    the expected standard behaviour.
    """
