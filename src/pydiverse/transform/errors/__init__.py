from __future__ import annotations


class DataTypeError(Exception):
    """
    Exception related to invalid types in an expression
    """


class FunctionTypeError(Exception):
    """
    Exception related to function type
    """


class NonStandardBehaviourWarning(UserWarning):
    """
    Category for when a specific backend deviates from
    the expected standard behaviour.
    """
