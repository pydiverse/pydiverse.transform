from __future__ import annotations


class OperatorNotSupportedError(Exception):
    """
    Exception raised when a specific operation is not supported by a backend.
    """


class ExpressionError(Exception):
    """
    Generic exception related to an invalid expression.
    """


class ExpressionTypeError(ExpressionError):
    """
    Exception related to invalid types in an expression
    """


class FunctionTypeError(ExpressionError):
    """
    Exception related to function type
    """


class AlignmentError(Exception):
    """
    Raised when something isn't aligned.
    """


# WARNINGS


class NonStandardBehaviourWarning(UserWarning):
    """
    Category for when a specific backend deviates from
    the expected standard behaviour.
    """
