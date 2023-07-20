from __future__ import annotations

from . import ElementWise, Unary

__all__ = [
    "NullsFirst",
    "NullsLast",
]


# Mark order-by column that it should be ordered with NULLs first
class NullsFirst(ElementWise, Unary):
    name = "nulls_first"
    signatures = ["T -> T"]


# Mark order-by column that it should be ordered with NULLs last
class NullsLast(ElementWise, Unary):
    name = "nulls_last"
    signatures = ["T -> T"]
