from __future__ import annotations

from pydiverse.transform.ops.core import Marker

__all__ = [
    "NullsFirst",
    "NullsLast",
]


# Mark order-by column that it should be ordered with NULLs first
class NullsFirst(Marker):
    name = "nulls_first"
    signatures = ["T -> T"]


# Mark order-by column that it should be ordered with NULLs last
class NullsLast(Marker):
    name = "nulls_last"
    signatures = ["T -> T"]
