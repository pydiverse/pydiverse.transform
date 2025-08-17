# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import polars as pl

from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree.col_expr import AlignedCol

# def aligned(*, with_: Table | None = None):
#     def decorator(fn):
#         @wraps(fn)
#         def wrapper(*args, **kwargs):
#             # inspect ...
#             return aligned_(fn(*args, **kwargs), with_)

#         return wrapper

#     return decorator


def aligned(col: pl.Series, with_: Table | None = None) -> AlignedCol:
    return AlignedCol(col, with_)
