from __future__ import annotations

import pandas as pd
import polars as pl

from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree.col_expr import AlignedCol
from pydiverse.transform._internal.tree.types import Dtype


def align(
    data: pl.Series | pd.Series | list, with_: Table, dtype: Dtype | None = None
) -> AlignedCol:
    if not isinstance(data, pl.Series):
        data = pl.Series(data)

    if dtype is None:
        from pydiverse.transform._internal.backend.polars import pdt_type

        dtype = pdt_type(data.dtype)

    return AlignedCol(data, with_._ast, dtype)
