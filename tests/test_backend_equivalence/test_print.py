from __future__ import annotations

from pydiverse.transform.extended import *


def test_preview_print(df3, df4, df_strings):
    def data_part(p: str):
        return p.split("\n", 1)[1]

    assert data_part(str(df3[0])) == data_part(str(df3[1]))
    assert data_part(str(df_strings[0])) == data_part(str(df_strings[1]))

    long_tbl = [
        # after a join, the order is arbitrary, so we need to enforce a specific order
        left >> cross_join(right) >> arrange(left.col4, right.col4.nulls_last())
        for left, right in zip(df3, df4, strict=True)
    ]
    assert data_part(str(long_tbl[0])) == data_part(str(long_tbl[1]))
