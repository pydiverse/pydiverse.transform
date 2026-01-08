# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform.extended import *


def test_preview_print(df3, df4, df_strings):
    # # might be needed again once SQL data print is configurable
    # def data_part(p: str):
    #     return p.split("\n", 1)[1].split("\n\n", 1)[0]
    # assert data_part(str(df3[0])) == data_part(str(df3[1]))

    long_tbl = [
        # after a join, the order is arbitrary, so we need to enforce a specific order
        left >> cross_join(right) >> arrange(left.col4, right.col4.nulls_last())
        for left, right in zip(df3, df4, strict=True)
    ]
    for df in df3, df4, df_strings, long_tbl:
        assert str(df[0] >> export(Polars)) == str(df[1] >> export(Polars))
