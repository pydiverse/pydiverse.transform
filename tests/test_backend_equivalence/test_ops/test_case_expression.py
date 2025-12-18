# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pydiverse.transform as pdt
from pydiverse.transform import C
from pydiverse.transform._internal.errors import DataTypeError, FunctionTypeError
from pydiverse.transform._internal.pipe.verbs import (
    export,
    group_by,
    mutate,
    select,
    summarize,
)
from tests.util import assert_result_equal


def test_map(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x1=t.col4.map({(4, 5, 6): -1}),
            x2=t.col1.map({0: 0, 1: None}),
            x3=t.col5.map({("a", "b", "c"): "abc"}).map({"abc": "abcd"}),
            x4=t.col4.map({(4, 5, 6): "minus one"}, default="default"),
            x5=t.col1.map({0: "zero", 1: None}, default="default"),
            x6=t.col5.map({("a", "b", "c"): 1}, default=0),
            x7=t.col4.map({(4, 5, 6): "minus one"}, default=pdt.lit(None)),
            x8=t.col1.map({0: "zero", 1: None}, default=pdt.lit(None)),
            x9=t.col5.map({("a", "b", "c"): 1}, default=pdt.lit(None)),
        ),
    )

    t = df4[0]
    res = (
        t
        >> select()
        >> mutate(
            x=t.col5.map({("a", "b", "c"): "abc"}).map({("abc",): "abcd"}, default=pdt.lit(None)),
            y=t.col5.map({("a", "b", "c"): "abc"}).map({"abc": "abcd"}, default=pdt.lit(None)),
        )
        >> export(pdt.DictOfLists)
    )
    assert res["x"] == ["abcd"] * 3 + [None] * 10
    assert res["y"] == ["abcd"] * 3 + [None] * 10


def test_mutate_case_ewise(df4):
    assert_result_equal(
        df4,
        lambda t: t >> mutate(x=C.col1.map({0: 1, (1, 2): 2}), y=C.col1.map({0: 0, 1: None}, default=10.4)),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=pdt.when(C.col1 == C.col2).then(1).when(C.col2 == C.col3).then(2).otherwise(C.col1 + C.col2),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=t.col1.map({0: 1}),
            y=t.col4.map({2: 2}, default=-1),
            z=t.col2.map({4: 3}, default=t.col1),
            w=t.col3.map({2: 2, 1: 5}, default=pdt.lit(None)),
        ),
    )


def test_mutate_case_window(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=pdt.when(C.col1.max() == 1)
            .then(1)
            .when(C.col1.max() == 2)
            .then(2)
            .when(C.col1.max() == 3)
            .then(3)
            .when(C.col1.max() == 4)
            .then(4)
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=C.col1.shift(1, 1729, arrange=[t.col3.descending().nulls_last(), t.col4.nulls_last()]),
            x=C.col1.shift(1, 0, arrange=[C.col4.nulls_first()]).map(
                {
                    1: C.col2.shift(1, -1, arrange=[C.col2.nulls_last(), C.col4.nulls_first()]),
                    2: C.col3.shift(2, -2, arrange=[C.col3.nulls_last(), C.col4.nulls_last()]),
                }
            ),
        ),
    )

    # Can't nest window in window
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.shift(1, 0, arrange=[C.col4.nulls_last()])
            .map(
                {
                    1: 2,
                    2: 3,
                }
            )
            .shift(1, -1, arrange=[C.col4.descending().nulls_first()])
        ),
        may_throw=True,
    )


def test_summarize_case(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(
            C.col1,
        )
        >> summarize(
            x=C.col2.max().map(
                {
                    0: C.col1.min(),
                    1: C.col2.mean() + 0.5,
                    2: 2,
                }
            ),
            y=pdt.when(C.col2.max() > 2).then(1).when(C.col2.max() < 2).then(C.col2.min()).otherwise(C.col3.mean()),
        ),
    )


def test_invalid_value_dtype(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.map(
                {
                    0: "a",
                    1: 1.1,
                }
            )
        ),
        exception=DataTypeError,
    )


def test_invalid_ftype(df1):
    assert_result_equal(
        df1,
        lambda t: t
        >> summarize(
            x=pdt.rank(arrange=[C.col1]).map(
                {
                    1: C.col1.max(),
                },
                default=None,
            )
        ),
        exception=FunctionTypeError,
    )

    assert_result_equal(
        df1,
        lambda t: t >> summarize(x=pdt.when(pdt.rank(arrange=[C.col1]) == 1).then(1).otherwise(None)),
        exception=FunctionTypeError,
    )
