from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform import C
from pydiverse.transform.errors import DataTypeError, FunctionTypeError
from pydiverse.transform.pipe.verbs import (
    group_by,
    mutate,
    summarise,
)
from tests.util import assert_result_equal


def test_mutate_case_ewise(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.map({0: 1, (1, 2): 2}), y=C.col1.map({0: 0, 1: None}, default=10.4)
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=pdt.when(C.col1 == C.col2)
            .then(1)
            .when(C.col2 == C.col3)
            .then(2)
            .otherwise(C.col1 + C.col2),
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
            u=C.col1.shift(1, 1729, arrange=[-t.col3, t.col4]),
            x=C.col1.shift(1, 0, arrange=[C.col4]).map(
                {
                    1: C.col2.shift(1, -1, arrange=[C.col2, C.col4]),
                    2: C.col3.shift(2, -2, arrange=[C.col3, C.col4]),
                }
            ),
        ),
    )

    # Can't nest window in window
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.shift(1, 0, arrange=[C.col4])
            .map(
                {
                    1: 2,
                    2: 3,
                }
            )
            .shift(1, -1, arrange=[-C.col4])
        ),
        may_throw=True,
    )


def test_summarise_case(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(
            C.col1,
        )
        >> summarise(
            x=C.col2.max().map(
                {
                    0: C.col1.min(),
                    1: C.col2.mean() + 0.5,
                    2: 2,
                }
            ),
            y=pdt.when(C.col2.max() > 2)
            .then(1)
            .when(C.col2.max() < 2)
            .then(C.col2.min())
            .otherwise(C.col3.mean()),
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
        >> summarise(
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
        lambda t: t
        >> summarise(
            x=pdt.when(pdt.rank(arrange=[C.col1]) == 1).then(1).otherwise(None)
        ),
        exception=FunctionTypeError,
    )
