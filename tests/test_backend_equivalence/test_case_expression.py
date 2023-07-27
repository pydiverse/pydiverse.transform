from __future__ import annotations

from pydiverse.transform import functions as f
from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    group_by,
    mutate,
    summarise,
)

from . import assert_result_equal


def test_mutate_case_ewise(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=λ.col1.case(
                (0, 1),
                (1, 2),
                (2, 2),
            ),
            y=λ.col1.case(
                (0, 0),
                (1, None),
                default=10.5,
            ),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=f.case(
                (λ.col1 == λ.col2, 1),
                (λ.col2 == λ.col3, 2),
                default=(λ.col1 + λ.col2),
            )
        ),
    )


def test_mutate_case_window(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=f.case(
                (λ.col1.max() == 1, 1),
                (λ.col1.max() == 2, 2),
                (λ.col1.max() == 3, 3),
                (λ.col1.max() == 4, 4),
            )
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=λ.col1.shift(1, 0, arrange=[λ.col4]).case(
                (1, λ.col2.shift(1, -1, arrange=[λ.col2, λ.col4])),
                (2, λ.col3.shift(2, -2, arrange=[λ.col3, λ.col4])),
            )
        ),
    )

    # Can't nest window in window
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=λ.col1.shift(1, 0, arrange=[λ.col4])
            .case(
                (1, 2),
                (2, 3),
            )
            .shift(1, -1, arrange=[-λ.col4])
        ),
        may_throw=True,
    )


def test_summarise_case(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(
            λ.col1,
        )
        >> summarise(
            x=λ.col2.max().case(
                (0, λ.col1.min()),  # Int
                (1, λ.col2.mean() + 0.5),  # Float
                (2, 2),  # ftype=EWISE
            ),
            y=f.case(
                (λ.col2.max() > 2, 1),
                (λ.col2.max() < 2, λ.col2.min()),
                default=λ.col3.mean(),
            ),
        ),
    )


def test_invalid_value_dtype(df4):
    # Incompatible types String and Float
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=λ.col1.case(
                (0, "a"),
                (1, 1.1),
            )
        ),
        exception=ValueError,
    )


def test_invalid_result_dtype(df4):
    # Invalid result type: none
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=f.case(
                default=None,
            )
        ),
        exception=TypeError,
    )


def test_invalid_ftype(df1):
    assert_result_equal(
        df1,
        lambda t: t
        >> summarise(
            x=λ.col1.rank().case(
                (1, λ.col1.max()),
                default=None,
            )
        ),
        exception=ValueError,
    )

    assert_result_equal(
        df1,
        lambda t: t
        >> summarise(
            x=f.case(
                (λ.col1.rank() == 1, 1),
                default=None,
            )
        ),
        exception=ValueError,
    )
