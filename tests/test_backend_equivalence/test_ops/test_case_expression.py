from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform import functions as f
from pydiverse.transform.core.verbs import (
    group_by,
    mutate,
    summarise,
)
from pydiverse.transform.errors import ExpressionTypeError, FunctionTypeError
from tests.util import assert_result_equal


def test_mutate_case_ewise(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.case(
                (0, 1),
                (1, 2),
                (2, 2),
            ),
            y=C.col1.case(
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
                (C.col1 == C.col2, 1),
                (C.col2 == C.col3, 2),
                default=(C.col1 + C.col2),
            )
        ),
    )


def test_mutate_case_window(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=f.case(
                (C.col1.max() == 1, 1),
                (C.col1.max() == 2, 2),
                (C.col1.max() == 3, 3),
                (C.col1.max() == 4, 4),
            )
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=C.col1.shift(1, 1729, arrange=[-t.col3, t.col4]),
            x=C.col1.shift(1, 0, arrange=[C.col4]).case(
                (1, C.col2.shift(1, -1, arrange=[C.col2, C.col4])),
                (2, C.col3.shift(2, -2, arrange=[C.col3, C.col4])),
            ),
        ),
    )

    # Can't nest window in window
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.shift(1, 0, arrange=[C.col4])
            .case(
                (1, 2),
                (2, 3),
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
            x=C.col2.max().case(
                (0, C.col1.min()),  # Int
                (1, C.col2.mean() + 0.5),  # Float
                (2, 2),  # ftype=EWISE
            ),
            y=f.case(
                (C.col2.max() > 2, 1),
                (C.col2.max() < 2, C.col2.min()),
                default=C.col3.mean(),
            ),
        ),
    )


def test_invalid_value_dtype(df4):
    # Incompatible types String and Float
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x=C.col1.case(
                (0, "a"),
                (1, 1.1),
            )
        ),
        exception=ExpressionTypeError,
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
        exception=ExpressionTypeError,
    )


def test_invalid_ftype(df1):
    assert_result_equal(
        df1,
        lambda t: t
        >> summarise(
            x=f.rank(arrange=[C.col1]).case(
                (1, C.col1.max()),
                default=None,
            )
        ),
        exception=FunctionTypeError,
    )

    assert_result_equal(
        df1,
        lambda t: t
        >> summarise(
            x=f.case(
                (f.rank(arrange=[C.col1]) == 1, 1),
                default=None,
            )
        ),
        exception=FunctionTypeError,
    )
