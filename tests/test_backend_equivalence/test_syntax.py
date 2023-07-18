from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)

from . import assert_result_equal, tables


@tables("df3")
def test_lambda_cols(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> select(λ.col1, λ.col2))
    assert_result_equal(df3_x, df3_y, lambda t: t >> mutate(col1=λ.col1, col2=λ.col1))

    assert_result_equal(
        df3_x, df3_y, lambda t: t >> select(λ.col10), exception=ValueError
    )


@tables("df3")
def test_columns_pipeable(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t.col1 >> mutate(x=t.col1))

    # Test invalid operations
    assert_result_equal(
        df3_x, df3_y, lambda t: t.col1 >> mutate(x=t.col2), exception=ValueError
    )

    assert_result_equal(
        df3_x, df3_y, lambda t: t.col1 >> mutate(x=λ.col2), exception=ValueError
    )

    assert_result_equal(
        df3_x, df3_y, lambda t: (t.col1 + 1) >> select(), exception=ValueError
    )
