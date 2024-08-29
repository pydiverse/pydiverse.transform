from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)
from tests.util import assert_result_equal


def test_lambda_cols(df3):
    assert_result_equal(df3, lambda t: t >> select(C.col1, C.col2))
    assert_result_equal(df3, lambda t: t >> mutate(col1=C.col1, col2=C.col1))

    assert_result_equal(df3, lambda t: t >> select(C.col10), exception=ValueError)


def test_columns_pipeable(df3):
    assert_result_equal(df3, lambda t: t.col1 >> mutate(x=t.col1))

    # Test invalid operations
    assert_result_equal(df3, lambda t: t.col1 >> mutate(x=t.col2), exception=ValueError)

    assert_result_equal(df3, lambda t: t.col1 >> mutate(x=C.col2), exception=ValueError)

    assert_result_equal(df3, lambda t: (t.col1 + 1) >> select(), exception=TypeError)
