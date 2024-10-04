from __future__ import annotations

from pydiverse.transform._internal.pipe.verbs import alias, filter, inner_join, mutate
from tests.util.assertion import assert_result_equal


def test_dtypes(df1):
    assert_result_equal(
        df1,
        lambda t: t
        >> filter(t.col1 % 2 == 1)
        >> inner_join(s := t >> mutate(u=t.col1 % 2) >> alias(), t.col1 == s.u),
    )
