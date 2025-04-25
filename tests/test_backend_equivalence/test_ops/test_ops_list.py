from __future__ import annotations

from pydiverse.transform.extended import *
from tests.fixtures.backend import skip_backends
from tests.util.assertion import assert_result_equal


@skip_backends("mssql", "sqlite")
def test_list_agg(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col3)
        >> summarize(s=t.col2.list.agg())
        >> arrange(C.s),
        check_row_order=True,
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col2)
        >> summarize(
            y=t.col5.list.agg(arrange=t.col4),
            z=t.col4.list.agg(filter=t.col1 > 0, arrange=[t.col4.descending()]),
        )
        >> arrange(t.col2),
        check_row_order=True,
    )


@skip_backends("mssql", "sqlite")
def test_list_agg_no_grouping(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> summarize(h=t.col5.list.agg(arrange=[t.col1, t.col4.descending()])),
        check_row_order=True,
    )
