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


# to make this work, use array(SELECT col5 FROM t ORDER BY col1)
# we can implement this with SELECT scalar
@skip_backends("mssql", "sqlite")
def test_list_agg_no_grouping(df3):
    assert_result_equal(
        df3,
        lambda t: t >> summarize(h=t.col5.list.agg(arrange=t.col1)),
        check_row_order=True,
    )
