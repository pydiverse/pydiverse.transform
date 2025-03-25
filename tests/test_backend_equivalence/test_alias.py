from __future__ import annotations

import pytest

from pydiverse.transform.extended import *
from tests.util import assert_result_equal


def test_alias(df3):
    assert_result_equal(df3, lambda t: t >> alias())


def test_alias_named(df3):
    def run(post_op):
        assert_result_equal(df3, lambda t: t >> alias("a") >> post_op)

    for post_op in [
        mutate(),
        mutate(y=C.col1),
        mutate(y=C.col1.count()),
        mutate(y=C.col1.count(partition_by=C.col2)),
        summarize(y=C.col1.count()),
    ]:
        run(post_op)


def test_alias_mutate(df3):
    def run(post_op):
        assert_result_equal(
            df3, lambda t: t >> mutate(x=t.col1) >> alias("a") >> post_op
        )

    for post_op in [
        mutate(),
        mutate(y=C.col1),
        mutate(y=C.col1.count()),
        mutate(y=C.col1.count(partition_by=C.col2)),
        summarize(y=C.col1.count()),
    ]:
        run(post_op)


@pytest.mark.xfail  # TODO: fix problem with errors
def test_alias_window(df3):
    def run(post_op):
        assert_result_equal(
            df3,
            lambda t: t
            >> mutate(x=t.col1.count(partition_by=t.col2))
            >> alias("a")
            >> post_op,
        )

    for post_op in [
        mutate(),
        mutate(y=C.col1),
        mutate(y=C.col1.count()),
        mutate(y=C.col1.count(partition_by=C.col2)),
        summarize(y=C.col1.count() + C.x.sum()),
        summarize(y=C.col1.count() + C.x),
    ]:
        run(post_op)


@pytest.mark.xfail  # TODO: fix problem with errors
def test_alias_summarize(df3):
    def run(post_op):
        assert_result_equal(
            df3,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarize(x=t.col1.count())
            >> alias("a")
            >> post_op,
        )

    for post_op in [
        mutate(),
        mutate(y=C.col1),
        mutate(y=C.col1.count()),
        mutate(y=C.col1.count(partition_by=C.col2)),
        summarize(y=C.col1.count()),
        mutate(y=C.x < row_number()),
    ]:
        run(post_op)
