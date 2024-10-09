from __future__ import annotations

import contextlib
import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pydiverse.transform._internal.backend.targets import Polars
from pydiverse.transform._internal.errors import NonStandardWarning
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.pipe.verbs import export, show_query


def assert_equal(left, right, check_dtypes=False, check_row_order=True):
    left_df = left >> export(Polars(lazy=False)) if isinstance(left, Table) else left
    right_df = (
        right >> export(Polars(lazy=False)) if isinstance(right, Table) else right
    )

    try:
        assert_frame_equal(
            left_df,
            right_df,
            check_column_order=False,
            check_row_order=check_row_order,
            check_dtypes=check_dtypes,
        )
    except AssertionError as e:
        print("First dataframe:")
        print(left_df)
        if isinstance(left, Table):
            left >> show_query()
        print()
        print("Second dataframe:")
        print(right_df)
        if isinstance(right, Table):
            right >> show_query()
        raise e


@contextlib.contextmanager
def catch_warnings():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def get_transform_warnings(
    warnings_: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [w for w in warnings_ if issubclass(w.category, NonStandardWarning)]


def assert_result_equal(
    input_tables,
    pipe_factory,
    *,
    exception=None,
    check_row_order: bool = False,
    may_throw: bool = False,
    xfail_warnings: bool = True,
):
    if not isinstance(input_tables[0], tuple | list):
        input_tables = (input_tables,)
    x, y = zip(*input_tables, strict=True)

    if exception and not may_throw:
        with pytest.raises(exception):
            pipe_factory(*x) >> export(Polars(lazy=False))
        with pytest.raises(exception):
            pipe_factory(*y) >> export(Polars(lazy=False))
        return

    did_raise_warning = False
    warnings_summary = None
    try:
        with catch_warnings() as warnings_record:
            query_x = pipe_factory(*x)
            query_y = pipe_factory(*y)

            dfx: pl.DataFrame = (query_x >> export(Polars(lazy=False))).with_columns(
                pl.col(pl.Decimal(scale=10)).cast(pl.Float64)
            )
            dfy: pl.DataFrame = (query_y >> export(Polars(lazy=False))).with_columns(
                pl.col(pl.Decimal(scale=10)).cast(pl.Float64)
            )

            # after a join, cols containing only null values get type Null on SQLite and
            # Postgres. maybe we can fix this but for now we just ignore such cols
            assert dfx.columns == dfy.columns
            null_cols = set(dfx.select(pl.col(pl.Null)).columns) | set(
                dfy.select(pl.col(pl.Null)).columns
            )
            assert all(
                all(d.get_column(col).is_null().all() for col in null_cols)
                for d in (dfx, dfy)
            )
            dfy = dfy.select(pl.all().exclude(null_cols))
            dfx = dfx.select(pl.all().exclude(null_cols))

        warnings_record = get_transform_warnings(warnings_record)
        if len(warnings_record):
            did_raise_warning = True
            warnings_summary = "\n".join({str(w.message) for w in warnings_record})

    except Exception as e:
        if may_throw:
            if exception is not None:
                if isinstance(exception, type):
                    exception = (exception,)
                if not isinstance(e, exception):
                    raise Exception(
                        f"Raised the wrong type of exception: {type(e)} instead of"
                        f" {exception}."
                    ) from e
            # TODO: Replace with logger
            print(f"An exception was thrown:\n{e}")
            return
        else:
            raise e

    try:
        assert_frame_equal(
            dfx,
            dfy,
            check_row_order=check_row_order,
            check_exact=False,
        )
    except Exception as e:
        if xfail_warnings and did_raise_warning:
            pytest.xfail(warnings_summary)

        print("First dataframe:")
        print(dfx)
        query_x >> show_query()
        print("")
        print("Second dataframe:")
        print(dfy)
        query_y >> show_query()
        print("")
        raise e
