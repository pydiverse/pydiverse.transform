from __future__ import annotations

from collections.abc import Iterable

import pytest
from pandas._testing import assert_frame_equal

from pydiverse.transform import Table, verb
from pydiverse.transform.core.verbs import arrange, collect, show_query


def assert_result_equal(
    input_tables,
    pipe_factory,
    *,
    exception=None,
    check_order=False,
    may_throw=False,
    **kwargs,
):
    if not isinstance(input_tables[0], (tuple, list)):
        input_tables = (input_tables,)
    x, y = zip(*input_tables)

    if exception and not may_throw:
        with pytest.raises(exception):
            pipe_factory(*x) >> collect()
        with pytest.raises(exception):
            pipe_factory(*y) >> collect()
        return
    else:
        try:
            query_x = pipe_factory(*x)
            query_y = pipe_factory(*y)
            dfx = (query_x >> collect()).reset_index(drop=True)
            dfy = (query_y >> collect()).reset_index(drop=True)

            if not check_order:
                dfx.sort_values(
                    by=dfx.columns.tolist(), inplace=True, ignore_index=True
                )
                dfy.sort_values(
                    by=dfy.columns.tolist(), inplace=True, ignore_index=True
                )
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
        assert_frame_equal(dfx, dfy, check_dtype=False, **kwargs)
    except Exception as e:
        print("First dataframe:")
        print(dfx)
        query_x >> show_query()
        print("")
        print("Second dataframe:")
        print(dfy)
        query_y >> show_query()
        print("")
        raise e


@verb
def full_sort(t: Table):
    """
    Ordering after join is not determined.
    This helper applies a deterministic ordering to a table.
    """
    return t >> arrange(*t)
