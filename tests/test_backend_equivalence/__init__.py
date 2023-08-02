from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable

import pytest
from pandas._testing import assert_frame_equal

from pydiverse.transform import Table, verb
from pydiverse.transform.core.verbs import arrange, collect, show_query
from pydiverse.transform.errors import NonStandardBehaviourWarning


@contextlib.contextmanager
def catch_warnings():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def get_transform_warnings(
    warnings_: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [w for w in warnings_ if issubclass(w.category, NonStandardBehaviourWarning)]


def assert_result_equal(
    input_tables,
    pipe_factory,
    *,
    exception=None,
    check_order=False,
    may_throw=False,
    xfail_warnings=True,
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

    did_raise_warning = False
    warnings_summary = None
    try:
        with catch_warnings() as warnings_record:
            query_x = pipe_factory(*x)
            query_y = pipe_factory(*y)

            dfx = (query_x >> collect()).reset_index(drop=True)
            dfy = (query_y >> collect()).reset_index(drop=True)

        warnings_record = get_transform_warnings(warnings_record)
        if len(warnings_record):
            did_raise_warning = True
            warnings_summary = "\n".join({str(w.message) for w in warnings_record})

        if not check_order:
            dfx.sort_values(by=dfx.columns.tolist(), inplace=True, ignore_index=True)
            dfy.sort_values(by=dfy.columns.tolist(), inplace=True, ignore_index=True)
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


@verb
def full_sort(t: Table):
    """
    Ordering after join is not determined.
    This helper applies a deterministic ordering to a table.
    """
    return t >> arrange(*t)
