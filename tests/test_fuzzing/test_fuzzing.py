from __future__ import annotations

import itertools
import sqlite3
from datetime import datetime

import pandas as pd
import pytest
import sqlalchemy as sa
from tqdm import tqdm

from pydiverse.transform.core.expressions import expr_repr
from pydiverse.transform.core.verbs import (
    build_query,
    collect,
    mutate,
    select,
)
from pydiverse.transform.errors import OperatorNotSupportedError
from tests import fuzzing
from tests.fixtures.backend import BACKEND_MARKS
from tests.util.backend import BACKEND_TABLES

sqlite3_math_supported = False
with sqlite3.connect(":memory:") as sqlite3_cx:
    try:
        sqlite3_cx.cursor().execute("SELECT PI()")
        sqlite3_math_supported = True
    except sqlite3.OperationalError:
        sqlite3_math_supported = False


fuzz_df = pd.DataFrame(
    {
        "int1": [1, 2, 3, 4, None, -2, -1],
        "int2": [1, 1, 2, 0, None, 2, None],
        "str1": ["a", "b", "c", "d", None, "", ""],
        "str2": [" ", "x", "y", "z", None, " ", None],
        "float1": [1.1, 2.2, -0.1, 4.5, None, 3.0, 3.0],
        "float2": [None, -1.2, 0.1, 4.5, None, 3.14, 0],
        "bool1": [True, False, False, True, None, True, None],
        "bool2": [True, False, True, None, False, False, False],
        "datetime1": [
            datetime(2000, 1, 1),
            datetime(2000, 1, 1, 12, 15, 59),
            datetime(1970, 1, 1),
            datetime(1700, 4, 27, 1, 2, 3),
            None,
            datetime(2250, 12, 24, 23, 59, 59),
            datetime(2023, 7, 31, 10, 16, 13),
        ],
        "datetime2": [
            datetime(2023, 7, 31, 10, 16, 13),
            datetime(2023, 7, 31, 10, 16, 14),
            datetime(1900, 1, 2, 3, 4, 5),
            datetime(1988, 5, 3, 12, 4, 54),
            None,
            datetime(2023, 7, 31, 10, 16, 13),
            None,
        ],
    }
)

acceptable_errors = {
    "postgres": [
        "DivisionByZero",
        "InvalidArgumentForPowerFunction",
        "NumericValueOutOfRange",
    ],
    "sqlite": [
        "Expression tree is too large",
        "parser stack overflow",
    ],
    "duckdb": [
        "Out of Range Error: Overflow in",  # DuckDB: #8359
    ],
    "mssql": [
        "Divide by zero error encountered.",
        "'LEAST' is not a recognized built-in function name.",
        "'GREATEST' is not a recognized built-in function name.",
    ],
}


@pytest.mark.parametrize(
    "max_depth, num_expr",
    [
        (1, 500),
        (2, 1_000),
        (4, 5_000),
        (10, 10_000),
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas"),
        pytest.param(
            "sqlite",
            marks=(
                pytest.mark.skip("SQLITE_ENABLE_MATH_FUNCTIONS not enabled")
                if not sqlite3_math_supported
                else ()
            ),
        ),
        pytest.param("duckdb", marks=pytest.mark.skip("DuckDB issue #8500")),
        pytest.param("postgres", marks=BACKEND_MARKS.get("postgres")),
        pytest.param("mssql", marks=BACKEND_MARKS.get("mssql")),
    ],
)
def test_basic(backend: str, max_depth: int, num_expr: int):
    if backend == "sqlite" and max_depth > 4:
        pytest.skip("Sipping because of exponentially growing SQLite expressions")

    table = BACKEND_TABLES[backend](fuzz_df, "fuzz_basic")
    acceptable_errors_ = acceptable_errors.get(backend, ())

    fuzzer = fuzzing.expression.ExpressionFuzzer(table)
    skip_counter = 0

    for expr_i, expr in tqdm(
        enumerate(
            itertools.islice(
                fuzzer.generate_expressions(max_depth=max_depth),
                num_expr,
            )
        ),
        desc=f"Fuzzing {backend} ({max_depth=})",
        total=num_expr,
    ):
        table_expression = None
        exception = None
        collected_successfully = False

        try:
            table_expression = table >> select() >> mutate(x=expr)
            table_expression >> collect()
            collected_successfully = True
        except OverflowError:
            pass
        except OperatorNotSupportedError:
            pass
        except (
            sa.exc.DataError,
            sa.exc.ProgrammingError,
            sa.exc.OperationalError,
        ) as e:
            e_str = str(e)
            if not any(msg in e_str for msg in acceptable_errors_):
                exception = e
        except Exception as e:
            exception = e

        if not collected_successfully:
            skip_counter += 1

        if exception:
            tqdm.write("-----------")
            tqdm.write(f"EXPR I: {expr_i}")
            tqdm.write("EXCEPTION:")
            tqdm.write(str(exception))
            tqdm.write("EXPRESSION:")
            tqdm.write(expr_repr(expr))

            if table_expression is not None:
                tqdm.write("QUERY:")
                tqdm.write(table_expression >> build_query())

            tqdm.write("-----------")
            raise exception

    print(f"{skip_counter} tests failed silently")
