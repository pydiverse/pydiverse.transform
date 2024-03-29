from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from tests.fixtures.backend import BACKEND_MARKS, flatten
from tests.util.backend import BACKEND_TABLES

dataframes = {
    "df1": pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
        }
    ),
    "df2": pd.DataFrame(
        {
            "col1": [1, 2, 2, 4, 5, 6],
            "col2": [2, 2, 0, 0, 2, None],
            "col3": [0.0, 0.1, 0.2, 0.3, 0.01, 0.02],
        }
    ),
    "df3": pd.DataFrame(
        {
            "col1": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "col2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "col3": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "col4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "col5": list("abcdefghijkl"),
        }
    ),
    "df4": pd.DataFrame(
        {
            "col1": [None, 0, 0, 0, 0, None, 1, 1, 1, 2, 2, 2, 2],
            "col2": [0, 0, 1, 1, 0, 0, 1, None, 1, 0, 0, 1, 1],
            "col3": [None, None, None, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "col4": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "col5": list("abcdefghijkl") + [None],
        }
    ),
    "df_strings": pd.DataFrame(
        {
            "col1": [
                "",
                " ",
                "xyz",
                " x ",
                "foo",
                "FooBar",
                "abracadabra",
                None,
                "x",
                None,
                "_ %",
            ],
            "col2": [
                "",
                "test_%",
                "xyz",
                " x",
                "FOO",
                "FooBar",
                "AbracadabA",
                "",
                "bar",
                None,
                "% _",
            ],
        }
    ),
    "df_datetime": pd.DataFrame(
        {
            "col1": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 12, 15, 59),
                datetime(1970, 1, 1, 12, 30, 30, 987_123),
                datetime(1700, 4, 27, 1, 2, 3),
                None,
                datetime(2250, 12, 24, 23, 0, 0),
                datetime(2023, 7, 31, 10, 16, 13),
                datetime(2004, 12, 31),
                datetime(1970, 1, 1),
            ],
            "col2": [
                datetime(2023, 7, 31, 10, 16, 13),
                datetime(2023, 7, 31, 10, 16, 14),
                datetime(1900, 1, 2, 3, 4, 5),
                datetime(1988, 5, 3, 12, 4, 54),
                None,
                datetime(2023, 7, 31, 10, 16, 13),
                None,
                datetime(2004, 12, 31, 23, 59, 59, 456_789),
                datetime(1970, 1, 1),
            ],
        }
    ),
}

# compare one dataframe and one SQL backend to all others
# (some tests are ignored if either backend does not support a feature)
reference_backends = ["pandas", "duckdb"]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    # Parametrize tests based on `backends` and `skip_backends` mark.

    backends = dict.fromkeys(BACKEND_TABLES)
    # if mark := metafunc.definition.get_closest_marker("backends"):
    #     backends = dict.fromkeys(mark.args)
    if mark := metafunc.definition.get_closest_marker("skip_backends"):
        for backend in mark.args:
            if backend in backends:
                del backends[backend]

    backends = {k: i for i, k in enumerate(backends)}

    backend_combinations = [
        (reference_backend, backend)
        for reference_backend in reference_backends
        for backend in backends
    ]

    params = []
    table_names = [name for name in metafunc.fixturenames if name in dataframes]
    for reference_backend, backend in backend_combinations:
        # Skip some redundant backend combinations
        if reference_backend == backend:
            continue
        if reference_backend not in backends:
            continue
        if backend in reference_backends:
            if backends[backend] < backends[reference_backend]:
                continue

        marks = list(
            flatten(
                [
                    BACKEND_MARKS.get(reference_backend, ()),
                    BACKEND_MARKS.get(backend, ()),
                ]
            )
        )

        arguments = [(reference_backend, backend)] * len(table_names)

        params.append(
            pytest.param(
                *arguments,
                marks=marks,
                id=f"{reference_backend}-{backend}",
            )
        )

    param_names = ",".join([f"{name}" for name in table_names])
    return metafunc.parametrize(param_names, params, indirect=True)


def generate_df_fixtures():
    def gen_fixture(table_name):
        @pytest.fixture(scope="function")
        def table_fixture(request):
            df = dataframes[table_name]
            name = f"equiv_{table_name}"

            table_x = BACKEND_TABLES[request.param[0]](df, name)
            table_y = BACKEND_TABLES[request.param[1]](df, name)
            return table_x, table_y

        return table_fixture

    for name in dataframes:
        globals()[f"{name}"] = gen_fixture(name)


generate_df_fixtures()
