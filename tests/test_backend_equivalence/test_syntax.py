# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform import C
from pydiverse.transform._internal.errors import ColumnNotFoundError
from pydiverse.transform._internal.pipe.verbs import mutate, select
from tests.util import assert_result_equal


def test_lambda_cols(df3):
    assert_result_equal(df3, lambda t: t >> select(C.col1, C.col2))
    assert_result_equal(df3, lambda t: t >> mutate(col1=C.col1, col2=C.col1))
    assert_result_equal(
        df3, lambda t: t >> select(C.col10), exception=ColumnNotFoundError
    )
