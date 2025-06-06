# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from ._internal.errors import (
    ColumnNotFoundError,
    FunctionTypeError,
    NotSupportedError,
    SubqueryError,
)

__all__ = [
    "SubqueryError",
    "FunctionTypeError",
    "NotSupportedError",
    "ColumnNotFoundError",
]
