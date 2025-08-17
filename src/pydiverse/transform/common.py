# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from ._internal.backend.targets import (
    Dict,
    DictOfLists,
    DuckDb,
    ListOfDicts,
    Pandas,
    Polars,
    Scalar,
    SqlAlchemy,
)
from ._internal.pipe.aligned import aligned
from ._internal.pipe.pipeable import verb
from ._internal.pipe.verbs import (
    arrange,
    ast_repr,
    cross_join,
    drop,
    full_join,
    group_by,
    inner_join,
    join,
    left_join,
    mutate,
    name,
    rename,
    select,
    slice_head,
    summarize,
    ungroup,
)
from .base import *  # noqa: F403
from .base import __all__ as __base

__all__ = __base + [
    "verb",
    "aligned",
    "arrange",
    "ast_repr",
    "drop",
    "group_by",
    "name",
    "join",
    "inner_join",
    "left_join",
    "full_join",
    "cross_join",
    "mutate",
    "rename",
    "select",
    "slice_head",
    "summarize",
    "ungroup",
    "DuckDb",
    "SqlAlchemy",
    "Polars",
    "Pandas",
    "Scalar",
    "Dict",
    "DictOfLists",
    "ListOfDicts",
]
