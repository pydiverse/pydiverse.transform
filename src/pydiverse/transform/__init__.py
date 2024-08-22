from __future__ import annotations

from pydiverse.transform.core import functions
from pydiverse.transform.core.alignment import aligned, eval_aligned
from pydiverse.transform.core.dispatchers import verb
from pydiverse.transform.core.expressions.lambda_getter import C
from pydiverse.transform.core.table import Table

__all__ = [
    "Table",
    "aligned",
    "eval_aligned",
    "functions",
    "verb",
    "C",
]
