from __future__ import annotations

from pydiverse.transform.backend.targets import DuckDb, Polars, SqlAlchemy
from pydiverse.transform.pipe import functions
from pydiverse.transform.pipe.c import C
from pydiverse.transform.pipe.pipeable import verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree.alignment import aligned, eval_aligned

__all__ = [
    "Polars",
    "SqlAlchemy",
    "DuckDb",
    "Table",
    "aligned",
    "eval_aligned",
    "functions",
    "verb",
    "C",
]
