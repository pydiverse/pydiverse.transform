# ruff: noqa: F405

from __future__ import annotations

import random
import string
from functools import partial

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import pydiverse.transform as pdt
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.ops.markers import Marker
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.col_expr import ColFn
from pydiverse.transform._internal.tree.types import Tyvar
from pydiverse.transform.common import *  # noqa: F403
from tests.util.backend import BACKEND_TABLES

rng = np.random.default_rng()
letters = list(string.printable)

ALL_TYPES = [pdt.Int(), pdt.Float(), pdt.Bool(), pdt.String()]
MEAN_HEIGHT = 3

RNG_FNS = {
    pdt.Float(): rng.standard_normal,
    pdt.Int(): partial(rng.integers, -(1 << 13), 1 << 13),
    pdt.Bool(): partial(rng.integers, 0, 1, dtype=bool),
    pdt.String(): (
        lambda rows: np.array(
            ["".join(random.choices(letters, k=rng.poisson(10))) for _ in range(rows)]
        )
    ),
}


def gen_table(rows: int, types: dict[pdt.Dtype, int]) -> pl.DataFrame:
    d = pl.DataFrame()

    for ty, fn in RNG_FNS.items():
        if ty in types:
            d = d.with_columns(
                **{
                    f"{ty.__class__.__name__.lower()} #{i+1}": pl.lit(fn(rows))
                    for i in range(types[ty])
                }
            )

    return d


ops_with_return_type: dict[pdt.Dtype, list[tuple[Operator, Signature]]] = {
    ty: [] for ty in ALL_TYPES
}

for op in ops.__dict__.values():
    if (
        not isinstance(op, Operator)
        or op.ftype != Ftype.ELEMENT_WISE
        or isinstance(op, Marker)
    ):
        continue
    for sig in op.signatures:
        if not all(
            t in (*ALL_TYPES, Tyvar("T")) for t in (*sig.types, sig.return_type)
        ):
            continue

        if isinstance(sig.return_type, Tyvar) or any(
            isinstance(param, Tyvar) for param in sig.types
        ):
            for ty in ALL_TYPES:
                rtype = ty if isinstance(sig.return_type, Tyvar) else sig.return_type
                ops_with_return_type[rtype].append(
                    (
                        op,
                        Signature(
                            *(
                                ty if isinstance(param, Tyvar) else param
                                for param in sig.types
                            ),
                            return_type=rtype,
                        ),
                    )
                )
        else:
            ops_with_return_type[sig.return_type].append((op, sig))


def gen_expr(
    dtype: pdt.Dtype, cols: dict[pdt.Dtype, list[str]], q: float = 0.0
) -> pdt.ColExpr:
    if dtype.const:
        return RNG_FNS[dtype.without_const()](1).item()

    if q > 1:
        # we always use C here so the expression does not have to be generated for each
        # backend
        return C[rng.choice(cols[dtype])]

    op, sig = rng.choice(ops_with_return_type[dtype])
    assert isinstance(op, Operator)
    assert isinstance(sig, Signature)

    args = []
    for param in sig.types[: len(sig.types) - sig.is_vararg]:
        args.append(gen_expr(param, cols, q + rng.exponential(1 / MEAN_HEIGHT)))

    if sig.is_vararg:
        nargs = int(rng.normal(2.5, 1 / 1.5))
        for _ in range(nargs):
            args.append(
                gen_expr(sig.types[-1], cols, q + rng.exponential(1 / MEAN_HEIGHT))
            )

    return ColFn(op, *args)


it = int(input("number of iterations: "))
rows = int(input("number of rows: "))
seed = int(input("seed: "))

rng = np.random.default_rng(seed)
NUM_COLS_PER_TYPE = 5

df = gen_table(rows, {dtype: NUM_COLS_PER_TYPE for dtype in ALL_TYPES})


tables = {backend: fn(df, "t") for backend, fn in BACKEND_TABLES.items()}
cols = {
    dtype: [col.name for col in tables["polars"] if col.dtype() <= dtype]
    for dtype in ALL_TYPES
}

for _ in range(it):
    expr = gen_expr(rng.choice(ALL_TYPES), cols)
    results = {
        backend: table >> mutate(y=expr) >> select(C.y) >> export(Polars())
        for backend, table in tables.items()
    }
    for _backend, res in results:
        assert_frame_equal(results["polars"], res)
