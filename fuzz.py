from __future__ import annotations

import random
import string
from functools import partial

import numpy as np
import polars as pl

import pydiverse.transform as pdt
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.col_expr import ColFn
from pydiverse.transform._internal.tree.types import Tvar

rng = np.random.default_rng()
letters = list(string.printable)

ALL_TYPES = [pdt.Int(), pdt.Float(), pdt.Bool(), pdt.String()]
MEAN_HEIGHT = 4


def gen_table(rows: int, types: dict[pdt.Dtype, int]) -> pl.DataFrame:
    d = pl.DataFrame()

    rng_fns = {
        pdt.Float(): rng.standard_normal,
        pdt.Int(): partial(rng.integers, -(1 << 13), 1 << 13),
        pdt.Bool(): partial(rng.integers, 0, 1, dtype=bool),
        pdt.String(): (
            lambda rows: np.array(
                [
                    "".join(random.choices(letters, k=rng.poisson(10)))
                    for _ in range(rows)
                ]
            )
        ),
    }

    for ty, fn in rng_fns.items():
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
    if not isinstance(op, Operator):
        continue
    for sig in op.signatures:
        if isinstance(sig.return_type, Tvar):
            for ty in ALL_TYPES:
                ops_with_return_type[ty].append(
                    (
                        op,
                        Signature(
                            *(
                                ty if isinstance(param, Tvar) else param
                                for param in sig.types
                            ),
                            return_type=ty,
                        ),
                    )
                )
        else:
            ops_with_return_type[sig.return_type].append((op, sig))


def gen_expr(
    dtype: pdt.Dtype, cols: dict[pdt.Dtype, list[str]], q: float = 0.0
) -> pdt.ColExpr:
    if q > 1:
        return rng.choice(cols[dtype])

    op, sig = rng.choice(ops_with_return_type[dtype])
    assert isinstance(op, Operator)
    assert isinstance(sig, Signature)

    args = []
    for param in sig.types[: len(sig.types) - sig.is_vararg]:
        args.append(gen_expr(param, cols, q + rng.exponential(1 / MEAN_HEIGHT)))

    if sig.is_vararg:
        nargs = int(rng.chisquare(4))
        for _ in range(nargs):
            args.append(
                gen_expr(sig.types[-1], cols, q + rng.exponential(1 / MEAN_HEIGHT))
            )

    return ColFn(op, *args)
