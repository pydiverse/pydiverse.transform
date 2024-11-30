from __future__ import annotations

import random
import string
from functools import partial

import numpy as np
import polars as pl

import pydiverse.transform as pdt
from pydiverse.transform._internal.backend.polars import polars_type

rng = np.random.default_rng()
letters = list(string.printable)


def gen_table(rows: int, types: dict[pdt.Dtype, int]) -> pl.DataFrame:
    d = pl.DataFrame()

    rng_fns = {
        pdt.Float64(): rng.standard_normal,
        pdt.Int64(): partial(rng.integers, -(1 << 13), 1 << 13),
        pdt.Bool(): partial(rng.integers, 0, 1),
        pdt.String(): (
            lambda rows: np.array(
                [
                    "".join(random.choices(letters, k=rng.poisson(10)))
                    for _ in range(rows)
                ]
            )
        ),
    }

    for t, fn in rng_fns.items():
        if t in types:
            d = d.with_columns(
                **{
                    f"{t.__class__.__name__.lower()} #{i+1}": pl.lit(fn(rows)).cast(
                        polars_type(t)
                    )
                    for i in range(types[t])
                }
            )

    return d
