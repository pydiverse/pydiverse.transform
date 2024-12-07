# DuckDB, Polars, and Parquet

Pydiverse.transform can swiftly switch between DuckDB and Polars based execution:

```python
from pydiverse.transform.extended import *
import pydiverse.transform as pdt

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]), name="A")
tbl2 = pdt.Table(dict(x=[2, 3], z=["b", "c"]), name="B") >> collect(DuckDb())

out = (
    tbl >> collect(DuckDb()) >> left_join(tbl2, tbl.x == tbl2.x) >> show_query()
        >> collect(Polars()) >> mutate(z=tbl.x + tbl.y) >> show()
)

df1 = out >> export(Polars())
print(type(df1))

df2 = out >> export(Polars(lazy=False))
print(type(df2))
```

In the future, it is also intended to allow both DuckDB and Polars backends to read and write Parquet files.
