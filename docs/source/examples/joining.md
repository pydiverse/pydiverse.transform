# Joining and Tables as Column namespaces

If you like to combine data from two tables, you need to describe which rows of one table should be combined with which
rows of the other table. This process is called joining. In case of a left_join, all rows of the table entering the join
will be at least once in the output. The join condition defines which rows exactly of both tables are combined. Columns
coming from the other table will be NULL for all rows where no match could be found. In case of an inner_join, only rows
that have a match in both tables will be in the output.

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

tbl1 = pdt.Table(dict(a=["a", "b", "c"], b=[1, 2, 3]))
tbl2 = pdt.Table(dict(a=["a", "b", "b", "d"], c=[1.1, 2.2, 2.3, 4.4]), name="tbl2")

tbl1 >> left_join(tbl2, tbl1.a == tbl2.a) >> show()
tbl1 >> inner_join(tbl2, tbl1.a == tbl2.a) >> show()
```

left_join result:
```text
Table ?, backend: PolarsImpl
shape: (4, 4)
┌─────┬─────┬────────┬────────┐
│ a   ┆ b   ┆ a_tbl2 ┆ c_tbl2 │
│ --- ┆ --- ┆ ---    ┆ ---    │
│ str ┆ i64 ┆ str    ┆ f64    │
╞═════╪═════╪════════╪════════╡
│ a   ┆ 1   ┆ a      ┆ 1.1    │
│ b   ┆ 2   ┆ b      ┆ 2.2    │
│ b   ┆ 2   ┆ b      ┆ 2.3    │
│ c   ┆ 3   ┆ null   ┆ null   │
└─────┴─────┴────────┴────────┘
```

inner_join result:
```text
Table ?, backend: PolarsImpl
shape: (3, 4)
┌─────┬─────┬────────┬────────┐
│ a   ┆ b   ┆ a_tbl2 ┆ c_tbl2 │
│ --- ┆ --- ┆ ---    ┆ ---    │
│ str ┆ i64 ┆ str    ┆ f64    │
╞═════╪═════╪════════╪════════╡
│ a   ┆ 1   ┆ a      ┆ 1.1    │
│ b   ┆ 2   ┆ b      ┆ 2.2    │
│ b   ┆ 2   ┆ b      ┆ 2.3    │
└─────┴─────┴────────┴────────┘
```

For DataFrame libraries, it is quite common that a join combines all columns of both tables, so the user then can pick
the columns of interest for further expressions. In SQL, the act of joining is actually not bringing in any new columns.
It only adds the columns of the joined tables to the namespace of usable columns in expressions of the `mutate` and
`summarize` verbs.

In pydiverse.transform, the empty `select()` verb can be used to hide all columns of a table. But all columns can still
be used in further expressions:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

tbl1 = pdt.Table(dict(a=["a", "b", "c"], b=[1, 2, 3]))
tbl2 = pdt.Table(dict(a=["a", "b", "b", "d"], c=[1.1, 2.2, 2.3, 4.4]), name="tbl2")

(
    tbl1
        >> left_join(tbl2 >> select(), tbl1.a == tbl2.a) >> show()
        >> mutate(d=tbl1.b + tbl2.c) >> show()
)
```

*dplyr* has also a verb called `transmute` which is very similar to `mutate`, but removes/hides all columns which were
not specified in the `mutate` call. This can be easily implemented in pydiverse.transform in user space:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

@verb
def transmute(tbl, **kwargs):
    # the empty select() is used to hide all columns; they can still be used in subsequent mutate statements
    return tbl >> select() >> mutate(**kwargs)

tbl1 = pdt.Table(dict(a=["a", "b", "c"], b=[1, 2, 3]))

tbl1 >> transmute(a=tbl1.a, b_sqr=tbl1.b * tbl1.b) >> show()
```
