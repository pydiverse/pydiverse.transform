# Quickstart

## Installation
pydiverse.transform is distributed on [PyPi](https://pypi.org/project/pydiverse-transform/)
and [Conda-Forge](https://anaconda.org/conda-forge/pydiverse-transform).
To use it, just install it with pip, conda, or pixi. Polars is installed as a dependency. We recommend also installing
duckdb since it is used in example code:

```shell
pip install pydiverse-transform duckdb duckdb-engine
```

```shell
conda install pydiverse-transform duckdb duckdb-engine
```

## Getting Started
### Ingesting Data from Pandas/Polars

The `Table` class of pydiverse.transform takes various dataframe formats as input:

```python
import pydiverse.transform as pdt
import polars as pl
import pandas as pd

tbl1 = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))  # implicitly calls pl.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl2 = pdt.Table(pl.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6])))
tbl3 = pdt.Table(pd.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6])))
```

### The pipe operator and printing

For doing something with a table or for describing a data transformation, we use the pipe operator `>>`. The individual
operations within the pipe are called verbs. The pipe operator is used to chain verbs together. We call them verbs
because they do something.

The `show` verb can be used to print a table. However, the python print function does pretty much the same:

```python
import pydiverse.transform as pdt
from pydiverse.transform import show

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl >> show()  # same as `print(tbl)` but can be used inside a pipe
```

### Importing verbs

Some verbs are extremely valuable in debugging (like `show`), but they might not be actually used in the final code.
Thus it is recommended to always import them with a wildcard import even though you might need to disable warnings for
your favorite linter:

```python
import pydiverse.transform as pdt
from pydiverse.transform.base import *

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl >> show()  # same as `print(tbl)` but can be used inside a pipe
```

For more convenience you might even consider to use `from pydiverse.transform.common import *` and if you don't mind
that some standard python functions like `filter` are overwritten in your scope, you can use:
`from pydiverse.transform.extended import *`

### Simple transformations

The `mutate` verb is used for adding new columns to a table. It can create more than one column, but it can also be
caused multiple times. Both lead to the exact same execution for the transformation described:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *
from polars.testing import assert_frame_equal

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))
out1 = tbl >> mutate(z=tbl.x + tbl.y, z2=tbl.x * tbl.y)
out2 = tbl >> mutate(z=tbl.x + tbl.y) >> mutate(z2=tbl.x * tbl.y)
assert_frame_equal(out1 >> export(Polars()), out2 >> export(Polars()))
```

### Referencing, selecting and deselecting columns

If column names are python identifiers, they can be referenced with `tbl.column_name`. If they are not, they can be
referenced with `tbl["column name"]`. Alternatively, columns can also be referenced by their name with `C.column_name`.
Even though it is very common in DataFrame libraries to only reference column names independent of their origin, it is
discouraged to do this in pydiverse.transform since it is very nice to show the reader from which source table a column
originated and then pydiverse.transform can provide better error messages in case of the errors can be forseen simply by
analyzing types within expressions.

The `select` verb is used to select columns and the `drop` verb is used to drop
them. Please bear in mind that `select` and `drop` only hide columns and that they can still be used in subsequent
`mutate`/`filter`/`group_by` expressions:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))
(
    tbl
        >> mutate(z=tbl.x + tbl.y) >> select(tbl.y, C.z) >> show()
        >> drop(tbl.y) >> show()
)
```

### Ingesting Data from SQL Database

You can reference a SQL Table within a database by providing its name and using a sqlalchemy engine:
`tbl = pdt.Table("my_tbl", SqlAlchemy(engine, schema="transform"))`

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *
import sqlalchemy as sa
import tempfile

def example(engine):
    # this pydiverse.transform code runs on database connection accessible with sqlalchemy engine
    pdt.Table("my_tbl", SqlAlchemy(engine, schema="transform")) >> show_query() >> show()

# initialize temporary database and call example() with sqlalchemy engine
with tempfile.TemporaryDirectory() as temp_dir:
    engine = sa.create_engine(f"duckdb:///{temp_dir}/duckdb.db")
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE SCHEMA transform"))
        conn.execute(sa.text("CREATE TABLE transform.my_tbl AS SELECT 'a' as a, 1 as b"))
        conn.commit()
    example(engine)
```

Output of show_query():
```sql
SELECT my_tbl.a AS a, my_tbl.b AS b
FROM transform.my_tbl AS my_tbl
```

Output of show():
```
Table my_tbl, backend: DuckDbImpl
shape: (1, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ str ┆ i32 │
╞═════╪═════╡
│ a   ┆ 1   │
└─────┴─────┘
```
