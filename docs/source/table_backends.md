# Table Backends

Pydiverse.transform is designed to describe data transformations on tabular data in a nice way with well defined \
semantics. Current table backends include Polars, and various SQL dialects.

The following examples show how to get data into a pydiverse.transform Table object and out again:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *
import polars as pl
import pandas as pd

tbl1 = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))  # implicitly calls pl.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl2 = pdt.Table(pl.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6])))
tbl3 = pdt.Table(pd.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6])))

df = tbl1 >> export(Polars())
print(df.collect())
```

The following SQL dialects are currently supported:

- Postgres
- Microsoft SQL Server/TSQL
- DuckDB
- Sqlite (rather used for testing so far)

Example sqlalchemy connection strings:
- Postgres: `postgresql://user:password@localhost:5432/{database}`
- Microsoft SQL Server: `mssql+pyodbc://user:password@127.0.0.1:1433/{database}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no`
- DuckDB: `duckdb:////tmp/db.duckdb`

See [Database Testing](database_testing.md) for an example how to spin up a database for testing.

If you don't have a database at hand, you can spin up a duckdb in a temporary directory like this:
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
