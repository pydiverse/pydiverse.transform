# Database testing

Relational databases are quite effective for analyzing medium size tabular data. You can leave the data in the database
and just describe the transformation in python. All that needs to be exchanged between python and the database is the
SQL string that is executed within the database as a `CREATE TABLE ... AS SELECT ...` statement. The database can
execute query in an optimal and parallelized way.

In practice, a relational database is already running somewhere and all you need is a connection URL and access
credentials. See [Table Backends](table_backends.md) for a list of currently supported databases.

The following example shows how to launch a postgres database in a container with docker-compose and how to work with it
using pydiverse.transform.

You can put the following example transform code in a file called `run_transform.py`:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *
import sqlalchemy as sa

# initialize database=pydiverse and schema=transform
base_engine = sa.create_engine(
    "postgresql://sa:Pydiverse23@127.0.0.1:6543/postgres",
    execution_options={"isolation_level": "AUTOCOMMIT"}
)
with base_engine.connect() as conn:
    exists = len(conn.execute(sa.text("SELECT FROM pg_database WHERE datname = 'pydiverse'")).fetchall()) > 0
    if not exists:
        conn.execute(sa.text("CREATE DATABASE pydiverse"))
        conn.commit()
engine = sa.create_engine("postgresql://sa:Pydiverse23@127.0.0.1:6543/pydiverse")
with engine.connect() as conn:
    conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS transform"))
    conn.execute(sa.text("DROP TABLE IF EXISTS transform.tbl1"))
    conn.execute(sa.text("CREATE TABLE transform.tbl1 AS SELECT 'a' as a, 1 as b"))
    conn.execute(sa.text("DROP TABLE IF EXISTS transform.tbl2"))
    conn.execute(sa.text("CREATE TABLE transform.tbl2 AS SELECT 'a' as a, 1.1 as c"))
    conn.commit()

# process tables
tbl1 = pdt.Table("tbl1", SqlAlchemy(engine, schema="transform"))
tbl2 = pdt.Table("tbl2", SqlAlchemy(engine, schema="transform"))
tbl1 >> left_join(tbl2, tbl1.a == tbl2.a) >> show_query() >> show()
```

If you don't have a postgres database at hand, you can start a postgres database, with the following `docker-compose.yaml` file:

```yaml
version: "3.9"
services:
  postgres:
    image: postgres
    environment:
      POSTGRES_USER: sa
      POSTGRES_PASSWORD: Pydiverse23
    ports:
      - "6543:5432"
```

Run `docker-compose up` in the directory of your `docker-compose.yaml` and then execute
the flow script as follows with a shell like `bash` and a python environment that
includes `pydiverse-transform` and `psycopg2`/`psycopg2-binary`:

```bash
python run_transform.py
```

Finally, you may connect to your localhost postgres database `pydiverse` and
look at tables in schema `transform`.

If you don't have a SQL UI at hand, you may use `psql` command line tool inside the docker container.
Check out the `NAMES` column in `docker ps` output. If the name of your postgres container is
`example_postgres_1`, then you can look at output tables like this:

```bash
docker exec example_postgres_1 psql --username=sa --dbname=pydiverse -c 'select * from transform.tbl1;'
```

Or more interactively:

```bash
docker exec -t -i example_postgres_1 bash
psql --username=sa --dbname=pydiverse
\dt transform.*
select * from transform.tbl2;
```
