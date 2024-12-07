---
hide-toc: true
---

# pydiverse.transform

Pydiverse.transform is best described by quoting from [dplyr documentation](https://dplyr.tidyverse.org/).
Pydiverse.transform "is a grammar of data manipulation, providing a consistent set of verbs that help you solve the most
common data manipulation challenges:

- mutate() adds new variables that are functions of existing variables
- select() picks variables based on their names.
- filter() picks cases based on their values.
- summarise() reduces multiple values down to a single summary.
- arrange() changes the ordering of the rows.

These all combine naturally with group_by() which allows you to perform any operation by group."

The following example describes how to create new columns, filter rows, and sort the result. The pipe operator `>>` is
used to chain multiple verbs to describe a data transformation in a functional way:

```python
from pydiverse.transform.extended import *
import pydiverse.transform as pdt

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl >> mutate(z=tbl.x + tbl.y) >> filter(tbl.x > 1) >> arrange(C.z) >> show()
```

Please note that you can reference columns either as members of table objects (i.e. `tbl.x`), or as generic reference
to a column in the expression with a specific name by prefixing with `C.` (i.e. `C.z`). If you are familiar with Polars,
this corresponds to `pl.col("z")` but is intentionally short.

```text
Table ?, backend: PolarsImpl
shape: (2, 3)
┌─────┬─────┬─────┐
│ x   ┆ y   ┆ z   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 2   ┆ 5   ┆ 7   │
│ 3   ┆ 6   ┆ 9   │
└─────┴─────┴─────┘
```

Pydiverse.transform is designed to provide a single syntax for data transformation code that can be executed reliably on
both in-memory dataframes and SQL databases. Focus is on predictable types, well defined semantics, and a nice syntax
that is pleasant and efficient to work with. The results should be identical across different backends, and good error
messages should be raised before sending a query to a backend if a specific feature is not supported.

Polars and DuckDB SQL based execution can be described in one pipe chain of data transformation tasks:

```python
from pydiverse.transform.extended import *
import pydiverse.transform as pdt

tbl = pdt.Table(dict(x=[1, 2, 3], y=[4, 5, 6]), name="A")
tbl2 = pdt.Table(dict(x=[2, 3], z=["b", "c"]), name="B") >> collect(DuckDb())

(
    tbl >> collect(DuckDb()) >> left_join(tbl2, tbl.x == tbl2.x) >> show_query()
        >> collect(Polars()) >> mutate(z=tbl.x + tbl.y) >> show()
)
```

The output of `show_query()` is:
```text
SELECT "A".x AS x, "A".y AS y, "B".x AS "x_B", "B".z AS "z_B"
FROM "A" LEFT OUTER JOIN "B" ON "A".x = "B".x
```

The output of `show()` is:
```text
Table A, backend: PolarsImpl
shape: (3, 5)
┌─────┬─────┬──────┬──────┬─────┐
│ x   ┆ y   ┆ x_B  ┆ z_B  ┆ z   │
│ --- ┆ --- ┆ ---  ┆ ---  ┆ --- │
│ i64 ┆ i64 ┆ i64  ┆ str  ┆ i64 │
╞═════╪═════╪══════╪══════╪═════╡
│ 2   ┆ 5   ┆ 2    ┆ b    ┆ 7   │
│ 3   ┆ 6   ┆ 3    ┆ c    ┆ 9   │
│ 1   ┆ 4   ┆ null ┆ null ┆ 5   │
└─────┴─────┴──────┴──────┴─────┘
```

Pydiverse.transform easily integrates with Pandas and Polars code:

```python
import pandas as pd
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

df = pd.DataFrame(dict(x=[1, 2, 3], y=[4, 5, 6]))
tbl = pdt.Table(df)
df_out = tbl >> mutate(z=tbl.x + tbl.y) >> export(Polars())
print(df_out.collect())
```

```text
shape: (3, 3)
┌─────┬─────┬─────┐
│ x   ┆ y   ┆ z   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 1   ┆ 4   ┆ 5   │
│ 2   ┆ 5   ┆ 7   │
│ 3   ┆ 6   ┆ 9   │
└─────┴─────┴─────┘
```

Using the pipe operator `>>` has two main advantages:
1. The member namespace of table objects is solely reserved for referencing column names (i.e. `tbl.x`).
2. It is much easier to create user defined verbs:

```python
from pydiverse.transform.extended import *
import pydiverse.transform as pdt

@verb
def strip_all_strings(tbl):
    return tbl >> mutate(**{col.name:"'" + col.str.strip() + "'" for col in tbl if col.dtype() == pdt.String})

tbl = pdt.Table(dict(x=[" crazy", "padded ", " strings "], y=[4, 5, 6]))
tbl >> strip_all_strings() >> show()
```

```text
Table ?, backend: PolarsImpl
shape: (3, 2)
┌─────┬───────────┐
│ y   ┆ x         │
│ --- ┆ ---       │
│ i64 ┆ str       │
╞═════╪═══════════╡
│ 4   ┆ 'crazy'   │
│ 5   ┆ 'padded'  │
│ 6   ┆ 'strings' │
└─────┴───────────┘
```


## The Pydiverse Library Collection

Pydiverse is a collection of libraries for describing data transformations and data processing pipelines.

Pydiverse.pipedag is designed to encapsulate any form of data processing pipeline code, providing immediate benefits.
It simplifies the operation of multiple pipeline instances with varying input data sizes and enhances performance
through automatic caching and cache invalidation.
A key objective is to facilitate the iterative improvement of data pipeline code, task by task, stage by stage.

Pydiverse.transform is designed to provide a single syntax for data transformation code that can be executed reliably on
both in-memory dataframes and SQL databases.
The interoperability of tasks in pipedag allows transform to narrow its scope and concentrate on quality.
The results should be identical across different backends, and good error messages should be raised before sending a
query to a backend if a specific feature is not supported.

We are placing increased emphasis on simplifying unit and integration testing across multiple pipeline instances,
which may warrant a separate library called pydiverse.pipetest.

In line with our goal to develop data pipeline code on small input data pipeline instances,
generating test data from the full input data could be an area worth exploring.
This may lead to the creation of a separate library, pydiverse.testdata.

Check out the Pydiverse libraries on GitHub:

- [pydiverse.pipedag](https://github.com/pydiverse/pydiverse.pipedag/)
- [pydiverse.transform](https://github.com/pydiverse/pydiverse.transform/)

Check out the Pydiverse libraries on Read the Docs:

- [pydiverse.pipedag](https://pydiversepipedag.readthedocs.io/en/latest/)
- [pydiverse.transform](https://pydiversetransform.readthedocs.io/en/latest/)

## Concepts for future feature extensions

Pydiverse.transform does not aim to be feature complete with either SQL or dataframe libraries. Instead, it aims to
provide a common interface for the most common data transformation tasks. And it aims to provide clear semantics for
how written code behaves on all backends.
A few features are still missing so transform can serve the majority of data transformation needs:

1. The verb `materialize` should allow executing a query directly within any SQL database. This functionality is common
   between pydiverse.pipedag and pydiverse.transform. In its lazy form it would just be an abstract node in the
   execution graph. Especially, the pipedag integration might greatly benefit from this because pipedag could contribute
   database connection and schema information.
2. Expansion of the type system in the direction of Duration, Categorical, and Array types.
3. Backend for producing ONNX graphs that can be used for reliable execution in production.
4. Precise semantics around numeric operations on IEEE 754 floating point numbers including NaN, Inf, and -Inf.

## Related Work: Standing on the Shoulders of Giants

We deeply admire the clean library design achievements of [tidyverse](https://www.tidyverse.org/),
especially [dplyr](https://dplyr.tidyverse.org/).
We highly recommend reading the dplyr documentation to get a feel of how clean data wrangling can look like.
However, we currently see Python's open-source ecosystem surpassing the R programming language due to software
engineering considerations.
We believe that the data science and software engineering tool stacks need to merge, to achieve optimal results in data
analytics and machine learning.

While other popular tools like Airflow, while very impressive, go too far in taking care of execution,
making interactive development within IDE debuggers overly complex.

[Pandas](https://pandas.pydata.org/) and [Polars](https://www.pola.rs/) are currently contending for what is the best
dataframe processing library in Python.
We appreciate, use, and support both. In pydiverse.transform, we decided to only provide a backend natively using
Polars. Pandas 2.0 can exchange data zero-copy with Polars when using Apache Arrow as storage format. Otherwise, Pandas
data might need to be copied when entering or exiting pydiverse.transform expressions.

[DuckDB](https://duckdb.org/) is a columnar relational database technology that very well integrates with the Parquet
file format, Polars, and Apache Arrow. Pydiverse.transform can use DuckDB as a SQL backend and even make it work based
on Polars DataFrames. Switching between DuckDB and Polars processing can be done even within one functional chain of
data processing transformations.

Leaving all intermediate calculation steps of a data pipeline in the database is beneficial for analyzing problems and
potential improvements with explorative SQL queries.
However, handwritten SQL code is notoriously difficult to test and maintain.
As a result, we aim to simplify the incremental transition from handwritten SQL or dataframe code to programmatically
created SQL where better suited.
[SQL Alchemy](https://www.sqlalchemy.org/) is one of the most fundamental and comprehensive tools to do so, but it lacks
convenience.
[Ibis](https://ibis-project.org/) aims to offer a dplyr-like user interface for generating SQL and also tries to support
dataframe backends, but we find the current quality unsatisfactory.
It may also be challenging to significantly enhance the quality across the multitude of supported backends and the
ambition to support all SQL capabilities.

[Siuba](https://siuba.org/) was the starting point for pydiverse.transform, but its maturity level was not satisfactory.
On the dataframe side, there's a dplyr-like user interface: [tidypolars](https://github.com/markfairbanks/tidypolars).
It appears promising, but it has not always kept up with the latest Polars version.
We would love to use Ibis and tidypolars in areas where pydiverse transform is not yet up to the task or where we
intentionally limit the scope to ensure quality.

There are translation tools that help going from one data transformation language to multiple target backends,
such as [substrait](https://substrait.io/) and [data_algebra](https://github.com/WinVector/data_algebra),
that are far more ambitious than pydiverse.
While we follow their developments with excitement,
we are more convinced that we can ensure reliable operation for a defined set of backends, types, and scopes with
pydiverse.transform.
However, all these techniques are candidates for pydiverse.pipedag integration if users demand it.


[//]: # (Contents of the Sidebar)

```{toctree}
:hidden:

quickstart
examples
table_backends
database_testing
best_practices
reference/api
```

```{toctree}
:caption: Development
:hidden:

changelog
license
