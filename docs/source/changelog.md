# Changelog

## 0.7.0 (2026-01-08)
- accept whitespace problems in MSSQL equality comparisons since the fix killed JOIN performance
- disable printing of table data for SQL backends (issue #100 is to make this configurable)
- fix sqlalchemy deprecation warnings

## 0.6.4 (2025-12-18)
- fix col.map() Null-type handling and string operations
- fix print display of short tables

## 0.6.3 (2025-12-17)
- implement `tbl1 >> union(tbl2)` and `union(tbl1,tbl2)`

## 0.6.2 (2025-12-15)
- drop support for python 3.10
- fix is_sql_backend(Table) and backend(Table)
- fix renaming a column to an existing hidden column

## 0.6.1 (2025-11-15)
- support python 3.14
- support pyarrow 22

## 0.6.0 (2025-10-03)
- bump dependency to pydiverse.common 0.4.0 (switch structlog config to stdlib logging)

## 0.5.6 (2025-08-22)
- improved regex behavior in .str.contains()
- AST repr is printing user friendly syntax (python-like black formatted)
- ColExpr repr is much more intuitive
- fixed handling of const Enum (mutate(enum_col="value"))
- added columns verb to simplify [c.name for c in tbl]
- compatible with pydiverse.common 0.3.12

## 0.5.5 (2025-08-20)
- don't suffix all joined columns if only the key columns overlap
- add query to __str__/__repr__ operators of Table
- add transfer_col_references to allow materialize with keep_col_references=True

## 0.5.4 (2025-08-19)
- support IBM DB2 dialect

## 0.5.3 (2025-06-29)

- improve repr for interactive development (a preview of the data is shown now)
- add basic Enum support
- add name verb
- add AST repr for column expressions

## 0.5.2 (2025-06-11)
- require pydiverse.common 0.3.4 to ensure its dependencies are installed

## 0.5.1 (2025-06-11)
- fixed pypi package dependencies

## 0.5.0 (2025-06-08)
- rename Uint type to UInt (see pydiverse.common 0.3)
- pydiverse.transform.__version__ (implemented via importlib.metadata)

## 0.4.0 (2025-06-06)
- adjust to pydiverse.common 0.2.0
- Decimal becomes subtype of Float

## 0.3.3 (2025-06-02)
- fix error messages
- fix polars count()

## 0.3.2 (2025-06-02)
- add Dict, DictOfLists, ListOfDicts as export types
- add ColExpr.uses_table to check whether a table occurs in a column expression
- fixes around float in SQL and integer truediv

## 0.3.1 (2025-04-27)
- fix conda feedstock build

## 0.3.0 (2025-04-25)
- moved DType code to pydiverse.common package
- various fixes
- make `list.agg` work on polars/postgres/duckdb
- add Scalar as export type

## 0.2.3 (2025-02-04)
- always set alias in query generation
- various fixes
- documentation improvements

## 0.2.2 (2025-10-09)
- require pandas/SQLAlchemy >= 2.0.0
- be more strict about iterables in column expressions

## 0.2.1 (2024-10-07)

- added some functions like exp / log
- make a stable public api
- casts
- duckdb execution on polars
- filter= arg for SQL
- new case expression syntax
- Date in SQL

## 0.2.0 (2024-08-31)

- add polars backend
- removed pandas backend (conversion to polars needed on ingest and export)
  * eventually, the syntax should look like this with hidden Polars conversion: `pdt.Table(df) >> ... >> export(Pandas())`
- add Date and Duration type
- string / datetime operations now have their separate namespace (.str / .dt)
- add partition_by=, arrange= and filter= arguments for window / aggregation functions (filter does not work on SQL yet)
- migrate project to pixi

## 0.1.5 (2024-04-20)
- support ArrowDtype based columns in pandas dataframes

## 0.1.4 (2024-04-20)
- better support for apache arrow backed pandas dataframes
- fix handling of boolean literals
- fix literal handling within SQL expressions
- support for operators/functions that take constant arguments

## 0.1.3 (2023-06-27)
- support pandas dataframes backed by pyarrow extension dtypes

## 0.1.2 (2023-06-01)
- relax python version requirement (>=3.9, <4).

## 0.1.1 (2023-05-04)
- development of pydiverse.transform is currently slow since pydiverse.pipedag
   adoption is currently prioritized: this will help to limit scope that then can
   be provided comprehensively with high quality
- added support for pandas >= 2.0.0
- added support for sqlalchemy >= 2.0.0

## 0.1.0 (2022-09-01)
- Initial release.
