# Changelog

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
