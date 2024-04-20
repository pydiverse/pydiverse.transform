# Changelog

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
