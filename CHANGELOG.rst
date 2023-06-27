.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.1.3 (2023-06-27)
------------------
- support pandas dataframes backed by pyarrow extension dtypes

0.1.2 (2023-06-01)
------------------
- relax python version requirement (>=3.9, <4).

0.1.1 (2023-05-04)
------------------
- development of pydiverse.transform is currently slow since pydiverse.pipedag
   adoption is currently prioritized: this will help to limit scope that then can
   be provided comprehensively with high quality
- added support for pandas >= 2.0.0
- added support for sqlalchemy >= 2.0.0

0.1.0 (2022-09-01)
------------------
- Initial release.
