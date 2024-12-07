# Best Practices: Beware the flatfile & embrace working with entities

In DataFrame libraries, joining different tables is often either cumbersome or slow. As a consequence, many data
pipelines bring their main pieces of information together in one big table called flatfile. While this might be nice
for quick exploration of the data, it causes several problems for long term maintenance and speed of adding new
features:
1. The number of columns grows very large and may become hard to overlook by the users that don't know all the prefixes
    and suffixes by heart.
2. Associated information with 1:n relationship either are duplicated (wasting space),
    or written to an array column (reducing flexibility for further joins), or simply make
    it prohibitively hard to add features on a certain granularity.
3. In case a table is historized, storing rows for each version of a data field, the table size grows quadratic with
    the number of columns.

The other alternative is to keep column groups with similar subject matter meaning or similar data sources together in
separate tables called entities. Especially when creating data transformation code programmatically with a nice syntax,
it can be made quite easy to work with typical groups of entities with code in the background joining underlying tables.

Often flatfiles are created before feature engineering. Due to the large number of features (columns), it becomes
necessary to build automatic tools for executing the code for each feature in the correct order and to avoid wasteful
execution. However, when using entity granularity (column groups of similar origin), it is more manageable to manually
wire all feature engineering computations. It is even very valuable code to see how the different computation steps /
entities build on each other. This makes tracking down problems much easier in debugging and helps new-joiners a chance
to step through the code.
