# Best Practices: start sql, finish polars

At the beginning of a data pipeline, there is typically the biggest amount of data touched with rather simple
operations: Data is combined, encodings are converted/harmonized, simple aggregations and computations are performed,
and data is heavily filtered. These operations lend themselves very well to using a powerful database, and converting
transformations to SQL `CREATE TABLE ... AS SELECT ...` statements. This way, the data stays within the database and
the communication heavy operations can be performed efficiently (i.e. parallelized) right where the data is stored.

Towards the end of the pipeline, the vast open source ecosystem of training libraries, evaluation, and
visualization tools is needed which are best interfaced with classical Polars / Pandas DataFrames in Memory.

In the middle with feature engineering, there is still a large part of logic, that is predominantly simple enough for
typical SQL expressiveness with some exceptions. Thus, it is super helpful if we can jump between SQL and Polars for
performance reasons, but stay within the same pydiverse.transform syntax for describing transformations for the most
part.

When moving code to production it is often the case that prediction calls are done with much less data than during
training. This it might not be worth setting up a sophisticated database technology, in that case. Pydiverse.transform
allows to take code written for SQL execution during training and use the exact same code for executing on Polars for
production. In the long run, we also want to be able to generate ONNX graphs from transform code to make long term
reliable deployments even easier.

The aim of pydiverse.transform is not feature completeness but rather versatility, ease of use, and very predictable
and reliable behavior. Thus it should always integrate nicely with other ways of writing data transformations. Together
with [pydiverse.pipedag](https://pydiversepipedag.readthedocs.io/en/latest/), this interoperability is made even much
easier.
