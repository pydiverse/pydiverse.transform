# Window functions

Pydiverse.transform offers window functions with the `mutate()` verb.
Window functions are functions that operate on a set of rows related to the current row.
They can be computed independently on groups, can use the order of rows, and can be computed only on a filtered
subset of the table. The most simple window function is `shift(n)` which shifts a column by `n` rows. Defining an
ordering is very important for this operation.

There are two notations which define the grouping and arranging in a different way.
The first is explicitly defining the `partition_by`, `order_by`, and `filter` arguments of the window function.
The second makes use of existing verbs like `group_by()` and `arrange()`. However, an additional verb `ungroup()` tells
that no `summarize()` will follow but rather that `group_by()` arguments should be used as `partition_by` and `arrange()`
arguments as `arrange` parameters to window functions.

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *
from polars.testing import assert_frame_equal

tbl1 = pdt.Table(dict(a=[1, 1, 2, 2, 2, 3], b=[4, 5, 8, 7, 6, 9]))

out1 = tbl1 >> mutate(b_shift=tbl1.b.shift(1, partition_by=tbl1.a, arrange=-tbl1.b, filter=tbl1.b < 8)) >> show()
out2 = tbl1 >> group_by(tbl1.a) >> arrange(-tbl1.b) >> mutate(b_shift=tbl1.b.shift(1, filter=tbl1.b < 8)) >> ungroup() >> show()
assert_frame_equal(out1 >> arrange(-tbl1.b) >> export(Polars()), out2 >> export(Polars()))
```
