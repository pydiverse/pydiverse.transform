# Aggregation functions

Pydiverse.transform offers aggregations either grouped or ungrouped with the `summarize()` verb:

```python
import pydiverse.transform as pdt
from pydiverse.transform.extended import *

tbl1 = pdt.Table(dict(a=[1, 1, 2], b=[4, 5, 6]))

tbl1 >> summarize(sum_a=a.sum(), sum_b=b.sum()) >> show()
tbl1 >> group_by(tbl1.a) >> summarize(sum_b=b.sum()) >> show()
```

Typical aggregation functions are `sum()`, `mean()`, `count()`, `min()`, `max()`, `any()`, and `all()`.
These functions can be used in the `summarize()` verb.
They can also be used as [window functions](/examples/window_functions) in the `mutate()` verb in case aggregated
values shall be projected back to the rows of the original table expression.
