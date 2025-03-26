from __future__ import annotations

from pydiverse.transform._internal.ops.op import ContextKwarg
from pydiverse.transform._internal.ops.ops.aggregation import Aggregation
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import D, List

list_agg = Aggregation(
    "list.agg",
    Signature(D, return_type=List()),
    context_kwargs=[
        ContextKwarg("partition_by"),
        ContextKwarg("arrange"),
        ContextKwarg("filter"),
    ],
    doc="""
Collect the elements of each group in a list.
""",
)
