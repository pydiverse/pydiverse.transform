# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform._internal.ops.op import ContextKwarg
from pydiverse.transform._internal.ops.ops.aggregation import Aggregation
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import List, S

list_agg = Aggregation(
    "list.agg",
    Signature(S, return_type=List(S)),
    context_kwargs=[
        ContextKwarg("partition_by"),
        ContextKwarg("arrange"),
        ContextKwarg("filter"),
    ],
    doc="""
Collect the elements of each group in a list.
""",
)
