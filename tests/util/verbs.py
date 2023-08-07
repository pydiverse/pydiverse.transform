from __future__ import annotations

from pydiverse.transform import Table, verb
from pydiverse.transform.core.verbs import arrange


@verb
def full_sort(t: Table):
    """
    Ordering after join is not determined.
    This helper applies a deterministic ordering to a table.
    """
    return t >> arrange(*t)
