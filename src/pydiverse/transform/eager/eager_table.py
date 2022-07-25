from __future__ import annotations

from pydiverse.transform.core.table_impl import AbstractTableImpl


def uuid_to_str(_uuid):
    # mod with 2^31-1  (prime number)
    return format(_uuid.int % 0x7FFFFFFF, "X")


class EagerTableImpl(AbstractTableImpl):
    pass
