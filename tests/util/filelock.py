# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def lock(lock_path: Path | str):
    if isinstance(lock_path, str):
        lock_path = Path(lock_path)
    import filelock

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(lock_path.with_suffix(".lock")):
        yield
