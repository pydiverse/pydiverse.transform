# `partition_by` vs `group_by` in Cache

## Overview

The `Cache` class in `pydiverse.transform` maintains two related but distinct fields for tracking grouping state:

- **`partition_by: list[UUID]`** - Active grouping state
- **`group_by: set[UUID]`** - Historical grouping state

## `partition_by` (Active Grouping State)

**Type:** `list[UUID]`
**Purpose:** Tracks the **currently active** grouping columns

### When it's set:
- When `group_by()` is called: `partition_by` is set to the grouping columns
- When `group_by(add=True)` is called: new columns are appended to existing `partition_by`

### When it's cleared:
- After `summarize()`: `partition_by = []` (aggregation consumes the grouping)
- After `ungroup()`: `partition_by = []`
- After `SubqueryMarker`: `partition_by = []` (subquery materializes the table)

### Usage:
1. **Window functions**: Window functions use `partition_by` to determine partitioning
2. **Backend compilation**: Backends use `partition_by` to generate GROUP BY clauses
3. **Subquery detection**: Tables with non-empty `partition_by` require subqueries for certain operations (joins, unions)

### Example:
```python
t >> group_by(t.col1)  # partition_by = [col1_uuid]
  >> mutate(x=t.col2.sum())  # window function uses partition_by
  >> summarize(y=t.col2.mean())  # partition_by cleared, group_by set
```

## `group_by` (Historical Grouping State)

**Type:** `set[UUID]`
**Purpose:** Tracks columns that the table **was grouped by** in the past

### When it's set:
- After `summarize()`: `group_by = group_by | set(partition_by)` (line 157 in cache.py)
  - This preserves the grouping information even after `partition_by` is cleared

### When it's cleared:
- After `SubqueryMarker`: `group_by = set()` (subquery materializes the table)
- After `Union`: `group_by = set()` (union resets grouping state)

### Usage:
1. **Nested summarize detection**: Detects if someone tries to call `summarize()` on an already-summarized table
   - Check: `if self.group_by and self.group_by != set(self.partition_by)` (line 243)
2. **Subquery detection**: Used alongside `partition_by` to detect grouped tables
   - Check: `if self.group_by or self.partition_by` (line 255)

### Example:
```python
t >> group_by(t.col1)  # partition_by = [col1_uuid], group_by = {}
  >> summarize(y=t.col2.mean())  # partition_by = [], group_by = {col1_uuid}
  >> summarize(z=t.col2.max())  # ERROR: nested summarize detected via group_by
```

## Key Differences

| Aspect | `partition_by` | `group_by` |
|--------|----------------|------------|
| **Type** | `list[UUID]` | `set[UUID]` |
| **State** | Active (current grouping) | Historical (past grouping) |
| **Set by** | `group_by()` verb | `summarize()` verb |
| **Cleared by** | `summarize()`, `ungroup()`, `SubqueryMarker` | `SubqueryMarker`, `Union` |
| **Used for** | Window functions, GROUP BY clauses | Nested operation detection |
| **After summarize** | Cleared (`[]`) | Set (from previous `partition_by`) |

## Why Both Are Needed

1. **`partition_by`** is needed for:
   - Active window function partitioning
   - Backend GROUP BY clause generation
   - Detecting if a table is currently grouped

2. **`group_by`** is needed for:
   - Detecting nested `summarize()` operations (table was grouped, then summarized, then someone tries to summarize again)
   - Historical tracking of grouping state even after aggregation

## Code References

- **Cache definition**: `src/pydiverse/transform/_internal/pipe/cache.py:22,28`
- **GroupBy update**: `cache.py:136-138`
- **Summarize update**: `cache.py:157-158` (sets `group_by` from `partition_by`, then clears `partition_by`)
- **SubqueryMarker update**: `cache.py:201-202` (clears both)
- **Nested summarize detection**: `cache.py:243`
- **Join/Union detection**: `cache.py:255`
