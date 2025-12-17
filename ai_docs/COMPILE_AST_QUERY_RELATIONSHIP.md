# Relationship between `compile_ast` and `compile_query`

## Overview

`compile_ast` and `compile_query` work together to convert the AST (Abstract Syntax Tree) into SQLAlchemy query objects:

- **`compile_ast`**: Recursively traverses the AST and builds up intermediate state
- **`compile_query`**: Converts the intermediate state into a concrete SQLAlchemy `Select` statement

## `compile_ast` - State Builder

**Signature:**
```python
compile_ast(nd: AstNode, needed_cols: dict[UUID, int])
  -> tuple[sqa.Table, Query, dict[UUID, sqa.Label]]
```

**What it does:**
- Recursively processes the AST from leaves to root
- Incrementally builds up three pieces of state:
  1. **`table`**: A SQLAlchemy `Table` or `Subquery` object representing the data source
  2. **`query`**: A `Query` object containing metadata (select list, where clauses, group_by, etc.)
  3. **`sqa_expr`**: A dictionary mapping column UUIDs to SQLAlchemy `Label` expressions

**Key behaviors:**
- Processes verbs by modifying the `query` and `sqa_expr` incrementally
- Tracks which columns are "needed" via `needed_cols` for subquery optimization
- Only calls `compile_query` when materializing a subquery (at `SubqueryMarker`)

## `compile_query` - Query Materializer

**Signature:**
```python
compile_query(table: sqa.Table, query: Query, sqa_expr: dict[UUID, sqa.ColumnElement])
  -> sqa.sql.Select
```

**What it does:**
- Takes the accumulated state from `compile_ast`
- Builds a concrete SQLAlchemy `Select` statement by:
  1. Starting from `table.select().select_from(table)`
  2. Adding WHERE clauses from `query.where`
  3. Adding GROUP BY from `query.group_by`
  4. Adding HAVING from `query.having`
  5. Adding LIMIT/OFFSET from `query.limit`/`query.offset`
  6. Adding ORDER BY from `query.order_by`
  7. Finally selecting only the columns in `query.select` using `with_only_columns()`

## Important Assumptions

### 1. **State Consistency**
When calling `compile_query`, the state must be consistent:
- All UUIDs in `query.select` must exist as keys in `sqa_expr`
- All UUIDs referenced in `query.where`, `query.group_by`, `query.having`, `query.order_by` must exist in `sqa_expr`
- The `table` must be a valid SQLAlchemy Table/Subquery that contains or can reference the columns in `sqa_expr`

### 2. **When to Call `compile_query`**
- **Only call `compile_query` when you need to materialize a query** (e.g., for subqueries)
- During normal AST traversal, `compile_ast` just modifies the `query` and `sqa_expr` state
- Don't call `compile_query` prematurely - let `compile_ast` build up the full state first

### 3. **Column References**
- `sqa_expr` maps UUIDs to SQLAlchemy expressions that can be used in:
  - SELECT clauses
  - WHERE/HAVING predicates
  - GROUP BY clauses
  - ORDER BY clauses
- These expressions must be valid in the context of the `table`

### 4. **Query Object State**
The `Query` object accumulates state as verbs are processed:
- `query.select`: List of UUIDs to select (final output columns)
- `query.where`: List of predicates for WHERE clause
- `query.having`: List of predicates for HAVING clause
- `query.group_by`: List of UUIDs for GROUP BY
- `query.order_by`: List of Order objects for ORDER BY
- `query.limit`/`query.offset`: For LIMIT/OFFSET

### 5. **Subquery Handling**
When `compile_query` is called (typically at `SubqueryMarker`):
- It creates a `Select` statement
- That `Select` is converted to a subquery via `.subquery()`
- The `sqa_expr` is updated to reference columns from the subquery
- The `query` is reset to only select the needed columns

## Example: Union Implementation

In the Union implementation, we:
1. Call `compile_ast` on both left and right to get their state
2. Call `compile_query` on both to get their `Select` statements
3. Use SQLAlchemy's `union`/`union_all` to combine them
4. Convert the result to a subquery

**Important**: Before calling `sa.union`, we check if either side is already a `Subquery` and unwrap it using `.original` to get the underlying `CompoundSelect`, because you can't union two subqueries directly - you need the original `CompoundSelect` objects.

## Common Pitfalls

1. **Calling `compile_query` too early**: Don't call it during AST traversal unless you're materializing a subquery
2. **Inconsistent state**: Make sure all UUIDs in `query` exist in `sqa_expr`
3. **Missing column references**: Ensure columns referenced in WHERE/HAVING/etc. are in `sqa_expr`
4. **Subquery unwrapping**: When combining queries (like UNION), unwrap subqueries to get the original `CompoundSelect`
