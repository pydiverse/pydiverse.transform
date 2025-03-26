=================
Column Operations
=================

.. toctree::
   :maxdepth: 1
   :hidden:

   arithmetic
   logical
   comparison
   numerical
   string
   datetime
   aggregation
   window
   sorting_markers
   horizontal_aggregation
   conditional_logic
   type_conversion


Expression methods
------------------

.. currentmodule:: pydiverse.transform.ColExpr

.. autosummary::
   :nosignatures:

   __add__
   __and__
   __eq__
   __floordiv__
   __ge__
   __gt__
   __invert__
   __le__
   __lt__
   __mod__
   __mul__
   __ne__
   __neg__
   __or__
   __pos__
   __pow__
   __sub__
   __truediv__
   __xor__
   abs
   all
   any
   ascending
   cast
   ceil
   count
   dense_rank
   descending
   dt.day
   dt.day_of_week
   dt.day_of_year
   dt.hour
   dt.microsecond
   dt.millisecond
   dt.minute
   dt.month
   dt.second
   dt.year
   dur.days
   dur.hours
   dur.microseconds
   dur.milliseconds
   dur.minutes
   dur.seconds
   exp
   fill_null
   floor
   is_in
   is_inf
   is_nan
   is_not_inf
   is_not_nan
   is_not_null
   is_null
   list.agg
   log
   map
   max
   mean
   min
   nulls_first
   nulls_last
   prefix_sum
   rank
   round
   shift
   str.contains
   str.ends_with
   str.join
   str.len
   str.lower
   str.replace_all
   str.slice
   str.starts_with
   str.strip
   str.to_date
   str.to_datetime
   str.upper
   sum

Global functions
----------------

.. currentmodule:: pydiverse.transform

.. autosummary::
   :nosignatures:

   all
   any
   coalesce
   count
   dense_rank
   lit
   max
   min
   rank
   row_number
   sum
   when
