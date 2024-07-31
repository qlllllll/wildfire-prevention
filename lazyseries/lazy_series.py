import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union

class LazySeries(pd.Series):
    _metadata = ["fn", "params", "_parent", "_val", "_slice", "_executed", "_row_check", "_fn_name", "_result_type"]

    def __init__(self, data=None, fn=None, params=None, parent=None, slice_=None, fn_name=None, result_type=None, *args, **kwargs):
        """
        Initialize a LazySeries object.

        Parameters:
        - data: The data for the series.
        - fn: The function to apply.
        - params: Parameters for the function.
        - parent: The parent series.
        - slice_: The slice to apply.
        - fn_name: The name of the function.
        - result_type: The result type of the operation.
        """
        self._val = data.copy()
        super().__init__(data=data, *args, **kwargs)
        self.fn = fn
        self.params = params or []
        self._parent = parent
        self._slice = slice_ 
        self._executed = False if parent is not None else True
        self._row_check = pd.Series(False, index=data.index)
        self._result_type = result_type if result_type is not None else self._val.dtype
        self._fn_name = fn_name or (fn.__name__ if hasattr(fn, '__name__') else str(fn))

    def execute(self, _slice=None):
        """
        Execute the lazy operation.

        Parameters:
        - _slice: The slice to apply.

        Returns:
        The resulting series after executing the function.
        """
        if self._executed:
            return self._val

        if self._parent is not None:
            if isinstance(self._slice, LazySeries):
                _slice = None
            if isinstance(self._parent, LazyDataFrame):
                parent_val = self._parent.execute(_slice, [self.name])[self.name]
            else:
                parent_val = self._parent.execute(_slice)
        else:
            parent_val = pd.Series(self._val)

        if self.fn is None: 
            self._val = parent_val
            if isinstance(self._slice, LazySeries):
                self._slice = self._slice.show()
                self._val = self._val[self._slice]
            return self._val

        mask = ~self._row_check
        if _slice is not None:
            mask &= self._val.index.isin(self._val[_slice].index)
        
        args, kwargs = self.params
        if self._fn_name == "merge":
            self._val = self.fn(*args, **kwargs)
            self._row_check = pd.Series(True, index=self._val.index)
        elif self._fn_name == "explode":
            self._val = parent_val.explode()
            self._row_check = pd.Series(True, index=self._val.index)
        try:
            result = self.fn(parent_val[mask], *args, **kwargs)
        except:
            result = parent_val[mask].apply(self.fn, *args, **kwargs)
        if self._val.dtype != result.dtype:
            self._val = self._val.astype(object)
        self._val[mask] = result
        inferred_dtype = pd.Series(self._val).infer_objects().dtype
        if inferred_dtype != object:
            self._val = self._val.astype(inferred_dtype)

        self._row_check[mask] = True
        if self._row_check.all():
            self._executed = True
            self.fn, self.params = None, None
            self._fn_name = None

        return self._val

    def show(self):
        """
        Show the resulting series after executing the function.
        """
        return self.execute(self._slice)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the series.

        Parameters:
        - func: The function to apply.
        - args: Additional arguments for the function.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazySeries object with the applied function.
        """
        fn_name = func.__name__ if hasattr(func, '__name__') else str(func)
        return LazySeries(data=self._val, fn=func, params=(args, kwargs), parent=self, fn_name=fn_name)

    def where(self, cond, other=pd.NA, **kwargs):
        """
        Filter elements of the series according to a condition.

        Parameters:
        - cond: The condition to apply.
        - other: Elements to replace where the condition is False.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazySeries object with the applied condition.
        """
        return LazySeries(data=self._val, fn=pd.Series.where, params=((cond, other), kwargs), parent=self)

    def take(self, indices, axis=0, **kwargs):
        """
        Return elements at the given positions.

        Parameters:
        - indices: The indices of the elements to take.
        - axis: The axis along which to take.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazySeries object with the taken elements.
        """
        return LazySeries(data=self._val, fn=pd.Series.take, params=((indices, axis), kwargs), parent=self)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """
        Subset the series rows or columns according to labels in the specified index.

        Parameters:
        - items: List of labels to filter.
        - like: String that the index starts with.
        - regex: Regular expression to filter by.
        - axis: The axis to filter on.

        Returns:
        A new LazySeries object with the filtered elements.
        """
        return LazySeries(data=self._val, fn=pd.Series.filter, params=((items, like, regex, axis), {}), parent=self)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=True, right_index=True, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        """
        Merge the series with another series or DataFrame.

        Parameters:
        - right: The right series or DataFrame to merge with.
        - how: Type of merge to be performed.
        - on: Column or index level names to join on.
        - left_on: Column or index level names to join on in the left DataFrame.
        - right_on: Column or index level names to join on in the right DataFrame.
        - left_index: Use the index from the left DataFrame as the join key(s).
        - right_index: Use the index from the right DataFrame as the join key(s).
        - sort: Sort the join keys lexicographically in the result DataFrame.
        - suffixes: Suffix to apply to overlapping column names.
        - copy: If False, avoid copy if possible.
        - indicator: If True, adds a column to the output DataFrame called "_merge".
        - validate: If specified, checks if merge is of specified type.

        Returns:
        A new LazyDataFrame object with the merged result.
        """
        if isinstance(right, LazySeries):
            right = right._val
        if not self._val.name:
            self._val.name = 'left'
        if not right.name:
            right.name = 'right'
        return LazyDataFrame(data=pd.DataFrame(self._val), fn=pd.merge, params=({}, (self._val, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate), {}), parent=self)

    def explode(self):
        """
        Transform each element of a list-like to a row.

        Returns:
        A new LazySeries object with the exploded elements.
        """
        return LazySeries(data=self._val, fn=pd.Series.explode, params=({}, {}), parent=self, fn_name="explode", result_type=self.dtype)

    def __getitem__(self, key):
        """
        Get item(s) from the series.

        Parameters:
        - key: The key(s) to get.

        Returns:
        The corresponding LazySeries or value.
        """
        if isinstance(key, (slice, int)):
            return LazySeries(data=self._val[key], parent=self, slice_=key)
        elif isinstance(key, LazySeries) and key._result_type == bool:
            return LazySeries(data=self._val, parent=self, slice_=key, result_type=self._result_type)
        elif isinstance(key, pd.Series) and key.dtype == 'bool':
            return LazySeries(data=self._val[key], parent=self, slice_=key)
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            return LazySeries(result, parent=self)
        else:
            return result

    @property
    def iloc(self):
        """
        Purely integer-location based indexing for selection by position.
        """
        class _iLocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.iloc[key]
                return LazySeries(data=data, parent=self.parent, slice_=key)

        return _iLocIndexer(self)

    @property
    def loc(self):
        """
        Access a group of rows and columns by labels or a boolean array.
        """
        class _LocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.loc[key]
                return LazySeries(data=data, parent=self.parent, slice_=key)

        return _LocIndexer(self)

    def _binary_op(self, other, op, result_dtype=None, fn_name=None):
        """
        Perform a binary operation.

        Parameters:
        - other: The other operand.
        - op: The binary operation function.
        - result_dtype: The result data type.
        - fn_name: The name of the function.

        Returns:
        A new LazySeries object with the result of the binary operation.
        """
        if isinstance(other, LazySeries):
            fn = lambda x: op(x, other._val)
        else:
            fn = lambda x: op(x, other)
        return LazySeries(data=self._val, fn=fn, params=({}, {}), parent=self, fn_name=op.__name__, result_type=result_dtype)

    def __gt__(self, other):
        """
        Greater than comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.gt, result_dtype=bool, fn_name='gt')

    def __ge__(self, other):
        """
        Greater than or equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.ge, result_dtype=bool, fn_name='ge')

    def __lt__(self, other):
        """
        Less than comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.lt, result_dtype=bool, fn_name='lt')

    def __le__(self, other):
        """
        Less than or equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.le, result_dtype=bool, fn_name='le')

    def __eq__(self, other):
        """
        Equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.eq, result_dtype=bool, fn_name='eq')

    def __ne__(self, other):
        """
        Not equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the comparison.
        """
        return self._binary_op(other, pd.Series.ne, result_dtype=bool, fn_name='ne')

    def __add__(self, other):
        """
        Addition operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the addition.
        """
        return self._binary_op(other, pd.Series.add, fn_name='add')

    def __sub__(self, other):
        """
        Subtraction operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the subtraction.
        """
        return self._binary_op(other, pd.Series.sub, fn_name='sub')

    def __mul__(self, other):
        """
        Multiplication operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the multiplication.
        """
        return self._binary_op(other, pd.Series.mul, fn_name='mul')

    def __truediv__(self, other):
        """
        True division operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the division.
        """
        return self._binary_op(other, pd.Series.truediv, fn_name='truediv')

    def __floordiv__(self, other):
        """
        Floor division operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the floor division.
        """
        return self._binary_op(other, pd.Series.floordiv, fn_name='floordiv')

    def __mod__(self, other):
        """
        Modulus operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the modulus.
        """
        return self._binary_op(other, pd.Series.mod, fn_name='mod')

    def __pow__(self, other):
        """
        Power operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazySeries object with the result of the power.
        """
        return self._binary_op(other, pd.Series.pow, fn_name='pow')

    def __repr__(self):
        """
        Return a string representation of the series.

        Returns:
        A string representation of the series.
        """
        val_repr = repr(self._val)
        graph_repr = self.show_computation_graph()
        return f"Value:\n{val_repr}\nComputation Graph: {graph_repr}"

    def show_computation_graph(self):
        """
        Show the computation graph.

        Returns:
        A string representation of the computation graph.
        """
        graph = []
        current = self
        while current is not None:
            if current._fn_name is not None: 
                graph.append(f"{current._fn_name}")
            elif current.fn is not None:
                graph.append(f"{current._fn_name}")
            current = current._parent
        return " -> ".join(graph[::-1])

    def __str__(self):
        """
        Return a string representation of the series.

        Returns:
        A string representation of the series.
        """
        return self.__repr__()


class LazyDataFrame(pd.DataFrame):
    _metadata = ["fn", "params", "_parent", "_val", "_row_slice", "_col_slice", "_mask", "_executed", "_bool", "_fn_name"]

    def __init__(self, data=None, fn=None, params=None, parent=None, fn_name=None, row_slice=None, col_slice=None, *args, **kwargs):
        """
        Initialize a LazyDataFrame object.

        Parameters:
        - data: The data for the DataFrame.
        - fn: The function to apply.
        - params: Parameters for the function.
        - parent: The parent DataFrame.
        - fn_name: The name of the function.
        - row_slice: The row slice to apply.
        - col_slice: The column slice to apply.
        """
        self._val = data.copy()
        super().__init__(data=data, *args, **kwargs)
        self.fn = fn
        self.params = params or []
        self._parent = parent
        self._row_slice = row_slice
        self._col_slice = col_slice
        self._executed = False
        self._mask = pd.DataFrame(False, index=self.index, columns=self.columns)
        self._fn_name = fn_name or (fn.__name__ if hasattr(fn, '__name__') else str(fn))

    def apply(self, func, axis=0, *args, **kwargs):
        """
        Apply a function to the DataFrame.

        Parameters:
        - func: The function to apply.
        - axis: The axis along which to apply the function.
        - args: Additional arguments for the function.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazyDataFrame object with the applied function.
        """
        fn_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        if self._fn_name == "merge":
            _axis, _args, _kwargs = self.params
            cols = self._val.columns.tolist()
            if 'right' in _kwargs:
                right_columns = _kwargs['right'].columns.tolist()
                cols.extend(right_columns)
            elif len(_args) > 1:
                right = _args[1]
                if isinstance(right, pd.DataFrame):
                    cols.extend(right.columns.tolist())
                elif isinstance(right, LazyDataFrame):
                    cols.extend(right._val.columns.tolist())
                elif isinstance(right, pd.Series):
                    cols.append(right.name)
        else: 
            cols = self._val.columns.tolist()

        return LazyDataFrame(data=self._val, fn=func, params=(axis, args, kwargs), parent=self, fn_name=fn_name, col_slice=cols)

    def where(self, cond, other=pd.NA, **kwargs):
        """
        Filter elements of the DataFrame according to a condition.

        Parameters:
        - cond: The condition to apply.
        - other: Elements to replace where the condition is False.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazyDataFrame object with the applied condition.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.where, params=({}, (cond, other), kwargs), parent=self)

    def take(self, indices, axis=0, **kwargs):
        """
        Return elements at the given positions.

        Parameters:
        - indices: The indices of the elements to take.
        - axis: The axis along which to take.
        - kwargs: Additional keyword arguments for the function.

        Returns:
        A new LazyDataFrame object with the taken elements.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.take, params=({}, (indices, axis), kwargs), parent=self)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """
        Subset the DataFrame rows or columns according to labels in the specified index.

        Parameters:
        - items: List of labels to filter.
        - like: String that the index starts with.
        - regex: Regular expression to filter by.
        - axis: The axis to filter on.

        Returns:
        A new LazyDataFrame object with the filtered elements.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.filter, params=({}, (items, like, regex, axis), {}), parent=self)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        """
        Merge the DataFrame with another DataFrame.

        Parameters:
        - right: The right DataFrame to merge with.
        - how: Type of merge to be performed.
        - on: Column or index level names to join on.
        - left_on: Column or index level names to join on in the left DataFrame.
        - right_on: Column or index level names to join on in the right DataFrame.
        - left_index: Use the index from the left DataFrame as the join key(s).
        - right_index: Use the index from the right DataFrame as the join key(s).
        - sort: Sort the join keys lexicographically in the result DataFrame.
        - suffixes: Suffix to apply to overlapping column names.
        - copy: If False, avoid copy if possible.
        - indicator: If True, adds a column to the output DataFrame called "_merge".
        - validate: If specified, checks if merge is of specified type.

        Returns:
        A new LazyDataFrame object with the merged result.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.merge, params=({}, (self._val, right._val, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate), {}), parent=self)

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None):
        """
        Join columns of another DataFrame.

        Parameters:
        - other: The other DataFrame to join with.
        - on: Column or index level names to join on.
        - how: Type of join to be performed.
        - lsuffix: Suffix to apply to overlapping column names in the left DataFrame.
        - rsuffix: Suffix to apply to overlapping column names in the right DataFrame.
        - sort: Sort the join keys lexicographically in the result DataFrame.
        - validate: If specified, checks if merge is of specified type.

        Returns:
        A new LazyDataFrame object with the joined result.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.join, params=({}, (self._val, other, on, how, lsuffix, rsuffix, sort, validate), {}), parent=self)

    def explode(self, column):
        """
        Transform each element of a list-like to a row.

        Parameters:
        - column: The column to explode.

        Returns:
        A new LazyDataFrame object with the exploded elements.
        """
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.explode, params=({}, [column], {}), parent=self, fn_name='explode')

    def execute(self, _row_slice=None, _col_slice=None):
        """
        Execute the lazy operation.

        Parameters:
        - _row_slice: The row slice to apply.
        - _col_slice: The column slice to apply.

        Returns:
        The resulting DataFrame after executing the function.
        """
        if _row_slice is None:
            _row_slice = self._row_slice

        if _col_slice is None:
            _col_slice = self._col_slice

        if self._parent is not None:
            if isinstance(self._parent, LazySeries):
                parent_val = self._parent.execute(_row_slice)
            elif self._parent._fn_name == "merge":
                if isinstance(_row_slice, LazySeries):
                    _row_slice._parent = None
                    _row_slice = self._row_slice.show()
                    self._row_slice = _row_slice
                aligned_mask = _row_slice.reindex(self.index, fill_value=False)
                parent_val = self._parent.execute(aligned_mask, None)   
                self._mask = pd.DataFrame(False, index=parent_val.index, columns=parent_val.columns)
                self._val = parent_val
            else: 
                if isinstance(_row_slice, LazySeries):
                    _row_slice = _row_slice.show()          
                parent_val = self._parent.execute(_row_slice, _col_slice)
        else:
            parent_val = pd.DataFrame(self._val)

        if self.fn is None: 
            self._val = parent_val
            if isinstance(self._row_slice, LazySeries):
                self._row_slice = self._row_slice.show()
                self._val = self._val[self._row_slice]
            return self._val

        if _row_slice is not None:
            if isinstance(_row_slice, pd.Series) and _row_slice.dtype == bool:
                row_mask = _row_slice
            else:
                row_mask = self.index.isin(parent_val.index[_row_slice])
        else:
            row_mask = slice(None)

        if _col_slice is not None:
            col_mask = self.columns.isin(parent_val[_col_slice].columns)
        else:
            col_mask = slice(None)

        axis, args, kwargs = self.params
        cell_mask = ~self._mask.loc[row_mask, col_mask]
        if self._fn_name in "merge":
            args = (args[0].loc[row_mask, col_mask],) + args[1:]
            self._val = self.fn(*args, **kwargs)
            self._mask.loc[:, :] = True
        elif self._fn_name == "explode":
            self._val = parent_val.explode(*args, **kwargs)
            self._mask.loc[:, :] = True
        else:
            try:
                self._val[cell_mask] = self.fn(parent_val[cell_mask], axis=axis, *args, **kwargs)
            except:
                self._val[cell_mask] = parent_val[cell_mask].apply(self.fn, axis=axis, *args, **kwargs)
        self._mask.loc[row_mask, col_mask] = True

        if self._mask.all().all():
            self.fn, self.params = None, None
            self._fn_name = None
            self._executed = True

        return self._val

    def show(self):
        """
        Show the resulting DataFrame after executing the function.
        """
        return self.execute(self._row_slice, self._col_slice)

    def __getitem__(self, key):
        """
        Get item(s) from the DataFrame.

        Parameters:
        - key: The key(s) to get.

        Returns:
        The corresponding LazyDataFrame or LazySeries.
        """
        if isinstance(key, str):
            data = self._val[key]
            return LazySeries(data=data, parent=self, fn_name=f"column: {key}")
        elif isinstance(key, list) and all(isinstance(i, str) for i in key):
            return LazyDataFrame(data=self._val[key], parent=self, col_slice=key, fn_name=f"columns: {key}")
        elif isinstance(key, (slice, int, list)):
            return LazyDataFrame(data=self._val.iloc[key], parent=self, row_slice=key, fn_name=f"rows: {key}")
        elif isinstance(key, pd.Series) and key.dtype == bool:
            return LazyDataFrame(data=self._val[key], parent=self, fn_name="boolean mask")
        elif isinstance(key, LazySeries) and key._result_type == bool:
            return LazyDataFrame(data=self._val, parent=self, row_slice=key, fn_name="lazy boolean mask")
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame):
            return LazyDataFrame(result, parent=self, fn_name="DataFrame slice")
        else:
            return result

    def _binary_op(self, other, op, fn_name=None):
        """
        Perform a binary operation.

        Parameters:
        - other: The other operand.
        - op: The binary operation function.
        - fn_name: The name of the function.

        Returns:
        A new LazyDataFrame object with the result of the binary operation.
        """
        fn_name = fn_name or op.__name__
        if isinstance(other, LazyDataFrame):
            return LazyDataFrame(data=self._val, fn=op, params=(self._val, other._val), parent=self, fn_name=fn_name)
        else:
            return LazyDataFrame(data=self._val, fn=op, params=(self._val, other), parent=self, fn_name=fn_name)

    def __gt__(self, other):
        """
        Greater than comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.gt, fn_name='gt')

    def __ge__(self, other):
        """
        Greater than or equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.ge, fn_name='ge')

    def __lt__(self, other):
        """
        Less than comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.lt, fn_name='lt')

    def __le__(self, other):
        """
        Less than or equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.le, fn_name='le')

    def __eq__(self, other):
        """
        Equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.eq, fn_name='eq')

    def __ne__(self, other):
        """
        Not equal to comparison.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the comparison.
        """
        return self._binary_op(other, pd.DataFrame.ne, fn_name='ne')

    def __add__(self, other):
        """
        Addition operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the addition.
        """
        return self._binary_op(other, pd.DataFrame.add, fn_name='add')

    def __sub__(self, other):
        """
        Subtraction operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the subtraction.
        """
        return self._binary_op(other, pd.DataFrame.sub, fn_name='sub')

    def __mul__(self, other):
        """
        Multiplication operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the multiplication.
        """
        return self._binary_op(other, pd.DataFrame.mul, fn_name='mul')

    def __truediv__(self, other):
        """
        True division operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the division.
        """
        return self._binary_op(other, pd.DataFrame.truediv, fn_name='truediv')

    def __floordiv__(self, other):
        """
        Floor division operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the floor division.
        """
        return self._binary_op(other, pd.DataFrame.floordiv, fn_name='floordiv')

    def __mod__(self, other):
        """
        Modulus operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the modulus.
        """
        return self._binary_op(other, pd.DataFrame.mod, fn_name='mod')

    def __pow__(self, other):
        """
        Power operation.

        Parameters:
        - other: The other operand.

        Returns:
        A new LazyDataFrame object with the result of the power.
        """
        return self._binary_op(other, pd.DataFrame.pow, fn_name='pow')

    def __repr__(self):
        """
        Return a string representation of the DataFrame.

        Returns:
        A string representation of the DataFrame.
        """
        val_repr = repr(self._val)
        graph_repr = self.show_computation_graph()
        return f"{val_repr}\nComputation Graph: {graph_repr}"

    def _repr_html_(self):
        """
        Return an HTML representation of the DataFrame.

        Returns:
        An HTML representation of the DataFrame.
        """
        val_repr = self._val._repr_html_()
        graph_repr = self.show_computation_graph().replace('<', '&lt;').replace('>', '&gt;')
        
        html = f"""
            <div>{val_repr}</div>
            <div>
                <strong style="font-size: 12px;">Computation Graph:</strong>
                <pre style="font-size: 12px; font-family: monospace; line-height: 1.4;">{graph_repr}</pre>
            </div>
        </div>
        """
        return html

    def show_computation_graph(self):
        """
        Show the computation graph.

        Returns:
        A string representation of the computation graph.
        """
        graph = []
        current = self
        while current is not None:
            if current.fn is not None:
                func_name = current.fn.__name__ if hasattr(current.fn, '__name__') else str(current.fn)
                graph.append(f"{func_name}")
            current = current._parent
        return " -> ".join(graph[::-1])

    def __str__(self):
        """
        Return a string representation of the DataFrame.

        Returns:
        A string representation of the DataFrame.
        """
        return self.__repr__()