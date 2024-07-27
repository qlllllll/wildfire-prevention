import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union

class LazySeries(pd.Series):
    _metadata = ["fn", "params", "_parent", "_val", "_slice", "_executed", "_row_check", "_fn_name", "_result_type"]

    def __init__(self, data=None, fn=None, params=None, parent=None, slice_=None, fn_name=None, result_type=None, *args, **kwargs):
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

    @property
    def _constructor(self):
        return LazySeries

    @property
    def values(self):
        return self._val.values
    
    @property
    def size(self):
        return self._val.size

    @property
    def shape(self):
        return self._val.shape

    @property
    def dtype(self):
        return self._val.dtype

    def execute(self, _slice=None):
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

        return self._val

    def show(self):
        return self.execute(self._slice)

    def apply(self, func, *args, **kwargs):
        fn_name = func.__name__ if hasattr(func, '__name__') else str(func)
        return LazySeries(data=self._val, fn=func, params=(args, kwargs), parent=self, fn_name=fn_name)

    def where(self, cond, other=pd.NA, **kwargs):
        return LazySeries(data=self._val, fn=pd.Series.where, params=((cond, other), kwargs), parent=self)

    def take(self, indices, axis=0, **kwargs):
        return LazySeries(data=self._val, fn=pd.Series.take, params=((indices, axis), kwargs), parent=self)

    def filter(self, items=None, like=None, regex=None, axis=None):
        return LazySeries(data=self._val, fn=pd.Series.filter, params=((items, like, regex, axis), {}), parent=self)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=True, right_index=True, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        if isinstance(right, LazySeries):
            right = right._val
        if not self._val.name:
            self._val.name = 'left'
        if not right.name:
            right.name = 'right'
        return LazyDataFrame(data=pd.DataFrame(self._val), fn=pd.merge, params=({}, (self._val, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate), {}), parent=self)
   
    def explode(self):
        return LazySeries(data=self._val, fn=pd.Series.explode, params=({}, {}), parent=self, fn_name="explode", result_type=self.dtype)

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, dropna=True):
        return LazyGroupBy(self, by, axis, level, as_index, sort, group_keys, squeeze, observed, dropna)

    def __getitem__(self, key):
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
        class _iLocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.iloc[key]
                return LazySeries(data=data, parent=self.parent, slice_=key)

        return _iLocIndexer(self)

    @property
    def loc(self):
        class _LocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.loc[key]
                return LazySeries(data=data, parent=self.parent, slice_=key)

        return _LocIndexer(self)

    def _binary_op(self, other, op, result_dtype=None, fn_name=None):
        if isinstance(other, LazySeries):
            fn = lambda x: op(x, other._val)
        else:
            fn = lambda x: op(x, other)
        return LazySeries(data=self._val, fn=fn, params=({}, {}), parent=self, fn_name=op.__name__, result_type=result_dtype)

    def __gt__(self, other):
        return self._binary_op(other, pd.Series.gt, result_dtype=bool, fn_name='gt')

    def __ge__(self, other):
        return self._binary_op(other, pd.Series.ge, result_dtype=bool, fn_name='ge')

    def __lt__(self, other):
        return self._binary_op(other, pd.Series.lt, result_dtype=bool, fn_name='lt')

    def __le__(self, other):
        return self._binary_op(other, pd.Series.le, result_dtype=bool, fn_name='le')

    def __eq__(self, other):
        return self._binary_op(other, pd.Series.eq, result_dtype=bool, fn_name='eq')

    def __ne__(self, other):
        return self._binary_op(other, pd.Series.ne, result_dtype=bool, fn_name='ne')

    def __add__(self, other):
        return self._binary_op(other, pd.Series.add, fn_name='add')

    def __sub__(self, other):
        return self._binary_op(other, pd.Series.sub, fn_name='sub')

    def __mul__(self, other):
        return self._binary_op(other, pd.Series.mul, fn_name='mul')

    def __truediv__(self, other):
        return self._binary_op(other, pd.Series.truediv, fn_name='truediv')

    def __floordiv__(self, other):
        return self._binary_op(other, pd.Series.floordiv, fn_name='floordiv')

    def __mod__(self, other):
        return self._binary_op(other, pd.Series.mod, fn_name='mod')

    def __pow__(self, other):
        return self._binary_op(other, pd.Series.pow, fn_name='pow')
        
    def __repr__(self):
        val_repr = repr(self._val)
        graph_repr = self.show_computation_graph()
        return f"Value:\n{val_repr}\nComputation Graph: {graph_repr}"

    def show_computation_graph(self):
        graph = []
        current = self
        while current is not None:
            if current.fn is not None:
                graph.append(f"{current._fn_name}")
            current = current._parent
        return " -> ".join(graph[::-1])

    def __str__(self):
        return self.__repr__()

class LazyDataFrame(pd.DataFrame):
    _metadata = ["fn", "params", "_parent", "_val", "_row_slice", "_col_slice", "_mask", "_executed", "_bool", "_fn_name"]

    def __init__(self, data=None, fn=None, params=None, parent=None, fn_name=None, row_slice=None, col_slice=None, *args, **kwargs):
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

    @property
    def _constructor(self):
        return LazyDataFrame

    def apply(self, func, axis=0, *args, **kwargs):
        fn_name = func.__name__ if hasattr(func, '__name__') else str(func)
        return LazyDataFrame(data=self._val, fn=func, params=(axis, args, kwargs), parent=self, fn_name=fn_name, col_slice=self.columns.tolist())

    def where(self, cond, other=pd.NA, **kwargs):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.where, params=({}, (cond, other), kwargs), parent=self)

    def take(self, indices, axis=0, **kwargs):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.take, params=({}, (indices, axis), kwargs), parent=self)

    def filter(self, items=None, like=None, regex=None, axis=None):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.filter, params=({}, (items, like, regex, axis), {}), parent=self)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.merge, params=({}, (self._val, right._val, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate), {}), parent=self)

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.join, params=({}, (self._val, other, on, how, lsuffix, rsuffix, sort, validate), {}), parent=self)

    def explode(self, column):
        return LazyDataFrame(data=self._val, fn=pd.DataFrame.explode, params=({}, [column], {}), parent=self, fn_name='explode')

    def execute(self, _row_slice=None, _col_slice=None):
        if self._executed:
            return self._val

        if _row_slice is None:
            _row_slice = self._row_slice

        if _col_slice is None:
            _col_slice = self._col_slice

        if self._parent is not None:
            if isinstance(self._parent, LazySeries):
                parent_val = self._parent.execute(_row_slice)
            elif isinstance(self._row_slice, LazySeries):
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
            row_mask = self.index.isin(parent_val.iloc[_row_slice].index)
        else:
            row_mask = slice(None)

        if _col_slice is not None:
            col_mask = self.columns.isin(parent_val[_col_slice].columns)
        else:
            col_mask = slice(None)

        axis, args, kwargs = self.params
        cell_mask = ~self._mask.loc[row_mask, col_mask]
        if self._fn_name == "merge":
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
            self._executed = True

        return self._val

    def show(self):
        return self.execute(self._row_slice, self._col_slice)

    def __getitem__(self, key):
        if isinstance(key, str):
            data = self._val[key]
            return LazySeries(data=data, parent=self, fn_name=f"Column: {key}")
        elif isinstance(key, list) and all(isinstance(i, str) for i in key):
            return LazyDataFrame(data=self._val[key], parent=self, col_slice=key)
        elif isinstance(key, (slice, int, list)):
            return LazyDataFrame(data=self._val.iloc[key], parent=self, row_slice=key)
        elif isinstance(key, pd.Series) and key.dtype == bool:
            return LazyDataFrame(data=self._val[mask], parent=self, row_slice=key)
        elif isinstance(key, LazySeries) and key._result_type == bool:
            return LazyDataFrame(data=self._val, parent=self, row_slice=key)
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame):
            return LazyDataFrame(result, parent=self)
        else:
            return result

    @property
    def iloc(self):
        class _iLocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.iloc[key]
                return LazyDataFrame(data=data, parent=self.parent, row_slice=key)

        return _iLocIndexer(self)

    @property
    def loc(self):
        class _LocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, key):
                data = self.parent._val.loc[key]
                return LazyDataFrame(data=data, parent=self.parent, row_slice=key)

        return _LocIndexer(self)

    def _binary_op(self, other, op, fn_name=None):
        fn_name = fn_name or op.__name__
        if isinstance(other, LazyDataFrame):
            return LazyDataFrame(data=self._val, fn=op, params=(self._val, other._val), parent=self, fn_name=fn_name)
        else:
            return LazyDataFrame(data=self._val, fn=op, params=(self._val, other), parent=self, fn_name=fn_name)
    
    def __gt__(self, other):
        return self._binary_op(other, pd.DataFrame.gt, fn_name='gt')

    def __ge__(self, other):
        return self._binary_op(other, pd.DataFrame.ge, fn_name='ge')

    def __lt__(self, other):
        return self._binary_op(other, pd.DataFrame.lt, fn_name='lt')

    def __le__(self, other):
        return self._binary_op(other, pd.DataFrame.le, fn_name='le')

    def __eq__(self, other):
        return self._binary_op(other, pd.DataFrame.eq, fn_name='eq')

    def __ne__(self, other):
        return self._binary_op(other, pd.DataFrame.ne, fn_name='ne')
        
    def __add__(self, other):
        return self._binary_op(other, pd.DataFrame.add, fn_name='add')

    def __sub__(self, other):
        return self._binary_op(other, pd.DataFrame.sub, fn_name='sub')

    def __mul__(self, other):
        return self._binary_op(other, pd.DataFrame.mul, fn_name='mul')

    def __truediv__(self, other):
        return self._binary_op(other, pd.DataFrame.truediv, fn_name='truediv')

    def __floordiv__(self, other):
        return self._binary_op(other, pd.DataFrame.floordiv, fn_name='floordiv')

    def __mod__(self, other):
        return self._binary_op(other, pd.DataFrame.mod, fn_name='mod')

    def __pow__(self, other):
        return self._binary_op(other, pd.DataFrame.pow, fn_name='pow')

    def __repr__(self):
        val_repr = repr(self._val)
        graph_repr = self.show_computation_graph()
        return f"{val_repr}\nComputation Graph: {graph_repr}"

    def _repr_html_(self):
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
        graph = []
        current = self
        while current is not None:
            if current.fn is not None:
                func_name = current.fn.__name__ if hasattr(current.fn, '__name__') else str(current.fn)
                graph.append(f"{func_name}")
            current = current._parent
        return " -> ".join(graph[::-1])

    def __str__(self):
        return self.__repr__()
