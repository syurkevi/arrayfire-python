#!/usr/bin/env python

#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Vector algorithms (sum, min, sort, etc).
"""

from .array import Array, _nan_parallel_dim, _nan_reduce_all, _parallel_dim, _reduce_all
from .defaults import BINARYOP, c_bool_t, c_double_t, c_int_t, c_pointer, c_uint_t
from .library import backend, safe_call


def product(a, dim=None, nan_val=None):
    """
    Calculate the product of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.
    nan_val: optional: scalar. default: None
         The value that replaces NaN in the array

    Returns
    -------
    out: af.Array or scalar number
         The product of all elements in `a` along dimension `dim`.
         If `dim` is `None`, product of the entire Array is returned.
    """
    if nan_val is not None:
        if dim is not None:
            return _nan_parallel_dim(a, dim, backend.get().af_product_nan, nan_val)
        return _nan_reduce_all(a, backend.get().af_product_nan_all, nan_val)

    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_product)
    return _reduce_all(a, backend.get().af_product_all)


def min(a, dim=None):
    """
    Find the minimum value of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the minimum value is required.

    Returns
    -------
    out: af.Array or scalar number
         The minimum value of all elements in `a` along dimension `dim`.
         If `dim` is `None`, minimum value of the entire Array is returned.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_min)
    return _reduce_all(a, backend.get().af_min_all)


def max(a, dim=None):
    """
    Find the maximum value of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the maximum value is required.

    Returns
    -------
    out: af.Array or scalar number
         The maximum value of all elements in `a` along dimension `dim`.
         If `dim` is `None`, maximum value of the entire Array is returned.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_max)
    return _reduce_all(a, backend.get().af_max_all)


def all_true(a, dim=None):
    """
    Check if all the elements along a specified dimension are true.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.

    Returns
    -------
    out: af.Array or scalar number
         Af.array containing True if all elements in `a` along the dimension are True.
         If `dim` is `None`, output is True if `a` does not have any zeros, else False.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_all_true)
    return _reduce_all(a, backend.get().af_all_true_all)


def any_true(a, dim=None):
    """
    Check if any the elements along a specified dimension are true.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.

    Returns
    -------
    out: af.Array or scalar number
         Af.array containing True if any elements in `a` along the dimension are True.
         If `dim` is `None`, output is True if `a` does not have any zeros, else False.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_any_true)
    return _reduce_all(a, backend.get().af_any_true_all)


def imin(a, dim=None):
    """
    Find the value and location of the minimum value along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the minimum value is required.

    Returns
    -------
    (val, idx): tuple of af.Array or scalars
                `val` contains the minimum value of `a` along `dim`.
                `idx` contains the location of where `val` occurs in `a` along `dim`.
                If `dim` is `None`, `val` and `idx` value and location of global minimum.
    """
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(backend.get().af_imin(c_pointer(out.arr), c_pointer(idx.arr), a.arr, c_int_t(dim)))
        return out, idx

    real = c_double_t(0)
    imag = c_double_t(0)
    idx = c_uint_t(0)
    safe_call(backend.get().af_imin_all(c_pointer(real), c_pointer(imag), c_pointer(idx), a.arr))
    real = real.value
    imag = imag.value
    val = real if imag == 0 else real + imag * 1j
    return val, idx.value


def imax(a, dim=None):
    """
    Find the value and location of the maximum value along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the maximum value is required.

    Returns
    -------
    (val, idx): tuple of af.Array or scalars
                `val` contains the maximum value of `a` along `dim`.
                `idx` contains the location of where `val` occurs in `a` along `dim`.
                If `dim` is `None`, `val` and `idx` value and location of global maximum.
    """
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(backend.get().af_imax(c_pointer(out.arr), c_pointer(idx.arr), a.arr, c_int_t(dim)))
        return out, idx

    real = c_double_t(0)
    imag = c_double_t(0)
    idx = c_uint_t(0)
    safe_call(backend.get().af_imax_all(c_pointer(real), c_pointer(imag), c_pointer(idx), a.arr))
    real = real.value
    imag = imag.value
    val = real if imag == 0 else real + imag * 1j
    return val, idx.value


def accum(a, dim=0):
    """
    Cumulative sum of an array along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the cumulative sum is required.

    Returns
    -------
    out: af.Array
         array of same size as `a` containing the cumulative sum along `dim`.
    """
    return _parallel_dim(a, dim, backend.get().af_accum)


def scan(a, dim=0, op=BINARYOP.ADD, inclusive_scan=True):
    """
    Generalized scan of an array.

    Parameters
    ----------
    a   : af.Array
        Multi dimensional arrayfire array.

    dim : optional: int. default: 0
        Dimension along which the scan is performed.

    op  : optional: af.BINARYOP. default: af.BINARYOP.ADD.
        Binary option the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan: optional: bool. default: True
        Specifies if the scan is inclusive

    Returns
    ---------
    out : af.Array
        - will contain scan of input.
    """
    out = Array()
    safe_call(backend.get().af_scan(c_pointer(out.arr), a.arr, dim, op.value, inclusive_scan))
    return out


def scan_by_key(key, a, dim=0, op=BINARYOP.ADD, inclusive_scan=True):
    """
    Generalized scan by key of an array.

    Parameters
    ----------
    key : af.Array
        key array.

    a   : af.Array
        Multi dimensional arrayfire array.

    dim : optional: int. default: 0
        Dimension along which the scan is performed.

    op  : optional: af.BINARYOP. default: af.BINARYOP.ADD.
        Binary option the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan: optional: bool. default: True
        Specifies if the scan is inclusive

    Returns
    ---------
    out : af.Array
        - will contain scan of input.
    """
    out = Array()
    safe_call(backend.get().af_scan_by_key(c_pointer(out.arr), key.arr, a.arr, dim, op.value, inclusive_scan))
    return out


def where(a):
    """
    Find the indices of non zero elements

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.

    Returns
    -------
    idx: af.Array
         Linear indices for non zero elements.
    """
    out = Array()
    safe_call(backend.get().af_where(c_pointer(out.arr), a.arr))
    return out


def diff1(a, dim=0):
    """
    Find the first order differences along specified dimensions

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the differences are required.

    Returns
    -------
    out: af.Array
         Array whose length along `dim` is 1 less than that of `a`.
    """
    return _parallel_dim(a, dim, backend.get().af_diff1)


def diff2(a, dim=0):
    """
    Find the second order differences along specified dimensions

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the differences are required.

    Returns
    -------
    out: af.Array
         Array whose length along `dim` is 2 less than that of `a`.
    """
    return _parallel_dim(a, dim, backend.get().af_diff2)


def sort(a, dim=0, is_ascending=True):
    """
    Sort the array along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    out: af.Array
         array containing the sorted values

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    out = Array()
    safe_call(backend.get().af_sort(c_pointer(out.arr), a.arr, c_uint_t(dim), c_bool_t(is_ascending)))
    return out


def sort_index(a, dim=0, is_ascending=True):
    """
    Sort the array along a specified dimension and get the indices.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    (val, idx): tuple of af.Array
         `val` is an af.Array containing the sorted values.
         `idx` is an af.Array containing the original indices of `val` in `a`.

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    out = Array()
    idx = Array()
    safe_call(backend.get().af_sort_index(
        c_pointer(out.arr), c_pointer(idx.arr), a.arr, c_uint_t(dim), c_bool_t(is_ascending)))
    return out, idx


def sort_by_key(ik, iv, dim=0, is_ascending=True):
    """
    Sort an array based on specified keys

    Parameters
    ----------
    ik  : af.Array
         An Array containing the keys
    iv  : af.Array
         An Array containing the values
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    (ok, ov): tuple of af.Array
         `ok` contains the values from `ik` in sorted order
         `ov` contains the values from `iv` after sorting them based on `ik`

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    ov = Array()
    ok = Array()
    safe_call(backend.get().af_sort_by_key(
        c_pointer(ok.arr), c_pointer(ov.arr), ik.arr, iv.arr, c_uint_t(dim), c_bool_t(is_ascending)))
    return ov, ok


def set_unique(a, is_sorted=False):
    """
    Find the unique elements of an array.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    is_sorted: optional: bool. default: False
         Specifies if the input is pre-sorted.

    Returns
    -------
    out: af.Array
         an array containing the unique values from `a`
    """
    out = Array()
    safe_call(backend.get().af_set_unique(c_pointer(out.arr), a.arr, c_bool_t(is_sorted)))
    return out


def set_union(a, b, is_unique=False):
    """
    Find the union of two arrays.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    b  : af.Array
         A 1D arrayfire array.
    is_unique: optional: bool. default: False
         Specifies if the both inputs contain unique elements.

    Returns
    -------
    out: af.Array
         an array values after performing the union of `a` and `b`.
    """
    out = Array()
    safe_call(backend.get().af_set_union(c_pointer(out.arr), a.arr, b.arr, c_bool_t(is_unique)))
    return out


def set_intersect(a, b, is_unique=False):
    """
    Find the intersect of two arrays.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    b  : af.Array
         A 1D arrayfire array.
    is_unique: optional: bool. default: False
         Specifies if the both inputs contain unique elements.

    Returns
    -------
    out: af.Array
         an array values after performing the intersect of `a` and `b`.
    """
    out = Array()
    safe_call(backend.get().af_set_intersect(c_pointer(out.arr), a.arr, b.arr, c_bool_t(is_unique)))
    return out
