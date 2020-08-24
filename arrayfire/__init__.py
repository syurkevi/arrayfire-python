#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
ArrayFire is a high performance scientific computing library with an easy to use API.


    >>> # Monte Carlo estimation of pi
    >>> def calc_pi_device(samples):
            # Simple, array based API
            # Generate uniformly distributed random numers
            x = af.randu(samples)
            y = af.randu(samples)
            # Supports Just In Time Compilation
            # The following line generates a single kernel
            within_unit_circle = (x * x + y * y) < 1
            # Intuitive function names
            return 4 * af.count(within_unit_circle) / samples

Programs written using ArrayFire are portable across CUDA, OpenCL and CPU devices.

The default backend is chosen in the following order of preference based on the available libraries:

    1. CUDA
    2. OpenCL
    3. CPU

The backend can be chosen at the beginning of the program by using the following function

    >>> af.set_backend(name)

where name is one of 'cuda', 'opencl' or 'cpu'.

The functionality provided by ArrayFire spans the following domains:

    1. Vector Algorithms
    2. Image Processing
    3. Signal Processing
    4. Computer Vision
    5. Linear Algebra
    6. Statistics

"""

from .algorithm import (
    accum, all_true, any_true, diff1, diff2, imax, imin, max, min, product, scan, scan_by_key, set_intersect,
    set_union, set_unique, sort, sort_by_key, sort_index, where)
from .arith import (
    abs, acos, acosh, arg, asin, asinh, atan, atan2, atanh, cbrt, ceil, clamp, conjg, cos, cosh, cplx, erf, erfc, exp,
    expm1, factorial, floor, hypot, imag, isinf, isnan, iszero, lgamma, log, log1p, log2, log10, maxof, minof, pow,
    pow2, real, rem, root, round, sigmoid, sign, sin, sinh, sqrt, tan, tanh, tgamma, trunc)
from .array import (
    Array, cast, constant_array, count, display, get_display_dims_limit, read_array, save_array,
    set_display_dims_limit, sum, transpose, transpose_inplace)
from .base import BaseArray
from .bcast import broadcast
from .blas import dot, matmul, matmulNT, matmulTN, matmulTT
from .cuda import get_native_id, get_stream, set_native_id
from .data import (
    constant, diag, flat, flip, identity, iota, join, lookup, lower, moddims, range, reorder, replace, select, shift,
    tile, upper)
from .library import (
    BACKEND, BINARYOP, CANNY_THRESHOLD, COLORMAP, CONNECTIVITY, CONV_DOMAIN, CONV_MODE, CSPACE, DIFFUSION, ERR, FLUX,
    HOMOGRAPHY, IMAGE_FORMAT, INTERP, MARKER, MATCH, MATPROP, MOMENT, NORM, PAD, STORAGE, TOPK, YCC_STD, Dtype, Source)
from .device import (
    alloc_device, alloc_host, alloc_pinned, device_gc, device_info, device_mem_info, eval, free_device, free_host,
    free_pinned, get_device, get_device_count, get_device_ptr, get_manual_eval_flag, get_reversion, get_version, info,
    info_str, init, is_dbl_supported, is_locked_array, lock_array, lock_device_ptr, print_mem_info, set_device,
    set_manual_eval_flag, sync, unlock_array, unlock_device_ptr)
from .graphics import Window
from .image import (
    anisotropic_diffusion, bilateral, canny, color_space, dilate, dilate3, erode, erode3, gaussian_kernel, gradient,
    gray2rgb, hist_equal, histogram, hsv2rgb, is_image_io_available, load_image, load_image_native, maxfilt,
    mean_shift, minfilt, moments, regions, resize, rgb2gray, rgb2hsv, rgb2ycbcr, rotate, sat, save_image,
    save_image_native, scale, skew, sobel_derivatives, sobel_filter, transform, translate, unwrap, wrap, ycbcr2rgb)
from .index import Index, ParallelRange, Seq
from .interop import AF_NUMBA_FOUND, AF_NUMPY_FOUND, AF_PYCUDA_FOUND, AF_PYOPENCL_FOUND
from .lapack import (
    cholesky, cholesky_inplace, det, inverse, is_lapack_available, lu, lu_inplace, norm, qr, qr_inplace, rank, solve,
    solve_lu, svd, svd_inplace)
from .library import (
    get_active_backend, get_available_backends, get_backend, get_backend_count, get_backend_id, get_device_id,
    get_size_of, safe_call, set_backend)
from .random import (
    RANDOM_ENGINE, Random_Engine, get_default_random_engine, get_seed, randn, randu, set_default_random_engine_type,
    set_seed)
from .signal import (
    approx1, approx2, convolve, convolve1, convolve2, convolve2_separable, convolve3, dft, fft, fft2, fft2_c2r,
    fft2_inplace, fft2_r2c, fft3, fft3_c2r, fft3_inplace, fft3_r2c, fft_c2r, fft_convolve, fft_convolve1,
    fft_convolve2, fft_convolve3, fft_inplace, fft_r2c, fir, idft, ifft, ifft2, ifft2_inplace, ifft3, ifft3_inplace,
    ifft_inplace, iir, medfilt, medfilt1, medfilt2, set_fft_plan_cache_size)
from .sparse import (
    convert_sparse, convert_sparse_to_dense, create_sparse, create_sparse_from_dense, create_sparse_from_host,
    sparse_get_col_idx, sparse_get_info, sparse_get_nnz, sparse_get_row_idx, sparse_get_storage, sparse_get_values)
from .statistics import corrcoef, cov, mean, median, stdev, topk, var
from .timer import timeit
from .util import dim4, dim4_to_tuple, implicit_dtype, number_dtype, to_str

try:
    # FIXME: pycuda imported but unused
    import pycuda.autoinit
except ImportError:
    pass


# do not export default modules as part of arrayfire
del ct
del inspect
del numbers
del os

if (AF_NUMPY_FOUND):
    del np

__all__ = [
    # algorithm
    "accum", "all_true", "any_true", "max", "min", "product", "scan", "scan_by_key", "set_intersect", "set_union",
    "set_unique", "sort", "sort_by_key", "sort_index", "imin", "imax", "where", "diff1", "diff2",
    # arith
    "abs", "acos", "acosh", "arg", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "clamp", "conjg", "cos",
    "cosh", "cplx", "erf", "erfc", "exp", "expm1", "factorial", "floor", "hypot", "imag", "isinf", "isnan", "iszero",
    "lgamma", "log", "log1p", "log2", "log10", "maxof", "minof", "pow", "pow2", "real", "rem", "root", "round",
    "sigmoid", "sign", "sin", "sinh", "sqrt", "tan", "tanh", "tgamma", "trunc",
    # array
    "Array", "cast", "count", "sum", "transpose", "transpose_inplace", "display", "set_display_dims_limit",
    "get_display_dims_limit", "constant_array", "save_array", "read_array",
    # base
    "BaseArray",
    # bcast
    "broadcast",
    # blas
    "dot", "matmul", "matmulTN", "matmulNT", "matmulTT",
    # cuda
    "get_native_id", "get_stream", "set_native_id",
    # data
    "constant", "diag", "flat", "flip", "identity", "iota", "join", "lower", "moddims", "range", "reorder", "replace",
    "select", "shift", "tile", "upper", "lookup",
    # library enums
    "BINARYOP", "CSPACE", "DIFFUSION", "FLUX", "MATPROP", "NORM", "Dtype", "COLORMAP", "ERR", "Source", "INTERP",
    "PAD", "CONNECTIVITY", "CONV_MODE", "CONV_DOMAIN", "MATCH", "YCC_STD", "IMAGE_FORMAT", "HOMOGRAPHY", "BACKEND",
    "MARKER", "MOMENT", "STORAGE", "CANNY_THRESHOLD", "FLUX", "TOPK",
    # device
    "device_gc", "device_info", "device_mem_info", "eval", "get_device", "get_device_count", "get_device_ptr",
    "get_manual_eval_flag", "info", "is_dbl_supported", "is_locked_array", "lock_array", "set_device",
    "set_manual_eval_flag", "sync", "unlock_array", "get_reversion", "get_version", "init", "info_str",
    "print_mem_info", "lock_device_ptr", "unlock_device_ptr", "alloc_device", "alloc_host", "alloc_pinned",
    "free_device", "free_host", "free_pinned",
    # graphics
    "Window",
    # image
    "anisotropic_diffusion", "bilateral", "canny", "color_space", "dilate", "dilate3", "erode", "erode3",
    "gaussian_kernel", "gradient", "gray2rgb", "hist_equal", "histogram", "hsv2rgb", "maxfilt", "mean_shift",
    "minfilt", "regions", "resize", "rgb2gray", "rgb2hsv", "rgb2ycbcr", "rotate", "sat", "scale", "skew",
    "sobel_derivatives", "sobel_filter", "transform", "translate", "unwrap", "wrap", "ycbcr2rgb", "load_image",
    "save_image", "load_image_native", "save_image_native", "moments", "is_image_io_available",
    # index
    "ParallelRange", "Seq", "Index",
    # interop
    "AF_NUMBA_FOUND", "AF_NUMPY_FOUND", "AF_PYCUDA_FOUND", "AF_PYOPENCL_FOUND",
    # lapack
    "cholesky", "cholesky_inplace", "det", "inverse", "lu", "lu_inplace", "norm", "qr", "qr_inplace", "rank", "solve",
    "solve_lu", "svd", "svd_inplace", "is_lapack_available",
    # library
    "get_active_backend", "set_backend", "get_backend", "get_backend_id", "get_backend_count", "safe_call",
    "get_available_backends", "get_device_id", "get_size_of",
    # random
    "RANDOM_ENGINE", "Random_Engine", "get_seed", "randn", "randu", "set_seed", "set_default_random_engine_type",
    "get_default_random_engine",
    # signal
    "approx1", "approx2", "convolve", "convolve1", "convolve2", "convolve3", "dft", "fft", "fft2", "fft2_c2r",
    "fft2_inplace", "fft2_r2c", "fft3", "fft3_c2r", "fft3_inplace", "fft3_r2c", "fft_c2r", "fft_convolve",
    "fft_convolve1", "fft_convolve2", "fft_convolve3", "fft_inplace", "fft_r2c", "fir", "idft", "ifft", "ifft2",
    "ifft2_inplace", "ifft3", "ifft3_inplace", "ifft_inplace", "iir", "medfilt", "medfilt1", "medfilt2",
    "convolve2_separable", "set_fft_plan_cache_size",
    # sparse
    "create_sparse_from_dense", "sparse_get_col_idx", "sparse_get_info", "sparse_get_nnz", "sparse_get_row_idx",
    "sparse_get_storage", "sparse_get_values", "create_sparse", "create_sparse_from_host", "convert_sparse_to_dense",
    "convert_sparse",
    # statistics
    "corrcoef", "mean", "median", "stdev", "topk", "var", "cov",
    # timer
    "timeit",
    # util
    "to_str", "dim4", "number_dtype", "implicit_dtype", "dim4_to_tuple"
]
