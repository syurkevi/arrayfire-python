#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import ctypes as ct
import platform

try:
    from enum import Enum as _Enum

    def _Enum_Type(v):
        return v
except ImportError:
    class _MetaEnum(type):
        def __init__(cls, name, attrs):
            for attrname, attrvalue in attrs.iteritems():
                if name != '_Enum' and isinstance(attrvalue, _Enum_Type):
                    attrvalue.__class__ = cls
                    attrs[attrname] = attrvalue

    class _Enum(object):
        __metaclass__ = _MetaEnum

    class _Enum_Type(object):
        def __init__(self, v):
            self.value = v

c_float_t = ct.c_float
c_double_t = ct.c_double
c_int_t = ct.c_int
c_uint_t = ct.c_uint
c_longlong_t = ct.c_longlong
c_ulonglong_t = ct.c_ulonglong
c_char_t = ct.c_char
c_bool_t = ct.c_bool
c_uchar_t = ct.c_ubyte
c_short_t = ct.c_short
c_ushort_t = ct.c_ushort
c_pointer = ct.pointer
c_void_ptr_t = ct.c_void_p
c_char_ptr_t = ct.c_char_p
c_size_t = ct.c_size_t

# Work around for unexpected architectures
# FIXME: c_dim_t_forced is undefined. Set source or remove as deprecated
if 'c_dim_t_forced' in globals():
    global c_dim_t_forced
    c_dim_t = c_dim_t_forced
else:
    # dim_t is long long by default
    c_dim_t = c_longlong_t
    # Change to int for 32 bit x86 and amr architectures
    if (platform.architecture()[0][0:2] == '32' and
            (platform.machine()[-2:] == '86' or platform.machine()[0:3] == 'arm')):
        c_dim_t = c_int_t


class ERR(_Enum):
    """
    Error values. For internal use only.
    """
    NONE = _Enum_Type(0)

    # 100-199 Errors in environment
    NO_MEM = _Enum_Type(101)
    DRIVER = _Enum_Type(102)
    RUNTIME = _Enum_Type(103)

    # 200-299 Errors in input parameters
    INVALID_ARRAY = _Enum_Type(201)
    ARG = _Enum_Type(202)
    SIZE = _Enum_Type(203)
    TYPE = _Enum_Type(204)
    DIFF_TYPE = _Enum_Type(205)
    BATCH = _Enum_Type(207)
    DEVICE = _Enum_Type(208)

    # 300-399 Errors for missing software features
    NOT_SUPPORTED = _Enum_Type(301)
    NOT_CONFIGURED = _Enum_Type(302)
    NONFREE = _Enum_Type(303)

    # 400-499 Errors for missing hardware features
    NO_DBL = _Enum_Type(401)
    NO_GFX = _Enum_Type(402)

    # 500-599 Errors specific to the heterogeneous API
    LOAD_LIB = _Enum_Type(501)
    LOAD_SYM = _Enum_Type(502)
    ARR_BKND_MISMATCH = _Enum_Type(503)

    # 900-999 Errors from upstream libraries and runtimes
    INTERNAL = _Enum_Type(998)
    UNKNOWN = _Enum_Type(999)


class Dtype(_Enum):
    """
    Error values. For internal use only.
    """
    f32 = _Enum_Type(0)
    c32 = _Enum_Type(1)
    f64 = _Enum_Type(2)
    c64 = _Enum_Type(3)
    b8 = _Enum_Type(4)
    s32 = _Enum_Type(5)
    u32 = _Enum_Type(6)
    u8 = _Enum_Type(7)
    s64 = _Enum_Type(8)
    u64 = _Enum_Type(9)
    s16 = _Enum_Type(10)
    u16 = _Enum_Type(11)


class Source(_Enum):
    """
    Source of the pointer
    """
    device = _Enum_Type(0)
    host = _Enum_Type(1)


class INTERP(_Enum):
    """
    Interpolation method
    """
    NEAREST = _Enum_Type(0)
    LINEAR = _Enum_Type(1)
    BILINEAR = _Enum_Type(2)
    CUBIC = _Enum_Type(3)
    LOWER = _Enum_Type(4)
    LINEAR_COSINE = _Enum_Type(5)
    BILINEAR_COSINE = _Enum_Type(6)
    BICUBIC = _Enum_Type(7)
    CUBIC_SPLINE = _Enum_Type(8)
    BICUBIC_SPLINE = _Enum_Type(9)


class PAD(_Enum):
    """
    Edge padding types
    """
    ZERO = _Enum_Type(0)
    SYM = _Enum_Type(1)


class CONNECTIVITY(_Enum):
    """
    Neighborhood connectivity
    """
    FOUR = _Enum_Type(4)
    EIGHT = _Enum_Type(8)


class CONV_MODE(_Enum):
    """
    Convolution mode
    """
    DEFAULT = _Enum_Type(0)
    EXPAND = _Enum_Type(1)


class CONV_DOMAIN(_Enum):
    """
    Convolution domain
    """
    AUTO = _Enum_Type(0)
    SPATIAL = _Enum_Type(1)
    FREQ = _Enum_Type(2)


class MATCH(_Enum):
    """
    Match type
    """

    """
    Sum of absolute differences
    """
    SAD = _Enum_Type(0)

    """
    Zero mean SAD
    """
    ZSAD = _Enum_Type(1)

    """
    Locally scaled SAD
    """
    LSAD = _Enum_Type(2)

    """
    Sum of squared differences
    """
    SSD = _Enum_Type(3)

    """
    Zero mean SSD
    """
    ZSSD = _Enum_Type(4)

    """
    Locally scaled SSD
    """
    LSSD = _Enum_Type(5)

    """
    Normalized cross correlation
    """
    NCC = _Enum_Type(6)

    """
    Zero mean NCC
    """
    ZNCC = _Enum_Type(7)

    """
    Sum of hamming distances
    """
    SHD = _Enum_Type(8)


class YCC_STD(_Enum):
    """
    YCC Standard formats
    """
    BT_601 = _Enum_Type(601)
    BT_709 = _Enum_Type(709)
    BT_2020 = _Enum_Type(2020)


class CSPACE(_Enum):
    """
    Colorspace formats
    """
    GRAY = _Enum_Type(0)
    RGB = _Enum_Type(1)
    HSV = _Enum_Type(2)
    YCbCr = _Enum_Type(3)


class MATPROP(_Enum):
    """
    Matrix properties
    """

    """
    None, general.
    """
    NONE = _Enum_Type(0)

    """
    Transposed.
    """
    TRANS = _Enum_Type(1)

    """
    Conjugate transposed.
    """
    CTRANS = _Enum_Type(2)

    """
    Upper triangular matrix.
    """
    UPPER = _Enum_Type(32)

    """
    Lower triangular matrix.
    """
    LOWER = _Enum_Type(64)

    """
    Treat diagonal as units.
    """
    DIAG_UNIT = _Enum_Type(128)

    """
    Symmetric matrix.
    """
    SYM = _Enum_Type(512)

    """
    Positive definite matrix.
    """
    POSDEF = _Enum_Type(1024)

    """
    Orthogonal matrix.
    """
    ORTHOG = _Enum_Type(2048)

    """
    Tri diagonal matrix.
    """
    TRI_DIAG = _Enum_Type(4096)

    """
    Block diagonal matrix.
    """
    BLOCK_DIAG = _Enum_Type(8192)


class NORM(_Enum):
    """
    Norm types
    """
    VECTOR_1 = _Enum_Type(0)
    VECTOR_INF = _Enum_Type(1)
    VECTOR_2 = _Enum_Type(2)
    VECTOR_P = _Enum_Type(3)
    MATRIX_1 = _Enum_Type(4)
    MATRIX_INF = _Enum_Type(5)
    MATRIX_2 = _Enum_Type(6)
    MATRIX_L_PQ = _Enum_Type(7)
    EUCLID = VECTOR_2


class COLORMAP(_Enum):
    """
    Colormaps
    """
    DEFAULT = _Enum_Type(0)
    SPECTRUM = _Enum_Type(1)
    COLORS = _Enum_Type(2)
    RED = _Enum_Type(3)
    MOOD = _Enum_Type(4)
    HEAT = _Enum_Type(5)
    BLUE = _Enum_Type(6)


class IMAGE_FORMAT(_Enum):
    """
    Image Formats
    """
    BMP = _Enum_Type(0)
    ICO = _Enum_Type(1)
    JPEG = _Enum_Type(2)
    JNG = _Enum_Type(3)
    PNG = _Enum_Type(13)
    PPM = _Enum_Type(14)
    PPMRAW = _Enum_Type(15)
    TIFF = _Enum_Type(18)
    PSD = _Enum_Type(20)
    HDR = _Enum_Type(26)
    EXR = _Enum_Type(29)
    JP2 = _Enum_Type(31)
    RAW = _Enum_Type(34)


class HOMOGRAPHY(_Enum):
    """
    Homography Types
    """
    RANSAC = _Enum_Type(0)
    LMEDS = _Enum_Type(1)


class BACKEND(_Enum):
    """
    Backend libraries
    """
    DEFAULT = _Enum_Type(0)
    CPU = _Enum_Type(1)
    CUDA = _Enum_Type(2)
    OPENCL = _Enum_Type(4)


class MARKER(_Enum):
    """
    Markers used for different points in graphics plots
    """
    NONE = _Enum_Type(0)
    POINT = _Enum_Type(1)
    CIRCLE = _Enum_Type(2)
    SQUARE = _Enum_Type(3)
    TRIANGE = _Enum_Type(4)
    CROSS = _Enum_Type(5)
    PLUS = _Enum_Type(6)
    STAR = _Enum_Type(7)


class MOMENT(_Enum):
    """
    Image Moments types
    """
    M00 = _Enum_Type(1)
    M01 = _Enum_Type(2)
    M10 = _Enum_Type(4)
    M11 = _Enum_Type(8)
    FIRST_ORDER = _Enum_Type(15)


class BINARYOP(_Enum):
    """
    Binary Operators
    """
    ADD = _Enum_Type(0)
    MUL = _Enum_Type(1)
    MIN = _Enum_Type(2)
    MAX = _Enum_Type(3)


class RANDOM_ENGINE(_Enum):
    """
    Random engine types
    """
    PHILOX_4X32_10 = _Enum_Type(100)
    THREEFRY_2X32_16 = _Enum_Type(200)
    MERSENNE_GP11213 = _Enum_Type(300)
    PHILOX = PHILOX_4X32_10
    THREEFRY = THREEFRY_2X32_16
    DEFAULT = PHILOX


class STORAGE(_Enum):
    """
    Matrix Storage types
    """
    DENSE = _Enum_Type(0)
    CSR = _Enum_Type(1)
    CSC = _Enum_Type(2)
    COO = _Enum_Type(3)


class CANNY_THRESHOLD(_Enum):
    """
    Canny Edge Threshold types
    """
    MANUAL = _Enum_Type(0)
    AUTO_OTSU = _Enum_Type(1)


class FLUX(_Enum):
    """
    Flux functions
    """
    DEFAULT = _Enum_Type(0)
    QUADRATIC = _Enum_Type(1)
    EXPONENTIAL = _Enum_Type(2)


class DIFFUSION(_Enum):
    """
    Diffusion equations
    """
    DEFAULT = _Enum_Type(0)
    GRAD = _Enum_Type(1)
    MCDE = _Enum_Type(2)


class TOPK(_Enum):
    """
    Top-K ordering
    """
    DEFAULT = _Enum_Type(0)
    MIN = _Enum_Type(1)
    MAX = _Enum_Type(2)
