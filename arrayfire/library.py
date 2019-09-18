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
Module containing enums and other constants.
"""

import ctypes as ct
import os
import traceback

from .defaults import BACKEND, ERR, Dtype, c_char_ptr_t, c_dim_t, c_int_t, c_pointer, c_size_t, c_void_ptr_t
from .util import to_str

AF_VER_MAJOR = "3"
FORGE_VER_MAJOR = "1"
_VER_MAJOR_PLACEHOLDER = "__VER_MAJOR__"


def _setup():
    import platform

    platform_name = platform.system()

    try:
        AF_PATH = os.environ["AF_PATH"]
    except KeyError:
        AF_PATH = None

    AF_SEARCH_PATH = AF_PATH

    try:
        CUDA_PATH = os.environ["CUDA_PATH"]
    except KeyError:
        CUDA_PATH = None

    CUDA_FOUND = False

    assert len(platform_name) >= 3
    if platform_name == "Windows" or platform_name[:3] == "CYG":
        # Windows specific setup
        pre = ""
        post = ".dll"
        if platform_name == "Windows":
            # Supressing crashes caused by missing dlls
            # http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            # https://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
            ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH = "C:/Program Files/ArrayFire/v" + AF_VER_MAJOR + "/"

        if CUDA_PATH is not None:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/bin") and os.path.isdir(CUDA_PATH + "/nvvm/bin/")

    elif platform_name == "Darwin":
        # OSX specific setup
        pre = "lib"
        post = "." + _VER_MAJOR_PLACEHOLDER + ".dylib"

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH = "/usr/local/"

        if CUDA_PATH is None:
            CUDA_PATH = "/usr/local/cuda/"

        CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib") and os.path.isdir(CUDA_PATH + "/nvvm/lib")

    elif platform_name == "Linux":
        pre = "lib"
        post = ".so." + _VER_MAJOR_PLACEHOLDER

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH = "/opt/arrayfire-" + AF_VER_MAJOR + "/"

        if CUDA_PATH is None:
            CUDA_PATH = "/usr/local/cuda/"

        if platform.architecture()[0][:2] == "64":
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib64") and os.path.isdir(CUDA_PATH + "/nvvm/lib64")
        else:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib") and os.path.isdir(CUDA_PATH + "/nvvm/lib")
    else:
        raise OSError(platform_name + " not supported")

    if AF_PATH is None:
        os.environ["AF_PATH"] = AF_SEARCH_PATH

    return pre, post, AF_SEARCH_PATH, CUDA_FOUND


class _clibrary(object):

    def __libname(self, name, head="af", ver_major=AF_VER_MAJOR):
        post = self.__post.replace(_VER_MAJOR_PLACEHOLDER, ver_major)
        libname = self.__pre + head + name + post
        libname_full = self.AF_PATH + "/lib/" + libname
        return (libname, libname_full)

    def set_unsafe(self, name):
        lib = self.__clibs[name]
        if lib is None:
            raise RuntimeError("Backend not found")
        self.__name = name

    def __init__(self):
        more_info_str = "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."

        pre, post, AF_PATH, CUDA_FOUND = _setup()

        self.__pre = pre
        self.__post = post
        self.AF_PATH = AF_PATH
        self.CUDA_FOUND = CUDA_FOUND

        self.__name = None

        self.__clibs = {
            "cuda": None,
            "opencl": None,
            "cpu": None,
            "unified": None}

        self.__backend_map = {
            0: "unified",
            1: "cpu",
            2: "cuda",
            4: "opencl"}

        self.__backend_name_map = {
            "default": 0,
            "unified": 0,
            "cpu": 1,
            "cuda": 2,
            "opencl": 4}

        # Try to pre-load forge library if it exists
        libnames = self.__libname("forge", head="", ver_major=FORGE_VER_MAJOR)

        try:
            VERBOSE_LOADS = os.environ["AF_VERBOSE_LOADS"] == "1"
        except KeyError:
            VERBOSE_LOADS = False

        for libname in libnames:
            try:
                ct.cdll.LoadLibrary(libname)
                if VERBOSE_LOADS:
                    print("Loaded " + libname)
                break
            except OSError:
                if VERBOSE_LOADS:
                    traceback.print_exc()
                    print("Unable to load " + libname)

        c_dim4 = c_dim_t*4
        out = c_void_ptr_t(0)
        dims = c_dim4(10, 10, 1, 1)

        # Iterate in reverse order of preference
        for name in {"cpu", "opencl", "cuda", ""}:
            libnames = self.__libname(name)
            for libname in libnames:
                try:
                    ct.cdll.LoadLibrary(libname)
                    __name = "unified" if name == "" else name
                    clib = ct.CDLL(libname)
                    self.__clibs[__name] = clib
                    err = clib.af_randu(c_pointer(out), 4, c_pointer(dims), Dtype.f32.value)
                    if err != ERR.NONE.value:
                        return
                    self.__name = __name
                    clib.af_release_array(out)
                    if VERBOSE_LOADS:
                        print("Loaded " + libname)
                    break
                except OSError:
                    if VERBOSE_LOADS:
                        traceback.print_exc()
                        print("Unable to load " + libname)

        if self.__name is None:
            raise RuntimeError("Could not load any ArrayFire libraries.\n" + more_info_str)

    def get_id(self, name):
        return self.__backend_name_map[name]

    def get_name(self, bk_id):
        return self.__backend_map[bk_id]

    def get(self):
        return self.__clibs[self.__name]

    def name(self):
        return self.__name

    def is_unified(self):
        return self.__name == "unified"

    def parse(self, res):
        lst = []
        for key, value in self.__backend_name_map.items():
            if value & res:
                lst.append(key)
        return tuple(lst)


backend = _clibrary()


def set_backend(name, unsafe=False):
    """
    Set a specific backend by name

    Parameters
    ----------

    name : str.

    unsafe : optional: bool. Default: False.
           If False, does not switch backend if current backend is not unified backend.
    """
    if not (backend.is_unified() or unsafe):
        raise RuntimeError("Can not change backend to %s after loading %s" % (name, backend.name()))

    if backend.is_unified():
        safe_call(backend.get().af_set_backend(backend.get_id(name)))

    backend.set_unsafe(name)


def get_backend():
    """
    Return the name of the backend
    """
    return backend.name()


def get_backend_id(A):
    """
    Get backend name of an array

    Parameters
    ----------
    A    : af.Array

    Returns
    ----------

    name : str.
         Backend name
    """
    backend_id = c_int_t(BACKEND.CPU.value)
    safe_call(backend.get().af_get_backend_id(c_pointer(backend_id), A.arr))
    return backend.get_name(backend_id.value)


def get_backend_count():
    """
    Get number of available backends

    Returns
    ----------

    count : int
          Number of available backends
    """
    count = c_int_t(0)
    safe_call(backend.get().af_get_backend_count(c_pointer(count)))
    return count.value


def get_available_backends():
    """
    Get names of available backends

    Returns
    ----------

    names : tuple of strings
          Names of available backends
    """
    available = c_int_t(0)
    safe_call(backend.get().af_get_available_backends(c_pointer(available)))
    return backend.parse(int(available.value))


def get_active_backend():
    """
    Get the current active backend

    name : str.
         Backend name
    """
    backend_id = c_int_t(BACKEND.CPU.value)
    safe_call(backend.get().af_get_active_backend(c_pointer(backend_id)))
    return backend.get_name(backend_id.value)


def get_device_id(A):
    """
    Get the device id of the array

    Parameters
    ----------
    A    : af.Array

    Returns
    ----------

    dev : Integer
         id of the device array was created on
    """
    device_id = c_int_t(0)
    safe_call(backend.get().af_get_device_id(c_pointer(device_id), A.arr))
    return device_id


def get_size_of(dtype):
    """
    Get the size of the type represented by arrayfire.Dtype
    """
    size = c_size_t(0)
    safe_call(backend.get().af_get_size_of(c_pointer(size), dtype.value))
    return size.value


def safe_call(af_error):
    if af_error == ERR.NONE.value:
        return
    err_str = c_char_ptr_t(0)
    err_len = c_dim_t(0)
    backend.get().af_get_last_error(c_pointer(err_str), c_pointer(err_len))
    raise RuntimeError(to_str(err_str))
