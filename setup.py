"""Build script for the C++ backend (lme4._lme4cpp).

Uses pybind11 + the vendored Eigen headers in third_party/eigen.  Any C++17
compiler works (MSVC, gcc, clang).  Without a compiler, see
``tools/build_ext.py --zig``.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

SRC = ["glmFamily.cpp", "predModule.cpp", "respModule.cpp", "optimizer.cpp", "bobyqa.cpp",
       "external.cpp", "bindings.cpp"]

ext = Pybind11Extension(
    "lme4._lme4cpp",
    [f"src/cpp/{f}" for f in SRC],
    include_dirs=["third_party/eigen"],
    define_macros=[("NDEBUG", None), ("_USE_MATH_DEFINES", None)],
    cxx_std=17,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
