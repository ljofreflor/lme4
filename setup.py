"""Build script for the C++ backend (lme4._lme4cpp).

Uses pybind11 + the vendored Eigen headers in third_party/eigen.  Any C++17
compiler works (MSVC, gcc, clang).  If no working compiler is found, the
extension is built with the Zig toolchain from the ``ziglang`` package
(declared as a build requirement on Windows in pyproject.toml), so that
``pip install`` / ``poetry add`` work on machines without Visual Studio.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
SRC = ["glmFamily.cpp", "predModule.cpp", "respModule.cpp", "optimizer.cpp", "bobyqa.cpp",
       "external.cpp", "bindings.cpp"]

ext = Pybind11Extension(
    "lme4._lme4cpp",
    [f"src/cpp/{f}" for f in SRC],
    include_dirs=["third_party/eigen"],
    define_macros=[("NDEBUG", None), ("_USE_MATH_DEFINES", None)],
    cxx_std=17,
)


def zig_compile(output: Path):
    """Compile the extension with ``python -m ziglang c++``."""
    import pybind11
    cmd = [sys.executable, "-m", "ziglang", "c++", "-std=c++17", "-shared", "-O2",
           "-DNDEBUG", "-D_USE_MATH_DEFINES", "-fvisibility=hidden",
           "-Wno-macro-redefined", "-Wno-deprecated-declarations",
           f"-I{sysconfig.get_paths()['include']}", f"-I{pybind11.get_include()}",
           f"-I{ROOT / 'third_party' / 'eigen'}",
           *[str(ROOT / "src" / "cpp" / f) for f in SRC]]
    if sys.platform == "win32":
        cmd[4:4] = ["-target", "x86_64-windows-gnu"]
        cmd += [f"-L{Path(sys.base_prefix) / 'libs'}",
                f"-lpython{sys.version_info.major}{sys.version_info.minor}"]
    elif sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]
    else:
        cmd += ["-fPIC"]
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-o", str(output)]
    print("building lme4._lme4cpp with zig:\n  " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    for suffix in (".lib", ".pdb"):          # zig leaves import libs next to the .pyd
        extra = output.with_suffix(suffix)
        if extra.exists():
            extra.unlink()


class build_ext_with_fallback(build_ext):
    """Try the platform compiler first; fall back to zig if it is missing."""

    def run(self):
        if os.environ.get("LME4_USE_ZIG") == "1":
            return self._zig_all()
        try:
            super().run()
        except Exception as e:  # missing MSVC raises DistutilsPlatformError, etc.
            print(f"C++ compiler build failed ({e}); trying the zig toolchain", flush=True)
            self._zig_all()

    def _zig_all(self):
        try:
            import ziglang  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "No C++17 compiler was found and the 'ziglang' package is not installed. "
                "Install a compiler (e.g. Visual Studio Build Tools) or `pip install ziglang`.")
        for e in self.extensions:
            zig_compile(Path(self.get_ext_fullpath(e.name)))


setup(ext_modules=[ext], cmdclass={"build_ext": build_ext_with_fallback})
