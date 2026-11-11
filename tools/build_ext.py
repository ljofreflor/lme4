"""Build the C++ backend (lme4._lme4cpp) in place.

Uses the regular setuptools/pybind11 machinery when a C++ compiler is
available (MSVC, gcc, clang).  On machines without one, ``--zig`` compiles
with the Zig toolchain shipped by the ``ziglang`` pip package.

    python tools/build_ext.py          # setuptools (python setup.py build_ext --inplace)
    python tools/build_ext.py --zig    # zig c++
"""
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ["glmFamily.cpp", "predModule.cpp", "respModule.cpp", "optimizer.cpp",
       "external.cpp", "bobyqa.cpp", "bindings.cpp"]


def build_zig(debug=False):
    import pybind11
    cppdir = ROOT / "src" / "cpp"
    out = ROOT / "src" / "lme4" / ("_lme4cpp" + sysconfig.get_config_var("EXT_SUFFIX"))
    pyinc = sysconfig.get_paths()["include"]
    cmd = [sys.executable, "-m", "ziglang", "c++", "-std=c++17", "-shared",
           "-O1" if debug else "-O2", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-fvisibility=hidden",
           "-Wno-macro-redefined", "-Wno-deprecated-declarations",
           f"-I{pyinc}", f"-I{pybind11.get_include()}", f"-I{ROOT / 'third_party' / 'eigen'}",
           *[str(cppdir / f) for f in SRC]]
    if sys.platform == "win32":
        cmd[4:4] = ["-target", "x86_64-windows-gnu"]
        cmd += [f"-L{Path(sys.base_prefix) / 'libs'}",
                f"-lpython{sys.version_info.major}{sys.version_info.minor}"]
    cmd += ["-o", str(out)]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    # zig on windows also leaves .lib/.pdb next to the extension
    for ext in (".lib", ".pdb"):
        p = out.with_suffix(ext)
        if p.exists():
            p.unlink()
    print("built", out)


if __name__ == "__main__":
    if "--zig" in sys.argv:
        build_zig(debug="--debug" in sys.argv)
    else:
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ROOT)
