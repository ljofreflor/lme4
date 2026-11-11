"""Loads the compiled C++ backend (``lme4._lme4cpp``)."""
try:
    from . import _lme4cpp as cpp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The C++ backend of lme4 (lme4._lme4cpp) is not built. Build it with\n"
        "    python tools/build_ext.py          (needs a C++17 compiler)\n"
        "or  python tools/build_ext.py --zig    (uses the 'ziglang' pip package)"
    ) from e

__all__ = ["cpp"]
