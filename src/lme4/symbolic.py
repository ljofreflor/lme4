"""Symbolic formulas written as Python expressions.

Python has no binary ``~`` operator, so the response is separated from the
predictors with ``==``::

    from lme4 import V, lmer, I, log
    Reaction, Days, Subject = V("Reaction Days Subject")
    lmer(Reaction == Days + (Days | Subject), data=sleepstudy)

Operator correspondences with R formulas:

=========  ==========  ===================================================
Python     R           meaning
=========  ==========  ===================================================
``==``     ``~``       response == predictors
``+``      ``+``       add terms
``-``      ``-``       remove terms (``- 1`` removes the intercept)
``*``      ``*``       crossing ``a*b = a + b + a:b``
``&``      ``:``       interaction (Python has no binary ``:``)
``/``      ``/``       nesting
``**``     ``^``       ``(a + b + c)**2``
``|``      ``|``       random-effects term ``(Days | Subject)``
``.dbar``  ``||``      ``Days.dbar(Subject)`` == ``(Days || Subject)``
=========  ==========  ===================================================

Note that in Python ``&`` and ``|`` bind *less* tightly than ``+``, so write
``a + (b & c)`` for R's ``a + b:c`` (exactly like ``(1 + Days | Subject)``,
which already needs parentheses in R).  Functions such as ``log``, ``I``,
``factor``, ``poly``, ``offset`` and numpy ufuncs (``np.log(x)``) build
calls.  Every symbolic object converts to the same R language tree as the
corresponding string formula, so both interfaces share one engine.
"""
from __future__ import annotations

from .rlang import Call, Const, Formula, Sym, deparse, parse, as_formula

__all__ = ["V", "Term", "to_formula", "I", "log", "exp", "sqrt", "factor", "poly", "offset",
           "scale", "cbind", "us", "diag", "cs", "ar1"]


def _lift(x):
    if isinstance(x, Term):
        return x.expr
    if isinstance(x, bool):
        return Const(x)
    if isinstance(x, (int, float)):
        return Const(float(x))
    if isinstance(x, str):
        return Const(x)
    raise TypeError(f"cannot use {type(x).__name__} in a formula")


def _op(op, *args):
    return Term(Call(Sym(op), tuple(_lift(a) for a in args)))


class Term:
    """A symbolic formula expression (wraps an R language object)."""

    __array_priority__ = 1000     # make numpy defer to our operators

    def __init__(self, expr):
        self.expr = expr

    # arithmetic / formula operators
    def __add__(self, o): return _op("+", self, o)
    def __radd__(self, o): return _op("+", o, self)
    def __sub__(self, o): return _op("-", self, o)
    def __rsub__(self, o): return _op("-", o, self)
    def __mul__(self, o): return _op("*", self, o)
    def __rmul__(self, o): return _op("*", o, self)
    def __truediv__(self, o): return _op("/", self, o)
    def __rtruediv__(self, o): return _op("/", o, self)
    def __pow__(self, o): return _op("^", self, o)
    def __rpow__(self, o): return _op("^", o, self)
    def __xor__(self, o): return _op("^", self, o)
    def __and__(self, o): return _op(":", self, o)
    def __rand__(self, o): return _op(":", o, self)
    def __or__(self, o): return Term(Call(Sym("("), (_op("|", self, o).expr,)))
    def __ror__(self, o): return Term(Call(Sym("("), (_op("|", o, self).expr,)))
    def __neg__(self): return _op("-", self)
    def __pos__(self): return _op("+", self)
    def __invert__(self): return _op("!", self)

    def interaction(self, other):
        """R's ``a:b``"""
        return _op(":", self, other)

    def dbar(self, group):
        """R's ``(x || g)``: uncorrelated random effects"""
        return Term(Call(Sym("("), (_op("||", self, group).expr,)))

    def in_(self, other):
        """R's ``a %in% b``"""
        return _op("%in%", self, other)

    # comparison operators used *inside* I(...) produce R comparison calls;
    # ``==`` at top level produces a Formula (response == predictors).
    def __eq__(self, o):
        return SymFormula(Call(Sym("~"), (self.expr, _lift(o))))

    def __ne__(self, o): return _op("!=", self, o)
    def __lt__(self, o): return _op("<", self, o)
    def __le__(self, o): return _op("<=", self, o)
    def __gt__(self, o): return _op(">", self, o)
    def __ge__(self, o): return _op(">=", self, o)

    __hash__ = object.__hash__

    def __bool__(self):
        raise TypeError("a symbolic formula term has no truth value")

    # numpy ufuncs -> R calls
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        names = {"log": "log", "exp": "exp", "sqrt": "sqrt", "absolute": "abs", "sin": "sin",
                 "cos": "cos", "tan": "tan", "log10": "log10", "log2": "log2",
                 "log1p": "log1p", "expm1": "expm1", "floor": "floor", "ceil": "ceiling"}
        if method != "__call__" or ufunc.__name__ not in names:
            return NotImplemented
        return Term(Call(Sym(names[ufunc.__name__]), tuple(_lift(i) for i in inputs)))

    def __repr__(self):
        return deparse(self.expr)

    def __str__(self):
        return deparse(self.expr)


class SymFormula(Formula):
    """Formula produced by ``response == predictors``"""

    def __bool__(self):
        raise TypeError("use a symbolic formula only as a model formula")


def V(names):
    """Create symbolic variables: ``y, x, g = V("y x g")`` (or a list of names,
    or a pandas DataFrame, whose column names are used)."""
    import pandas as pd
    if isinstance(names, pd.DataFrame):
        names = [str(c) for c in names.columns]
    elif isinstance(names, str):
        names = names.replace(",", " ").split()
    out = [Term(Sym(n)) for n in names]
    return out[0] if len(out) == 1 else out


def _fn(name):
    def f(*args, **kw):
        a = tuple(_lift(x) for x in args) + tuple(_lift(v) for v in kw.values())
        nms = (None,) * len(args) + tuple(kw)
        return Term(Call(Sym(name), a, nms))
    f.__name__ = name
    f.__doc__ = f"symbolic R function ``{name}()``"
    return f


I = _fn("I")
log = _fn("log")
exp = _fn("exp")
sqrt = _fn("sqrt")
factor = _fn("factor")
poly = _fn("poly")
offset = _fn("offset")
scale = _fn("scale")
cbind = _fn("cbind")
us = _fn("us")
diag = _fn("diag")
cs = _fn("cs")
ar1 = _fn("ar1")


def to_formula(f, env=None) -> Formula:
    """accept a string, a Formula, a SymFormula, or a one-sided Term"""
    if isinstance(f, Formula):
        if not f.env and env is not None:
            return Formula(f.expr, env)
        return f
    if isinstance(f, Term):
        return Formula(Call(Sym("~"), (f.expr,)), env)
    return as_formula(f, env)
