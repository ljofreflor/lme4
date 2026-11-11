"""R's ``terms()``, ``model.frame()`` and ``model.matrix()`` for pandas data.

This is a re-implementation of the parts of R's stats package (model.c,
models.R, contrast.R) that lme4 relies upon, producing the same term
ordering, the same contrast coding decisions and the same column names as R.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .rlang import Call, Const, Formula, Sym, NULL, deparse, is_call, parse, mkcall

__all__ = ["Factor", "Terms", "terms", "ModelFrame", "model_frame", "model_matrix",
           "contr_treatment", "contr_sum", "contr_helmert", "contr_poly", "poly",
           "eval_expr"]


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

class Factor:
    """An R factor: integer codes (-1 for NA) and character levels."""

    __slots__ = ("codes", "levels", "ordered", "contrasts")

    def __init__(self, codes, levels, ordered=False, contrasts=None):
        self.codes = np.asarray(codes, dtype=np.int64)
        self.levels = [str(x) for x in levels]
        self.ordered = bool(ordered)
        self.contrasts = contrasts

    def __len__(self):
        return len(self.codes)

    @property
    def nlevels(self):
        return len(self.levels)

    def isna(self):
        return self.codes < 0

    def subset(self, idx):
        return Factor(self.codes[idx], self.levels, self.ordered, self.contrasts)

    def droplevels(self):
        used = np.unique(self.codes[self.codes >= 0])
        if len(used) == len(self.levels):
            return self
        remap = -np.ones(len(self.levels), dtype=np.int64)
        remap[used] = np.arange(len(used))
        codes = np.where(self.codes >= 0, remap[np.maximum(self.codes, 0)], -1)
        return Factor(codes, [self.levels[i] for i in used], self.ordered, self.contrasts)

    def as_strings(self):
        lev = np.array(self.levels + [None], dtype=object)
        return lev[np.where(self.codes >= 0, self.codes, len(self.levels))]

    def to_pandas(self):
        return pd.Categorical.from_codes(self.codes, categories=self.levels, ordered=self.ordered)

    def __repr__(self):
        return f"Factor({list(self.as_strings()[:6])}{'...' if len(self) > 6 else ''}, levels={self.levels})"


def _fmt_level(v):
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return repr(float("%.15g" % v))
    return str(v)


def as_factor(x, levels=None, ordered=None, exclude_na=True) -> Factor:
    """R's factor()/as.factor()."""
    if isinstance(x, Factor):
        if levels is None:
            return x if ordered is None else Factor(x.codes, x.levels, ordered, x.contrasts)
        x = x.as_strings()
    if isinstance(x, pd.Series):
        x = x.array
    if isinstance(x, (pd.Categorical, pd.arrays.Categorical if hasattr(pd.arrays, "Categorical") else pd.Categorical)):
        cats = [_fmt_level(c) for c in x.categories]
        if levels is None:
            return Factor(x.codes, cats, x.ordered if ordered is None else ordered)
        x = np.array([None if c < 0 else cats[c] for c in x.codes], dtype=object)
    arr = np.asarray(x, dtype=object) if not isinstance(x, np.ndarray) else x
    if arr.ndim != 1:
        arr = arr.ravel()
    na = pd.isna(arr) if arr.dtype == object else (np.isnan(arr) if arr.dtype.kind == "f" else np.zeros(len(arr), bool))
    if levels is None:
        vals = arr[~na]
        if arr.dtype.kind in "biuf":
            uniq = np.unique(vals.astype(float) if arr.dtype.kind == "f" else vals)
            levels_v = list(uniq)
        elif arr.dtype.kind == "b":
            levels_v = sorted(set(vals.tolist()))
        else:
            try:
                levels_v = sorted(set(vals.tolist()))
            except TypeError:
                levels_v = sorted(set(map(str, vals.tolist())))
        lev_str = [_fmt_level(v) for v in levels_v]
    else:
        lev_str = [_fmt_level(v) for v in levels]
    key = {s: i for i, s in enumerate(lev_str)}
    codes = np.array([-1 if n else key.get(_fmt_level(v), -1) for v, n in zip(arr.tolist(), na)],
                     dtype=np.int64)
    return Factor(codes, lev_str, bool(ordered))


def interaction_factor(f1: Factor, f2: Factor, fix_order=True) -> Factor:
    """reformulas' ``%i%``: interaction of two factors with observed levels only"""
    s1, s2 = f1.as_strings(), f2.as_strings()
    f12 = np.array([f"{a}:{b}" for a, b in zip(s1, s2)], dtype=object)
    _, first = np.unique(f12, return_index=True)
    u = np.sort(first)
    if not fix_order:
        levs = list(f12[u])
    else:
        rank = len(f2.levels) * (f1.codes[u] + 1) + (f2.codes[u] + 1)
        levs = list(f12[u][np.argsort(rank, kind="stable")])
    return as_factor(f12, levels=levs)


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------

def contr_treatment(levels, base=1):
    n = len(levels)
    m = np.eye(n)
    keep = [i for i in range(n) if i != base - 1]
    return m[:, keep], [levels[i] for i in keep]


def contr_sas(levels):
    return contr_treatment(levels, base=len(levels))


def contr_sum(levels):
    n = len(levels)
    m = np.vstack([np.eye(n - 1), -np.ones((1, n - 1))])
    return m, None


def contr_helmert(levels):
    n = len(levels)
    m = np.zeros((n, n - 1))
    for j in range(1, n):
        m[:j, j - 1] = -1
        m[j, j - 1] = j
    return m, None


def contr_poly(levels, scores=None):
    n = len(levels)
    if scores is None:
        scores = np.arange(1, n + 1, dtype=float)
    x = np.asarray(scores, float)
    xc = x - x.mean()
    X = np.vander(xc, n, increasing=True)
    q, r = np.linalg.qr(X)
    z = q * np.diag(r)
    Z = z / np.sqrt((z ** 2).sum(axis=0))
    Z = Z[:, 1:]
    names = [".L", ".Q", ".C"] + [f"^{i}" for i in range(4, n)]
    return Z, names[: n - 1]


_CONTRASTS = {"contr.treatment": contr_treatment, "contr.sum": contr_sum,
              "contr.helmert": contr_helmert, "contr.poly": contr_poly,
              "contr.SAS": contr_sas}


def _contrast_matrix(f: Factor, spec=None):
    """returns (matrix nlev x ncol, column suffixes)"""
    spec = spec if spec is not None else f.contrasts
    if spec is None:
        spec = "contr.poly" if f.ordered else "contr.treatment"
    if isinstance(spec, str):
        fun = _CONTRASTS.get(spec)
        if fun is None:
            raise ValueError(f"unknown contrasts {spec!r}")
        m, names = fun(f.levels)
    elif callable(spec):
        res = spec(f.levels)
        m, names = res if isinstance(res, tuple) else (np.asarray(res), None)
    else:
        m = np.asarray(spec, float)
        names = list(spec.columns) if isinstance(spec, pd.DataFrame) else None
    if names is None:
        names = [str(i + 1) for i in range(m.shape[1])]
    return np.asarray(m, float), list(names)


# ---------------------------------------------------------------------------
# Orthogonal polynomials (R's poly) and scale
# ---------------------------------------------------------------------------

def poly(x, degree=1, coefs=None, raw=False):
    """R's ``poly(x, degree)``; returns (matrix, coefs) so predictions can reuse them"""
    x = np.asarray(x, float)
    if raw:
        Z = np.column_stack([x ** d for d in range(1, degree + 1)])
        return Z, None
    if coefs is None:
        xbar = x.mean()
        xc = x - xbar
        X = np.vander(xc, degree + 1, increasing=True)
        q, r = np.linalg.qr(X)
        z = q * np.diag(r)
        norm2 = (z ** 2).sum(axis=0)
        alpha = ((xc[:, None] * z ** 2).sum(axis=0) / norm2 + xbar)[:degree]
        norm2 = np.concatenate([[1.0], norm2])
        Z = z / np.sqrt(norm2[1:])
        coefs = {"alpha": alpha, "norm2": norm2}
        return Z[:, 1:], coefs
    alpha, norm2 = coefs["alpha"], coefs["norm2"]
    n = len(x)
    Z = np.ones((n, degree + 1))
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    for i in range(1, degree):
        Z[:, i + 1] = (x - alpha[i]) * Z[:, i] - (norm2[i + 1] / norm2[i]) * Z[:, i - 1]
    Z = Z / np.sqrt(norm2[1:])
    return Z[:, 1:], coefs


# ---------------------------------------------------------------------------
# Evaluation of R expressions against a data frame
# ---------------------------------------------------------------------------

class _Matrix(np.ndarray):
    """2-d numeric variable carrying column names (e.g. poly(), cbind())"""

    def __new__(cls, arr, colnames=None):
        obj = np.asarray(arr, float).view(cls)
        obj.colnames = colnames
        return obj

    def __array_finalize__(self, obj):
        self.colnames = getattr(obj, "colnames", None)


def _as_num(x):
    if isinstance(x, Factor):
        return (x.codes + 1).astype(float)
    a = np.asarray(x)
    if a.dtype == bool:
        return a.astype(float)
    return a.astype(float) if a.dtype.kind in "iu" else a


def _vec(x, n):
    a = np.asarray(x)
    if a.ndim == 0:
        return np.full(n, a.item())
    return a


class EvalContext:
    def __init__(self, data: Dict[str, Any], env=None, n=None, predvars=None):
        self.data = data
        self.env = env or {}
        self.n = n
        self.predvars = predvars   # dict: deparsed var -> replacement Call (makepredictcall)

    def lookup(self, name):
        if name in self.data:
            return self.data[name]
        if name in self.env:
            return self.env[name]
        if name == "pi":
            return math.pi
        if name == "T":
            return True
        if name == "F":
            return False
        raise NameError(f"object '{name}' not found")


def _num_args(ctx, e):
    return [eval_expr(a, ctx) for a in e.args]


_UFUNCS = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin,
           "cos": np.cos, "tan": np.tan, "log10": np.log10, "log2": np.log2,
           "log1p": np.log1p, "expm1": np.expm1, "floor": np.floor, "ceiling": np.ceil,
           "sign": np.sign, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
           "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "trunc": np.trunc}


def eval_expr(e, ctx: EvalContext, record=None):
    """Evaluate an R expression.  ``record`` collects predvars substitutions
    (R's makepredictcall) for data-dependent transformations."""
    if isinstance(e, Const):
        return e.value
    if e is NULL:
        return None
    if isinstance(e, Sym):
        return ctx.lookup(e.name)
    if not isinstance(e, Call):
        return e
    fn = e.fn.name if isinstance(e.fn, Sym) else None
    if fn == "::":
        return eval_expr(Call(e.args[1], ()), ctx) if False else ctx.lookup(e.args[1].name)
    if isinstance(e.fn, Call) and is_call(e.fn, "::"):
        return eval_expr(Call(e.fn.args[1], e.args, e.names), ctx, record)
    if fn == "(":
        return eval_expr(e.args[0], ctx, record)
    if fn in ("I", "{"):
        return eval_expr(e.args[0], ctx, record)
    if fn in ("+", "-") and len(e.args) == 1:
        v = _as_num(eval_expr(e.args[0], ctx))
        return v if fn == "+" else -v
    if fn == "!":
        return np.logical_not(eval_expr(e.args[0], ctx))
    binops = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
              "^": np.power, "%%": np.mod, "%/%": np.floor_divide,
              "==": np.equal, "!=": np.not_equal, "<": np.less, ">": np.greater,
              "<=": np.less_equal, ">=": np.greater_equal,
              "&": np.logical_and, "&&": np.logical_and, "|": np.logical_or,
              "||": np.logical_or}
    if fn in binops and len(e.args) == 2:
        a = eval_expr(e.args[0], ctx)
        b = eval_expr(e.args[1], ctx)
        if fn in ("==", "!="):
            if isinstance(a, Factor):
                a = a.as_strings()
            if isinstance(b, Factor):
                b = b.as_strings()
            return binops[fn](np.asarray(a, dtype=object) if np.asarray(a).dtype.kind in "OUS" else a,
                              b)
        if fn in ("&", "&&", "|", "||"):
            return binops[fn](a, b)
        return binops[fn](_as_num(a), _as_num(b))
    if fn in _UFUNCS:
        args = _num_args(ctx, e)
        x = _as_num(args[0])
        if fn == "log" and (len(args) > 1 or e.arg("base") is not None):
            base = e.arg("base", 1)
            return np.log(x) / np.log(_as_num(eval_expr(base, ctx)))
        return _UFUNCS[fn](x)
    if fn == "round":
        x = _as_num(eval_expr(e.args[0], ctx))
        d = e.arg("digits", 1)
        return np.round(x, int(eval_expr(d, ctx)) if d is not None else 0)
    if fn in ("pmin", "pmax"):
        args = [_as_num(a) for a in _num_args(ctx, e)]
        f = np.minimum if fn == "pmin" else np.maximum
        out = args[0]
        for a in args[1:]:
            out = f(out, a)
        return out
    if fn in ("as.numeric", "as.double", "as.integer", "c"):
        v = eval_expr(e.args[0], ctx)
        if fn == "c" and len(e.args) > 1:
            return np.concatenate([np.atleast_1d(_as_num(eval_expr(a, ctx))) for a in e.args])
        if isinstance(v, Factor):
            return (v.codes + 1).astype(float)
        v = np.asarray(v)
        if v.dtype.kind in "OUS":
            return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
        return v.astype(float) if fn != "as.integer" else np.trunc(v.astype(float))
    if fn in ("as.character",):
        v = eval_expr(e.args[0], ctx)
        if isinstance(v, Factor):
            return v.as_strings()
        return np.array([_fmt_level(x) for x in np.asarray(v).tolist()], dtype=object)
    if fn in ("factor", "as.factor", "ordered", "as.ordered"):
        v = eval_expr(e.args[0], ctx)
        lev = e.arg("levels")
        levels = None
        if lev is not None:
            levels = list(np.atleast_1d(eval_expr(lev, ctx)))
        ordered = fn in ("ordered", "as.ordered")
        o = e.arg("ordered")
        if o is not None:
            ordered = bool(eval_expr(o, ctx))
        f = as_factor(v, levels=levels, ordered=ordered)
        if fn in ("factor", "ordered"):
            f = f if levels is not None else f.droplevels() if not isinstance(v, Factor) else Factor(f.codes, f.levels, f.ordered).droplevels()
        return f
    if fn == "interaction":
        fs = [as_factor(eval_expr(a, ctx)) for a, n in zip(e.args, e.names) if n is None]
        out = fs[0]
        for f in fs[1:]:
            s = np.array([f"{a}.{b}" for a, b in zip(out.as_strings(), f.as_strings())], dtype=object)
            levs = [f"{a}.{b}" for b in f.levels for a in out.levels]
            out = as_factor(s, levels=levs)
        return out.droplevels()
    if fn == "%i%":
        return interaction_factor(as_factor(eval_expr(e.args[0], ctx)),
                                  as_factor(eval_expr(e.args[1], ctx)))
    if fn == "offset":
        return _as_num(eval_expr(e.args[0], ctx))
    if fn == "cbind":
        cols = [np.asarray(_as_num(eval_expr(a, ctx)), float) for a in e.args]
        n = max(len(np.atleast_1d(c)) for c in cols)
        cols = [_vec(c, n) for c in cols]
        names = [n_ if n_ is not None else (a.name if isinstance(a, Sym) else "")
                 for a, n_ in zip(e.args, e.names)]
        return _Matrix(np.column_stack(cols), names)
    if fn == "poly":
        x = _as_num(eval_expr(e.args[0], ctx))
        dg = e.arg("degree", 1)
        degree = int(eval_expr(dg, ctx)) if dg is not None else 1
        rawa = e.arg("raw")
        raw = bool(eval_expr(rawa, ctx)) if rawa is not None else False
        ca = e.arg("coefs")
        coefs = ca.value if isinstance(ca, Const) and isinstance(ca.value, dict) else None
        Z, coefs = poly(x, degree, coefs=coefs, raw=raw)
        if record is not None and not raw and ca is None:
            record[deparse(e)] = Call(e.fn, e.args + (Const(coefs),), e.names + ("coefs",))
        return _Matrix(Z, [str(i) for i in range(1, degree + 1)])
    if fn == "scale":
        x = _as_num(eval_expr(e.args[0], ctx))
        ce = e.arg("center", 1)
        sc = e.arg("scale", 2)
        cen = True if ce is None else eval_expr(ce, ctx)
        scl = True if sc is None else eval_expr(sc, ctx)
        cval = np.nanmean(x) if cen is True else (0.0 if cen is False else float(np.asarray(cen)))
        xc = x - cval
        if scl is True:
            sval = np.sqrt(np.nansum(xc ** 2) / (np.sum(~np.isnan(xc)) - 1))
        elif scl is False:
            sval = 1.0
        else:
            sval = float(np.asarray(scl))
        if record is not None and (cen is True or scl is True):
            record[deparse(e)] = Call(e.fn, (e.args[0], Const(float(cval)), Const(float(sval))),
                                      (None, "center", "scale"))
        return xc / sval
    if fn == "rep":
        v = eval_expr(e.args[0], ctx)
        times = e.arg("times", 1) or e.arg("length.out")
        return np.repeat(v, int(eval_expr(times, ctx))) if times is not None else np.atleast_1d(v)
    if fn == "ifelse":
        t, a, b = (eval_expr(x, ctx) for x in e.args[:3])
        return np.where(np.asarray(t, bool), a, b)
    if fn == "is.na":
        v = eval_expr(e.args[0], ctx)
        return v.isna() if isinstance(v, Factor) else pd.isna(np.asarray(v, dtype=object))
    if fn == "exp" or fn == "sqrt":
        return _UFUNCS[fn](_as_num(eval_expr(e.args[0], ctx)))
    # user-supplied callables in the environment
    if fn is not None:
        try:
            f = ctx.lookup(fn)
        except NameError:
            raise NameError(f'could not find function "{fn}"')
        if callable(f):
            args = [eval_expr(a, ctx) for a, n in zip(e.args, e.names) if n is None]
            kwargs = {n: eval_expr(a, ctx) for a, n in zip(e.args, e.names) if n is not None}
            args = [a.to_pandas() if isinstance(a, Factor) else a for a in args]
            return f(*args, **kwargs)
    raise NotImplementedError(f"cannot evaluate {deparse(e)}")


# ---------------------------------------------------------------------------
# terms()
# ---------------------------------------------------------------------------

@dataclass
class Terms:
    """The R ``terms`` object (subset of attributes)."""
    formula: Formula
    variables: List[Any]                 # language objects, response first if present
    varnames: List[str]
    response: int                        # 1 if response present else 0 (R convention)
    terms: List[frozenset]               # sets of variable indices
    term_labels: List[str]
    intercept: int
    offset: List[int]                    # indices of offset variables
    factors: np.ndarray                  # nvar x nterm pattern (0/1/2)
    order: List[int]
    predvars: Optional[List[Any]] = None
    dataClasses: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        import copy
        return copy.copy(self)


def _is_number(e, v):
    return isinstance(e, Const) and isinstance(e.value, (int, float)) and not isinstance(
        e.value, bool) and e.value == v


def terms(formula, data_columns: Optional[Sequence[str]] = None, keep_order=False) -> Terms:
    """R's terms.formula()"""
    formula = formula if isinstance(formula, Formula) else Formula(formula)
    variables: List[Any] = []
    varkeys: List[str] = []

    def install(e):
        k = deparse(e)
        if k not in varkeys:
            varkeys.append(k)
            variables.append(e)
        return varkeys.index(k)

    state = {"intercept": 1}
    has_resp = len(formula.expr.args) == 2
    if has_resp:
        install(formula.expr.args[0])
    resp_vars = set(_all_symbol_names(formula.expr.args[0])) if has_resp else set()

    def encode(e):
        """returns list of terms (frozensets)"""
        if _is_number(e, 1):
            state["intercept"] = 1
            return []
        if _is_number(e, 0):
            state["intercept"] = 0
            return []
        if isinstance(e, Sym) and e.name == "." and data_columns is not None:
            out = []
            for c in data_columns:
                if c in resp_vars:
                    continue
                out.append(frozenset([install(Sym(c))]))
            return out
        if is_call(e, "(") and len(e.args) == 1:
            return encode(e.args[0])
        if is_call(e, "+"):
            if len(e.args) == 1:
                return encode(e.args[0])
            return _trim(encode(e.args[0]) + encode(e.args[1]))
        if is_call(e, ":"):
            l, r = encode(e.args[0]), encode(e.args[1])
            return _trim([a | b for a in l for b in r])
        if is_call(e, "*"):
            l, r = encode(e.args[0]), encode(e.args[1])
            return _trim(l + r + [a | b for a in l for b in r])
        if is_call(e, "%in%"):
            l, r = encode(e.args[0]), encode(e.args[1])
            allr = frozenset().union(*r) if r else frozenset()
            return _trim([a | allr for a in l])
        if is_call(e, "/"):
            l, r = encode(e.args[0]), encode(e.args[1])
            alll = frozenset().union(*l) if l else frozenset()
            return _trim(l + [b | alll for b in r])
        if is_call(e, "^"):
            l = encode(e.args[0])
            pw = e.args[1]
            k = int(pw.value) if isinstance(pw, Const) else int(eval_expr(pw, EvalContext({})))
            res = list(l)
            for _ in range(1, k):
                res = _trim(res + [a | b for a in res for b in l])
            return res
        if is_call(e, "-"):
            if len(e.args) == 1:
                sub = e.args[0]
                if _is_number(sub, 1):
                    state["intercept"] = 0
                    return []
                if _is_number(sub, 0):
                    state["intercept"] = 1
                    return []
                encode(sub)
                return []
            l = encode(e.args[0])
            saved = state["intercept"]
            state["intercept"] = None          # sentinel: did the RHS mention 1 or 0?
            r = encode(e.args[1])
            ri = state["intercept"]
            state["intercept"] = saved
            if ri == 1:
                state["intercept"] = 0         # "- 1"
            elif ri == 0:
                state["intercept"] = 1         # "- 0"
            rs = set(r)
            return [t for t in l if t not in rs]
        # a variable
        return [frozenset([install(e)])]

    rhs_terms = encode(formula.rhs)
    nresp = 1 if has_resp else 0
    # drop terms equal to the response
    if has_resp:
        rhs_terms = [t for t in rhs_terms if t != frozenset([0])]
    # offsets
    offset_vars = [i for i, v in enumerate(variables) if is_call(v, "offset")]
    rhs_terms = [t for t in rhs_terms if not any(i in t for i in offset_vars)]
    rhs_terms = [t for t in rhs_terms if len(t) > 0]
    if not keep_order:
        rhs_terms = sorted(rhs_terms, key=len)   # stable sort by order
    labels = [":".join(varkeys[i] for i in sorted(t)) for t in rhs_terms]
    nvar = len(variables)
    fac = np.zeros((nvar, len(rhs_terms)), dtype=int)
    for j, t in enumerate(rhs_terms):
        for i in t:
            fac[i, j] = _term_code(rhs_terms, j, i)
    return Terms(formula=formula, variables=variables, varnames=varkeys, response=nresp,
                 terms=rhs_terms, term_labels=labels, intercept=state["intercept"],
                 offset=offset_vars, factors=fac, order=[len(t) for t in rhs_terms])


def _contains_number(e, v):
    return _is_number(e, v)


def _trim(lst):
    out = []
    for t in lst:
        if t not in out:
            out.append(t)
    return out


def _term_code(termlist, j, var):
    """R's TermCode: 1 = code by contrasts, 2 = full indicator coding"""
    term = termlist[j] - {var}
    if not term:
        return 1
    for t in termlist[:j]:
        if term <= t:
            return 1
    return 2


def _all_symbol_names(e):
    from .rlang import all_vars
    return all_vars(e)


# ---------------------------------------------------------------------------
# model.frame()
# ---------------------------------------------------------------------------

class ModelFrame:
    """R model frame: named columns (numpy arrays, 2-d matrices or Factors)
    with the ``terms`` attribute and other attributes stored in ``attrs``."""

    def __init__(self, columns: Dict[str, Any], terms: Optional[Terms], row_names, attrs=None):
        self.columns = dict(columns)
        self.terms = terms
        self.row_names = np.asarray(row_names)
        self.attrs = attrs or {}

    @property
    def nrow(self):
        return len(self.row_names)

    def __len__(self):
        return self.nrow

    def __getitem__(self, k):
        return self.columns[k]

    def __contains__(self, k):
        return k in self.columns

    def names(self):
        return list(self.columns)

    def get(self, k, default=None):
        return self.columns.get(k, default)

    def subset_rows(self, idx):
        cols = {}
        for k, v in self.columns.items():
            cols[k] = v.subset(idx) if isinstance(v, Factor) else np.asarray(v)[idx]
        return ModelFrame(cols, self.terms, self.row_names[idx], dict(self.attrs))

    def response(self):
        """model.response()"""
        if self.terms is None or not self.terms.response:
            return None
        return self.columns[self.terms.varnames[0]]

    def weights(self):
        return self.columns.get("(weights)")

    def offset(self):
        """model.offset(): sum of offset() terms and the (offset) column"""
        out = None
        if self.terms is not None:
            for i in self.terms.offset:
                v = np.asarray(self.columns[self.terms.varnames[i]], float)
                out = v if out is None else out + v
        if "(offset)" in self.columns:
            v = np.asarray(self.columns["(offset)"], float)
            out = v if out is None else out + v
        return out

    def to_pandas(self):
        d = {}
        for k, v in self.columns.items():
            if isinstance(v, Factor):
                d[k] = v.to_pandas()
            elif np.ndim(v) == 2:
                cn = getattr(v, "colnames", None) or [str(i + 1) for i in range(v.shape[1])]
                for j in range(v.shape[1]):
                    d[f"{k}.{cn[j]}"] = np.asarray(v)[:, j]
            else:
                d[k] = np.asarray(v)
        return pd.DataFrame(d, index=self.row_names)

    def __repr__(self):
        return repr(self.to_pandas())


def data_to_dict(data) -> Dict[str, Any]:
    """convert a pandas DataFrame (or dict) into a dict of columns,
    turning categoricals into Factors"""
    if data is None:
        return {}
    if isinstance(data, ModelFrame):
        return dict(data.columns)
    if isinstance(data, dict):
        return {k: (as_factor(v) if isinstance(v, (pd.Categorical,)) or
                    (isinstance(v, pd.Series) and isinstance(v.dtype, pd.CategoricalDtype)) else
                    (np.asarray(v) if not isinstance(v, Factor) else v))
                for k, v in data.items()}
    out = {}
    for c in data.columns:
        s = data[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[str(c)] = as_factor(s)
        elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            out[str(c)] = np.asarray(s.astype(object).where(s.notna(), None), dtype=object)
        elif s.dtype == bool:
            out[str(c)] = s.to_numpy(bool)
        elif pd.api.types.is_bool_dtype(s.dtype):
            out[str(c)] = s.to_numpy(dtype=object)
        else:
            out[str(c)] = s.to_numpy(dtype=float, na_value=np.nan)
    return out


def _isna_value(v):
    if isinstance(v, Factor):
        return v.isna()
    a = np.asarray(v)
    if a.ndim == 2:
        return np.isnan(a.astype(float)).any(axis=1) if a.dtype.kind in "fiu" else pd.isna(a).any(axis=1)
    if a.dtype.kind == "f":
        return np.isnan(a)
    if a.dtype.kind in "OUS":
        return pd.isna(a)
    return np.zeros(len(a), bool)


def model_frame(formula, data=None, subset=None, weights=None, offset=None,
                na_action="na.omit", drop_unused_levels=True, env=None, extras=None,
                predvars=None, xlev=None) -> ModelFrame:
    """R's model.frame.default()"""
    formula = formula if isinstance(formula, Formula) else Formula(formula, env)
    env = env if env is not None else formula.env
    ddict = data_to_dict(data)
    if isinstance(data, ModelFrame):
        nrow = data.nrow
    elif isinstance(data, pd.DataFrame):
        nrow = len(data)
    else:
        nrow = None
    if nrow is None:
        nrow = len(next(iter(ddict.values()))) if ddict else 0
    if isinstance(data, pd.DataFrame):
        row_names = np.asarray(data.index)
    elif isinstance(data, ModelFrame):
        row_names = data.row_names
    else:
        row_names = np.arange(1, nrow + 1)
    tt = terms(formula, data_columns=list(ddict) if data is not None else None)
    ctx = EvalContext(ddict, env, nrow)
    record: Dict[str, Any] = {}
    columns: Dict[str, Any] = {}
    vars_eval = predvars if predvars is not None else tt.variables
    for key, v in zip(tt.varnames, vars_eval):
        val = eval_expr(v, ctx, record)
        if isinstance(val, (pd.Series, pd.Categorical)):
            val = as_factor(val) if isinstance(val, pd.Categorical) or isinstance(
                getattr(val, "dtype", None), pd.CategoricalDtype) else val.to_numpy()
        if not isinstance(val, Factor):
            a = np.asarray(val)
            if a.ndim == 0:
                a = np.full(nrow, a.item())
            val = val if isinstance(val, _Matrix) else a
        columns[key] = val
    # extras: (weights), (offset), etc.
    extras = dict(extras or {})
    if weights is not None:
        extras["(weights)"] = weights
    if offset is not None:
        extras["(offset)"] = offset
    for k, v in extras.items():
        if v is None:
            continue
        if isinstance(v, str):
            v = eval_expr(parse(v), ctx)
        elif isinstance(v, (Call, Sym)):
            v = eval_expr(v, ctx)
        a = np.asarray(v, float)
        if a.ndim == 0:
            a = np.full(nrow, float(a))
        columns[k] = a
    # lengths
    for k, v in columns.items():
        if len(v) != nrow:
            raise ValueError(f"variable lengths differ (found for '{k}')")
    # subset
    keep = np.ones(nrow, bool)
    if subset is not None:
        if isinstance(subset, str):
            sv = eval_expr(parse(subset), ctx)
        elif isinstance(subset, (Call, Sym)):
            sv = eval_expr(subset, ctx)
        else:
            sv = subset
        sv = np.asarray(sv)
        if sv.dtype == bool:
            keep &= sv
        else:
            m = np.zeros(nrow, bool)
            m[sv.astype(int)] = True
            keep &= m
    # NA handling
    na_rows = np.zeros(nrow, bool)
    for k, v in columns.items():
        na_rows |= _isna_value(v)
    na_rows &= keep
    na_info = None
    if na_rows.any():
        if na_action in ("na.omit", "na.exclude", None):
            na_info = {"type": na_action or "na.omit",
                       "omitted": np.nonzero(na_rows)[0], "names": row_names[na_rows]}
            keep &= ~na_rows
        elif na_action == "na.fail":
            raise ValueError("missing values in object")
        elif na_action == "na.pass":
            pass
    idx = np.nonzero(keep)[0]
    out = {}
    for k, v in columns.items():
        if isinstance(v, Factor):
            f = v.subset(idx)
            if drop_unused_levels:
                f = f.droplevels()
            out[k] = f
        elif isinstance(v, _Matrix):
            out[k] = _Matrix(np.asarray(v)[idx], v.colnames)
        else:
            out[k] = np.asarray(v)[idx]
    if record:
        tt.predvars = [record.get(k, v) for k, v in zip(tt.varnames, tt.variables)]
    else:
        tt.predvars = list(vars_eval)
    tt.dataClasses = {k: _data_class(v) for k, v in out.items()}
    attrs = {"na.action": na_info, "orig_index": idx}
    return ModelFrame(out, tt, row_names[idx], attrs)


def _data_class(v):
    if isinstance(v, Factor):
        return "ordered" if v.ordered else "factor"
    a = np.asarray(v)
    if a.ndim == 2:
        return f"nmatrix.{a.shape[1]}"
    if a.dtype == bool:
        return "logical"
    if a.dtype.kind in "OUS":
        return "character"
    return "numeric"


# ---------------------------------------------------------------------------
# model.matrix()
# ---------------------------------------------------------------------------

def _var_for_matrix(v, contrasts_arg, name):
    """returns ('num', matrix n x k, colnames) or ('fac', Factor)"""
    if isinstance(v, Factor):
        return ("fac", v)
    a = np.asarray(v)
    if a.dtype == bool:
        return ("fac", Factor(a.astype(np.int64), ["FALSE", "TRUE"]))
    if a.dtype.kind in "OUS":
        return ("fac", as_factor(a))
    if a.ndim == 2:
        cn = getattr(v, "colnames", None)
        if cn is None or all(c == "" for c in cn):
            cn = [str(i + 1) for i in range(a.shape[1])]
        return ("num", a.astype(float), [name + c for c in cn])
    return ("num", a.astype(float).reshape(-1, 1), [name])


def model_matrix(tt: Terms, frame: ModelFrame, contrasts=None):
    """R's model.matrix.default().  Returns (X, colnames, assign, contrasts_used)."""
    contrasts = contrasts or {}
    n = frame.nrow
    resp = tt.response
    vinfo = {}
    for i, key in enumerate(tt.varnames):
        if resp and i == 0:
            continue
        if any(i in t for t in tt.terms):
            vinfo[i] = _var_for_matrix(frame[key], contrasts.get(key), key)
    factors = tt.factors.copy()
    if not tt.intercept:
        done = False
        for j in range(len(tt.terms)):
            for i in range(resp, len(tt.variables)):
                if i in vinfo and vinfo[i][0] == "fac" and factors[i, j] > 0:
                    factors[i, j] = 2
                    done = True
                    break
            if done:
                break
    cols, names, assign = [], [], []
    contr_used = {}
    if tt.intercept:
        cols.append(np.ones((n, 1)))
        names.append("(Intercept)")
        assign.append(0)
    for j, t in enumerate(tt.terms):
        mats = [np.ones((n, 1))]
        nms = [""]
        for i in sorted(t):
            info = vinfo[i]
            key = tt.varnames[i]
            if info[0] == "num":
                m, cn = info[1], info[2]
            else:
                f = info[1]
                codes = f.codes
                if factors[i, j] == 1:
                    cm, suffix = _contrast_matrix(f, contrasts.get(key))
                    contr_used[key] = contrasts.get(key) or f.contrasts or (
                        "contr.poly" if f.ordered else "contr.treatment")
                else:
                    cm, suffix = np.eye(f.nlevels), list(f.levels)
                m = np.full((n, cm.shape[1]), np.nan)
                ok = codes >= 0
                m[ok] = cm[codes[ok]]
                cn = [key + s for s in suffix]
            newm, newn = [], []
            for b in range(m.shape[1]):          # previous variables vary fastest
                for a in range(len(nms)):
                    newm.append(mats[a][:, 0] * m[:, b])
                    newn.append(cn[b] if nms[a] == "" else nms[a] + ":" + cn[b])
            mats = [x.reshape(-1, 1) for x in newm]
            nms = newn
        for mm, nm in zip(mats, nms):
            cols.append(mm)
            names.append(nm)
            assign.append(j + 1)
    X = np.hstack(cols) if cols else np.zeros((n, 0))
    return X, names, np.array(assign, dtype=int), contr_used
