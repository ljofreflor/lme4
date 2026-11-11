"""Nonlinear mixed-effects models: transcription of ``nlmer()``
(lme4/R/lmer.R), ``nlformula()`` and the nlmer branch of ``mkRespMod()``
(lme4/R/utilities.R), plus the R ``stats`` self-starting models that lme4's
documentation and tests use (``SSlogis``, ``SSfol``, ...), with analytic
gradients.

The model is given as R's three-part formula::

    nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym | Tree",
          data=Orange, start=dict(Asym=200, xmid=725, scal=350))

The nonlinear model in the middle part can be

* one of the self-starting models defined here (``SSlogis`` etc.);
* a Python function found in the caller's namespace and called like an R
  function, e.g. ``nfun(age, Asym, xmid, scal)``.  It must return either an
  array carrying a ``gradient`` attribute, or a tuple ``(value, gradient)``.
  The gradient is an ``n x k`` array (columns named after the parameter
  arguments, in their order in the call), a ``pandas.DataFrame`` or a dict
  ``{name: column}``;
* a bare name of a Python callable ``model(params, data)``, e.g.
  ``"circumference ~ mymodel ~ Asym | Tree"``, where ``params`` is a dict
  ``{pname: ndarray(n)}`` (in ``start`` order) and ``data`` a dict of the
  data columns for the rows used; it returns ``(mu, gradient)`` with the
  gradient as above (an array has its columns in ``start`` order).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .control import nlmerControl, nlmerControlObj
from .covariance import mkMkPar, mkMkTheta, upReTrms
from .lmer import Call as _McCall, _caller_env, _name_of, mkMerMod
from .modelframe import (EvalContext, Factor, ModelFrame, data_to_dict, eval_expr, model_frame,
                         model_matrix, terms)
from .modular import Rho, mkdevfun, _rank_qr
from .modules import merPredD, nlsResp
from .optimizers import optwrap
from .reformulas import (findbars_x, lme4_specials, mkReTrms, nobars, noSpecials, sub_specials)
from .rlang import Call, Const, Formula, Sym, all_vars, as_formula, deparse, is_call, parse
from . import options

__all__ = ["nlmer", "nlformula", "mkNlRespMod", "SSlogis", "SSasymp", "SSasympOff",
           "SSasympOrig", "SSmicmen", "SSfol", "SSfpl", "SSgompertz", "SSbiexp", "SSweibull",
           "selfStart_models", "NLValue"]


# ---------------------------------------------------------------------------
# self-starting models (R's stats/R/zzModels.R) with analytic gradients
# ---------------------------------------------------------------------------

class NLValue(np.ndarray):
    """model value with R's ``gradient`` attribute (an n x k matrix whose
    column names are in ``gradient_names``)"""

    def __new__(cls, value, gradient=None, gradient_names=None):
        obj = np.asarray(value, float).view(cls)
        obj.gradient = None if gradient is None else np.asarray(gradient, float)
        obj.gradient_names = None if gradient_names is None else list(gradient_names)
        return obj

    def __array_finalize__(self, obj):
        self.gradient = getattr(obj, "gradient", None)
        self.gradient_names = getattr(obj, "gradient_names", None)


def _f(x):
    return np.asarray(x, float)


def _ss(value, cols, names):
    n = np.broadcast(*([value] + list(cols))).shape
    g = np.column_stack([np.broadcast_to(_f(c), n) for c in cols]) if cols else None
    return NLValue(np.broadcast_to(_f(value), n), g, names)


def SSlogis(input, Asym, xmid, scal):
    """Asym/(1+exp((xmid-input)/scal))"""
    input, Asym, xmid, scal = map(_f, (input, Asym, xmid, scal))
    e2 = (xmid - input) / scal
    e3 = np.exp(e2)
    e4 = 1 + e3
    val = Asym / e4
    xm = Asym * e3 / scal / e4 ** 2
    return _ss(val, [1 / e4, -xm, xm * e2], ["Asym", "xmid", "scal"])


def SSasymp(input, Asym, R0, lrc):
    """Asym+(R0-Asym)*exp(-exp(lrc)*input)"""
    input, Asym, R0, lrc = map(_f, (input, Asym, R0, lrc))
    k = np.exp(lrc)
    E = np.exp(-k * input)
    val = Asym + (R0 - Asym) * E
    return _ss(val, [1 - E, E, -(R0 - Asym) * E * k * input], ["Asym", "R0", "lrc"])


def SSasympOff(input, Asym, lrc, c0):
    """Asym*(1 - exp(-exp(lrc)*(input - c0)))"""
    input, Asym, lrc, c0 = map(_f, (input, Asym, lrc, c0))
    k = np.exp(lrc)
    d = input - c0
    E = np.exp(-k * d)
    val = Asym * (1 - E)
    return _ss(val, [1 - E, Asym * E * k * d, -Asym * E * k], ["Asym", "lrc", "c0"])


def SSasympOrig(input, Asym, lrc):
    """Asym*(1 - exp(-exp(lrc)*input))"""
    input, Asym, lrc = map(_f, (input, Asym, lrc))
    k = np.exp(lrc)
    E = np.exp(-k * input)
    val = Asym * (1 - E)
    return _ss(val, [1 - E, Asym * E * k * input], ["Asym", "lrc"])


def SSmicmen(input, Vm, K):
    """Vm*input/(K+input)"""
    input, Vm, K = map(_f, (input, Vm, K))
    d = K + input
    val = Vm * input / d
    return _ss(val, [input / d, -Vm * input / d ** 2], ["Vm", "K"])


def SSfol(Dose, input, lKe, lKa, lCl):
    """Dose*exp(lKe+lKa-lCl)*(exp(-exp(lKe)*input)-exp(-exp(lKa)*input))/(exp(lKa)-exp(lKe))"""
    Dose, input, lKe, lKa, lCl = map(_f, (Dose, input, lKe, lKa, lCl))
    ke, ka = np.exp(lKe), np.exp(lKa)
    A = Dose * np.exp(lKe + lKa - lCl)
    e1, e2 = np.exp(-ke * input), np.exp(-ka * input)
    B = e1 - e2
    D = ka - ke
    val = A * B / D
    g_lKe = val - A * input * ke * e1 / D + A * B * ke / D ** 2
    g_lKa = val + A * input * ka * e2 / D - A * B * ka / D ** 2
    return _ss(val, [g_lKe, g_lKa, -val], ["lKe", "lKa", "lCl"])


def SSfpl(input, A, B, xmid, scal):
    """A+(B-A)/(1+exp((xmid-input)/scal))"""
    input, A, B, xmid, scal = map(_f, (input, A, B, xmid, scal))
    e2 = (xmid - input) / scal
    e3 = np.exp(e2)
    e4 = 1 + e3
    val = A + (B - A) / e4
    xm = (B - A) * e3 / scal / e4 ** 2
    return _ss(val, [1 - 1 / e4, 1 / e4, -xm, xm * e2], ["A", "B", "xmid", "scal"])


def SSgompertz(x, Asym, b2, b3):
    """Asym*exp(-b2*b3^x)"""
    x, Asym, b2, b3 = map(_f, (x, Asym, b2, b3))
    p = b3 ** x
    E = np.exp(-b2 * p)
    val = Asym * E
    return _ss(val, [E, -Asym * E * p, -Asym * E * b2 * x * b3 ** (x - 1)], ["Asym", "b2", "b3"])


def SSbiexp(input, A1, lrc1, A2, lrc2):
    """A1*exp(-exp(lrc1)*input)+A2*exp(-exp(lrc2)*input)"""
    input, A1, lrc1, A2, lrc2 = map(_f, (input, A1, lrc1, A2, lrc2))
    k1, k2 = np.exp(lrc1), np.exp(lrc2)
    E1, E2 = np.exp(-k1 * input), np.exp(-k2 * input)
    val = A1 * E1 + A2 * E2
    return _ss(val, [E1, -A1 * E1 * k1 * input, E2, -A2 * E2 * k2 * input],
               ["A1", "lrc1", "A2", "lrc2"])


def SSweibull(x, Asym, Drop, lrc, pwr):
    """Asym-Drop*exp(-exp(lrc)*x^pwr)"""
    x, Asym, Drop, lrc, pwr = map(_f, (x, Asym, Drop, lrc, pwr))
    k = np.exp(lrc)
    xp = x ** pwr
    E = np.exp(-k * xp)
    val = Asym - Drop * E
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), 0.0)
    return _ss(val, [np.ones_like(E), -E, Drop * E * k * xp, Drop * E * k * xp * lx],
               ["Asym", "Drop", "lrc", "pwr"])


# name -> (function, formal argument names, parameter formals)
selfStart_models = {
    "SSlogis": (SSlogis, ["input", "Asym", "xmid", "scal"], ["Asym", "xmid", "scal"]),
    "SSasymp": (SSasymp, ["input", "Asym", "R0", "lrc"], ["Asym", "R0", "lrc"]),
    "SSasympOff": (SSasympOff, ["input", "Asym", "lrc", "c0"], ["Asym", "lrc", "c0"]),
    "SSasympOrig": (SSasympOrig, ["input", "Asym", "lrc"], ["Asym", "lrc"]),
    "SSmicmen": (SSmicmen, ["input", "Vm", "K"], ["Vm", "K"]),
    "SSfol": (SSfol, ["Dose", "input", "lKe", "lKa", "lCl"], ["lKe", "lKa", "lCl"]),
    "SSfpl": (SSfpl, ["input", "A", "B", "xmid", "scal"], ["A", "B", "xmid", "scal"]),
    "SSgompertz": (SSgompertz, ["x", "Asym", "b2", "b3"], ["Asym", "b2", "b3"]),
    "SSbiexp": (SSbiexp, ["input", "A1", "lrc1", "A2", "lrc2"], ["A1", "lrc1", "A2", "lrc2"]),
    "SSweibull": (SSweibull, ["x", "Asym", "Drop", "lrc", "pwr"], ["Asym", "Drop", "lrc", "pwr"]),
}


# ---------------------------------------------------------------------------
# evaluation of the nonlinear model expression
# ---------------------------------------------------------------------------

def _match_args(call: Call, formals):
    """R's match.call() for a closure with the given formals (exact names,
    then positional)"""
    out = {}
    pos = []
    for a, nm in zip(call.args, call.names):
        if nm is not None:
            if nm not in formals:
                raise TypeError(f"unused argument ({nm} = {deparse(a)})")
            out[nm] = a
        else:
            pos.append(a)
    rest = [f for f in formals if f not in out]
    if len(pos) > len(rest):
        raise TypeError(f"unused argument(s) in {deparse(call)}")
    for f, a in zip(rest, pos):
        out[f] = a
    missing = [f for f in formals if f not in out]
    if missing:
        raise TypeError(f'argument "{missing[0]}" is missing, with no default')
    return out


def _as_grad(gr, n, default_names):
    """normalize a gradient into (ndarray n x k, colnames)"""
    if gr is None:
        return None, None
    if isinstance(gr, pd.DataFrame):
        return gr.to_numpy(float), [str(c) for c in gr.columns]
    if isinstance(gr, dict):
        names = list(gr)
        return np.column_stack([np.broadcast_to(_f(gr[k]), (n,)) for k in names]), names
    g = np.asarray(gr, float)
    if g.ndim == 1:
        g = g.reshape(n, -1)
    names = getattr(gr, "colnames", None) or default_names
    return g, (list(names) if names is not None else None)


def _split_value(res, n, default_names):
    if isinstance(res, tuple) and len(res) == 2:
        val, gr = res
    else:
        val, gr = res, getattr(res, "gradient", None)
        if isinstance(res, NLValue) and gr is not None:
            gr = (gr, res.gradient_names)
    names = default_names
    if isinstance(gr, tuple):
        gr, names = gr
    g, gnames = _as_grad(gr, n, names)
    return np.asarray(val, float).ravel(), g, gnames


class _NLModel:
    """the evaluation of ``nlmod`` in ``nlenv`` (what nlsResp does in R via
    ``eval(nlmod, nlenv)`` after assigning the parameter columns)"""

    def __init__(self, nlmod, nlenv, env, pnames, data_cols=None):
        self.nlmod = nlmod
        self.nlenv = nlenv          # dict: frame columns (+ parameter columns)
        self.env = env
        self.pnames = list(pnames)
        self.data_cols = data_cols  # data dict for the bare-callable interface

    def _resolve(self, name):
        if name in self.env and callable(self.env[name]):
            return self.env[name], None
        if name in selfStart_models:
            return selfStart_models[name][0], selfStart_models[name]
        raise NameError(f'could not find function "{name}"')

    def evaluate(self, nlenv=None):
        """returns (value, gradient matrix, gradient colnames)"""
        nlenv = self.nlenv if nlenv is None else nlenv
        e = self.nlmod
        n = len(next(iter(nlenv.values()))) if nlenv else 0
        if isinstance(e, Sym):
            fun = self.env.get(e.name)
            if not callable(fun):
                raise ValueError(f"'{e.name}' is not a function")
            params = {p: np.asarray(nlenv[p], float) for p in self.pnames}
            return _split_value(fun(params, self.data_cols), len(params[self.pnames[0]]),
                                self.pnames)
        if not (isinstance(e, Call) and isinstance(e.fn, Sym)):
            ctx = EvalContext(nlenv, self.env, n)
            return _split_value(eval_expr(e, ctx), n, None)
        fun, ss = self._resolve(e.fn.name)
        ctx = EvalContext(nlenv, self.env, n)

        def ev(a):
            v = eval_expr(a, ctx)
            return v.to_pandas() if isinstance(v, Factor) else v
        if ss is not None:
            _, formals, parformals = ss
            m = _match_args(e, formals)
            res = fun(**{k: ev(a) for k, a in m.items()})
            actual = [m[p] for p in parformals]
            if not all(isinstance(a, Sym) for a in actual):
                # R's selfStart models only return a gradient when all the
                # parameter arguments are names
                return np.asarray(res, float).ravel(), None, None
            # dimnames(.grad) <- list(NULL, .actualArgs)
            return np.asarray(res, float).ravel(), res.gradient, [a.name for a in actual]
        args = [ev(a) for a, nm in zip(e.args, e.names) if nm is None]
        kwargs = {nm: ev(a) for a, nm in zip(e.args, e.names) if nm is not None}
        default = [a.name for a in e.args if isinstance(a, Sym) and a.name in self.pnames]
        return _split_value(fun(*args, **kwargs), n, default)

    def __call__(self, pars):
        """the callable handed to the C++ nlsResp: ``pars`` is n x npar
        (column j = parameter pnames[j]); returns (mu, gradient n x npar)"""
        pars = np.asarray(pars, float)
        env = dict(self.nlenv)
        for j, p in enumerate(self.pnames):
            env[p] = pars[:, j]
        val, g, gn = self.evaluate(env)
        if g is None:
            raise ValueError("the nonlinear model did not return a gradient")
        return val, _order_grad(g, gn, self.pnames)


def _order_grad(g, gnames, pnames):
    if gnames is None or list(gnames) == list(pnames):
        return np.asfortranarray(g)
    idx = [list(gnames).index(p) for p in pnames]
    return np.asfortranarray(g[:, idx])


# ---------------------------------------------------------------------------
# mkRespMod (nlmer branch)
# ---------------------------------------------------------------------------

def mkNlRespMod(fr: ModelFrame, nlenv, nlmod: _NLModel, pnames):
    """``mkRespMod(fr, nlenv=nlenv, nlmod=nlmod)``"""
    y = fr.response()
    offset = fr.offset()
    weights = fr.weights()
    n = fr.nrow
    if y is not None:
        ya = np.asarray(y)
        if ya.dtype == bool:
            y = ya.astype(int)
        elif ya.dtype.kind not in "biuf":
            raise ValueError("response must be numeric")
        if not np.all(np.isfinite(np.asarray(y, float))):
            raise ValueError("NA/NaN/Inf in 'y'")
    y = np.zeros(0) if y is None else np.asarray(y, float)
    val, gr, gnames = nlmod.evaluate()
    if len(val) != n:
        raise ValueError("length(val) == n is not TRUE")
    if gr is None or gr.ndim != 2:
        raise ValueError("is.matrix(gr <- attr(val, \"gradient\")) is not TRUE")
    if gr.shape[0] != n:
        raise ValueError("nrow(gr) == n is not TRUE")
    if gnames is None:
        raise ValueError("!is.null(pnames <- colnames(gr)) is not TRUE")
    unknown = [g for g in gnames if g not in pnames]
    if unknown:
        raise ValueError(f"object '{unknown[0]}' not found")
    if sorted(gnames) != sorted(pnames):
        raise ValueError("the columns of the gradient do not match the parameter names "
                         f"({', '.join(gnames)} vs {', '.join(pnames)})")
    # NB: R takes 'pnames' from colnames(gr), whereas X (and beta0) follow the
    # order of 'start'; we keep the gradient columns in 'start' order so that
    # both agree (identical to R whenever the two orders coincide).
    gr = _order_grad(gr, gnames, pnames)
    N = gr.size
    mu = val
    sqrtXwt = gr.ravel(order="F")
    gam = np.concatenate([np.asarray(nlenv[p], float) for p in pnames])
    if offset is not None:
        offset = np.asarray(offset, float)
        if offset.size == 1:
            offset = np.full(N, float(offset))
        elif offset.size != N:
            raise ValueError("length(offset) == N is not TRUE")
    else:
        offset = np.zeros(N)
    if weights is not None:
        weights = np.asarray(weights, float)
        if len(weights) != n or np.any(weights < 0):
            raise ValueError("length(weights) == n && all(weights >= 0) is not TRUE")
    else:
        weights = np.ones(n)
    return nlsResp(y=y, weights=weights, offset=offset, mu=mu, sqrtXwt=sqrtXwt, gam=gam,
                   model=nlmod, pnames=list(pnames))


# ---------------------------------------------------------------------------
# nlformula
# ---------------------------------------------------------------------------

def _chck1(expr):
    """check for a literal 1 in the mixed-effects part"""
    if isinstance(expr, Const):
        if isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool) \
                and expr.value == 1:
            raise ValueError("1 is not meaningful in a nonlinear model formula")
        return
    if isinstance(expr, Call):
        for a in expr.args:
            _chck1(a)


def _sum_expr(names):
    e = Sym(names[0])
    for nm in names[1:]:
        e = Call(Sym("+"), (e, Sym(nm)))
    return e


def _rbind_frames(fr: ModelFrame, s: int):
    cols = {}
    for k, v in fr.columns.items():
        if isinstance(v, Factor):
            cols[k] = Factor(np.tile(v.codes, s), v.levels, v.ordered, v.contrasts)
        else:
            a = np.asarray(v)
            cols[k] = np.concatenate([a] * s, axis=0)
    return cols


def _start_nlpars(start):
    if start is None:
        raise ValueError("is.numeric(nlpars <- start$nlpars) is not TRUE")
    if isinstance(start, pd.Series):
        start = start.to_dict()
    if isinstance(start, dict) and "nlpars" in start:
        start = start["nlpars"]
        if isinstance(start, pd.Series):
            start = start.to_dict()
    if not isinstance(start, dict) or not start:
        raise ValueError("is.numeric(nlpars <- start$nlpars) is not TRUE")
    out = {}
    for k, v in start.items():
        if np.size(v) != 1:
            raise ValueError("lengths(nlpars) == 1L is not TRUE")
        out[str(k)] = float(np.asarray(v, float).ravel()[0])
    return out


def nlformula(formula, data=None, start=None, subset=None, weights=None,
              na_action="na.omit", offset=None, env=None):
    """lme4::nlformula -- returns dict(respMod, frame, X, reTrms, pnames, Xattrs)"""
    env = env if env is not None else {}
    nlpars = _start_nlpars(start)
    pnames = list(nlpars)
    form = as_formula(formula, env)
    if len(form) != 3:
        raise ValueError("length(form <- as.formula(mc$formula)) == 3L is not TRUE")
    nlform = form[1]
    if not is_call(nlform, "~"):
        raise ValueError('is(nlform <- eval(form[[2]]), "formula") is not TRUE')
    nlmod = nlform[len(nlform) - 1]
    bare = isinstance(nlmod, Sym) and callable(env.get(nlmod.name))
    if not bare and not all(p in all_vars(nlmod) for p in pnames):
        raise ValueError("pnames %in% all.vars(nlmod) is not TRUE")
    # nlform[[lnl]] <- (all variables but the parameters)
    fvars = [v for v in all_vars(form) if v not in pnames]
    if bare:
        fvars = [v for v in fvars if v != nlmod.name]
    lhs = nlform[1] if len(nlform) == 3 else None
    # the response is the LHS; the remaining variables are model-frame columns
    rhs_vars = [v for v in fvars if not (isinstance(lhs, Sym) and v == lhs.name)]
    rhs = _sum_expr(rhs_vars) if rhs_vars else Const(1.0)
    fr_formula = Formula(Call(Sym("~"), (lhs, rhs)) if lhs is not None else Call(Sym("~"), (rhs,)),
                         env)
    fr = model_frame(fr_formula, data, subset=subset, weights=weights, offset=offset,
                     na_action=na_action, drop_unused_levels=True, env=env)
    n = fr.nrow
    nlenv = dict(fr.columns)
    for nm in pnames:
        nlenv[nm] = np.repeat(nlpars[nm], n)
    data_cols = None
    if bare:
        d = data_to_dict(data) if data is not None else {}
        idx = fr.attrs.get("orig_index")
        data_cols = {k: (v.subset(idx).to_pandas() if isinstance(v, Factor) else np.asarray(v)[idx])
                     for k, v in d.items()}
        for k, v in fr.columns.items():
            data_cols.setdefault(k, v.to_pandas() if isinstance(v, Factor) else v)
    model = _NLModel(nlmod, nlenv, env, pnames, data_cols)
    respMod = mkNlRespMod(fr, nlenv, model, pnames)
    # fixed and random parts
    meform = form[2]
    _chck1(meform)
    pnameexpr = _sum_expr(pnames)
    formula2 = Formula(Call(Sym("~"), (Call(Sym("+"), (Call(Sym("+"), (
        Const(0.0), Call(Sym("("), (pnameexpr,)))), Call(Sym("("), (meform,)))),)), env)
    fr_form_ = noSpecials(formula2, specials=lme4_specials, delete=False)
    fixedform = fr_form_.with_rhs(nobars(fr_form_.rhs))
    frE_cols = _rbind_frames(fr, len(pnames))
    for nm in pnames:
        frE_cols[nm] = np.repeat(np.array([float(nm == p) for p in pnames]), n)
    tfix = terms(fixedform)
    cols = {}
    for key, v in zip(tfix.varnames, tfix.variables):
        if key in frE_cols:
            cols[key] = frE_cols[key]
        else:
            cols[key] = eval_expr(v, EvalContext(frE_cols, env, n * len(pnames)))
    frE_fix = ModelFrame(cols, tfix, np.arange(1, n * len(pnames) + 1))
    X, cn, assign, contr = model_matrix(tfix, frE_fix)
    frE = ModelFrame(frE_cols, None, np.arange(1, n * len(pnames) + 1))
    bb1 = findbars_x(formula2, specials=lme4_specials, default_special="us", target="|",
                     expand_doublevert_method=options.get("lme4.doublevert.default"))
    bb0 = []
    for call in bb1:
        bar = call[1]
        bar = bar.replace(1, Call(Sym("+"), (Const(0.0), Call(Sym("("), (bar[1],)))))
        bb0.append(bar)
    reTrms = mkReTrms(bb0, frE, calc_lambdat=False, env=env)
    bb1 = [bb1[i] for i in reTrms["ord"]]
    reTrms = upReTrms(reTrms, bb1)
    Xattrs = {"assign": assign, "contrasts": contr, "colnames": cn}
    return dict(respMod=respMod, frame=fr, X=X, reTrms=reTrms, pnames=pnames, Xattrs=Xattrs,
                nlenv=nlenv)


# ---------------------------------------------------------------------------
# nlmer
# ---------------------------------------------------------------------------

def nlmer(formula, data=None, control=None, start=None, verbose=0, nAGQ=1, subset=None,
          weights=None, na_action="na.omit", offset=None, contrasts=None, devFunOnly=False,
          _env=None):
    """Fit a nonlinear mixed-effects model (lme4::nlmer).

    ``formula`` is a three-part formula ``resp ~ Nonlin(...) ~ mixed``, e.g.
    ``"circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym | Tree"``;
    ``start`` gives the starting values of the nonlinear-model parameters
    (a dict, a ``pandas.Series`` or ``{"nlpars": ...}``)."""
    env = _env if _env is not None else _caller_env()
    if control is None:
        control = nlmerControl()
    elif not isinstance(control, nlmerControlObj):
        if not isinstance(control, dict):
            raise TypeError("'control' is not a list; use nlmerControl()")
        control = nlmerControl(**control)
    control = control.copy()
    nAGQ = int(nAGQ)
    form = as_formula(formula, env)
    mc = _McCall("nlmer", formula=form, data=_name_of(data, env), start=start,
                 weights=weights if isinstance(weights, str) else None,
                 offset=offset if isinstance(offset, str) else None,
                 subset=subset if isinstance(subset, str) else None, control=control,
                 nAGQ=nAGQ, verbose=verbose, contrasts=contrasts, na_action=na_action,
                 _data=data)
    vals = nlformula(form, data, start=start, subset=subset, weights=weights,
                     na_action=na_action, offset=offset, env=env)
    X = vals["X"]
    p = X.shape[1]
    rankX = _rank_qr(X)
    if rankX < p:
        raise ValueError(f"rank of X = {rankX} < ncol(X) = {p}")
    reTrms = vals["reTrms"]
    resp = vals["respMod"]
    rho = Rho(verbose=verbose, tolPwrss=0.001, resp=resp)
    gam0 = np.concatenate([np.asarray(vals["nlenv"][nm], float) for nm in vals["pnames"]])
    beta0 = np.linalg.lstsq(X, gam0, rcond=None)[0]       # qr.coef(qr(X), ...)
    rho.pp = merPredD(X, reTrms["Zt"], reTrms["Lambdat"], reTrms["Lind"], reTrms["theta"],
                      n=len(resp.mu), Xwts=resp.sqrtXwt, beta0=beta0, Xattrs=vals["Xattrs"])
    rho.mkPar = mkMkPar(reTrms["reCovs"])
    rho.mkTheta = mkMkTheta(reTrms["reCovs"])
    rho.reCovs = reTrms["reCovs"]
    rho.u0 = np.asarray(rho.pp.u0).copy()
    rho.beta0 = np.asarray(rho.pp.beta0).copy()
    # deviance as a function of theta only:
    devfun = mkdevfun(rho, 0, verbose=verbose, control=control)
    if devFunOnly and not nAGQ:
        return devfun
    devfun(rho.mkPar(np.asarray(rho.pp.theta)))   # initial coarse evaluation (u0, beta0)
    rho.u0 = np.asarray(rho.pp.u0).copy()
    rho.beta0 = np.asarray(rho.pp.beta0).copy()
    rho.tolPwrss = control["tolPwrss"]
    rho.nAGQ = nAGQ
    # bounds: user-specified ones are restricted to the random-effects parameters
    optCtrl = dict(control["optCtrl"] or {})
    lower = np.asarray(reTrms["lower"], float)
    lwr = optCtrl.pop("lower", None)
    if lwr is not None:
        lwr = np.asarray(lwr, float)
        lower = lwr[:len(lower)]
    upper = np.asarray(reTrms["upper"], float)
    upr = optCtrl.pop("upper", None)
    if upr is not None:
        upr = np.asarray(upr, float)
        upper = upr[:len(upper)]
    control["optCtrl"] = optCtrl
    rho.lower, rho.upper = lower, upper
    opt = optwrap(control["optimizer"][0], devfun, rho.mkPar(np.asarray(rho.pp.theta)),
                  lower=lower, upper=upper, control=optCtrl, adj=False, rho=rho)
    if nAGQ > 0:
        pb = len(rho.pp.beta0)
        lower = lwr if lwr is not None else np.concatenate([lower, np.full(pb, -np.inf)])
        upper = upr if upr is not None else np.concatenate([upper, np.full(pb, np.inf)])
        rho.dpars = np.arange(len(lower) - pb)
        rho.u0 = np.asarray(rho.pp.u0).copy()
        nd = len(rho.dpars)
        rho.beta0 = np.minimum(upper[nd:], np.maximum(np.asarray(rho.pp.beta0), lower[nd:]))
        if nAGQ > 1:
            if len(reTrms["flist"]) != 1 or len(reTrms["cnms"][0][1]) != 1:
                raise ValueError("nAGQ > 1 is only available for models with a single, "
                                 "scalar random-effects term")
            rho.fac = list(reTrms["flist"].values())[0]
        rho.lower, rho.upper = lower, upper
        devfun = mkdevfun(rho, nAGQ, verbose=verbose, control=control)
        if devFunOnly:
            return devfun
        opt = optwrap(control["optimizer"][1], devfun,
                      np.concatenate([rho.mkPar(np.asarray(rho.pp.theta)), rho.beta0]),
                      lower=lower, upper=upper, control=optCtrl, adj=True, verbose=verbose,
                      rho=rho)
    return mkMerMod(rho, opt, reTrms, fr=vals["frame"], mc=mc)
