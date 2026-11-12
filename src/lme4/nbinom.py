"""Negative binomial GLMMs: transcription of lme4/R/nbinom.R
(``glmer.nb``, ``getNBdisp``, ``setNBdisp``, ``refitNB``, ``optTheta``,
``est_theta``) together with the pieces of other packages it relies on:

* ``MASS::theta.ml``  (:func:`theta_ml`)
* ``stats::optimize`` with R's C implementation of Brent's ``fmin``
  (``src/library/stats/src/optimize.c``)  (:func:`optimize`, :func:`Brent_fmin`)
* the GLMM branch of ``refit.merMod`` (lme4/R/lmer.R), needed by
  ``refitNB`` (:func:`refit_glmerMod`)
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
import sys
import warnings

import numpy as np
from scipy.special import digamma, polygamma

from .control import glmerControl, glmerControlObj
from .covariance import mkMkPar, mkMkTheta
from .family import negative_binomial, poisson, isNBfamily
from .modular import Rho, mkRespMod, mkdevfun, glmerPwrssUpdate
from .optimizers import optwrap, checkConv, GHrule

DBL_EPSILON = np.finfo(float).eps
DBL_MAX = np.finfo(float).max


# ---------------------------------------------------------------------------
# stats::optimize  (Brent_fmin from R's src/library/stats/src/optimize.c)
# ---------------------------------------------------------------------------

def Brent_fmin(ax, bx, f, tol):
    """R's ``Brent_fmin()``: one-dimensional minimisation of ``f`` on
    ``[ax, bx]`` by golden-section search + successive parabolic
    interpolation (Brent 1973).  Literal transcription of the C code."""
    # c is the squared inverse of the golden ratio
    c = (3. - math.sqrt(5.)) * .5
    # eps is approximately the square root of the relative machine precision.
    eps = DBL_EPSILON
    tol1 = eps + 1.          # the smallest 1.000... > 1
    eps = math.sqrt(eps)

    a = ax
    b = bx
    v = a + c * (b - a)
    w = v
    x = v

    d = 0.
    e = 0.
    fx = f(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.

    # main loop starts here -----------------------------------
    while True:
        xm = (a + b) * .5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.

        # check stopping criterion
        if abs(x - xm) <= t2 - (b - a) * .5:
            break
        p = 0.
        q = 0.
        r = 0.
        if abs(e) > tol1:    # fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.
            if q > 0.:
                p = -p
            else:
                q = -q
            r = e
            e = d

        if abs(p) >= abs(q * .5 * r) or p <= q * (a - x) or p >= q * (b - x):
            # a golden-section step
            if x < xm:
                e = b - x
            else:
                e = a - x
            d = c * e
        else:
            # a parabolic-interpolation step
            d = p / q
            u = x + d
            # f must not be evaluated too close to ax or bx
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d

        # f must not be evaluated too close to x
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.:
            u = x + tol1
        else:
            u = x - tol1

        fu = f(u)

        # update  a, b, v, w, and x
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    # end of main loop
    return x


def optimize(f, interval=None, lower=None, upper=None, maximum=False,
             tol=DBL_EPSILON ** 0.25):
    """``stats::optimize()``.  Returns ``dict(minimum=, objective=)`` (or
    ``maximum=`` when ``maximum=True``); as in R, ``f`` is re-evaluated at
    the optimum to obtain the objective."""
    if interval is not None:
        lower = min(interval) if lower is None else lower
        upper = max(interval) if upper is None else upper
    lower, upper = float(lower), float(upper)
    if not (lower < upper):
        raise ValueError("'xmin' not less than 'xmax'")

    def fcn1(x):   # do_fmin's fcn1(): non-finite values replaced by DBL_MAX
        val = float(-f(x) if maximum else f(x))
        if not math.isfinite(val):
            warnings.warn("NA/Inf replaced by maximum positive value")
            val = DBL_MAX
        return val

    val = Brent_fmin(lower, upper, fcn1, float(tol))
    if maximum:
        return dict(maximum=val, objective=f(val))
    return dict(minimum=val, objective=f(val))


# ---------------------------------------------------------------------------
# MASS::theta.ml
# ---------------------------------------------------------------------------

class ThetaML(float):
    """a float carrying R's attributes ``SE`` (and possibly ``warn``)"""
    SE = float("nan")
    warn = None


def theta_ml(y, mu, n=None, weights=None, limit=10, eps=DBL_EPSILON ** 0.25, trace=False):
    """``MASS::theta.ml()``: ML estimate of the negative binomial theta for
    given means ``mu``, by Newton-Raphson with expected information."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)

    def score(n, th, mu, y, w):
        return np.sum(w * (digamma(th + y) - digamma(th) + math.log(th) + 1 -
                           np.log(th + mu) - (y + th) / (mu + th)))

    def info(n, th, mu, y, w):
        return np.sum(w * (-polygamma(1, th + y) + polygamma(1, th) - 1 / th +
                           2 / (mu + th) - (y + th) / (mu + th) ** 2))

    if weights is None:
        weights = np.ones(len(y))
    weights = np.asarray(weights, float)
    if n is None:
        n = np.sum(weights)
    t0 = n / np.sum(weights * (y / mu - 1) ** 2)
    it = 0
    dl = 1
    i = float("nan")
    if trace:
        print(f"theta.ml: iter {it} 'theta = {_signif(t0, 6):f}'", file=sys.stderr)
    while True:
        it += 1
        if not (it < limit and abs(dl) > eps):
            break
        t0 = abs(t0)
        i = info(n, t0, mu, y, weights)
        dl = score(n, t0, mu, y, weights) / i
        t0 = t0 + dl
        if trace:
            print(f"theta.ml: iter{it} theta ={_signif(t0, 6):g}", file=sys.stderr)
    warn = None
    if t0 < 0:
        t0 = 0
        warnings.warn("estimate truncated at zero")
        warn = "estimate truncated at zero"
    if it == limit:
        warnings.warn("iteration limit reached")
        warn = "iteration limit reached"
    out = ThetaML(t0)
    out.SE = math.sqrt(1 / i) if i > 0 else float("nan")
    out.warn = warn
    return out


def _signif(x, digits):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


# ---------------------------------------------------------------------------
# getNBdisp / setNBdisp
# ---------------------------------------------------------------------------

def getNBdisp(object):
    """the negative binomial dispersion parameter theta (``getME(., "glmer.nb.theta")``)"""
    return object.resp.family.theta


def setNBdisp(object, theta):
    """NB-object with changed [DISP]ersion parameter 'theta' (and all that entails)"""
    rr = object.resp
    newresp = rr.copy()
    newresp.setTheta(theta)          # also replaces family by negative.binomial(theta)
    newresp.setOffset(rr.offset)
    newresp.ptr.updateMu(rr.eta - rr.offset)
    obj = copy.copy(object)
    obj.resp = newresp
    return obj


# ---------------------------------------------------------------------------
# refit.merMod, GLMM branch with newresp = NULL (lme4/R/lmer.R)
# ---------------------------------------------------------------------------

def refit_glmerMod(object, control=None, maxit=100, verbose=0):
    """``refit(object, control=control)`` for a ``glmerMod``: rebuild the
    response module from the model frame (with the *current* family of
    ``object@resp``), and re-optimise starting from the current estimates."""
    from .lmer import mkMerMod
    if not object.isGLMM():
        raise ValueError("refit_glmerMod() requires a glmerMod object")
    ctrl_arg = control
    if ctrl_arg is not None:
        control = copy.deepcopy(ctrl_arg)
        if not isinstance(control, glmerControlObj):
            control = glmerControl(**control) if isinstance(control, dict) else control
        if len(control.get("optCtrl") or {}) == 0:   # use object's version
            obj_control = dict(object.optinfo.get("control") or {})
            for k in ("xst", "xt"):
                obj_control.pop(k, None)
            control["optCtrl"] = obj_control
    else:
        control = glmerControl()
    pp = object.pp.copy()
    dc = object.devcomp
    nAGQ = dc["dims"]["nAGQ"]
    nAGQ = 0 if nAGQ is None else int(nAGQ)
    rr = mkRespMod(object.frame, family=object.resp.family)
    dispProfile = (control.get("disp_method") or "moment") == "moment"
    maxPhiIter = int(control.get("maxPhiIter") or 100)
    GQmat = GHrule(nAGQ)
    grpFac = list(object.flist.values())[0]
    if nAGQ <= 1:
        glmerPwrssUpdate(pp, rr, control["tolPwrss"], GQmat, maxit=maxit,
                         dispProfile=dispProfile, maxPhiIter=maxPhiIter)
    else:
        glmerPwrssUpdate(pp, rr, control["tolPwrss"], GQmat, maxit=maxit, grpFac=grpFac,
                         dispProfile=dispProfile, maxPhiIter=maxPhiIter)
    reCovs = object.getReCovs()
    dims = dc["dims"]
    qEff = dims.get("qEff")
    rho = Rho(pp=pp, resp=rr, mkPar=mkMkPar(reCovs), mkTheta=mkMkTheta(reCovs),
              reCovs=reCovs,
              tolPwrss=dc["cmp"]["tolPwrss"], compDev=dims["compDev"],
              dispProfile=True if dims.get("dispProfile") is None else bool(dims["dispProfile"]),
              maxPhiIter=100 if dims.get("maxPhiIter") is None else int(dims["maxPhiIter"]),
              qEff=float("nan") if qEff is None else float(qEff),
              nAGQ=nAGQ, lp0=np.asarray(pp.linPred(1.0)),
              baseOffset=np.array(object.resp.offset, copy=True),
              GQmat=GHrule(nAGQ), fac=grpFac, verbose=verbose,
              dpars=np.arange(object.getParLength()))
    ff = mkdevfun(rho, nAGQ=nAGQ, maxit=maxit, verbose=verbose)
    x0 = rho.mkPar(np.asarray(pp.theta))
    lower = np.asarray(object.lower, float)
    upper = np.asarray(object.getUpper(), float)
    rho.lower, rho.upper = lower, upper
    if nAGQ > 0:
        n_beta = len(object.beta)
        x0 = np.concatenate([x0, np.asarray(object.beta, float)])
        lower = np.concatenate([lower, np.full(n_beta, -np.inf)])
        upper = np.concatenate([upper, np.full(n_beta, np.inf)])
        rho.lower, rho.upper = lower, upper
    calc_derivs = object.optinfo.get("derivs") is not None
    optimizer = object.optinfo.get("optimizer")
    newopt = ctrl_arg.get("optimizer") if ctrl_arg is not None else None
    if newopt is not None:
        # we might end up with a length-2 optimizer vector ... use the *last* element
        optimizer = newopt[-1] if isinstance(newopt, (list, tuple)) else newopt
    opt = optwrap(optimizer, ff, x0, lower=lower, upper=upper, control=control["optCtrl"],
                  calc_derivs=calc_derivs)
    cc = checkConv(opt.attrs.get("derivs"), opt["par"], ctrl=control["checkConv"],
                   lbound=lower, ubound=upper)
    rr.setOffset(rho.baseOffset)
    reTrms = dict(flist=object.flist, flist_assign=object.flist_assign, cnms=object.cnms,
                  Gp=object.Gp, lower=object.lower, upper=object.getUpper(), reCovs=reCovs)
    return mkMerMod(rho, opt, reTrms, fr=object.frame, mc=object.call, lme4conv=cc,
                    wmsgs=getattr(object, "wmsgs", None))


def refitNB(object, theta, control=None):
    return refit_glmerMod(setNBdisp(object, theta), control=control)


# ---------------------------------------------------------------------------
# optTheta
# ---------------------------------------------------------------------------

def optTheta(object, interval=(-5, 5), tol=DBL_EPSILON ** 0.25, verbose=False, control=None):
    """Optimize over the negative binomial parameter theta (on the log scale)
    by ``optimize()``; returns the last fit (an object like ``object``)."""
    state = dict(lastfit=object, it=0, wlist=[])

    def NBfun(t):
        # Kluge to retain last value and evaluation count  (lme4's factory():
        # warnings are recorded with the fit, not signalled)
        # messages (R's message(), printed to stderr here) are captured too
        mbuf = io.StringIO()
        with warnings.catch_warnings(record=True) as wl, contextlib.redirect_stderr(mbuf):
            warnings.simplefilter("always")
            state["lastfit"] = refitNB(state["lastfit"], theta=math.exp(t), control=control)
        state["wlist"] = list(wl)
        state["mlist"] = mbuf.getvalue()
        lastfit = state["lastfit"]
        dev = -2 * lastfit.logLik().value
        state["it"] += 1
        if verbose:
            print("%2d: th=%#15.10g, dev=%#14.8f, beta[1]=%#14.8f" %
                  (state["it"], math.exp(t), dev, lastfit.beta[0]))
        return dev

    optval = optimize(NBfun, interval=interval, tol=tol)
    lastfit = state["lastfit"]
    if not _all_equal(optval["minimum"], math.log(getNBdisp(lastfit))):
        raise AssertionError("all.equal(optval$minimum, log(getNBdisp(lastfit))) is not TRUE")
    lastfit.nevals = state["it"]
    # fix up the 'th' expression, replacing it by the real number
    cl = type(lastfit.call)(lastfit.call.fun, **dict(lastfit.call))
    cl["family"] = negative_binomial(theta=math.exp(optval["minimum"]))
    lastfit.call = cl
    # output warnings/messages from last fit
    for w in state["wlist"]:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if state.get("mlist"):
        print(state["mlist"], end="", file=sys.stderr)
    return lastfit


@contextlib.contextmanager
def _suppressMessages():
    """R's ``suppressMessages()``: discard messages (printed to stderr by this
    package), but let warnings through."""
    with warnings.catch_warnings(record=True) as wl:
        with contextlib.redirect_stderr(io.StringIO()):
            yield
    for w in wl:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _all_equal(target, current, tolerance=1.5e-8):
    xy = abs(target - current)
    xn = abs(target)
    if math.isfinite(xn) and xn > tolerance:
        xy = xy / xn
    return xy <= tolerance


# ---------------------------------------------------------------------------
# est_theta
# ---------------------------------------------------------------------------

def est_theta(object, limit=20, eps=DBL_EPSILON ** 0.25, trace=0):
    """use MASS machinery to estimate theta from residuals"""
    Y = np.asarray(object.frame.response(), float)
    mu = np.asarray(object.fitted(), float)
    mu = mu[~np.isnan(mu)]
    return theta_ml(Y, mu, weights=object.resp.weights, limit=limit, eps=eps, trace=trace)


# ---------------------------------------------------------------------------
# glmer.nb
# ---------------------------------------------------------------------------

def glmer_nb(formula, data=None, *args, interval=None, tol=5e-5, verbose=False,
             nb_control=None, initCtrl=None, _env=None, **kwargs):
    """``glmer.nb()``: fit a negative binomial GLMM, optimizing the
    dispersion parameter theta by profiling (``optimize()`` over log(theta)).

    All further arguments (``control``, ``nAGQ``, ``subset``, ``weights``,
    ``offset``, ``na_action``, ``contrasts``, ``start``, ...) are passed to
    :func:`glmer` (apart from ``family``).  ``interval`` defaults to
    ``log(th) + (-3, 3)`` where ``th`` is the initial theta estimate.
    """
    from .lmer import glmer, _caller_env
    if "family" in kwargs:
        raise TypeError("glmer_nb() does not take a 'family' argument")
    env = _env if _env is not None else _caller_env()
    ic = dict(limit=20, eps=2 * tol, trace=verbose, theta=None)
    if initCtrl is not None:
        ic.update(initCtrl)
    initCtrl = ic
    gverbose = int(verbose >= 2)
    user_control = kwargs.get("control")
    th = initCtrl.get("theta")
    if th is None:
        with _suppressMessages():
            g0 = glmer(formula, data, *args, family=poisson(), verbose=gverbose,
                       _env=env, **kwargs)
        th = est_theta(g0, limit=initCtrl["limit"], eps=initCtrl["eps"],
                       trace=initCtrl["trace"])
        if verbose:
            print("th := est_theta(glmer(..)) =", format(float(th), ".7g"), end="")
    th = float(th)
    if interval is None:
        interval = (math.log(th) - 3, math.log(th) + 3)
    with _suppressMessages():
        g1 = glmer(formula, data, *args, family=negative_binomial(theta=th), verbose=gverbose,
                   _env=env, **kwargs)
    if verbose:
        print(" --> dev.= -2*logLik(.) =", format(-2 * g1.logLik().value, ".7g"))
    # fix the call: 'verbose' / 'control' as given by the user
    cl = type(g1.call)(g1.call.fun, **dict(g1.call))
    cl["verbose"] = 0
    if user_control is None:
        cl["control"] = None
    g1.call = cl
    if data is None:
        warnings.warn("no 'data = *' in glmer.nb() call ... Not much is guaranteed")
    # control = c(control, nb.control): the elements of 'control' take precedence
    if user_control is not None and nb_control is not None:
        ctrl = dict(nb_control)
        ctrl.update(user_control)
        ctrl = type(user_control)(ctrl) if isinstance(user_control, dict) else ctrl
    else:
        ctrl = user_control if user_control is not None else nb_control
    return optTheta(g1, interval=interval, tol=tol, verbose=verbose, control=ctrl)
