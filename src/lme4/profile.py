"""Transcription of lme4/R/profile.R: likelihood profiles of fitted mixed
models (``profile.merMod``), the forward/backward profile splines,
``confint.thpr`` and ``confint.merMod`` (methods "profile", "Wald", "boot"),
plus ``logProf``, ``varianceProf`` and ``as.data.frame.thpr``.

The lattice plotting methods (xyplot/splom/densityplot) are not transcribed.

The interpolation-spline machinery of R's *splines* package
(``interpSpline``, ``backSpline``, ``predict.polySpline`` /
``predict.npolySpline``) used by the profile code is re-implemented here.

Index conventions: integer ``which`` / ``parm`` arguments are **0-based**
(Python convention); R's 1-based indices are used only internally.
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ._backend import cpp
from .covariance import mkMkPar, mkMkTheta, anyStructured
from .optimizers import optwrap

__all__ = ["profile_merMod", "confint_merMod", "confint_thpr", "thpr", "logProf",
           "varianceProf", "profnames", "get_which", "devfun2", "convParToProfPar",
           "convProfParToPar", "getProfPar_merMod", "interpSpline", "backSpline",
           "polySpline", "predy", "isMonotoneSpline", "profSplines"]


# ---------------------------------------------------------------------------
# splines (R package 'splines'): polySpline / npolySpline, interpSpline,
# backSpline, predict
# ---------------------------------------------------------------------------

class polySpline:
    """piecewise cubic polynomial spline in the 'polySpline' representation:
    row ``i`` of ``coefficients`` holds c0 + c1*d + c2*d^2 + c3*d^3 with
    d = x - knots[i].  ``natural=True`` corresponds to R's class
    ``npolySpline`` (linear extrapolation outside the knots); otherwise
    predictions outside the knot range are NA (as in predict.polySpline,
    which uses ``cut(x, knots)``)."""

    def __init__(self, knots, coefficients, xname="x", yname="y", natural=False):
        self.knots = np.asarray(knots, float)
        self.coefficients = np.asarray(coefficients, float)
        self.xname = xname
        self.yname = yname
        self.natural = bool(natural)

    @property
    def cls(self):
        return "npolySpline" if self.natural else "polySpline"

    def predict(self, x, deriv=0):
        """predict.polySpline / predict.npolySpline: returns the y values"""
        x = np.atleast_1d(np.asarray(x, float))
        kn = self.knots
        coeff = self.coefficients
        nk = len(kn)
        ord_ = coeff.shape[1]
        deriv = int(deriv)
        if deriv < 0 or deriv >= ord_:
            raise ValueError(f"'deriv' must be between 0 and {ord_ - 1}")
        # i <- as.numeric(cut(x, knots)); i[x == knots[1]] <- 1
        idx = np.searchsorted(kn, x, side="left")      # kn[idx-1] < x <= kn[idx]
        i = idx - 1
        i = np.where(x == kn[0], 0, i)
        ok = (i >= 0) & (i <= nk - 2) & ~np.isnan(x)
        cf = coeff.copy()
        o = ord_
        d = deriv
        while d > 0:
            o -= 1
            cf = cf[:, 1:] * np.arange(1, o + 1)[None, :]
            d -= 1
        y = np.full(len(x), np.nan)
        if ok.any():
            ii = i[ok]
            delx = x[ok] - kn[ii]
            yy = cf[ii, o - 1].copy()
            for j in range(o - 2, -1, -1):
                yy = yy * delx + cf[ii, j]
            y[ok] = yy
        if self.natural and np.any(np.isnan(y)):
            lower = x < kn[0]
            upper = x > kn[-1]
            if deriv == 0:
                y[lower] = coeff[0, 0] + coeff[0, 1] * (x[lower] - kn[0])
                y[upper] = coeff[-1, 0] + coeff[-1, 1] * (x[upper] - kn[-1])
            elif deriv == 1:
                y[lower] = coeff[0, 1]
                y[upper] = coeff[-1, 1]
            else:
                y[lower] = 0.0
                y[upper] = 0.0
        return y

    def __repr__(self):
        return (f"{self.cls}: {self.yname} ~ {self.xname}, {len(self.knots)} knots "
                f"[{self.knots[0]:g}, {self.knots[-1]:g}]")


class SplineError(Exception):
    """stands in for an R 'error' object stored in place of a spline"""


def interpSpline(x, y, xname="x", yname="y"):
    """splines::interpSpline: the natural cubic interpolation spline through
    (x, y), returned in the 'npolySpline' representation.  NA pairs are
    omitted (``na.action = na.omit``)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    n = len(x)
    if n > 1 and np.any(np.diff(x) == 0):
        raise SplineError("values of 'x' must be distinct")
    if n < 4:
        # interpSpline() needs x[4] to construct its boundary knots
        raise SplineError("interpSpline() needs at least 4 distinct points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise SplineError("non-finite values in interpSpline()")
    h = np.diff(x)
    s = np.diff(y) / h
    # second derivatives M, natural end conditions M[0] = M[n-1] = 0
    M = np.zeros(n)
    m = n - 2
    A = np.zeros((m, m))
    rhs = np.zeros(m)
    for k in range(m):
        i = k + 1
        A[k, k] = 2 * (h[i - 1] + h[i])
        if k > 0:
            A[k, k - 1] = h[i - 1]
        if k < m - 1:
            A[k, k + 1] = h[i]
        rhs[k] = 6 * (s[i] - s[i - 1])
    M[1:-1] = np.linalg.solve(A, rhs)
    coeff = np.zeros((n, 4))
    coeff[:, 0] = y
    coeff[:-1, 1] = s - h * (2 * M[:-1] + M[1:]) / 6
    coeff[:-1, 2] = M[:-1] / 2
    coeff[:-1, 3] = (M[1:] - M[:-1]) / (6 * h)
    coeff[-1, 1] = s[-1] + h[-1] * (M[-2] + 2 * M[-1]) / 6
    coeff[-1, 2] = M[-1] / 2
    coeff[-1, 3] = coeff[-2, 3]
    return polySpline(x, coeff, xname=xname, yname=yname, natural=True)


def backSpline(sp):
    """splines::backSpline (backSpline.npolySpline): the inverse of a monotone
    cubic spline.  Interior intervals are cubic Hermite pieces through
    (y_i, x_i) with slopes 1/f'(x_i); the first and last intervals are
    quadratic (natural-spline ends), with the curvature clipped to zero if it
    has the wrong sign.  Only the values at the knots are checked for
    monotonicity."""
    if not isinstance(sp, polySpline):
        raise SplineError("backSpline() needs a spline")
    knots = sp.knots
    nk = len(knots)
    nkm1 = nk - 1
    kdiff = np.diff(knots)
    if np.any(kdiff <= 0):
        raise SplineError("knot positions must be strictly increasing")
    coeff = sp.coefficients
    if coeff.shape[1] != 4:
        raise SplineError("currently implemented only for cubic splines")
    bknots = coeff[:, 0]
    adiff = np.diff(bknots)
    if np.all(adiff < 0):
        revKnots = True
    elif np.all(adiff > 0):
        revKnots = False
    else:
        raise SplineError("spline must be monotone")
    bcoeff = np.zeros(coeff.shape)
    bcoeff[:, 0] = knots
    with np.errstate(divide="ignore"):
        bcoeff[:, 1] = 1.0 / coeff[:, 1]
    for i in range(nkm1):
        a = np.array([[adiff[i] ** 2, adiff[i] ** 3], [2 * adiff[i], 3 * adiff[i] ** 2]])
        b = np.array([kdiff[i] - adiff[i] * bcoeff[i, 1], bcoeff[i + 1, 1] - bcoeff[i, 1]])
        bcoeff[i, 2:4] = np.linalg.solve(a, b)
    bcoeff[nk - 1, 1:4] = np.nan
    if nk > 2:
        bcoeff[0, 3] = bcoeff[nkm1 - 1, 3] = 0.0
        bcoeff[0, 1:3] = np.linalg.solve(np.array([[adiff[0], adiff[0] ** 2], [1.0, 2 * adiff[0]]]),
                                         np.array([kdiff[0], 1.0 / coeff[1, 1]]))
        bcoeff[nkm1 - 1, 2] = ((kdiff[nkm1 - 1] - adiff[nkm1 - 1] * bcoeff[nkm1 - 1, 1])
                               / adiff[nkm1 - 1] ** 2)
    if bcoeff[0, 2] > 0:
        bcoeff[0, 2] = 0.0
        bcoeff[0, 1] = kdiff[0] / adiff[0]
    if bcoeff[nkm1 - 1, 2] < 0:
        bcoeff[nkm1 - 1, 2] = 0.0
        bcoeff[nkm1 - 1, 1] = kdiff[nkm1 - 1] / adiff[nkm1 - 1]
    if revKnots:
        bknots = bknots[::-1]
        bcoeff = bcoeff[::-1].copy()
    return polySpline(bknots, bcoeff, xname=sp.yname, yname=sp.xname, natural=False)


def predy(sp, vv):
    """extract only the y component from a prediction (NA for a failed spline)"""
    vv = np.atleast_1d(np.asarray(vv, float))
    if sp is None or isinstance(sp, Exception):
        return np.full(len(vv), np.nan)
    return sp.predict(vv)


def splineDerivRange(sp):
    """exact range of the first derivative of a cubic polySpline on each
    interval between knots"""
    kn = sp.knots
    cf = sp.coefficients
    h = np.diff(kn)
    nint = len(h)
    c1 = cf[:nint, 1]
    c2 = cf[:nint, 2]
    c3 = cf[:nint, 3]
    qh = c1 + 2 * c2 * h + 3 * c3 * h ** 2
    lo = np.minimum(c1, qh)
    hi = np.maximum(c1, qh)
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = -c2 / (3 * c3)
        inside = (c3 != 0) & (ds > 0) & (ds < h)
        qs = (c1 - c2 ** 2 / (3 * c3))
    lo[inside] = np.minimum(lo[inside], qs[inside])
    hi[inside] = np.maximum(hi[inside], qs[inside])
    return lo, hi


def isMonotoneSpline(sp):
    if sp is None or isinstance(sp, Exception):
        return False
    lo, hi = splineDerivRange(sp)
    return bool(np.all(lo >= 0) or np.all(hi <= 0))


def monoHermiteSpline(x, y, xname="x", yname="y"):
    """monotone cubic Hermite interpolant (Fritsch-Butland / PCHIP slopes)
    through strictly monotone (x, y), as a polySpline"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    n = len(x)
    h = np.diff(x)
    s = np.diff(y) / h
    m = np.zeros(n)
    m[0] = s[0]
    m[-1] = s[-1]
    if n > 2:
        k = np.arange(1, n - 1)
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        m[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])
    i = np.arange(n - 1)
    coeff = np.column_stack([y, m,
                             np.append((3 * s - 2 * m[i] - m[i + 1]) / h, 0.0),
                             np.append((m[i] + m[i + 1] - 2 * s) / h ** 2, 0.0)])
    return polySpline(x, coeff, xname=xname, yname=yname, natural=False)


def profSplines(data, pname):
    """forward (zeta as a function of the parameter) and backward (inverse)
    splines for one profiled parameter (with the monotone Hermite fallback
    of GH #834).  ``data`` is a DataFrame with columns ``pname`` and '.zeta'."""
    forspl = None
    try:
        forspl = interpSpline(data[pname].values, data[".zeta"].values, xname=pname,
                              yname=".zeta")
        bakspl = backSpline(forspl)
    except Exception as e:   # tryCatch(..., error = function(e) e)
        bakspl = e if isinstance(e, SplineError) else SplineError(str(e))
    if not (isMonotoneSpline(forspl) and isMonotoneSpline(bakspl)):
        d = data[[pname, ".zeta"]].dropna()
        d = d.sort_values(pname, kind="stable")
        dz = np.diff(d[".zeta"].values)
        if len(d) >= 2 and not d[pname].duplicated().any() and (np.all(dz > 0) or np.all(dz < 0)):
            forspl = monoHermiteSpline(d[pname].values, d[".zeta"].values, xname=pname,
                                       yname=".zeta")
            bakspl = monoHermiteSpline(d[".zeta"].values, d[pname].values, xname=".zeta",
                                       yname=pname)
    return forspl, bakspl


# ---------------------------------------------------------------------------
# profile-scale parameter conversions (lme4/R/covariance.R)
# ---------------------------------------------------------------------------

def _validProfScale(profscale):
    if profscale is None:
        return "sdcor"
    for v in ("sdcor", "varcov"):
        if v.startswith(profscale):
            return v
    raise ValueError("'profscale' should be one of \"sdcor\", \"varcov\"")


def getProfPar_merMod(obj, profscale="sdcor", sc=None):
    """getProfPar(<merMod>)"""
    profscale = _validProfScale(profscale)
    L = [np.asarray(c.getProfPar(profscale, sc), float) for c in obj.getReCovs()]
    if sc is not None:
        L.append(np.array([sc ** 2 if profscale == "varcov" else sc], float))
    return np.concatenate(L) if L else np.zeros(0)


def convParToProfPar(value, obj, profscale="sdcor", sc=None):
    """convert 'par' (covariance parameters) to the profiling scale"""
    profscale = _validProfScale(profscale)
    value = np.asarray(value, float)
    L = []
    pos = 0
    for c in obj.getReCovs():
        npar = c.getParLength()
        L.append(np.asarray(c.setPar(value, pos).getProfPar(profscale, sc), float))
        pos += npar
    if sc is not None:
        L.append(np.array([sc ** 2 if profscale == "varcov" else sc], float))
    return np.concatenate(L) if L else np.zeros(0)


def convProfParToPar(value, obj, profscale="sdcor", sc=None):
    """convert profiling-scale parameters back to 'par'"""
    profscale = _validProfScale(profscale)
    value = np.asarray(value, float)
    L = []
    pos = 0
    for c in obj.getReCovs():
        npar = c.getParLength()
        if npar > len(value) - pos:
            raise ValueError("attempt to read past end of 'value'")
        L.append(np.asarray(c.setProfPar(value, profscale, sc, pos).getPar(), float))
        pos += npar
    return np.concatenate(L) if L else np.zeros(0)


# ---------------------------------------------------------------------------
# names and 'which'
# ---------------------------------------------------------------------------

def profnames(obj, signames=True, useSc=None, prefix=("sd", "cor")):
    if useSc is None:
        useSc = obj.isLMM()
    if signames:
        res = [".sig%02d" % (i + 1) for i in range(obj.getParLength())]
    else:
        res = []
        for c, (g, cn) in zip(obj.getReCovs(), obj.cnms):
            v, cc = c.getVCNames(cn, g, list(prefix), False)
            res += list(v) + list(cc)
    if useSc:
        res.append(".sigma" if signames else "sigma")
    return res


def get_which(which, nvp, nptot, parnames, verbose=False):
    """Transform 'which' (names, "theta_", "beta_" or 0-based integers) into
    **1-based** integer indices (as R's get.which)."""
    if which is None:
        return list(range(1, nptot + 1))
    if isinstance(which, str):
        which = [which]
    which = list(np.atleast_1d(which)) if not isinstance(which, list) else which
    if len(which) and all(isinstance(w, str) for w in which):
        wi = []
        wh = list(which)
        if "theta_" in wh:
            wi = list(range(1, nvp + 1))
            wh = [w for w in wh if w != "theta_"]
        if "beta_" in wh and nptot > nvp:
            wi = wi + list(range(nvp + 1, nptot + 1))
            wh = [w for w in wh if w != "beta_"]
        j = [k + 1 for k, nm in enumerate(parnames) if nm in wh]
        if j:
            wi = sorted(set(wi) | set(j))
        if verbose:
            print(f"From original which = {which}: new which <- {wi}", file=sys.stderr)
        if len(wi) == 0:
            warnings.warn(f"Nothing selected by 'which={which}'")
        return wi
    return [int(w) + 1 for w in which]


# ---------------------------------------------------------------------------
# devfun2
# ---------------------------------------------------------------------------

class _DevFun2:
    """deviance function on the (sd/cor or var/cov) profiling scale; R's
    devfun2() closure with its attributes"""

    def __init__(self, f, optimum, basedev, paropt, stderr, pp, n):
        self.f = f
        self.optimum = optimum        # pd.Series (named)
        self.basedev = basedev
        self.paropt = paropt
        self.stderr = stderr
        self.pp = pp
        self.n = n

    def __call__(self, pars):
        return self.f(np.asarray(pars, float))


def devfun2(fm, useSc=None, signames=True, prefix=("sd", "cor"), scale="sdcor"):
    """The deviance profiled with respect to the fixed-effects parameters but
    not with respect to sigma; the other parameters are on the standard
    deviation (or variance) scale, not the theta scale."""
    scale = _validProfScale(scale)
    if useSc is None:
        useSc = True if fm.isLMM() else None
    if scale == "varcov" and anyStructured(fm.getReCovs()):
        raise ValueError("haven't thought about varcov scale for structured cov matrices")
    fm = fm.refitML()
    mkTheta = mkMkTheta(fm.getReCovs())
    basedev = -2 * fm.logLik().value
    hasSc = bool(fm.devcomp["dims"]["useSc"])
    stdErr = np.sqrt(np.diag(fm.vcov().values))
    pp = fm.pp.copy()
    sig = fm.sigma() if useSc else None
    opt = getProfPar_merMod(fm, profscale=scale, sc=sig)
    names = profnames(fm, signames=signames, useSc=bool(useSc), prefix=prefix)
    fe = fm.fixef()
    opt = pd.Series(np.concatenate([opt, fe.values]), index=names + list(fe.index))
    resp = fm.resp.copy()
    np_ = len(pp.theta)
    nf = len(fe)
    if hasSc:
        np_ += 1
    n = pp.V.shape[0]
    if fm.isLMM():
        ldW = float(np.sum(np.log(resp.weights)))
        log2pi = math.log(2 * math.pi)

        def ans(pars):
            ppl = fm.pp.copy()       # fresh copy of predictor object
            sig = pars[-1]
            try:
                thpars = convProfParToPar(pars, fm, profscale=scale, sc=sig)
            except Exception:
                return np.nan
            cpp.lmer_Deviance(ppl.ptr, resp.ptr, mkTheta(thpars))
            sigsq = sig if scale == "varcov" else sig ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                return (ppl.ldL2() - ldW + (resp.wrss() + ppl.sqrL(1.0)) / sigsq +
                        n * (log2pi + np.log(sigsq)))
    else:
        d0 = fm.as_function()
        ntheta = len(pp.theta) + (1 if hasSc else 0)

        def ans(pars):
            if len(pars) != ntheta + nf:
                raise ValueError("is.numeric(pars) && length(pars) == np + nf is not TRUE")
            try:
                thpars = convProfParToPar(pars, fm, profscale=scale, sc=None)
            except Exception:
                return np.nan
            fixpars = pars[ntheta:]
            return d0(np.concatenate([thpars, fixpars]))
    paropt = np.asarray(fm.optinfo["val"], float)[:len(fm.lower)]
    return _DevFun2(ans, opt, basedev, paropt, stdErr, pp, n)


def _unique_rows(mat):
    """unique(<matrix>) -- rows compared as R does (15 significant digits)"""
    seen = set()
    keep = []
    for i, row in enumerate(mat):
        key = tuple("NA" if np.isnan(v) else "%.15g" % v for v in row)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return mat[keep]


# ---------------------------------------------------------------------------
# the profile object ('thpr')
# ---------------------------------------------------------------------------

class thpr:
    """result of profile(<merMod>) (R class 'thpr'): a data frame of
    profile points plus forward/backward splines and parameter bounds.

    ``frame`` has columns '.zeta', the parameter names and '.par'."""

    def __init__(self, frame, forward, backward, lower=None, upper=None):
        self.frame = frame
        self.forward = forward
        self.backward = backward
        self.lower = lower
        self.upper = upper

    # data-frame like access
    def __getitem__(self, k):
        return self.frame[k]

    def __len__(self):
        return len(self.frame)

    @property
    def columns(self):
        return self.frame.columns

    def __repr__(self):
        return repr(self.frame)

    def copy(self):
        return thpr(self.frame.copy(), dict(self.forward), dict(self.backward),
                    None if self.lower is None else self.lower.copy(),
                    None if self.upper is None else self.upper.copy())

    def confint(self, parm=None, level=0.95, zeta=None, non_mono_tol=1e-2):
        return confint_thpr(self, parm=parm, level=level, zeta=zeta, non_mono_tol=non_mono_tol)

    def as_data_frame(self):
        """as.data.frame.thpr: adds a '.focal' column (the value of the
        profiled parameter in each row)"""
        x = self.frame.copy()
        par = x[".par"].astype(str).values
        vals = np.array([x[p].iloc[i] for i, p in enumerate(par)], float) if len(x) else np.zeros(0)
        x[".focal"] = vals
        x[".par"] = pd.Categorical(par, categories=list(dict.fromkeys(par)))
        return x

    def logProf(self, base=math.e, ranef=True, sigIni=None):
        return logProf(self, base=base, ranef=ranef, sigIni=sigIni)

    def varianceProf(self, ranef=True):
        return varianceProf(self, ranef=ranef)

    # lattice methods (matplotlib versions in lme4.plots_profile)
    def xyplot(self, **kw):
        """xyplot.thpr: profile zeta plots"""
        from .plots_profile import xyplot_thpr
        return xyplot_thpr(self, **kw)

    def densityplot(self, **kw):
        """densityplot.thpr: densities implied by the profiles"""
        from .plots_profile import densityplot_thpr
        return densityplot_thpr(self, **kw)

    def splom(self, **kw):
        """splom.thpr: profile pairs plot"""
        from .plots_profile import splom_thpr
        return splom_thpr(self, **kw)


def _format_perc(a, digits=3):
    from .rformat import r_format
    return [f"{s.strip()} %" for s in r_format(list(100 * np.asarray(a, float)), digits=digits)]


def confint_thpr(obj, parm=None, level=0.95, zeta=None, non_mono_tol=1e-2):
    """confint() method for profile() results"""
    bak = obj.backward
    bnms = list(bak.keys())

    def nameBnds(b):
        v = pd.Series(np.nan, index=bnms, dtype=float)
        if b is not None:
            b = np.asarray(b, float)
            v.iloc[:len(b)] = b
        return v
    lower = nameBnds(obj.lower)
    upper = nameBnds(obj.upper)
    if parm is None:
        parm = bnms
    elif isinstance(parm, (int, np.integer)) or (
            not isinstance(parm, str) and len(parm) and all(isinstance(p, (int, np.integer)) for p in parm)):
        parm = [bnms[int(i)] for i in np.atleast_1d(parm)]
    elif isinstance(parm, str) or len(parm) == 1:
        p = parm if isinstance(parm, str) else parm[0]
        import re
        pat = re.compile(r"^(sd_|cor_|.sig|sigma$)")
        if p == "theta_":
            parm = [b for b in bnms if pat.search(b)]
        elif p == "beta_":
            parm = [b for b in bnms if not pat.search(b)]
        elif p in bnms:
            parm = [p]
        else:
            parm = []
    else:
        parm = [p for p in parm if p in bnms]
    if zeta is None:
        a = (1 - level) / 2
        a = np.array([a, 1 - a])
        zeta = stats.norm.ppf(a)
        cn = _format_perc(a, 3)
    else:
        zeta = np.asarray(zeta, float)
        cn = None
    ci = np.full((len(parm), 2), np.nan)
    for i, nm in enumerate(parm):
        badprof = False
        p = np.full(2, np.nan)
        b = bak.get(nm)
        if not isinstance(b, Exception):
            p = predy(b, zeta)
        else:
            fr = obj.frame
            obj1 = fr.loc[fr[".par"] == nm, [nm, ".zeta"]]
            z1 = obj1[".zeta"].values.astype(float)
            if np.all(np.isnan(z1)):
                badprof = True
                warnings.warn(f"bad profile for {nm}")
            elif _all_true_narm(np.diff(z1) < -non_mono_tol, np.diff(z1)):
                badprof = True
                warnings.warn(f"non-monotonic profile for {nm}")
            else:
                warnings.warn(f"bad spline fit for {nm}: falling back to linear interpolation")
                p = _approxfun(z1, obj1[nm].values.astype(float), zeta)
        if not badprof:
            if np.isnan(p[0]):
                p[0] = lower[nm]
            if np.isnan(p[1]):
                p[1] = upper[nm]
        ci[i] = p[:2]
    return pd.DataFrame(ci, index=parm, columns=cn if cn is not None else None)


def _all_true_narm(cond, d):
    """R's  min(<logical>, na.rm=TRUE)  (TRUE iff all non-NA are TRUE)"""
    cond = np.asarray(cond)[~np.isnan(d)]
    if len(cond) == 0:
        return True   # min(numeric(0)) is Inf -> TRUE
    return bool(np.all(cond))


def _approxfun(x, y, xout):
    """stats::approxfun(x, y)(xout) -- linear, rule = 1 (NA outside),
    ties = mean"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.full(len(np.atleast_1d(xout)), np.nan)
    df = pd.DataFrame({"x": x, "y": y}).groupby("x", sort=True)["y"].mean()
    xs, ys = df.index.values, df.values
    xout = np.atleast_1d(np.asarray(xout, float))
    if len(xs) == 1:
        return np.where(xout == xs[0], ys[0], np.nan)
    out = np.interp(xout, xs, ys)
    out[(xout < xs[0]) | (xout > xs[-1]) | np.isnan(xout)] = np.nan
    return out


# ---------------------------------------------------------------------------
# profile.merMod
# ---------------------------------------------------------------------------

def profile_merMod(obj, which=None, alphamax=0.01, maxpts=100, delta=None, delta_cutoff=1 / 8,
                   verbose=0, devtol=1e-9, maxmult=10, startmethod="prev", optimizer=None,
                   control=None, signames=True, profscale="sdcor", prof_scale=None,
                   devmatchtol=1e-5):
    """Profile the deviance of a fitted mixed model (profile.merMod).

    ``which``: None (all parameters), parameter names, "theta_", "beta_" or
    0-based integer positions in c(variance parameters [, sigma], fixed effects).
    Returns a :class:`thpr` object."""
    fitted = obj
    prof_scale = _validProfScale(prof_scale if prof_scale is not None else profscale)
    oi = fitted.optinfo or {}
    if optimizer is None:
        optimizer = oi.get("optimizer")
    ignore_pars = ("xst", "xt")
    control_internal = dict(oi.get("control") or {}) if optimizer == oi.get("optimizer") else {}
    for k in ignore_pars:
        control_internal.pop(k, None)
    if control is not None:
        control_internal.update(control)
    control = control_internal
    useSc = fitted.isLMM() or fitted.isNLMM()
    prof_prefix = {"sdcor": ("sd", "cor"), "varcov": ("var", "cov")}[prof_scale]
    if fitted.isGLMM() and fitted.devcomp["dims"]["useSc"]:
        raise ValueError("can't (yet) profile GLMMs with non-fixed scale parameters")
    if fitted.isNLMM():
        raise NotImplementedError("profiling nlmer fits is not implemented in Python")
    dd = devfun2(fitted, useSc=useSc, signames=signames, prefix=prof_prefix, scale=prof_scale)
    if devtol < 0:
        raise ValueError("devtol >= 0 is not TRUE")
    base = dd.basedev
    paropt = np.array(dd.paropt, copy=True)
    stderr = dd.stderr
    pp = dd.pp
    X_orig = np.asarray(pp.X)
    n = dd.n
    p = len(pp.beta0)
    opt = dd.optimum
    optv = opt.values.astype(float)
    optnames = list(opt.index)
    nptot = len(opt)
    nvp = nptot - p
    which = get_which(which, nvp, nptot, optnames, verbose)

    ncol = 1 + nptot
    res = np.tile(np.concatenate([[0.0], optv]), (maxpts, 1))
    cutoff = math.sqrt(stats.chi2.ppf(1 - alphamax, nptot))
    if delta is None:
        delta = cutoff * delta_cutoff
    isLMM = fitted.isLMM()

    # ---- helper functions (0-based: r = row index, cc = column index)
    def nextpar(mat, cc, r, absstep, lower=-np.inf, upper=np.inf, minstep=1e-6):
        pvals = mat[[r - 1, r], cc]
        zeta = mat[[r - 1, r], 0]
        num = pvals[1] - pvals[0]
        denom = zeta[1] - zeta[0]
        if np.isnan(denom) or denom == 0:
            warnings.warn("Last two rows have identical or NA .zeta values: using minstep")
            step = minstep
        else:
            step = absstep * num / denom
            if step < 0:
                warnings.warn("unexpected decrease in profile: using minstep")
                step = minstep
            elif r > 0:
                maxstep = abs(maxmult * num)
                if abs(step) > maxstep:
                    maxstep = np.sign(step) * maxstep
                    if verbose:
                        print("capped step at %1.2f (multiplier=%1.2f > %1.2f)" %
                              (maxstep, abs(step / num), maxmult))
                    step = maxstep
        return min(upper, max(lower, pvals[1] + np.sign(num) * step))

    npar1 = nvp if isLMM else nptot
    seqpar1 = np.arange(npar1)

    def nextstart(mat, pind, r, method):
        if method == "global":
            v = optv[:npar1]
        elif method == "prev":
            v = mat[r, 1:1 + npar1]
        elif method == "extrap":
            raise NotImplementedError("not yet implemented")
        else:
            raise ValueError("invalid nextstart method")
        return np.delete(v, pind) if pind < npar1 else v.copy()

    def mkpar(np_, w, pw, pmw):
        par = np.zeros(np_)
        par[w] = pw
        par[np.arange(np_) != w] = pmw
        return par

    def fillmat(mat, lowcut, upcut, zetafun, cc):
        nr = mat.shape[0]
        i = 1
        while (i < nr - 1 and not np.isnan(mat[i, cc]) and mat[i, cc] > lowcut and
               mat[i, cc] < upcut and not np.isnan(mat[i, 0]) and abs(mat[i, 0]) <= cutoff):
            np_ = nextpar(mat, cc, i, delta, lowcut, upcut)
            ns = nextstart(mat, pind=cc - 1, r=i, method=startmethod)
            mat[i + 1, :] = zetafun(np_, ns)
            if verbose > 0:
                print(i + 1, cc + 1, mat[i + 1, :])
            i += 1
        return mat[~np.isnan(mat[:, 0])]

    # bounds
    covs = fitted.getReCovs()
    lower = np.concatenate([np.asarray(c.getProfLower(prof_scale), float) for c in covs])
    upper = np.concatenate([np.asarray(c.getProfUpper(prof_scale), float) for c in covs])
    if useSc:
        lower = np.append(lower, 0.0)
        upper = np.append(upper, np.inf)
    lower = np.concatenate([lower, np.full(p, -np.inf)])
    upper = np.concatenate([upper, np.full(p, np.inf)])
    # check that devfun2() at the base parameters matches the original deviance
    d_opt = dd(optv[:npar1])
    if not (np.isfinite(d_opt) and abs(d_opt - base) <= devmatchtol * abs(d_opt)):
        basediff = abs(d_opt / base - 1)
        raise ValueError("Profiling over both the residual variance and\n"
                         "fixed effects is not numerically consistent with\n"
                         "profiling over the fixed effects only (relative difference: %1.4g);\n"
                         "consider adjusting devmatchtol" % basediff)
    seqnvp = [k for k in range(npar1) if (k + 1) in which]
    lowvp = lower[:npar1]
    upvp = upper[:npar1]

    def FUN(w):
        if verbose:
            print(("var-cov " if isLMM else "") + f"parameter {w + 1}:")
        cc = w + 1
        pw = optv[w]
        lowcut = lower[w]
        upcut = upper[w]
        others = seqpar1 != w

        def zeta(xx, start):
            try:
                ores = optwrap(optimizer, lambda x: dd(mkpar(npar1, w, xx, x)), start,
                               lower=lowvp[others], upper=upvp[others], control=control,
                               calc_derivs=False)
            except Exception:
                ores = None
            if ores is None:
                devdiff = np.nan
            else:
                devdiff = ores["fval"] - base
                pars = np.asarray(ores["par"], float)
            if np.isnan(devdiff):
                warnings.warn("NAs detected in profiling")
                return np.full(ncol, np.nan)
            if verbose and devdiff < 0:
                print(f"old deviance {base},\nnew deviance {ores['fval']},\nnew params "
                      + ",".join(str(v) for v in mkpar(npar1, w, xx, pars)))
            if devdiff < -devtol:
                raise RuntimeError("profiling detected new, lower deviance "
                                   "(deviance diff = %1.3g, tolerance = %1.3g)" % (abs(devdiff), devtol))
            if devdiff < 0:
                warnings.warn("slightly lower deviances (diff=%g) detected" % devdiff)
            if devdiff <= 0:
                devdiff = devtol
            zz = np.sign(xx - pw) * math.sqrt(devdiff)
            r = np.concatenate([[zz], mkpar(npar1, w, xx, pars)])
            if isLMM:
                r = np.concatenate([r, np.asarray(pp.beta(1.0), float)])
            return r

        pres = res.copy()
        nres = res.copy()
        shiftpar = 1e-3 if pw == 0 else pw * 1.01
        if shiftpar >= upcut:
            shiftpar = (pw + upcut) / 2 if pw < upcut else 2 * pw - shiftpar
        elif shiftpar <= lowcut:
            shiftpar = (pw + lowcut) / 2 if pw > lowcut else 2 * pw - shiftpar
        row = zeta(shiftpar, start=optv[:npar1][others])
        nres[0, :] = row
        pres[1, :] = row
        upperf = fillmat(pres, lowcut, upcut, zeta, cc)
        lowerf = fillmat(nres, lowcut, upcut, zeta, cc) if pw > lowcut else nres
        bres = _unique_rows(np.vstack([upperf, lowerf]))
        pname = optnames[w]
        bdf = pd.DataFrame(bres, columns=[".zeta"] + optnames)
        bdf[".par"] = pname
        bdf = bdf.iloc[np.argsort(bdf[pname].values, kind="stable")].reset_index(drop=True)
        forspl, bakspl = profSplines(bdf, pname)
        if isinstance(bakspl, Exception):
            warnings.warn(f"non-monotonic profile for {pname}")
        return bdf, bakspl, forspl

    L = [FUN(w) for w in seqnvp]
    nn = [optnames[w] for w in seqnvp]
    ans = {nm: l[0] for nm, l in zip(nn, L)}
    bakspl = {nm: l[1] for nm, l in zip(nn, L)}
    forspl = {nm: l[2] for nm, l in zip(nn, L)}

    # profile fixed effects separately (for LMMs)
    if isLMM:
        from .modular import Rho, mkdevfun
        from .modules import merPredD, lmerResp
        reCovs = fitted.getReCovs()
        mkPar = mkMkPar(reCovs)
        mkTheta = mkMkTheta(reCovs)
        offset_orig = np.asarray(fitted.resp.offset, float)
        fp = [j for j in range(p) if (j + nvp + 1) in which]
        fe_names = optnames[nvp:]
        for j in fp:
            if verbose:
                print(f"fixed-effect parameter {j + 1}:")
            pres = res.copy()
            nres = res.copy()
            est = optv[nvp + j]
            std = stderr[j]
            Xw = X_orig[:, j]
            Xdrop = np.delete(X_orig, j, axis=1)
            pp1 = merPredD(Xdrop, pp.Zt, pp.Lambdat, np.asarray(pp.Lind), np.asarray(pp.theta),
                           n=Xdrop.shape[0])
            rr = lmerResp(y=np.asarray(fitted.resp.y, float))
            rr.ptr.setWeights(np.asarray(fitted.resp.weights, float))

            def fe_zeta(fw, start=None, j=j, est=est, Xw=Xw, pp1=pp1, rr=rr):
                rr.ptr.setOffset(Xw * fw + offset_orig)
                rho = Rho(pp=pp1, resp=rr, mkPar=mkPar, mkTheta=mkTheta)
                ores = optwrap(optimizer, mkdevfun(rho, 0), paropt, lower=fitted.lower,
                               calc_derivs=False)
                fv = ores["fval"]
                sig = math.sqrt((rr.wrss() + pp1.sqrL(1.0)) / n)
                ppars = convParToProfPar(ores["par"], fitted, profscale=prof_scale, sc=sig)
                with np.errstate(invalid="ignore"):
                    z = np.sign(fw - est) * np.sqrt(fv - base)
                return np.concatenate([[z], ppars, mkpar(p, j, fw, np.asarray(pp1.beta(1.0), float))])

            row = fe_zeta(est + delta * std)
            nres[0, :] = row
            pres[1, :] = row
            poff = nvp + 1 + j
            upperf = fillmat(pres, -np.inf, np.inf, fe_zeta, poff)
            lowerf = fillmat(nres, -np.inf, np.inf, fe_zeta, poff)
            bres = _unique_rows(np.vstack([upperf, lowerf]))
            n_j = fe_names[j]
            bdf = pd.DataFrame(bres, columns=[".zeta"] + optnames)
            bdf[".par"] = n_j
            bdf = bdf.iloc[np.argsort(bdf[n_j].values, kind="stable")].reset_index(drop=True)
            ans[n_j] = bdf
            fs, bs = profSplines(bdf, n_j)
            forspl[n_j] = fs
            bakspl[n_j] = bs
            if isinstance(bs, Exception):
                warnings.warn(f"non-monotonic profile for {n_j}")

    frame = pd.concat(list(ans.values()), ignore_index=True) if ans else \
        pd.DataFrame(columns=[".zeta"] + optnames + [".par"])
    lo = pd.Series(lower[seqnvp], index=[optnames[k] for k in seqnvp])
    up = pd.Series(upper[seqnvp], index=[optnames[k] for k in seqnvp])
    return thpr(frame, forward=forspl, backward=bakspl, lower=lo, upper=up)


# ---------------------------------------------------------------------------
# logProf / varianceProf
# ---------------------------------------------------------------------------

def _rename_prof(x, pat, rep):
    import re
    rx = re.compile(pat)
    newcols = [rx.sub(rep, c, count=1) for c in x.frame.columns]
    out = x.copy()
    out.frame.columns = newcols
    out.frame[".par"] = [rx.sub(rep, v, count=1) for v in out.frame[".par"].astype(str)]
    out.forward = {rx.sub(rep, k, count=1): v for k, v in x.forward.items()}
    out.backward = {rx.sub(rep, k, count=1): v for k, v in x.backward.items()}
    if out.lower is not None:
        out.lower.index = [rx.sub(rep, k, count=1) for k in out.lower.index]
        out.upper.index = [rx.sub(rep, k, count=1) for k in out.upper.index]
    return out


def logProf(x, base=math.e, ranef=True, sigIni=None):
    """return an lmer profile like x with all the .sigNN parameters replaced
    by .lsigNN (log scale); the splines are recomputed"""
    import re
    if sigIni is None:
        sigIni = "sig" if ranef else "sigma"
    cn = list(x.frame.columns)
    sigP = "^\\." + sigIni
    sigs = [i for i, c in enumerate(cn) if re.search(sigP, c)]
    if not sigs:
        return x
    repP = "." + sigIni.replace("sig", "lsig", 1)
    x = _rename_prof(x, sigP, repP.replace("\\", "\\\\"))
    cn = list(x.frame.columns)
    fr0 = x.frame
    for i in sigs:
        nm = cn[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            fr0[nm] = np.log(fr0[nm].values.astype(float)) / math.log(base)
        sel = (fr0[".par"] == nm) & np.isfinite(fr0[nm].values)
        f, b = profSplines(fr0.loc[sel], nm)
        x.forward[nm] = f
        x.backward[nm] = b
    keep = np.all(np.isfinite(fr0.iloc[:, sigs].values.astype(float)), axis=1)
    x.frame = fr0.loc[keep].reset_index(drop=True)
    return x


def varianceProf(x, ranef=True):
    """transform a mixed-effects profile to the variance scale"""
    import re
    cn = list(x.frame.columns)
    sigs = [i for i, c in enumerate(cn) if re.search("^\\." + ("sig" if ranef else "sigma"), c)]
    if not sigs:
        return x
    sigP = "^(\\.sig" + ("(ma)?" if ranef else "ma") + ")"
    x = _rename_prof(x, sigP, r"\1sq")
    cn = list(x.frame.columns)
    fr0 = x.frame
    for i in sigs:
        nm = cn[i]
        fr0[nm] = fr0[nm].values.astype(float) ** 2
        f, b = profSplines(fr0.loc[fr0[".par"] == nm], nm)
        x.forward[nm] = f
        x.backward[nm] = b
    return x


# ---------------------------------------------------------------------------
# confint.merMod
# ---------------------------------------------------------------------------

def confint_merMod(obj, parm=None, level=0.95, method="profile", zeta=None, nsim=500,
                   boot_type="perc", FUN=None, quiet=False, oldNames=None, seed=None,
                   signames=True, boot_scale="sdcor", **kwargs):
    """Confidence intervals on the parameters of a fitted mixed model
    (confint.merMod).

    method: "profile" (default), "Wald" or "boot".  ``parm``: None, names,
    "theta_", "beta_" or 0-based integer positions.  Extra keyword arguments
    go to :func:`profile_merMod` (method "profile") or :func:`lme4.bootmer.bootMer`
    (method "boot")."""
    meths = ["profile", "Wald", "boot"]
    m = [mm for mm in meths if mm.lower().startswith(str(method).lower())]
    if len(m) != 1:
        raise ValueError("'arg' should be one of \"profile\", \"Wald\", \"boot\"")
    method = m[0].lower()
    if boot_type not in ("perc", "basic", "norm"):
        raise ValueError("'arg' should be one of \"perc\", \"basic\", \"norm\"")
    if boot_scale in ("vcov", "varcov"):
        boot_scale = "varcov"
    elif boot_scale != "sdcor":
        raise ValueError("'arg' should be one of \"sdcor\", \"vcov\"")
    if oldNames is not None:
        warnings.warn("'oldNames' is deprecated. Please use 'signames' instead.")
        signames = oldNames
    custom_FUN = method == "boot" and FUN is not None
    if custom_FUN:
        pass   # custom boot function, don't expand parameter names
    else:
        useSc = bool(obj.devcomp["dims"]["useSc"])
        vn = profnames(obj, signames, useSc=useSc)
        an = vn + list(obj.fixef().index)
        parm = list(range(1, len(an) + 1)) if parm is None else \
            get_which(parm, nvp=len(vn), nptot=len(an), parnames=an)
        if not quiet and method in ("profile", "boot"):
            mtype = {"profile": "profile", "boot": "bootstrap"}[method]
            print(f"Computing {mtype} confidence intervals ...", file=sys.stderr)
    if method == "profile":
        pp = profile_merMod(obj, which=[k - 1 for k in parm], signames=signames, **kwargs)
        return confint_thpr(pp, level=level, zeta=zeta)
    if method == "wald":
        a = (1 - level) / 2
        a = np.array([a, 1 - a])
        cols = _format_perc(a, 3)
        ci_vcov = pd.DataFrame(np.nan, index=vn, columns=cols)
        cf = obj.fixef()
        ses = np.sqrt(np.diag(obj.vcov().values))
        q = stats.norm.ppf(a)
        ci_fixed = pd.DataFrame(cf.values[:, None] + ses[:, None] * q[None, :],
                                index=cf.index, columns=cols)
        ci_all = pd.concat([ci_vcov, ci_fixed])
        return ci_all.iloc[[k - 1 for k in parm]]
    # boot
    from .bootmer import bootMer, print_bootWarnings
    from .family import hasNoScale
    if FUN is None:
        fe_names = list(obj.fixef().index)

        def FUN(x):
            useSc_ = not (x.isGLMM() and hasNoScale(x.family()))
            sc = x.sigma() if useSc_ else None
            ss = convParToProfPar(x.optinfo["val"], x, profscale=boot_scale, sc=sc)
            return pd.Series(np.concatenate([ss, x.fixef().values]), index=vn + fe_names)
    bb = bootMer(obj, FUN=FUN, nsim=nsim, seed=seed, **kwargs)
    if np.all(np.isnan(bb.t)):
        raise RuntimeError("*all* bootstrap runs failed!")
    print_bootWarnings(bb, verbose=False)
    citab = bb.confint(level=level, type=boot_type)
    if custom_FUN:
        # parm was not expanded: names or 0-based positions (all rows if None)
        if parm is None:
            return citab
        if isinstance(parm, str):
            parm = [parm]
        if all(isinstance(k, str) for k in parm):
            return citab.loc[parm]
        return citab.iloc[[int(k) for k in np.atleast_1d(parm)]]
    # parm from get_which() is 1-based
    return citab.iloc[[k - 1 for k in parm]]
