"""Lists of linear / generalized linear models fitted per group:
transcription of lme4/R/lmList.R (``lmList``, class ``lmList4`` and its
methods), together with the nlme methods lme4 re-uses for it
(``summary``, ``fixef``, ``ranef``, ``logLik``, ``fitted``, ``residuals``,
``predict``) and minimal transcriptions of the stats functions involved:
``lm()`` / ``lm.wfit()`` (LINPACK ``dqrdc2``/``dqrls`` QR least squares),
``glm()`` / ``glm.fit()`` (IRLS), ``summary.lm``, ``summary.glm``,
``confint.lm`` (t-based) and ``confint.glm`` (profile likelihood, via
``MASS::profile.glm`` + ``confint.profile.glm`` incl. R's ``spline()`` and
``approx()``).
"""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .family import Family, get_family
from .modelframe import Factor, ModelFrame, as_factor, model_frame, model_matrix
from .modular import dqrdc2
from .reformulas import subbars
from .rlang import Call, Const, Formula, Sym, deparse, is_call
from .symbolic import to_formula

DBL_EPSILON = np.finfo(float).eps


# ---------------------------------------------------------------------------
# formula helpers (lmList.R)
# ---------------------------------------------------------------------------

def dropOffset(form):
    """remove offset() terms from a formula / expression"""
    if isinstance(form, (Sym, Const)) or not isinstance(form, Call):
        return form
    if is_call(form, "offset"):
        return None
    if len(form) == 2:        # unary operator
        return form.replace(1, dropOffset(form[1]))
    nb2 = dropOffset(form[1])
    nb3 = dropOffset(form[2])
    if nb2 is None:
        return nb3
    if nb3 is None:
        return nb2
    return form.replace(1, nb2).replace(2, nb3)


def _unparen(e):
    """strip enclosing parentheses (as produced by update.formula())"""
    while is_call(e, "(") and len(e) == 2:
        e = e[1]
    return e


def modelFormula(form: Formula):
    """split ``y ~ x | g`` into model ``y ~ x`` (offsets dropped) and groups ``g``"""
    if not isinstance(form, Formula) or len(form) != 3:
        raise ValueError("formula must be a two-sided formula object")
    rhs = _unparen(form[2])
    if not is_call(rhs, "|"):
        raise ValueError("rhs of formula must be a conditioning expression")
    model = Call(Sym("~"), (form[1], rhs[1]))
    return dict(model=Formula(dropOffset(model), form.env), groups=rhs[2])


def get_grpvars(form: Formula):
    """reformulas::get_grpvars: the grouping variables after the bar"""
    rhs = _unparen(form[2] if len(form) == 3 else form[1])
    if not is_call(rhs, "|"):
        return []
    g = rhs[2]
    out = []

    def rec(e):
        if is_call(e, ("*", "/", ":", "+")) and len(e) == 3:
            rec(e[1])
            rec(e[2])
        else:
            out.append(e)
    rec(g)
    return out


# ---------------------------------------------------------------------------
# LINPACK QR least squares (dqrdc2 + dqrsl), as in R's lm.fit / Cdqrls
# ---------------------------------------------------------------------------

def _qty(qr, qraux, k, y):
    """dqrsl: Q'y for the Householder QR in (qr, qraux), first k columns"""
    y = np.array(y, float, copy=True)
    n = qr.shape[0]
    for j in range(min(k, n - 1)):
        if qraux[j] != 0:
            temp = qr[j, j]
            qr[j, j] = qraux[j]
            t = -np.dot(qr[j:, j], y[j:]) / qr[j, j]
            y[j:] += t * qr[j:, j]
            qr[j, j] = temp
    return y


def _qy(qr, qraux, k, y):
    """dqrsl: Qy"""
    y = np.array(y, float, copy=True)
    n = qr.shape[0]
    for j in reversed(range(min(k, n - 1))):
        if qraux[j] != 0:
            temp = qr[j, j]
            qr[j, j] = qraux[j]
            t = -np.dot(qr[j:, j], y[j:]) / qr[j, j]
            y[j:] += t * qr[j:, j]
            qr[j, j] = temp
    return y


def Cdqrls(x, y, tol=1e-7):
    """R's C_Cdqrls: least squares fit by dqrdc2 + dqrsl.  Returns dict with
    coefficients (in pivoted order, aliased ones 0), residuals, effects,
    rank, pivot (0-based), qr, qraux."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, p = x.shape
    if p == 0:
        return dict(coefficients=np.zeros(0), residuals=y.copy(), effects=y.copy(), rank=0,
                    pivot=np.arange(0), qr=x.copy(), qraux=np.zeros(0))
    qr, k, pivot, qraux = dqrdc2(x, tol)
    qr = np.array(qr, float, order="F")
    qty = _qty(qr, qraux, k, y)
    coef = np.zeros(p)
    if k > 0:
        R = np.triu(qr[:k, :k])
        coef[:k] = _backsolve(R, qty[:k])
    tmp = qty.copy()
    tmp[:k] = 0.0
    resid = _qy(qr, qraux, k, tmp)
    return dict(coefficients=coef, residuals=resid, effects=qty, rank=k, pivot=pivot, qr=qr,
                qraux=qraux)


def _backsolve(R, b):
    from scipy.linalg import solve_triangular
    return solve_triangular(R, b, lower=False)


def _chol2inv(R):
    k = R.shape[0]
    if k == 0:
        return np.zeros((0, 0))
    Ri = _backsolve(R, np.eye(k))
    return Ri @ Ri.T


# ---------------------------------------------------------------------------
# lm() / glm() fits (minimal transcriptions of stats::lm, stats::glm)
# ---------------------------------------------------------------------------

def _check_contrasts(fr: ModelFrame):
    """model.matrix(): `contrasts<-` fails for factors with < 2 levels"""
    tt = fr.terms
    used = set()
    for t in tt.terms:
        used |= set(t)
    for i, key in enumerate(tt.varnames):
        if tt.response and i == 0:
            continue
        if i not in used:
            continue
        v = fr[key]
        nlev = None
        if isinstance(v, Factor):
            nlev = v.nlevels
        else:
            a = np.asarray(v)
            if a.dtype.kind in "OUS" or a.dtype == bool:
                nlev = len(pd.unique(a[~pd.isna(a)]))
        if nlev is not None and nlev < 2:
            raise ValueError("contrasts can be applied only to factors with 2 or more levels")


class _FitBase:
    """common parts of the lm / glm fits"""

    def coef(self):
        return self.coefficients

    def fitted(self):
        return pd.Series(self.fitted_values, index=self.row_names)

    def nobs(self):
        w = self.prior_weights
        return int(np.sum(w != 0)) if w is not None else len(self.y)

    def formula(self):
        return self.call_formula

    def model_matrix(self):
        return pd.DataFrame(self.X, columns=self.colnames, index=self.row_names)

    def _cov_unscaled(self):
        """chol2inv of R (for non-aliased coefficients, original order)"""
        p1 = self.pivot[:self.rank]
        R = np.triu(self.qr[:self.rank, :self.rank])
        cu = _chol2inv(R)
        nms = [self.colnames[i] for i in p1]
        # put into coefficient order (as summary.lm does with Qr$pivot[p1])
        order = np.argsort(p1)
        cu = cu[np.ix_(order, order)]
        nms = [nms[i] for i in order]
        return pd.DataFrame(cu, index=nms, columns=nms)

    def vcov(self):
        s = self.summary()
        cu = s["cov.unscaled"] * s["dispersion"]
        full = pd.DataFrame(np.nan, index=self.colnames, columns=self.colnames)
        full.loc[cu.index, cu.columns] = cu.values
        return full


class LmFit(_FitBase):
    """result of :func:`lm` (an R ``lm`` object)"""
    kind = "lm"

    def residuals(self, type="working"):
        r = self.resid_raw
        if type in ("deviance", "pearson") and self.prior_weights is not None:
            r = r * np.sqrt(self.prior_weights)
        return pd.Series(r, index=self.row_names)

    resid = residuals

    def summary(self):
        """summary.lm (the numeric parts)"""
        p = self.rank
        r = self.resid_raw
        w = self.prior_weights
        rdf = self.df_residual
        if w is not None:
            ok = w != 0
            r = r[ok]
            w = w[ok]
            rss = np.sum(w * r ** 2)
        else:
            rss = np.sum(r ** 2)
        resvar = rss / rdf if rdf > 0 else np.nan
        cu = self._cov_unscaled()
        nms = list(cu.index)
        est = self.coefficients[nms].values
        se = np.sqrt(np.diag(cu.values) * resvar)
        tval = est / se
        coefs = pd.DataFrame({"Estimate": est, "Std. Error": se, "t value": tval,
                              "Pr(>|t|)": 2 * stats.t.sf(np.abs(tval), rdf)}, index=nms)
        sigma = math.sqrt(resvar) if np.isfinite(resvar) else np.nan
        f = self.fitted_values if w is None else self.fitted_values[self.prior_weights != 0]
        if p != (1 if self.intercept else 0):
            ww = np.ones(len(f)) if w is None else w
            if self.intercept:
                m = np.sum(ww * f / np.sum(ww))
                mss = np.sum(ww * (f - m) ** 2)
            else:
                mss = np.sum(ww * f ** 2)
            r_squared = mss / (mss + rss)
            df_int = 1 if self.intercept else 0
            adj_r_squared = 1 - (1 - r_squared) * ((len(f) - df_int) / rdf)
        else:
            r_squared = adj_r_squared = 0.0
        return {"coefficients": coefs, "sigma": sigma, "df": (p, rdf, len(self.colnames)),
                "r.squared": r_squared, "adj.r.squared": adj_r_squared,
                "cov.unscaled": cu, "dispersion": resvar, "aliased": self.coefficients.isna()}

    def sigma(self):
        return self.summary()["sigma"]

    def confint(self, parm=None, level=0.95):
        """confint.lm: t-based intervals"""
        cf = self.coefficients
        pnames = list(cf.index)
        if parm is None:
            parm = pnames
        elif isinstance(parm, str):
            parm = [parm]
        a = (1 - level) / 2
        a = np.array([a, 1 - a])
        fac = stats.t.ppf(a, self.df_residual)
        ses = np.sqrt(np.diag(self.vcov().values))
        ses = pd.Series(ses, index=pnames)
        ci = np.outer(ses[parm].values, fac) + cf[parm].values[:, None]
        return pd.DataFrame(ci, index=parm, columns=[_pct(x) for x in a])

    def logLik(self, REML=False):
        """logLik.lm"""
        res = self.resid_raw
        p = self.rank
        N = len(res)
        w = self.prior_weights
        if w is None:
            w = np.ones(N)
        else:
            excl = w == 0
            if np.any(excl):
                res = res[~excl]
                N = len(res)
                w = w[~excl]
        N0 = N
        if REML:
            N = N - p
        val = .5 * (np.sum(np.log(w)) - N * (math.log(2 * math.pi) + 1 - math.log(N) +
                                             math.log(np.sum(w * res ** 2))))
        if REML:
            val -= np.sum(np.log(np.abs(np.diag(self.qr)[:p])))
        return _LogLik(val, df=p + 1, nobs=N, nall=N0)

    def deviance(self):
        r = self.resid_raw
        w = self.prior_weights
        return float(np.sum(r ** 2) if w is None else np.sum(w * r ** 2))

    def predict(self, newdata=None):
        if newdata is None:
            return self.fitted()
        X, off = _newX(self, newdata)
        cf = np.nan_to_num(self.coefficients.values)
        return pd.Series(X @ cf + off, index=newdata.index)


def lm(formula, data, weights=None, offset=None, na_action="na.omit", env=None):
    """stats::lm (QR via LINPACK dqrdc2, as R's lm.fit / lm.wfit)"""
    formula = to_formula(formula, env)
    fr = model_frame(formula, data, weights=weights, offset=offset, na_action=na_action,
                     drop_unused_levels=True)
    _check_contrasts(fr)
    X, cn, assign, _ = model_matrix(fr.terms, fr)
    y = np.asarray(fr.response(), float)
    if y.ndim > 1:
        y = y[:, 0] if y.shape[1] == 1 else y
    n = X.shape[0]
    if n == 0:
        raise ValueError("0 (non-NA) cases")
    off = fr.offset()
    w = fr.weights()
    yy = y - off if off is not None else y
    if w is not None:
        w = np.asarray(w, float)
        if np.any(w < 0):
            raise ValueError("missing or negative weights not allowed")
        ok = w != 0
        wts = np.sqrt(w[ok])
        z = Cdqrls(X[ok] * wts[:, None], yy[ok] * wts, 1e-7)
    else:
        ok = np.ones(n, bool)
        z = Cdqrls(X, yy, 1e-7)
    p = X.shape[1]
    coef = np.full(p, np.nan)
    piv = z["pivot"]
    coef[piv[:z["rank"]]] = z["coefficients"][:z["rank"]]
    fitted = X[:, piv[:z["rank"]]] @ coef[piv[:z["rank"]]] if z["rank"] > 0 else np.zeros(n)
    if off is not None:
        fitted = fitted + off
    ans = LmFit()
    ans.coefficients = pd.Series(coef, index=cn)
    ans.fitted_values = fitted
    ans.resid_raw = y - fitted
    ans.rank = z["rank"]
    ans.pivot = piv
    ans.qr = z["qr"]
    ans.qraux = z["qraux"]
    ans.df_residual = int(np.sum(ok)) - z["rank"]
    ans.prior_weights = w
    ans.offset = off
    ans.y = y
    ans.X = X
    ans.colnames = cn
    ans.assign = assign
    ans.intercept = bool(fr.terms.intercept)
    ans.row_names = fr.row_names
    ans.terms = fr.terms
    ans.frame = fr
    ans.call_formula = formula
    return ans


class GlmFit(_FitBase):
    """result of :func:`glm` (an R ``glm`` object)"""
    kind = "glm"

    def residuals(self, type="deviance"):
        y, mu, wts = self.y, self.fitted_values, self.prior_weights
        if type == "deviance":
            d = self.family.dev_resids(y, mu, wts)
            r = np.where(y > mu, np.sqrt(np.maximum(d, 0)), -np.sqrt(np.maximum(d, 0)))
        elif type == "pearson":
            r = (y - mu) * np.sqrt(wts) / np.sqrt(self.family.variance(mu))
        elif type == "working":
            r = self.working_residuals
        elif type == "response":
            r = y - mu
        else:
            raise ValueError(type)
        return pd.Series(r, index=self.row_names)

    resid = residuals

    def summary(self, dispersion=None):
        """summary.glm (numeric parts)"""
        est_disp = False
        df_r = self.df_residual
        if dispersion is None:
            if self.family.base in ("poisson", "binomial", "negative.binomial"):
                dispersion = 1.0
            elif df_r > 0:
                est_disp = True
                ok = self.weights > 0
                dispersion = np.sum((self.weights * self.working_residuals ** 2)[ok]) / df_r
            else:
                est_disp = True
                dispersion = np.nan
        cu = self._cov_unscaled()
        nms = list(cu.index)
        est = self.coefficients[nms].values
        se = np.sqrt(np.diag(cu.values) * dispersion)
        tval = est / se
        if not est_disp:
            pv = 2 * stats.norm.sf(np.abs(tval))
            cols = ["Estimate", "Std. Error", "z value", "Pr(>|z|)"]
        else:
            pv = 2 * stats.t.sf(np.abs(tval), df_r)
            cols = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
        coefs = pd.DataFrame(np.column_stack([est, se, tval, pv]), index=nms, columns=cols)
        return {"coefficients": coefs, "dispersion": dispersion, "cov.unscaled": cu,
                "df": (self.rank, df_r, len(self.colnames)), "deviance": self.deviance_,
                "null.deviance": self.null_deviance, "aic": self.aic,
                "aliased": self.coefficients.isna(), "sigma": math.sqrt(dispersion)}

    def deviance(self):
        return self.deviance_

    def sigma(self):
        return self.summary()["sigma"]

    def logLik(self, REML=False):
        """logLik.glm"""
        p = self.rank
        if self.family.base in ("gaussian", "Gamma", "inverse.gaussian"):
            p += 1
        return _LogLik(p - self.aic / 2, df=p, nobs=int(np.sum(self.prior_weights != 0)))

    def confint(self, parm=None, level=0.95, trace=False):
        """confint.glm: profile likelihood intervals (MASS::profile.glm +
        confint.profile.glm)"""
        pnames = list(self.coefficients.index)
        if parm is None:
            which = list(range(len(pnames)))
        else:
            if isinstance(parm, str):
                parm = [parm]
            which = [pnames.index(p) if isinstance(p, str) else int(p) for p in parm]
        prof = profile_glm(self, which=which, alpha=(1 - level) / 4)
        a = (1 - level) / 2
        a = np.array([a, 1 - a])
        ci = np.full((len(which), 2), np.nan)
        cutoff = stats.norm.ppf(a)
        for r, pm in enumerate(which):
            pro = prof.get(pnames[pm])
            if pro is None:
                continue
            sx, sy = spline(pro["par.vals"][:, pm], pro["z"])
            ci[r] = approx(sy, sx, cutoff)
        return pd.DataFrame(ci, index=[pnames[i] for i in which], columns=[_pct(x) for x in a])

    def predict(self, newdata=None, type="link"):
        if newdata is None:
            eta = self.linear_predictors
            idx = self.row_names
        else:
            X, off = _newX(self, newdata)
            eta = X @ np.nan_to_num(self.coefficients.values) + off
            idx = newdata.index
        if type == "response":
            eta = self.family.linkinv(eta)
        return pd.Series(eta, index=idx)


def _valid(fam: Family, eta, mu):
    if not (np.all(np.isfinite(eta)) and np.all(np.isfinite(mu))):
        return False
    if fam.base == "binomial":
        return bool(np.all((mu > 0) & (mu < 1)))
    if fam.base in ("poisson", "Gamma", "negative.binomial"):
        ok = bool(np.all(mu > 0))
        if fam.base == "Gamma" and fam.link == "inverse":
            ok = ok and bool(np.all(eta != 0))
        return ok
    if fam.base == "inverse.gaussian":
        ok = bool(np.all(mu > 0))
        if fam.link == "1/mu^2":
            ok = ok and bool(np.all(eta > 0))
        return ok
    return True


def glm_fit(x, y, weights=None, start=None, etastart=None, mustart=None, offset=None,
            family=None, epsilon=1e-8, maxit=25, intercept=True):
    """stats::glm.fit (IRLS)"""
    x = np.asarray(x, float)
    nobs = x.shape[0]
    nvars = x.shape[1]
    EMPTY = nvars == 0
    weights = np.ones(nobs) if weights is None else np.asarray(weights, float)
    offset = np.zeros(nobs) if offset is None else np.asarray(offset, float)
    fam = family
    ini = fam.initialize(y, weights, nobs)
    y, weights, n = ini["y"], ini["weights"], ini["n"]
    if mustart is None:
        mustart = ini["mustart"]
    linkinv, dev_resids, mu_eta, variance = fam.linkinv, fam.dev_resids, fam.mu_eta, fam.variance
    fit = None
    if EMPTY:
        eta = np.zeros(nobs) + offset
        mu = linkinv(eta)
        dev = float(np.sum(dev_resids(y, mu, weights)))
        w = np.sqrt((weights * mu_eta(eta) ** 2) / variance(mu))
        residuals = (y - mu) / mu_eta(eta)
        good = np.ones(nobs, bool)
        boundary = conv = True
        coef = np.zeros(0)
        it = 0
        fit = dict(rank=0, pivot=np.arange(0), qr=np.zeros((nobs, 0)), qraux=np.zeros(0))
    else:
        coefold = None
        if etastart is not None:
            eta = np.asarray(etastart, float).ravel()
        elif start is not None:
            start = np.asarray(start, float)
            if len(start) != nvars:
                raise ValueError(f"length of 'start' should equal {nvars}")
            coefold = start
            eta = offset + x @ start
        else:
            eta = fam.linkfun(mustart)
        mu = linkinv(eta)
        if not _valid(fam, eta, mu):
            raise ValueError("cannot find valid starting values: please specify some")
        devold = float(np.sum(dev_resids(y, mu, weights)))
        boundary = conv = False
        start = np.zeros(nvars) if start is None else np.array(start, float)
        coef = start.copy()
        it = 0
        for it in range(1, maxit + 1):
            good = weights > 0
            mu_eta_val = mu_eta(eta)
            good = (weights > 0) & (mu_eta_val != 0)
            if not np.any(good):
                conv = False
                warnings.warn(f"no observations informative at iteration {it}")
                break
            z = (eta - offset)[good] + (y - mu)[good] / mu_eta_val[good]
            w = np.sqrt((weights[good] * mu_eta_val[good] ** 2) / variance(mu)[good])
            fit = Cdqrls(x[good] * w[:, None], z * w, min(1e-7, epsilon / 1000))
            if np.any(~np.isfinite(fit["coefficients"])):
                conv = False
                warnings.warn(f"non-finite coefficients at iteration {it}")
                break
            if nobs < fit["rank"]:
                raise ValueError(f"X matrix has rank {fit['rank']}, but only {nobs} observations")
            start = start.copy()
            start[fit["pivot"]] = fit["coefficients"]
            eta = x @ start
            eta = eta + offset
            mu = linkinv(eta)
            dev = float(np.sum(dev_resids(y, mu, weights)))
            boundary = False
            if not np.isfinite(dev):
                if coefold is None:
                    raise ValueError("no valid set of coefficients has been found: please supply starting values")
                warnings.warn("step size truncated due to divergence")
                ii = 1
                while not np.isfinite(dev):
                    if ii > maxit:
                        raise ValueError("inner loop 1; cannot correct step size")
                    ii += 1
                    start = (start + coefold) / 2
                    eta = x @ start + offset
                    mu = linkinv(eta)
                    dev = float(np.sum(dev_resids(y, mu, weights)))
                boundary = True
            if not _valid(fam, eta, mu):
                if coefold is None:
                    raise ValueError("no valid set of coefficients has been found: please supply starting values")
                warnings.warn("step size truncated: out of bounds")
                ii = 1
                while not _valid(fam, eta, mu):
                    if ii > maxit:
                        raise ValueError("inner loop 2; cannot correct step size")
                    ii += 1
                    start = (start + coefold) / 2
                    eta = x @ start + offset
                    mu = linkinv(eta)
                boundary = True
                dev = float(np.sum(dev_resids(y, mu, weights)))
            if abs(dev - devold) / (abs(dev) + 0.1) < epsilon:
                conv = True
                coef = start
                break
            devold = dev
            coef = coefold = start
        if not conv:
            warnings.warn("glm.fit: algorithm did not converge")
        if boundary:
            warnings.warn("glm.fit: algorithm stopped at boundary value")
        eps = 10 * DBL_EPSILON
        if fam.base == "binomial" and np.any((mu > 1 - eps) | (mu < eps)):
            warnings.warn("glm.fit: fitted probabilities numerically 0 or 1 occurred")
        if fam.base == "poisson" and np.any(mu < eps):
            warnings.warn("glm.fit: fitted rates numerically 0 occurred")
        coef = np.array(coef, float)
        if fit["rank"] < nvars:
            coef[fit["pivot"][fit["rank"]:]] = np.nan
        residuals = (y - mu) / mu_eta(eta)
    wt = np.zeros(nobs)
    wt[good] = w ** 2
    wtdmu = np.full(nobs, np.sum(weights * y) / np.sum(weights)) if intercept else linkinv(offset)
    nulldev = float(np.sum(dev_resids(y, wtdmu, weights)))
    n_ok = nobs - int(np.sum(weights == 0))
    nulldf = n_ok - int(intercept)
    rank = 0 if EMPTY else fit["rank"]
    resdf = n_ok - rank
    aic_model = fam.aic(y, n, mu, weights, dev) + 2 * rank
    return dict(coefficients=coef, residuals=residuals, fitted_values=mu, rank=rank,
                linear_predictors=eta, deviance=dev, aic=aic_model, null_deviance=nulldev,
                iter=it, weights=wt, prior_weights=weights, df_residual=resdf,
                df_null=nulldf, y=y, converged=conv, boundary=boundary, qr=fit["qr"],
                qraux=fit["qraux"], pivot=fit["pivot"], n=n)


def glm(formula, family, data, weights=None, offset=None, na_action="na.omit", env=None,
        epsilon=1e-8, maxit=25):
    """stats::glm"""
    family = get_family(family)
    formula = to_formula(formula, env)
    fr = model_frame(formula, data, weights=weights, offset=offset, na_action=na_action,
                     drop_unused_levels=True)
    _check_contrasts(fr)
    X, cn, assign, _ = model_matrix(fr.terms, fr)
    Y = fr.response()
    nobs = X.shape[0]
    if nobs == 0:
        raise ValueError("0 (non-NA) cases")
    w = fr.weights()
    off = fr.offset()
    intercept = bool(fr.terms.intercept)
    f = glm_fit(X, Y, weights=w, offset=off, family=family, intercept=intercept,
                epsilon=epsilon, maxit=maxit)
    ans = GlmFit()
    ans.family = family
    ans.coefficients = pd.Series(f["coefficients"], index=cn)
    ans.fitted_values = f["fitted_values"]
    ans.linear_predictors = f["linear_predictors"]
    ans.working_residuals = f["residuals"]
    ans.rank = f["rank"]
    ans.pivot = f["pivot"]
    ans.qr = f["qr"]
    ans.qraux = f["qraux"]
    ans.deviance_ = f["deviance"]
    ans.null_deviance = f["null_deviance"]
    ans.aic = f["aic"]
    ans.df_residual = f["df_residual"]
    ans.df_null = f["df_null"]
    ans.weights = f["weights"]
    ans.prior_weights = f["prior_weights"]
    ans.y = f["y"]
    ans.converged = f["converged"]
    ans.iter = f["iter"]
    ans.offset = off
    ans.Y = Y
    ans.X = X
    ans.colnames = cn
    ans.assign = assign
    ans.intercept = intercept
    ans.row_names = fr.row_names
    ans.terms = fr.terms
    ans.frame = fr
    ans.call_formula = formula
    ans.control = dict(epsilon=epsilon, maxit=maxit)
    return ans


def _newX(fit, newdata):
    tt = fit.terms
    rhs = Formula(Call(Sym("~"), (tt.formula.rhs,)), tt.formula.env)
    xlev = {k: v.levels for k, v in fit.frame.columns.items() if isinstance(v, Factor)}
    fr = model_frame(rhs, newdata, na_action="na.pass", drop_unused_levels=False,
                     predvars=None, xlev=xlev)
    for k, lev in xlev.items():
        if k in fr.columns:
            fr.columns[k] = as_factor(np.asarray(fr.columns[k].as_strings()
                                                 if isinstance(fr.columns[k], Factor)
                                                 else fr.columns[k], dtype=object).astype(str),
                                      levels=lev)
    X, cn, _, _ = model_matrix(fr.terms, fr)
    X = pd.DataFrame(X, columns=cn).reindex(columns=fit.colnames, fill_value=0.0).values
    off = fr.offset()
    return X, (np.zeros(X.shape[0]) if off is None else off)


# ---------------------------------------------------------------------------
# profile.glm / confint.profile.glm  (MASS / stats >= 4.4)
# ---------------------------------------------------------------------------

def profile_glm(fitted: GlmFit, which=None, alpha=0.01, maxsteps=10, del_=None):
    """MASS:::profile.glm (LRT version)"""
    B0 = fitted.coefficients.values
    Pnames = list(fitted.coefficients.index)
    nonA = ~np.isnan(B0)
    pv0 = B0.reshape(1, -1)
    p = len(Pnames)
    if which is None:
        which = list(range(p))
    summ = fitted.summary()
    std_err = summ["coefficients"]["Std. Error"]
    Y = fitted.Y
    n = fitted.X.shape[0]
    O = fitted.offset if fitted.offset is not None else np.zeros(n)
    W = fitted.frame.weights()
    W = np.ones(n) if W is None else np.asarray(W, float)
    OriginalDeviance = fitted.deviance_
    DispersionParameter = summ["dispersion"]
    X = fitted.X
    fam = fitted.family
    if fam.base in ("binomial", "poisson", "negative.binomial"):
        zmax = math.sqrt(stats.chi2.ppf(1 - alpha, 1))
        profName = "z"
    else:
        zmax = math.sqrt(stats.f.ppf(1 - alpha, 1, n - p))
        profName = "tau"
    dl = zmax / 5 if del_ is None else del_
    prof = {}
    for i in which:
        if not nonA[i]:
            continue
        zi = [0.0]
        pvi = [pv0[0].copy()]
        a = nonA.copy()
        a[i] = False
        Xi = X[:, a]
        pi = Pnames[i]
        for sgn in (-1, 1):
            step = 0
            z = 0.0
            LP = X[:, nonA] @ B0[nonA] + O
            while True:
                step += 1
                if not (step < maxsteps and abs(z) < zmax):
                    break
                bi = B0[i] + sgn * step * dl * std_err[pi]
                o = O + X[:, i] * bi
                fm = glm_fit(Xi, Y, weights=W, etastart=LP, offset=o, family=fam,
                             epsilon=fitted.control["epsilon"], maxit=fitted.control["maxit"],
                             intercept=fitted.intercept)
                cf = np.nan_to_num(fm["coefficients"])
                LP = Xi @ cf + o
                ri = pv0[0].copy()
                ri[a] = fm["coefficients"]
                ri[i] = bi
                pvi.append(ri)
                zz = (fm["deviance"] - OriginalDeviance) / DispersionParameter
                if zz > -1e-3:
                    zz = max(zz, 0)
                else:
                    raise ValueError("profiling has found a better solution, so original fit had not converged")
                z = sgn * math.sqrt(zz)
                zi.append(z)
        zi = np.asarray(zi)
        si = np.argsort(zi, kind="stable")
        prof[pi] = {profName: zi[si], "z": zi[si], "par.vals": np.asarray(pvi)[si]}
    return prof


def _regularize(x, y):
    """regularize.values(x, y, ties=mean): sort by x, average y over ties"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) < len(x):
        yy = np.zeros(len(ux))
        np.add.at(yy, inv, y)
        y = yy / np.bincount(inv)
        x = ux
    return x, y


def _fmm_spline(x, y):
    """R's fmm_spline (Forsythe, Malcolm & Moler) coefficients b, c, d"""
    n = len(x)
    x = np.concatenate([[0.0], x])     # 1-based
    y = np.concatenate([[0.0], y])
    b = np.zeros(n + 1)
    c = np.zeros(n + 1)
    d = np.zeros(n + 1)
    if n < 2:
        raise ValueError("need at least two points")
    if n < 3:
        t = y[2] - y[1]
        b[1] = t / (x[2] - x[1])
        b[2] = b[1]
        return b[1:], c[1:], d[1:]
    nm1 = n - 1
    d[1] = x[2] - x[1]
    c[2] = (y[2] - y[1]) / d[1]
    for i in range(2, n):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]
    b[1] = -d[1]
    b[n] = -d[nm1]
    c[1] = c[n] = 0.0
    if n > 3:
        c[1] = c[3] / (x[4] - x[2]) - c[2] / (x[3] - x[1])
        c[n] = c[nm1] / (x[n] - x[n - 2]) - c[n - 2] / (x[nm1] - x[n - 3])
        c[1] = c[1] * d[1] * d[1] / (x[4] - x[1])
        c[n] = -c[n] * d[nm1] * d[nm1] / (x[n] - x[n - 3])
    for i in range(2, n + 1):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]
    c[n] = c[n] / b[n]
    for i in range(nm1, 0, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
    b[n] = (y[n] - y[n - 1]) / d[n - 1] + d[n - 1] * (c[n - 1] + 2.0 * c[n])
    for i in range(1, nm1 + 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]
    c[n] = 3.0 * c[n]
    d[n] = d[nm1]
    return b[1:], c[1:], d[1:]


def _spline_eval(u, x, y, b, c, d):
    n = len(x)
    v = np.empty(len(u))
    i = 0
    for l, ul in enumerate(u):
        if ul < x[i] or (i < n - 1 and x[i + 1] < ul):
            i = 0
            j = n
            while True:
                k = (i + j) // 2
                if ul < x[k]:
                    j = k
                else:
                    i = k
                if not (j > i + 1):
                    break
        dx = ul - x[i]
        v[l] = y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))
    return v


def spline(x, y, n=None):
    """stats::spline(x, y, n = 3*length(x), method = "fmm")"""
    x, y = _regularize(x, y)
    nx = len(x)
    if n is None:
        n = 3 * nx
    b, c, d = _fmm_spline(x, y)
    xmin, xmax = x[0], x[-1]
    xout = xmin + np.arange(n) * ((xmax - xmin) / (n - 1))
    xout[0] = xmin
    xout[-1] = xmax
    return xout, _spline_eval(xout, x, y, b, c, d)


def approx(x, y, xout):
    """stats::approx(x, y, xout) (linear, rule = 1, ties = mean)"""
    x, y = _regularize(x, y)
    xout = np.atleast_1d(np.asarray(xout, float))
    out = np.full(len(xout), np.nan)
    for k, v in enumerate(xout):
        if np.isnan(v) or v < x[0] or v > x[-1]:
            continue
        i, j = 0, len(x) - 1
        while i < j - 1:
            ij = (i + j) // 2
            if v < x[ij]:
                j = ij
            else:
                i = ij
        if v == x[j]:
            out[k] = y[j]
        elif v == x[i]:
            out[k] = y[i]
        else:
            out[k] = y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    return out


# ---------------------------------------------------------------------------
# lmList
# ---------------------------------------------------------------------------

def _hasScale_fam(famname):
    return famname in ("gaussian", "Gamma", "inverse.gaussian")


class LmList(dict):
    """R's ``lmList4`` object: a dict group name -> fitted lm/glm (or None
    when the fit failed), with attributes ``call``, ``pool``, ``groups``
    (ordered factor), ``origOrder`` and ``groupsForm``."""

    call = None
    pool = True
    groups = None
    origOrder = None
    groupsForm = None
    warningMsg = None

    # ------------------------------------------------------------ helpers
    def isGLMlist(self):
        vals = list(self.values())
        return len(vals) >= 1 and isinstance(vals[0], GlmFit)

    def hasScale(self):
        if not self.isGLMlist():
            return True
        return _hasScale_fam(next(iter(self.values())).family.base)

    def _nonnull(self):
        return [(k, v) for k, v in self.items() if v is not None]

    # ------------------------------------------------------------ coef
    def coef(self):
        """coef.lmList4: data frame (groups x coefficients), NA where not estimated"""
        coefs = {k: (v.coef() if v is not None else None) for k, v in self.items()}
        nn = [c for c in coefs.values() if c is not None]
        if not nn:
            return pd.DataFrame(index=list(self.keys()))
        allnames = []
        for c in nn:
            for nm in c.index:
                if nm not in allnames:
                    allnames.append(nm)
        co = pd.DataFrame(np.nan, index=list(self.keys()), columns=allnames)
        for k, c in coefs.items():
            if c is not None:
                co.loc[k, list(c.index)] = c.values
        co.attrs.update(label="Coefficients", effectNames=allnames, standardized=False)
        return co

    coefficients = coef

    def fixef(self):
        """nlme's fixef.lmList: column means of the coefficients"""
        return self.coef().mean(axis=0, skipna=True)

    def ranef(self):
        """nlme's ranef.lmList: coefficients minus their means"""
        cf = self.coef()
        return cf - cf.mean(axis=0, skipna=True)

    # ------------------------------------------------------------ sigma
    def pooledSD(self, allow_0_df=True):
        """pooled residual standard deviation, with attribute ``df``"""
        if not self.hasScale():
            if allow_0_df:
                return _PooledSD(1.0, df=np.nan)
            raise ValueError("no scale, hence no pooled SD for this object")
        ss, dfr = 0.0, 0
        for el in self.values():
            if el is None:
                continue
            res = el.residuals().values
            ss += np.sum(res ** 2)
            dfr += el.df_residual
        if dfr == 0:
            raise ValueError("No degrees of freedom for estimating std. dev.")
        return _PooledSD(math.sqrt(ss / dfr), df=dfr)

    def sigma(self):
        return float(self.pooledSD()) if self.hasScale() else 1.0

    # ------------------------------------------------------------ confint
    def confint(self, parm=None, level=0.95, pool=None):
        """confint.lmList4.  Returns a DataFrame indexed by group with
        MultiIndex columns (coefficient, bound) -- R's 3-d array
        ``[group, bound, coefficient]``.  As in lme4 (where the default
        ``pool <- object$pool`` is NULL), intervals are *not* pooled unless
        ``pool=True`` is given."""
        nn = self._nonnull()
        if not nn:
            return pd.DataFrame()
        template = nn[0][1].confint(parm=parm, level=level)
        parms = list(template.index)
        pct = list(template.columns)
        val = np.full((len(self), 2, len(parms)), np.nan)
        keys = list(self.keys())
        if pool:
            sd = self.pooledSD()
            a = (1 - level) / 2
            fac = float(sd) * stats.t.ppf([a, 1 - a], sd.df)
            for i, k in enumerate(keys):
                ob = self[k]
                if ob is None:
                    continue
                cu = ob.summary()["cov.unscaled"]
                se = np.array([math.sqrt(cu.loc[p, p]) if p in cu.index else np.nan
                               for p in parms])
                cf = ob.coef().reindex(parms).values
                val[i] = (cf[:, None] + np.outer(se, fac)).T
        else:
            for i, k in enumerate(keys):
                ob = self[k]
                if ob is None:
                    continue
                ci = ob.confint(parm=parm, level=level)
                ci = ci.reindex(parms)
                val[i] = ci.values.T
        cols = pd.MultiIndex.from_product([parms, pct])
        out = pd.DataFrame(val.transpose(0, 2, 1).reshape(len(keys), -1), index=keys,
                           columns=cols)
        return out

    # ------------------------------------------------------------ logLik
    def logLik(self, REML=False, pool=None):
        """nlme's logLik.lmList"""
        pool = self.pool if pool is None else pool
        if any(v is None for v in self.values()):
            raise ValueError("log-likelihood not available with NULL fits")
        if pool:
            aux = np.zeros(5)
            for el in self.values():
                res = el.residuals(type="working").values if isinstance(el, LmFit) else \
                    el.residuals().values
                p = el.rank
                n = len(res)
                w = el.prior_weights
                if w is None:
                    w = np.ones(n)
                else:
                    excl = w == 0
                    if np.any(excl):
                        res = res[~excl]
                        n = len(res)
                        w = w[~excl]
                aux += [n, np.sum(w * res ** 2), p, np.sum(np.log(w)),
                        np.sum(np.log(np.abs(np.diag(el.qr)[:p])))]
            N = aux[0] - REML * aux[2]
            val = (aux[3] - N * (math.log(2 * math.pi) + 1 - math.log(N) + math.log(aux[1]))) / 2 \
                - REML * aux[4]
            return _LogLik(val, df=aux[2] + 1, nobs=N, nall=aux[0])
        aux = [el.logLik(REML) for el in self.values()]
        return _LogLik(sum(a.value for a in aux), df=sum(a.df for a in aux),
                       nobs=sum(a.nobs for a in aux))

    # ------------------------------------------------------------ fitted etc.
    def _ordered_concat(self, fun):
        grp_counts = pd.Series(self.groups.codes).value_counts()
        keys = list(self.keys())
        pieces, names = [], []
        for j in self.origOrder:
            k = keys[j]
            ob = self[k]
            if ob is None:
                ng = int(grp_counts.get(self.groups.levels.index(k), 0))
                v = np.full(ng, np.nan)
            else:
                v = np.asarray(fun(ob))
            pieces.append(v)
            names += [k] * len(v)
        return pd.Series(np.concatenate(pieces) if pieces else np.zeros(0), index=names)

    def fitted(self):
        """nlme's fitted.lmList (values in original group order, named by group)"""
        return self._ordered_concat(lambda ob: ob.fitted().values)

    def residuals(self, type=None):
        return self._ordered_concat(
            lambda ob: (ob.residuals() if type is None else ob.residuals(type=type)).values)

    resid = residuals

    def predict(self, newdata=None):
        if newdata is None:
            return self.fitted()
        gname = deparse(self.groupsForm)
        g = np.asarray(newdata[gname]).astype(str)
        out = np.full(len(newdata), np.nan)
        for k, ob in self.items():
            sel = g == str(k)
            if ob is None or not np.any(sel):
                continue
            out[sel] = ob.predict(newdata.loc[sel]).values
        return pd.Series(out, index=newdata.index)

    def formula(self):
        return self.call["formula"]

    def getGroups(self):
        return self.groups

    def update(self, formula=None, **kw):
        """update.lmList4"""
        args = dict(self.call)
        if formula is not None:
            from .mermod import update_formula
            args["formula"] = update_formula(self.formula(), formula)
        args.update(kw)
        return lmList(**args)

    # ------------------------------------------------------------ summary / print
    def summary(self, pool=None):
        return SummaryLmList(self, pool=self.pool if pool is None else pool)

    def __str__(self):
        lines = ["Call: " + _deparse_call(getattr(self, "call_print", self.call)), "Coefficients:",
                 self.coef().to_string(float_format=lambda v: f"{v:.7g}")]
        if self.pool:
            lines.append("")
            psd = self.pooledSD()
            nobs = sum(len(v.fitted()) for v in self.values() if v is not None)
            lines.append(f"Degrees of freedom: {nobs} total; {_fmt_num(psd.df)} residual")
            lines.append(f"Residual standard error: {float(psd):.7g}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def show(self):
        print(self)


def _fmt_num(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _deparse_call(call):
    if call is None:
        return "lmList()"
    parts = []
    for k, v in call.items():
        if v is None or k.startswith("_"):
            continue
        if k == "formula":
            parts.append(f"formula = {deparse(v.expr) if isinstance(v, Formula) else v}")
        elif k == "data":
            parts.append(f"data = {v if isinstance(v, str) else '<data>'}")
        elif isinstance(v, (str, int, float, bool)):
            parts.append(f"{k} = {v!r}")
        elif isinstance(v, Family):
            parts.append(f"{k} = {v.base}")
    return "lmList(" + ", ".join(parts) + ")"


class _PooledSD(float):
    def __new__(cls, value, df):
        obj = super().__new__(cls, value)
        obj.df = df
        return obj


class _LogLik:
    def __init__(self, value, df, nobs, nall=None):
        self.value = float(value)
        self.df = df
        self.nobs = nobs
        self.nall = nall if nall is not None else nobs

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"'log Lik.' {self.value:.7g} (df={_fmt_num(float(self.df))})"


def _pct(p):
    s = f"{round(100 * p, 1):g}"
    return f"{s} %"


class SummaryLmList:
    """nlme's summary.lmList: per-coefficient tables (groups x statistics),
    pooled standard errors if ``pool``"""

    def __init__(self, obj: LmList, pool=True):
        sums = {k: (v.summary() if v is not None else None) for k, v in obj.items()}
        nn = [s for s in sums.values() if s is not None]
        if not nn:
            raise ValueError("all fits failed")
        keys = list(obj.keys())
        cnames = list(obj.coef().columns)
        statnames = list(nn[0]["coefficients"].columns)
        arr = np.full((len(keys), len(statnames), len(cnames)), np.nan)
        for i, k in enumerate(keys):
            s = sums[k]
            if s is None:
                continue
            c = s["coefficients"]
            for j, cn in enumerate(cnames):
                if cn in c.index:
                    arr[i, :, j] = c.loc[cn].values
        self.call = obj.call
        self.sigma = pd.Series([s["sigma"] if s is not None else np.nan for s in sums.values()],
                               index=keys)
        self.df = pd.DataFrame([list(s["df"]) if s is not None else [np.nan] * 3
                                for s in sums.values()], index=keys)
        self.r_squared = pd.Series([s.get("r.squared", np.nan) if s is not None else np.nan
                                    for s in sums.values()], index=keys)
        self.df_residual = None
        self.RSE = None
        if pool:
            poolSD = obj.pooledSD()
            dfRes = poolSD.df
            RSE = float(poolSD)
            corRSE = RSE / self.sigma.values
            arr[:, 1, :] = arr[:, 1, :] * corRSE[:, None]
            arr[:, 2, :] = arr[:, 0, :] / arr[:, 1, :]
            arr[:, 3, :] = 2 * stats.t.sf(np.abs(arr[:, 2, :]), dfRes)
            statnames = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
            self.df_residual = dfRes
            self.RSE = RSE
        self.coefficients_array = arr
        self.coefficients = {cn: pd.DataFrame(arr[:, :, j], index=keys, columns=statnames)
                             for j, cn in enumerate(cnames)}
        self.pool = pool
        self.groupsForm = obj.groupsForm

    def __getitem__(self, k):
        return {"coefficients": self.coefficients, "RSE": self.RSE, "sigma": self.sigma,
                "df.residual": self.df_residual, "df": self.df,
                "r.squared": self.r_squared, "pool": self.pool}[k]

    def __str__(self):
        out = ["Call:", "  Model: " + (deparse(self.call["formula"].expr)
                                        if self.call is not None else ""), "",
               "Coefficients:"]
        for cn, tab in self.coefficients.items():
            out.append(f"   {cn} ")
            out.append(tab.to_string(float_format=lambda v: f"{v:.6g}"))
        if self.pool:
            out.append("")
            out.append(f"Residual standard error: {self.RSE:.6g} on {_fmt_num(float(self.df_residual))}"
                       " degrees of freedom")
        return "\n".join(out)

    __repr__ = __str__


def _warnErrList(errs, fname):
    """utils::warnErrList message"""
    if not errs:
        return None
    msgs = [str(e) for e in errs]
    uniq = pd.Series(msgs).value_counts()
    if len(uniq) == 1:
        n = int(uniq.iloc[0])
        if n == 1:
            return f"{n} error caught in {fname}: {uniq.index[0]}"
        return f"{n} times caught the same error in {fname}: {uniq.index[0]}"
    tab = "\n".join(f"{c} {m}" for m, c in uniq.sort_values().items())
    return (f"{int(uniq.sum())} errors caught in {fname}.  The error messages and their "
            f"frequencies are\n{tab}")


def lmList(formula, data, family=None, subset=None, weights=None, na_action="na.omit",
           offset=None, pool=None, warn=True, _env=None, **kw):
    """List of lm (or glm, if ``family`` is given) objects with a common
    model, fitted separately for each level of the grouping factor:
    ``formula = "y ~ x1 + ... | g"``."""
    from .lmer import _caller_env, _name_of
    env = _env if _env is not None else _caller_env()
    formula = to_formula(formula, env)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    isGLM = family is not None
    if isGLM:
        family = get_family(family)
    if pool is None:
        pool = (not isGLM) or _hasScale_fam(family.base)
    ff = get_grpvars(formula)
    if len(ff) > 1:
        raise ValueError("lme4::lmList only works with simple grouping formulas (not involving * or /); \n"
                         "            use interaction(...) to construct an appropriate grouping variable first.")
    mCall = dict(formula=formula, data=data, family=family, subset=subset, weights=weights,
                 na_action=na_action, offset=offset, pool=pool, warn=warn)
    mCall_print = dict(mCall)
    mCall_print["data"] = _name_of(data, env)
    # model frame (NAs passed) -- only needed for the groups, weights, offsets
    sbf = Formula(subbars(formula.expr), formula.env)
    frm = model_frame(sbf, data, subset=subset, weights=weights, offset=offset,
                      na_action="na.pass", drop_unused_levels=True)
    idx = frm.attrs.get("orig_index")
    if idx is None:
        idx = np.arange(len(data))
    mform = modelFormula(formula)
    gkey = deparse(mform["groups"])
    if gkey in frm.columns:
        groups = frm[gkey]
    else:
        from .reformulas import eval_in_frame
        groups = eval_in_frame(mform["groups"], frm, env=formula.env)
    if not isinstance(groups, Factor):
        groups = as_factor(groups)
    groups = groups.droplevels()
    # apply subset to the original data
    data = data.iloc[np.asarray(idx)].copy()
    data["(weights)"] = frm.weights() if frm.weights() is not None else np.nan
    data["(offset)"] = frm.offset() if frm.offset() is not None else np.nan
    has_w = frm.weights() is not None
    has_o = frm.offset() is not None
    fname = "glm" if isGLM else "lm"

    def fitfun(d):
        try:
            with warnings.catch_warnings():
                if not warn:
                    warnings.simplefilter("ignore")
                w = d["(weights)"].values if has_w else None
                o = d["(offset)"].values if has_o else None
                d = d.drop(columns=["(weights)", "(offset)"])
                if isGLM:
                    return glm(mform["model"], family, d, weights=w, offset=o, **kw)
                return lm(mform["model"], d, weights=w, offset=o, **kw)
        except Exception as e:      # tryCatch(..., error = function(x) x)
            return e
    nms = list(groups.levels)
    val = {}
    for j, lev in enumerate(nms):
        sel = groups.codes == j
        val[lev] = fitfun(data.loc[sel] if not isinstance(data.index, pd.RangeIndex)
                          else data.iloc[np.nonzero(sel)[0]])
    errs = [v for v in val.values() if isinstance(v, Exception)]
    wMsg = _warnErrList(errs, f"{fname}(formula, data, ...)")
    for k in list(val):
        if isinstance(val[k], Exception):
            val[k] = None
    if warn and wMsg:
        if re.search(r"contrasts.* factors? .* 2 ", wMsg):
            warnings.warn(f"Fitting failed for {sum(v is None for v in val.values())} group(s), "
                          "probably because a factor only had one level" +
                          re.sub(r"(?s).*:", ":\n ", wMsg, count=1))
        else:
            warnings.warn(wMsg)
    ans = LmList(val)
    ans.call = mCall
    ans.call_print = mCall_print
    ans.pool = pool
    ans.groups = Factor(groups.codes, groups.levels, ordered=True)
    first_seen = pd.unique(np.asarray([nms[c] for c in groups.codes if c >= 0], dtype=object))
    ans.origOrder = [nms.index(x) for x in first_seen]
    ans.groupsForm = ff[0] if ff else None
    ans.warningMsg = wMsg
    return ans
