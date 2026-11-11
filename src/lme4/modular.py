"""Transcription of lme4/R/modular.R (plus mkRespMod and mkdevfun)."""
from __future__ import annotations

import math
import sys
import warnings
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

from ._backend import cpp
from .control import lmerControl, glmerControl
from .covariance import mkMkPar, mkMkTheta, upReTrms, anyStructured
from .family import Family, get_family, hasNoScale
from .modelframe import Factor, ModelFrame, model_frame, model_matrix, terms, as_factor
from .modules import merPredD, lmerResp, glmResp
from .optimizers import optwrap, GHrule
from .reformulas import (findbars_x, nobars, noSpecials, sub_specials, expandDoubleVerts,
                         mkReTrms, lme4_specials, lme4_reOnly, factorize)
from .rlang import Call, Const, Formula, Sym, as_formula, deparse, is_call
from . import options


class Rho(SimpleNamespace):
    """the environment of a deviance function"""


class DevFun:
    """a deviance function together with its environment ``rho``"""

    def __init__(self, f, rho):
        self.f = f
        self.rho = rho

    def __call__(self, par):
        return float(self.f(np.asarray(par, float)))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def doCheck(x):
    return isinstance(x, str) and x != "ignore"


def _act(cc, msg, cstr):
    if cc in ("warning", "warningSmall"):
        warnings.warn(msg)
    elif cc in ("stop", "stopSmall"):
        raise ValueError(msg)
    elif cc == "message":
        print(msg, file=sys.stderr)
    else:
        raise ValueError(f"unknown check level for '{cstr}'")


def wmsg(n, cmp_val, allow_n, msg1="", msg2="", msg3=""):
    if allow_n:
        unident = n < cmp_val
        cmp = "<"
        rstr = ""
    else:
        unident = n <= cmp_val
        cmp = "<="
        rstr = " and the residual variance (or scale parameter)"
    wstr = (f"{msg1} (={n}) {cmp} {msg2} (={cmp_val}){msg3}; the random-effects parameters"
            f"{rstr} are probably unidentifiable")
    return unident, wstr


def checkZdims(Ztlist, n, ctrl, allow_n=False):
    cc = ctrl["check_nobs_vs_nRE"]
    out = []
    if doCheck(cc):
        for nm, Zt in Ztlist.items():
            unident, w = wmsg(Zt.shape[1], Zt.shape[0], allow_n, "number of observations",
                              "number of random effects", f" for term ({nm})")
            if unident:
                _act(cc, w, "check.nobs.vs.nRE")
                if w not in out:
                    out.append(w)
    return out


def _rank_qr(M, tol=None):
    if M.size == 0:
        return 0
    r = np.linalg.qr(M, mode="r")
    d = np.abs(np.diag(r))
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps
    return int(np.sum(d > tol * d.max())) if d.max() > 0 else 0


def checkZrank(Zt, n, ctrl, nonSmall=1e6, allow_n=False):
    cc = ctrl["check_nobs_vs_rankZ"]
    if not doCheck(cc):
        return []
    d = Zt.shape
    if "Small" in cc and d[0] * d[1] > nonSmall:
        return []
    M = Zt.T.toarray() if d[0] < d[1] else Zt.toarray()
    rankZ = _rank_qr(M)
    unident, w = wmsg(n, rankZ, allow_n, "number of observations", "rank(Z)")
    if unident:
        _act(cc, w, "check.nobs.vs.rankZ")
        return [w]
    return []


def checkNlevels(flist, n, ctrl, allow_n=False):
    nlev = {k: f.droplevels().nlevels for k, f in flist.items()}
    out = []
    cc = ctrl["check_nlev_gtr_1"]
    if doCheck(cc) and any(v < 2 for v in nlev.values()):
        w = "grouping factors must have > 1 sampled level"
        _act(cc, w, "check.nlev.gtr.1")
        out.append(w)
    cc = ctrl["check_nobs_vs_nlev"]
    bad = [k for k, v in nlev.items() if (v > n if allow_n else v >= n)]
    if doCheck(cc) and bad:
        w = (f"number of levels of each grouping factor must be {'<=' if allow_n else '<'} "
             f"number of observations (problems: {', '.join(bad)})")
        _act(cc, w, "check.nobs.vs.nlev")
        out.append(w)
    cc = ctrl["check_nlev_gtreq_5"]
    if doCheck(cc) and any(v < 5 for v in nlev.values()):
        w = "grouping factors with < 5 sampled levels may give unreliable estimates"
        _act(cc, w, "check.nlev.gtreq.5")
        out.append(w)
    return out


def dqrdc2(x, tol=1e-7):
    """LINPACK dqrdc2 (R's default QR, with limited column pivoting).
    Returns (qr, rank, pivot (0-based), qraux)."""
    x = np.array(x, float, order="F", copy=True)
    n, p = x.shape
    jpvt = np.arange(p)
    qraux = np.array([np.linalg.norm(x[:, j]) for j in range(p)])
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0] = 1.0
    lup = min(n, p)
    k = p + 1   # 1-based as in Fortran
    for l in range(1, lup + 1):
        L = l - 1
        while l < k and qraux[L] < work2[L] * tol:
            # move column l to the end
            x[:, L:] = np.roll(x[:, L:], -1, axis=1)
            jpvt[L:] = np.roll(jpvt[L:], -1)
            qraux[L:] = np.roll(qraux[L:], -1)
            work1[L:] = np.roll(work1[L:], -1)
            work2[L:] = np.roll(work2[L:], -1)
            k -= 1
        if l == n:
            break
        nrmxl = np.linalg.norm(x[L:, L])
        if nrmxl == 0:
            continue
        if x[L, L] != 0:
            nrmxl = math.copysign(nrmxl, x[L, L])
        x[L:, L] /= nrmxl
        x[L, L] = 1 + x[L, L]
        for j in range(L + 1, p):
            t = -np.dot(x[L:, L], x[L:, j]) / x[L, L]
            x[L:, j] += t * x[L:, L]
            if qraux[j] == 0:
                continue
            tt = 1 - (abs(x[L, j]) / qraux[j]) ** 2
            tt = max(tt, 0.0)
            if abs(tt) < 1e-6:
                qraux[j] = np.linalg.norm(x[L + 1:, j])
                work1[j] = qraux[j]
            else:
                qraux[j] = qraux[j] * math.sqrt(tt)
        qraux[L] = x[L, L]
        x[L, L] = -nrmxl
    k = min(k - 1, n)
    return x, k, jpvt, qraux


def chkRank_drop_cols(X, colnames, kind, tol=1e-7, attrs=None):
    attrs = attrs if attrs is not None else {}
    if kind == "ignore":
        return X, colnames, attrs
    p = X.shape[1]
    if kind == "stop.deficient":
        _, r, _, _ = dqrdc2(X, tol)
        if r < p:
            raise ValueError(f"the fixed-effects model matrix is column rank deficient (rank(X) = {r} < {p} = p);\n"
                             "the fixed effects will be jointly unidentifiable")
        return X, colnames, attrs
    _, rnkX, pivot, _ = dqrdc2(X, tol)
    if rnkX == p:
        return X, colnames, attrs
    nd = p - rnkX
    msg = (f"fixed-effect model matrix is rank deficient so dropping {nd} column / coefficient"
           if nd == 1 else
           f"fixed-effect model matrix is rank deficient so dropping {nd} columns / coefficients")
    if kind != "silent.drop.cols":
        if kind == "warn+drop.cols":
            warnings.warn(msg)
        else:
            print(msg, file=sys.stderr)
    keep = pivot[:rnkX]
    dropped = [colnames[i] for i in range(p) if i not in set(keep)]
    X2 = X[:, keep]
    if _rank_qr(X2) < X2.shape[1]:
        raise ValueError("Dropping columns failed to produce full column rank design matrix")
    if "assign" in attrs:
        attrs["assign"] = np.asarray(attrs["assign"])[keep]
    attrs["msgRankdrop"] = msg
    attrs["col.dropped"] = dict(zip(dropped, (pivot[rnkX:] + 1).tolist()))
    return X2, [colnames[i] for i in keep], attrs


def checkScaleX(X, colnames, kind="warning", tol=1e3, attrs=None):
    attrs = attrs if attrs is not None else {}
    if kind is None or kind == "ignore":
        return X, attrs
    cont = np.array([not np.all(np.isin(X[:, j], [0, 1])) for j in range(X.shape[1])], bool)
    if not cont.any():
        attrs["msgScaleX"] = ""
        return X, attrs
    col_sd = X[:, cont].std(axis=0, ddof=1)
    with np.errstate(divide="ignore"):
        sdcomp = col_sd[:, None] / col_sd[None, :]
        logcomp = np.abs(np.log(sdcomp[np.tril_indices(len(col_sd), -1)]))
        logsd = np.abs(np.log(col_sd))
    if np.any(np.concatenate([logcomp, logsd]) > math.log(tol)):
        w = "Some predictor variables are on very different scales:"
        if kind in ("warning", "stop"):
            w += " consider rescaling. \nYou may also use (g)lmerControl(autoscale = TRUE) to improve numerical stability."
            if kind == "warning":
                warnings.warn(w)
            else:
                raise ValueError(w)
        else:
            w += " auto-rescaled (results NOT adjusted)"
            X = X.copy()
            X[:, cont] = X[:, cont] / col_sd
            attrs["scaled:scale"] = dict(zip([c for c, k in zip(colnames, cont) if k], col_sd))
            if kind == "warn+rescale":
                warnings.warn(w)
        attrs["msgScaleX"] = w
    else:
        attrs["msgScaleX"] = ""
    return X, attrs


def checkResponse(y, ctrl):
    cc = ctrl.get("check_response_not_const")
    if doCheck(cc) and len(np.unique(np.asarray(y))) < 2:
        _act(cc, "Response is constant", "check.response.not.const")


# ---------------------------------------------------------------------------
# mkFormula / lFormula / glFormula
# ---------------------------------------------------------------------------

def mkFormula(formula: Formula, data, contrasts, control, isGLMM=False, allow_n=False,
              check_zero_rows=False, check_na_Zt=False, set_varnames_fixed=False,
              subset=None, weights=None, na_action="na.omit", offset=None,
              mustart=None, etastart=None, env=None):
    if formula.lhs is None and control.get("check_formula_LHS") == "stop":
        raise ValueError("formula must have a LHS")
    if options.get("lme4.doublevert.default") == "split":
        formula = formula.with_rhs(expandDoubleVerts(formula.rhs))
    fr_form_ = noSpecials(formula, delete=False, specials=lme4_specials)
    fr_form = sub_specials(fr_form_, specials=("|", "||"), keep_args=(2, 2))
    extras = {}
    if isGLMM:
        if mustart is not None:
            extras["(mustart)"] = mustart
        if etastart is not None:
            extras["(etastart)"] = etastart
    fr = model_frame(fr_form, data, subset=subset, weights=weights, offset=offset,
                     na_action=na_action, env=env, extras=extras)
    if check_zero_rows and fr.nrow == 0:
        raise ValueError("0 (non-NA) cases")
    fr = factorize(fr_form, fr, char_only=True)
    fr.attrs["formula"] = formula
    fr.attrs["offset"] = offset
    n = fr.nrow
    bb1 = findbars_x(formula, specials=lme4_specials, default_special="us", target="|",
                     expand_doublevert_method=options.get("lme4.doublevert.default"))
    bb0 = [b[1] for b in bb1]
    extra_data = _data_rows(data, fr)
    reTrms = mkReTrms(bb0, fr, calc_lambdat=False, env=env, extra=extra_data)
    bb1 = [bb1[i] for i in reTrms["ord"]]
    reTrms = upReTrms(reTrms, bb1)
    ctrl = dict(control)
    if anyStructured(reTrms["reCovs"]):
        ctrl["check_nobs_vs_nRE"] = "ignore"
    wNlev = checkNlevels(reTrms["flist"], n, ctrl, allow_n)
    wZd = checkZdims(reTrms["Ztlist"], n, ctrl, allow_n)
    if check_na_Zt and np.any(np.isnan(reTrms["Zt"].data)):
        raise ValueError("NA in Z (random-effects model matrix): please use "
                         "\"na.action='na.omit'\" or \"na.action='na.exclude'\"")
    wZr = checkZrank(reTrms["Zt"], n, ctrl, 1e6, allow_n)
    # fixed-effects model frame/matrix
    fixedform = fr_form_.with_rhs(nobars(fr_form_.rhs))
    fixedfr = model_frame(fixedform, data, subset=subset, weights=weights, offset=offset,
                          na_action=na_action, env=env)
    fr.terms.extra["predvars.fixed"] = fixedfr.terms.predvars
    fr.terms.extra["fixed.terms"] = fixedfr.terms
    if set_varnames_fixed:
        fr.terms.extra["varnames.fixed"] = list(fixedfr.columns)
    ranform = fr_form_.with_rhs(sub_specials(noSpecials(lme4_reOnly(fr_form_).rhs, delete=False,
                                                        specials=lme4_specials),
                                             specials=("|", "||"), keep_args=(2, 2)))
    ranfr = model_frame(ranform, data, subset=subset, na_action=na_action, env=env)
    fr.terms.extra["predvars.random"] = ranfr.terms.predvars
    # model matrix, computed on the full model frame (as R does)
    from .modelframe import data_to_dict
    tfix = terms(fixedform, data_columns=list(data_to_dict(data)) if data is not None else None)
    # evaluate the fixed terms' variables from the full frame / data rows
    fr_fix = _frame_for_terms(tfix, fr, fixedfr)
    X, cn, assign, contr = model_matrix(tfix, fr_fix, contrasts)
    Xattrs = {"assign": assign, "contrasts": contr}
    if control.get("autoscale"):
        X = X.copy()
        if "(Intercept)" in cn:
            sl = [j for j, c in enumerate(cn) if c != "(Intercept)"]
        else:
            sl = list(range(X.shape[1]))
        ce = X[:, sl].mean(axis=0)
        sc = X[:, sl].std(axis=0, ddof=1)
        X[:, sl] = (X[:, sl] - ce) / sc
        Xattrs["scaled:center"] = dict(zip([cn[j] for j in sl], ce))
        Xattrs["scaled:scale"] = dict(zip([cn[j] for j in sl], sc))
    X, cn, Xattrs = chkRank_drop_cols(X, cn, control.get("check_rankX") or "message+drop.cols", 1e-7, Xattrs)
    X, Xattrs = checkScaleX(X, cn, control.get("check_scaleX") or "warning", attrs=Xattrs)
    Xattrs["colnames"] = cn
    wmsgs = {"Nlev": wNlev, "Zdims": wZd, "Zrank": wZr}
    return dict(fr=fr, X=X, Xattrs=Xattrs, reTrms=reTrms, formula=formula, wmsgs=wmsgs,
                fixedfr=fixedfr)


def _data_rows(data, fr):
    """original data columns restricted to the rows kept in the model frame"""
    from .modelframe import data_to_dict
    if data is None:
        return None
    d = data_to_dict(data)
    idx = fr.attrs.get("orig_index")
    out = {}
    for k, v in d.items():
        out[k] = v.subset(idx) if isinstance(v, Factor) else np.asarray(v)[idx]
    return out


def _frame_for_terms(tt, fr, fallback):
    cols = {}
    for key in tt.varnames:
        if key in fr.columns:
            cols[key] = fr.columns[key]
        else:
            cols[key] = fallback.columns[key]
    return ModelFrame(cols, tt, fr.row_names)


def lFormula(formula, data=None, REML=True, subset=None, weights=None, na_action="na.omit",
             offset=None, contrasts=None, control=None, env=None, **kw):
    control = control or lmerControl()
    formula = as_formula(formula, env)
    if kw.get("family") is not None:
        return glFormula(formula, data, family=kw["family"], subset=subset, weights=weights,
                         na_action=na_action, offset=offset, contrasts=contrasts, env=env)
    res = mkFormula(formula, data, contrasts, control["checkControl"], isGLMM=False, allow_n=False,
                    check_zero_rows=True, check_na_Zt=True, set_varnames_fixed=True,
                    subset=subset, weights=weights, na_action=na_action, offset=offset,
                    env=formula.env)
    res["REML"] = REML
    return res


def glFormula(formula, data=None, family="gaussian", subset=None, weights=None,
              na_action="na.omit", offset=None, contrasts=None, start=None, mustart=None,
              etastart=None, control=None, env=None, **kw):
    control = control or glmerControl()
    formula = as_formula(formula, env)
    family = get_family(family)
    if family.is_gaussian_identity():
        return lFormula(formula, data, subset=subset, weights=weights, na_action=na_action,
                        offset=offset, contrasts=contrasts, env=env)
    res = mkFormula(formula, data, contrasts, control["checkControl"], isGLMM=True, allow_n=True,
                    subset=subset, weights=weights, na_action=na_action, offset=offset,
                    mustart=mustart, etastart=etastart, env=formula.env)
    if isinstance(start, dict):
        fixef = start.get("fixef", start.get("beta"))
        res["fr"].attrs["start"] = fixef
    res["family"] = family
    return res


# ---------------------------------------------------------------------------
# mkRespMod (lme4/R/utilities.R)
# ---------------------------------------------------------------------------

def mkRespMod(fr: ModelFrame = None, REML=None, family: Family = None, y=None, weights=None,
              offset=None, etastart=None, mustart=None):
    if fr is not None:
        y = fr.response()
        offset = fr.offset()
        weights = fr.weights()
        etastart = fr.get("(etastart)")
        mustart = fr.get("(mustart)")
        n = fr.nrow
    else:
        n = 0 if y is None else (len(y) if not isinstance(y, Factor) else len(y))
    isGLMM = family is not None
    if y is not None:
        if isinstance(y, Factor):
            if not (isGLMM and family.base == "binomial"):
                raise ValueError("response must be numeric")
        else:
            ya = np.asarray(y)
            if ya.dtype == bool and not (isGLMM and family.base == "binomial"):
                y = ya.astype(int)
            elif ya.dtype.kind not in "biuf":
                raise ValueError("response must be numeric" if not (isGLMM and family.base == "binomial")
                                 else "response must be numeric or factor")
            ya = np.asarray(y, float)
            if not np.all(np.isfinite(ya)):
                raise ValueError("NA/NaN/Inf in 'y'")
    if y is None:
        y = np.zeros(0)
    N = n
    offset = np.zeros(N) if offset is None else (np.full(N, float(offset)) if np.ndim(offset) == 0 else np.asarray(offset, float))
    if weights is not None:
        weights = np.asarray(weights, float)
        if len(weights) != n or np.any(weights < 0):
            raise ValueError("length(weights) == n && all(weights >= 0) is not TRUE")
    else:
        weights = np.ones(n)
    if isGLMM:
        if len(y) > 0:
            ini = family.initialize(y, weights, n)
            y, weights, nn, mst = ini["y"], ini["weights"], ini["n"], ini["mustart"]
        else:
            nn, mst = np.ones(0), np.zeros(0)
        if mustart is not None:
            mst = np.asarray(mustart, float)
        elif family.base == "Gamma" and etastart is None and len(y) > 0:
            mst = np.full(len(mst), np.mean(mst))
        ans = glmResp(family=family, y=y, weights=weights, offset=offset, n=nn, etastart=etastart)
        if len(y) > 0:
            ans.ptr.updateMu(np.asarray(etastart, float) if etastart is not None else family.linkfun(mst))
        ans.start = fr.attrs.get("start") if fr is not None else None
        return ans
    return lmerResp(y=np.asarray(y, float), weights=weights, offset=offset, REML=REML or 0)


# ---------------------------------------------------------------------------
# start values
# ---------------------------------------------------------------------------

def getStart(start, rho, nAGQ):
    par = par0 = rho.mkPar(np.asarray(rho.pp.theta))
    fixef = fixef0 = np.asarray(rho.pp.delb)
    if start is None:
        pass
    elif isinstance(start, (list, tuple, np.ndarray)) and not isinstance(start, dict):
        par = np.asarray(start, float)
    elif isinstance(start, dict):
        valid = ["par", "theta"] + (["fixef", "beta"] if nAGQ > 0 else [])
        invalid = [k for k in start if k not in valid]
        if invalid:
            raise ValueError(f"'start' has invalid names {invalid}")
        if "par" in start or "theta" in start:
            if "par" in start and "theta" in start:
                print("ignoring start$theta")
            par = np.asarray(start.get("par", start.get("theta")), float)
        if "fixef" in start or "beta" in start:
            if "fixef" in start and "beta" in start:
                print("ignoring start$beta")
            fx = start.get("fixef", start.get("beta"))
            if isinstance(fx, dict):
                cn = rho.pp.colnames or []
                fx = [fx[c] for c in cn]
            fixef = np.asarray(fx, float)
    else:
        raise ValueError("'start' is not NULL, a numeric vector, or a list")
    if len(par) != len(par0):
        raise ValueError(f"starting value of 'par' has length not equal to {len(par0)}")
    if len(fixef) != len(fixef0):
        raise ValueError(f"starting value of 'fixef' has length not equal to {len(fixef0)}")
    return np.concatenate([par, fixef]) if nAGQ > 0 else np.asarray(par, float)


def updateStart(start, par):
    if start is None or not isinstance(start, dict):
        return par
    start = dict(start)
    if "par" in start:
        start["par"] = par
    if "theta" in start:
        start["theta"] = par
    return start


# ---------------------------------------------------------------------------
# lmer deviance function
# ---------------------------------------------------------------------------

def _new_pp(reTrms, X, Xattrs):
    return merPredD(X, reTrms["Zt"], reTrms["Lambdat"], reTrms["Lind"], reTrms["theta"],
                    n=X.shape[0], Xattrs=Xattrs)


def _var(x):
    x = np.asarray(x, float)
    return x.var(ddof=1) if len(x) > 1 else np.nan


def _ave(y, f: Factor):
    out = np.empty(len(y))
    for k in range(f.nlevels):
        m = f.codes == k
        if m.any():
            out[m] = y[m].mean()
    return out


def mkLmerDevfun(fr, X, reTrms, REML=True, start=None, verbose=0, control=None, Xattrs=None,
                 **kw):
    control = control or lmerControl()
    p = X.shape[1]
    rho = Rho()
    rho.pp = _new_pp(reTrms, X, Xattrs)
    REMLpass = p if REML else 0
    rho.resp = mkRespMod(fr, REML=REMLpass)
    rho.mkPar = mkMkPar(reTrms["reCovs"])
    rho.mkTheta = mkMkTheta(reTrms["reCovs"])
    rho.reCovs = reTrms["reCovs"]
    pp, resp, mkTheta = rho.pp, rho.resp, rho.mkTheta

    def devfun(par):
        return cpp.lmer_Deviance(pp.ptr, resp.ptr, mkTheta(np.asarray(par, float)))
    # simple starting values when all terms are (1|f)
    y = fr.response() if fr is not None else None
    if start is None and all(cn == ["(Intercept)"] for _, cn in reTrms["cnms"]) and \
            len(reTrms["flist"]) == len(reTrms["lower"]) and y is not None:
        y = np.asarray(y, float)
        v = np.array([_var(_ave(y, f)) for f in reTrms["flist"].values()])
        v_e = _var(y) - v.sum()
        if not np.isnan(v_e) and v_e > 0:
            v_rel = v / v_e
            if np.all(v_rel >= np.asarray(reTrms["lower"]) ** 2):
                rho.pp.setTheta(np.sqrt(v_rel))
    df = DevFun(devfun, rho)
    if len(rho.resp.y) > 0:
        df(rho.mkPar(np.asarray(rho.pp.theta)))
    rho.lower = np.asarray(reTrms["lower"], float)
    rho.upper = np.asarray(reTrms.get("upper", np.full(len(rho.lower), np.inf)), float)
    return df


def optimizeLmer(devfun, optimizer="nloptwrap", restart_edge=True, boundary_tol=1e-5,
                 start=None, verbose=0, control=None, calc_derivs=None, force_calc_derivs=False,
                 use_last_params=False):
    rho = devfun.rho
    start = getStart(start, rho, 0)
    lower, upper = rho.lower, rho.upper
    kw = dict(calc_derivs=calc_derivs, force_calc_derivs=force_calc_derivs,
              use_last_params=use_last_params)
    opt = optwrap(optimizer, devfun, start, lower=lower, upper=upper, control=control,
                  adj=False, verbose=verbose, rho=rho, **kw)
    if restart_edge:
        par0 = opt.attrs.get("last_par")
        par0 = opt["par"] if par0 is None else par0
        wl = np.nonzero(par0 == lower)[0]
        wu = np.nonzero(par0 == upper)[0]
        if len(wl) > 0 or len(wu) > 0:
            par0 = np.array(par0, float)
            d0 = devfun(par0)
            btol = 1e-5
            bgrad = []
            for i, bval, bt in zip(list(wl) + list(wu), list(lower[wl]) + list(upper[wu]),
                                   [btol] * len(wl) + [-btol] * len(wu)):
                par = par0.copy()
                par[i] = bval + bt
                bgrad.append((devfun(par) - d0) / bt)
            devfun(par0)
            if any(np.isnan(bgrad)):
                warnings.warn("some gradient components are NA near boundaries, skipping boundary check")
                return opt
            if any(b < 0 for b in bgrad):
                if verbose:
                    print("some theta parameters on the boundary, restarting", file=sys.stderr)
                opt = optwrap(optimizer, devfun, opt["par"], lower=lower, upper=upper,
                              control=control, adj=False, verbose=verbose, rho=rho, **kw)
    if boundary_tol > 0:
        return check_boundary(rho, opt, devfun, boundary_tol)
    return opt


def check_boundary(rho, opt, devfun, boundary_tol):
    par0 = np.array(opt["par"], float)
    lower, upper = rho.lower, rho.upper
    dl = par0 - lower
    du = upper - par0
    if getattr(rho, "dpars", None) is not None:
        dl = dl[rho.dpars]
        du = du[rho.dpars]
    wl = np.nonzero((0 < dl) & (dl < boundary_tol))[0]
    wu = np.nonzero((0 < du) & (du < boundary_tol))[0]
    if len(wl) > 0 or len(wu) > 0:
        for i in wl:
            par = par0.copy()
            par[i] = lower[i]
            if devfun(par) < opt["fval"]:
                par0[i] = par[i]
        for i in wu:
            par = par0.copy()
            par[i] = upper[i]
            if devfun(par) < opt["fval"]:
                par0[i] = par[i]
        opt["par"] = par0
        opt["fval"] = devfun(par0)
    return opt


# ---------------------------------------------------------------------------
# glmer deviance functions
# ---------------------------------------------------------------------------

def computeQEff(X, reTrms, enable, family):
    if not enable or hasNoScale(family):
        return float("nan")
    try:
        full = np.hstack([X, reTrms["Zt"].T.toarray()])
        return float(_rank_qr(full))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"glmerControl(disp_dof_correction=TRUE): computing the degrees-of-freedom "
                      f"correction failed ({e}); proceeding without it.")
        return float("nan")


def glmerPwrssUpdate(pp, resp, tol, GQmat, compDev=True, grpFac=None, maxit=70, verbose=0,
                     dispProfile=True, maxPhiIter=100, qEff=float("nan")):
    nAGQ = GQmat.shape[0]
    if compDev:
        if nAGQ < 2:
            return cpp.glmerLaplace(pp.ptr, resp.ptr, nAGQ, tol, int(maxit), int(verbose),
                                    bool(dispProfile), int(maxPhiIter), float(qEff))
        return cpp.glmerAGQ(pp.ptr, resp.ptr, tol, int(maxit), GQmat,
                            (grpFac.codes + 1).astype(np.int32), int(verbose), bool(dispProfile),
                            int(maxPhiIter), float(qEff))
    oldpdev = np.finfo(float).max
    uOnly = nAGQ != 0
    i = 0
    while True:
        pp.updateXwts(np.asarray(resp.sqrtWrkWt()))
        pp.updateDecomp()
        pp.updateRes(np.asarray(resp.wtWrkResp()))
        if uOnly:
            pp.solveU()
        else:
            pp.solve()
        resp.ptr.updateMu(np.asarray(pp.linPred(1.0)))
        pdev = resp.resDev() + pp.sqrL(1.0)
        if verbose > 2:
            print(f"{i}: {pdev}")
        if np.isnan(pdev):
            raise RuntimeError("encountered NA in PWRSS update")
        if abs((oldpdev - pdev) / pdev) < tol:
            break
        if pdev > oldpdev:
            raise RuntimeError("PIRLS update failed")
        oldpdev = pdev
        i += 1
    pp.updateXwts(np.asarray(resp.sqrtWrkWt()))
    pp.updateDecomp()
    return resp.Laplace(pp.ldL2(), 0.0, pp.sqrL(1.0))


def mkdevfun(rho, nAGQ=1, maxit=None, verbose=0, control=None):
    """lme4:::mkdevfun"""
    control = control or {}
    resp = rho.resp
    if isinstance(resp, lmerResp):
        pp, mkTheta = rho.pp, rho.mkTheta

        def ff(par):
            return cpp.lmer_Deviance(pp.ptr, resp.ptr, mkTheta(np.asarray(par, float)))
        return DevFun(ff, rho)
    if isinstance(resp, glmResp):
        maxit = 100 if maxit is None else maxit
        if control.get("tolPwrss") is not None:
            rho.tolPwrss = control["tolPwrss"]
        if control.get("compDev") is not None:
            rho.compDev = control["compDev"]
        if control.get("disp_method") is not None:
            rho.dispProfile = control["disp_method"] == "moment"
        if control.get("maxPhiIter") is not None:
            rho.maxPhiIter = int(control["maxPhiIter"])
        pp = rho.pp
        if nAGQ == 0:
            def ff(par):
                resp.ptr.updateMu(rho.lp0)
                pp.setTheta(rho.mkTheta(np.asarray(par, float)))
                p = glmerPwrssUpdate(pp, resp, rho.tolPwrss, GHrule(0), rho.compDev,
                                     maxit=maxit, verbose=verbose, dispProfile=rho.dispProfile,
                                     maxPhiIter=rho.maxPhiIter, qEff=rho.qEff)
                resp.updateWts()
                return p
        else:
            def ff(pars):
                pars = np.asarray(pars, float)
                resp.setOffset(rho.baseOffset)
                resp.ptr.updateMu(rho.lp0)
                pp.setTheta(rho.mkTheta(pars[rho.dpars]))
                spars = np.delete(pars, rho.dpars)
                offset = rho.baseOffset if len(spars) == 0 else rho.baseOffset + pp.X @ spars
                resp.setOffset(offset)
                p = glmerPwrssUpdate(pp, resp, rho.tolPwrss, rho.GQmat, rho.compDev,
                                     grpFac=rho.fac, maxit=maxit, verbose=verbose,
                                     dispProfile=rho.dispProfile, maxPhiIter=rho.maxPhiIter,
                                     qEff=rho.qEff)
                resp.updateWts()
                return p
        return DevFun(ff, rho)
    from .modules import nlsResp
    if isinstance(resp, nlsResp):
        maxit = 300 if maxit is None else maxit
        if nAGQ > 1:
            raise NotImplementedError("nAGQ > 1  not yet implemented for nlmer models")
        rho.tolPwrss = control.get("tolPwrss", getattr(rho, "tolPwrss", 1e-10))
        rho.maxit = maxit
        pp = rho.pp
        if nAGQ == 0:
            def ff(par):
                return cpp.nlmerLaplace(pp.ptr, resp.ptr, rho.mkTheta(np.asarray(par, float)),
                                        np.asarray(rho.u0, float), np.asarray(rho.beta0, float),
                                        int(verbose), False, float(rho.tolPwrss), int(maxit))
        else:
            def ff(pars):
                pars = np.asarray(pars, float)
                return cpp.nlmerLaplace(pp.ptr, resp.ptr, rho.mkTheta(pars[rho.dpars]),
                                        np.asarray(rho.u0, float), np.delete(pars, rho.dpars),
                                        int(verbose), True, float(rho.tolPwrss), int(maxit))
        return DevFun(ff, rho)
    raise NotImplementedError("code not yet written")


def mkGlmerDevfun(fr, X, reTrms, family, nAGQ=None, verbose=0, maxit=100, control=None,
                  Xattrs=None, **kw):
    control = control or glmerControl()
    if nAGQ is None:
        nAGQ = 0 if control["nAGQ0initStep"] else 1
    nAGQ = int(nAGQ)
    if not (0 <= nAGQ <= 100):
        raise ValueError("0 <= nAGQ <= 100 is not TRUE")
    rho = Rho(verbose=int(verbose), maxit=int(maxit), tolPwrss=control["tolPwrss"],
              compDev=control["compDev"],
              dispProfile=(control.get("disp_method") or "moment") == "moment",
              maxPhiIter=int(control.get("maxPhiIter") or 100),
              qEff=computeQEff(X, reTrms, control.get("disp_dof_correction", True), family))
    rho.pp = _new_pp(reTrms, X, Xattrs)
    rho.resp = mkRespMod(fr, family=family)
    rho.mkPar = mkMkPar(reTrms["reCovs"])
    rho.mkTheta = mkMkTheta(reTrms["reCovs"])
    rho.reCovs = reTrms["reCovs"]
    if len(rho.resp.y) > 0:
        checkResponse(rho.resp.y, control["checkControl"])
        cpp.glmerLaplace(rho.pp.ptr, rho.resp.ptr, int(nAGQ > 0), control["tolPwrss"], int(maxit),
                         int(verbose), bool(rho.dispProfile), int(rho.maxPhiIter), float(rho.qEff))
        rho.lp0 = np.asarray(rho.pp.linPred(1.0))
    devfun = mkdevfun(rho, 0, maxit=maxit, verbose=verbose, control=control)
    if nAGQ > 0:
        return updateGlmerDevfun(devfun, reTrms, nAGQ)
    rho.nAGQ = 0
    rho.lower = np.asarray(reTrms["lower"], float)
    rho.upper = np.asarray(reTrms.get("upper", np.full(len(rho.lower), np.inf)), float)
    return devfun


def updateGlmerDevfun(devfun, reTrms, nAGQ=1):
    nAGQ = int(nAGQ)
    if not (0 <= nAGQ <= 100):
        raise ValueError("0 <= nAGQ <= 100 is not TRUE")
    if nAGQ > 1:
        if len(reTrms["flist"]) != 1 or len(reTrms["cnms"][0][1]) != 1:
            raise ValueError("nAGQ > 1 is only available for models with a single, scalar random-effects term")
    rho = devfun.rho
    rho.nAGQ = nAGQ
    p = len(rho.pp.beta0)
    rho.lower = np.concatenate([np.asarray(reTrms["lower"], float), np.full(p, -np.inf)])
    rho.upper = np.concatenate([np.asarray(reTrms.get("upper"), float), np.full(p, np.inf)])
    rho.lp0 = np.asarray(rho.pp.linPred(1.0))
    rho.dpars = np.arange(len(reTrms["lower"]))
    rho.baseOffset = np.array(rho.resp.offset, copy=True)
    rho.GQmat = GHrule(nAGQ)
    rho.fac = list(reTrms["flist"].values())[0]
    return mkdevfun(rho, nAGQ)


def optimizeGlmer(devfun, optimizer=None, restart_edge=False, boundary_tol=1e-5, verbose=0,
                  control=None, nAGQ=None, stage=None, start=None, calc_derivs=None,
                  force_calc_derivs=False, use_last_params=False):
    if nAGQ is None:
        nAGQ = 0 if (stage is None or stage == 1) else 1
    if stage is not None and (nAGQ > 0) != (stage != 1):
        raise ValueError("incompatible 'nAGQ' and 'stage'")
    if optimizer is None:
        optimizer = "Nelder_Mead" if nAGQ > 0 else "bobyqa"
    rho = devfun.rho
    start = getStart(start, rho, nAGQ)
    opt = optwrap(optimizer, devfun, start, lower=rho.lower, upper=rho.upper, control=control,
                  adj=nAGQ > 0, verbose=verbose, rho=rho, calc_derivs=calc_derivs,
                  force_calc_derivs=force_calc_derivs, use_last_params=use_last_params)
    if nAGQ > 0:
        rho.resp.setOffset(rho.baseOffset)
    if restart_edge:
        raise NotImplementedError("restart_edge not implemented for optimizeGlmer yet")
    if boundary_tol > 0:
        opt = check_boundary(rho, opt, devfun, boundary_tol)
        if nAGQ > 0:
            rho.resp.setOffset(rho.baseOffset)
    return opt
