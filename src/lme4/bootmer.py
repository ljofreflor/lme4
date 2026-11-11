"""Transcription of lme4/R/bootMer.R (model-based parametric and
semiparametric bootstrap for mixed models), the fast ``refit()`` of
lme4/R/lmer.R that it relies on, and the parts of the *boot* package
(``boot.ci`` percentile / basic / normal intervals) needed by
``confint.bootMer`` and ``confint.merMod(method = "boot")``.

Random numbers come from numpy (``np.random.default_rng(seed)``), so
bootstrap replicates cannot reproduce R's streams bit-for-bit.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["bootMer", "BootMer", "refit_merMod", "print_bootWarnings", "boot_ci"]


# ---------------------------------------------------------------------------
# refit.merMod (fast: replace the response, re-optimize)
# ---------------------------------------------------------------------------

def refit_merMod(object, newresp=None, control=None, maxit=100, verbose=0):
    """refit.merMod: refit a model with a new response vector (or two-column
    successes/failures matrix for binomial models) re-using the model
    structures of ``object`` (no re-evaluation of the formula)."""
    from .control import lmerControl, glmerControl
    from .covariance import mkMkPar, mkMkTheta
    from .lmer import mkMerMod
    from .modular import Rho, mkdevfun, mkRespMod, glmerPwrssUpdate
    from .modules import lmerResp
    from .optimizers import optwrap, checkConv, GHrule

    if object.isNLMM():
        raise NotImplementedError("'refit.merMod' does not yet support nonlinear mixed models; "
                                  "try 'update.merMod' instead")
    haveGLMM = object.isGLMM()
    oi = object.optinfo or {}
    ctrl_arg = control
    if ctrl_arg is not None:
        control = ctrl_arg.copy() if hasattr(ctrl_arg, "copy") else dict(ctrl_arg)
        if len(control.get("optCtrl") or {}) == 0:
            obj_control = {k: v for k, v in (oi.get("control") or {}).items()
                           if k not in ("xst", "xt")}
            control["optCtrl"] = obj_control
    else:
        control = glmerControl() if haveGLMM else lmerControl()
    pp = object.pp.copy()
    dc = object.devcomp
    nAGQ = dc["dims"]["nAGQ"] if haveGLMM else None
    rr_old = object.resp
    weights = np.asarray(rr_old.weights, float)
    offset = np.asarray(rr_old.offset, float)
    if newresp is None:
        newresp = rr_old.y
    newresp = np.asarray(newresp)
    nobs = len(rr_old.y)
    if newresp.shape[0] != nobs:
        raise ValueError(f"'newresp' has {newresp.shape[0]} rows: expected {nobs} "
                         "(the number of observations used in the fit)")
    if not haveGLMM:
        rr = lmerResp(y=np.asarray(newresp, float).ravel(), weights=weights, offset=offset,
                      REML=rr_old.REML)
    else:
        fam = object.family()
        if newresp.ndim == 2 and newresp.shape[1] == 2:
            ntot = newresp.sum(axis=1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = newresp[:, 0] / ntot
            weights = ntot
        else:
            y = np.asarray(newresp, float).ravel()
        rr = mkRespMod(fr=None, family=fam, y=y, weights=weights, offset=offset)
    if haveGLMM:
        GQmat = GHrule(nAGQ)
        dispProfile = (control.get("disp_method") or "moment") == "moment"
        maxPhiIter = int(control.get("maxPhiIter") or 100)
        glmerPwrssUpdate(pp, rr, control.get("tolPwrss", 1e-7), GQmat, maxit=maxit,
                         grpFac=list(object.flist.values())[0] if nAGQ > 1 else None,
                         dispProfile=dispProfile, maxPhiIter=maxPhiIter)
    reCovs = object.getReCovs()
    rho = Rho(pp=pp, resp=rr, mkPar=mkMkPar(reCovs), mkTheta=mkMkTheta(reCovs))
    if haveGLMM:
        dims = dc["dims"]
        qEff = dims.get("qEff")
        rho.tolPwrss = dc["cmp"]["tolPwrss"]
        rho.compDev = dims.get("compDev", True)
        rho.dispProfile = True if dims.get("dispProfile") is None else bool(dims["dispProfile"])
        rho.maxPhiIter = 100 if dims.get("maxPhiIter") is None else int(dims["maxPhiIter"])
        rho.qEff = float("nan") if qEff is None else float(qEff)
        rho.nAGQ = nAGQ
        rho.lp0 = np.asarray(pp.linPred(1.0))
        rho.baseOffset = np.array(offset, copy=True)
        rho.GQmat = GHrule(nAGQ)
        rho.fac = list(object.flist.values())[0]
        rho.verbose = verbose
        rho.dpars = np.arange(object.getParLength())
    ff = mkdevfun(rho, nAGQ=nAGQ if haveGLMM else 0, maxit=maxit, verbose=verbose)
    x0 = rho.mkPar(np.asarray(rho.pp.theta))
    lower = np.asarray(object.lower, float)
    upper = np.asarray(object.getUpper(), float)
    if haveGLMM and nAGQ > 0:
        fe = np.asarray(object.fixef().values, float)
        x0 = np.concatenate([x0, fe])
        lower = np.concatenate([lower, np.full(len(fe), -np.inf)])
        upper = np.concatenate([upper, np.full(len(fe), np.inf)])
    rho.lower, rho.upper = lower, upper
    calc_derivs = oi.get("derivs") is not None
    optimizer = oi.get("optimizer")
    if ctrl_arg is not None and ctrl_arg.get("optimizer") is not None:
        newopt = ctrl_arg["optimizer"]
        optimizer = newopt[-1] if isinstance(newopt, (list, tuple)) else newopt
    opt = optwrap(optimizer, ff, x0, lower=lower, upper=upper, control=control.get("optCtrl"),
                  calc_derivs=calc_derivs)
    cc = checkConv(opt.attrs.get("derivs"), opt["par"], ctrl=control["checkConv"],
                   lbound=lower, ubound=upper)
    if haveGLMM:
        rr.setOffset(rho.baseOffset)
    reTrms = dict(flist=object.flist, flist_assign=object.flist_assign, cnms=object.cnms,
                  Gp=object.Gp, lower=np.asarray(object.lower, float),
                  upper=np.asarray(object.getUpper(), float), reCovs=reCovs)
    return mkMerMod(rho, opt, reTrms, fr=object.frame, mc=object.call, lme4conv=cc,
                    wmsgs=getattr(object, "wmsgs", None))


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def _simulate_list(x, nsim, rng, use_u, re_form):
    """list of nsim simulated responses (vectors or 2-column matrices)"""
    try:
        from .predict import simulate_merMod
    except ImportError:
        simulate_merMod = None
    if simulate_merMod is not None:
        kw = {}
        if re_form is not None:
            kw["re_form"] = re_form
        else:
            kw["use_u"] = use_u
        ss = simulate_merMod(x, nsim=nsim, seed=rng, **kw)
        return _sim_frame_to_list(ss, x)
    if re_form is not None:
        raise NotImplementedError("'re_form' needs lme4.predict.simulate_merMod")
    return _simulate_fallback(x, nsim, rng, use_u)


def _sim_frame_to_list(ss, x):
    n = len(x.resp.y)
    rn = list(x.frame.row_names)

    def fix_len(v):
        if v.shape[0] == n:
            return v
        try:
            return np.asarray(pd.DataFrame(v, index=ss.index).loc[rn].values).squeeze()
        except Exception:
            return v
    out = []
    if isinstance(ss.columns, pd.MultiIndex):
        for k in dict.fromkeys(ss.columns.get_level_values(0)):
            out.append(fix_len(np.asarray(ss[k].values, float)))
        return out
    for c in ss.columns:
        col = ss[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            v = (col.cat.codes.values > 0).astype(float)
        else:
            v = np.asarray(col.values, float)
        out.append(fix_len(v))
    return out


def _simulate_fallback(x, nsim, rng, use_u):
    """minimal parametric simulation (gaussian / binomial / poisson):
    y* = X beta + Z b* + eps, b* = sigma Lambda u*, u* ~ N(0, I)"""
    import scipy.sparse as sp
    X = np.asarray(x.pp.X)
    off = np.asarray(x.resp.offset, float)
    eta0 = X @ np.asarray(x.beta, float) + off
    Zt = sp.csc_matrix(x.pp.Zt)
    Lt = sp.csc_matrix(x.pp.Lambdat)
    sigma = x.sigma()
    n = len(eta0)
    w = np.asarray(x.resp.weights, float)
    q = Zt.shape[0]
    if use_u:
        reff = np.tile((Zt.T @ (Lt.T @ x.u))[:, None] * (sigma if x.isLMM() else 1.0), (1, nsim))
    else:
        U = (Lt @ Zt).T.tocsr()
        u = rng.standard_normal(q * nsim).reshape(q, nsim, order="F")
        reff = np.asarray(U @ u) * sigma
    eta = eta0[:, None] + reff
    out = []
    if x.isLMM():
        eps = rng.standard_normal(n * nsim).reshape(n, nsim, order="F")
        val = eta + sigma * eps / np.sqrt(w)[:, None]
        return [val[:, i] for i in range(nsim)]
    fam = x.family()
    mu = fam.linkinv(eta.ravel(order="F")).reshape(n, nsim, order="F")
    for i in range(nsim):
        if fam.family == "binomial":
            out.append(rng.binomial(np.round(w).astype(int), mu[:, i]) / w)
        elif fam.family == "poisson":
            out.append(rng.poisson(mu[:, i]).astype(float))
        else:
            raise NotImplementedError(f"simulation not implemented for family '{fam.family}'")
    return out


# ---------------------------------------------------------------------------
# bootMer
# ---------------------------------------------------------------------------

class BootMer:
    """result of bootMer() (R classes 'bootMer', 'boot')"""

    def __init__(self, t0, t, R, data, seed, statistic, sim, mle, msgs, numFail, fail_msgs):
        self.t0 = t0           # pd.Series (named statistic at the original fit)
        self.t = t             # ndarray (R x length(t0))
        self.R = R
        self.data = data
        self.seed = seed
        self.statistic = statistic
        self.sim = sim
        self.mle = mle
        self.boot_all_msgs = msgs
        self.bootFail = numFail
        self.boot_fail_msgs = fail_msgs
        self.boot_type = "boot"

    def as_data_frame(self):
        return pd.DataFrame(self.t, columns=list(self.t0.index))

    def confint(self, parm=None, level=0.95, type="perc"):
        return confint_bootMer(self, parm=parm, level=level, type=type)

    def __repr__(self):
        t = self.t
        with np.errstate(invalid="ignore"):
            bias = np.nanmean(t, axis=0) - self.t0.values
            se = np.nanstd(t, axis=0, ddof=1) if t.shape[0] > 1 else np.full(t.shape[1], np.nan)
        tab = pd.DataFrame({"original": self.t0.values, "bias": bias, "std. error": se},
                           index=[f"t{i + 1}*" for i in range(len(self.t0))])
        s = (f"\nPARAMETRIC BOOTSTRAP\n\nBootstrap Statistics :\n{tab}\n")
        if self.bootFail:
            s += f"\n{self.bootFail} bootstrap run(s) failed\n"
        return s


def bootMer(x, FUN, nsim=1, seed=None, use_u=False, re_form=None, type="parametric",
            verbose=False, control=None):
    """Model-based (semi-)parametric bootstrap for mixed models.

    ``FUN(fit)`` must return a numeric vector (or pandas Series).  ``seed``
    is passed to ``np.random.default_rng``.  ``re_form=None`` means "not
    specified" (R's missing): then ``use_u`` decides whether the random
    effects are re-simulated (False) or held at their conditional modes
    (True).  ``control`` defaults to the control stored in the model's call.
    """
    nsim = int(np.atleast_1d(nsim)[0])
    if nsim <= 0:
        raise ValueError("(nsim <- as.integer(nsim[1])) > 0 is not TRUE")
    if type not in ("parametric", "semiparametric"):
        raise ValueError("'arg' should be one of \"parametric\", \"semiparametric\"")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    seed_state = rng.bit_generator.state
    t0 = FUN(x)
    t0 = _as_stat(t0)
    mle = dict(beta=np.asarray(x.beta), theta=np.asarray(x.theta))
    if x.isLMM():
        mle["sigma"] = x.sigma()
    if type == "parametric":
        ss = _simulate_list(x, nsim, rng, use_u, re_form)
    else:
        if re_form is not None:
            raise ValueError("'re.form' cannot be used with semiparametric bootstrapping")
        if use_u:
            if x.isGLMM():
                warnings.warn("semiparametric bootstrapping is questionable for GLMMs")
            ftd = np.asarray(x.fitted().values, float)
            res = np.asarray(x.residuals(type="response").values, float)
            ss = [ftd + rng.choice(res, size=len(res), replace=True) for _ in range(nsim)]
        else:
            raise NotImplementedError("semiparametric bootstrapping with use.u=FALSE not yet "
                                      "implemented")
    if control is None and x.call is not None:
        control = x.call.get("control")
    lt0 = len(t0)
    res = []
    msgs = {"message": [], "warning": [], "error": []}
    for i in range(nsim):
        # factory(): capture messages (printed output), warnings and errors
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as wl, contextlib.redirect_stdout(buf):
            warnings.simplefilter("always")
            try:
                val = np.asarray(_as_stat(FUN(refit_merMod(x, ss[i], control=control))).values,
                                 float)
                if len(val) != lt0:
                    raise ValueError("FUN returned a value of the wrong length")
            except Exception as e:
                val = np.full(lt0, np.nan)
                msgs["error"].append(str(e).strip())
        msgs["warning"] += [str(w.message).strip() for w in wl]
        msgs["message"] += [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
        if verbose:
            print("%5d :" % (i + 1), val)
        res.append(val)
    t_star = np.vstack(res) if res else np.zeros((0, lt0))
    tab = {k: Counter(v) for k, v in msgs.items()}
    numFail = sum(tab["error"].values())
    if numFail > 0:
        warnings.warn(f"some bootstrap runs failed ({numFail}/{nsim})")
    fail_msgs = None if numFail == 0 else tab["error"]
    return BootMer(t0=t0, t=t_star, R=nsim, data=x.model_frame(), seed=seed_state,
                   statistic=FUN, sim="parametric", mle=mle, msgs=tab, numFail=numFail,
                   fail_msgs=fail_msgs)


def _as_stat(v):
    if isinstance(v, pd.Series):
        return v.astype(float)
    if isinstance(v, dict):
        return pd.Series(v, dtype=float)
    a = np.asarray(v)
    if a.dtype.kind not in "biuf":
        raise TypeError("bootMer currently only handles functions that return numeric vectors")
    a = np.atleast_1d(a.astype(float)).ravel()
    return pd.Series(a)


def print_bootWarnings(x, verbose=False):
    msgs = getattr(x, "boot_all_msgs", None)
    if not msgs or all(len(m) == 0 for m in msgs.values()):
        return None
    out = ["\n"]
    for i in ("message", "warning", "error"):
        m = msgs.get(i) or {}
        if len(m) > 0:
            items = sorted(m.items(), key=lambda kv: kv[1])
            if not verbose:
                out.append(f"{sum(m.values())} {i}(s): {items[0][0]}")
                if len(m) > 1:
                    out.append(" (and others)")
                out.append("\n")
            else:
                out = [f"{i}(s):\n"] + [f"   {c} {k}\n" for k, c in items] + ["\n"]
    print("".join(out), file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# boot::boot.ci (perc / basic / norm) and confint.bootMer
# ---------------------------------------------------------------------------

def norm_inter(t, alpha):
    """boot:::norm.inter -- interpolation on the normal quantile scale"""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    alpha = np.asarray(alpha, float)
    R = len(t)
    rk = (R + 1) * alpha
    if not np.all((rk > 1) & (rk < R)):
        warnings.warn("extreme order statistics used as endpoints")
    k = np.trunc(rk).astype(int)
    out = np.zeros(len(k))
    tstar = np.sort(t)
    ints = k == rk
    for j in range(len(k)):
        if ints[j] and 0 < k[j] <= R:
            out[j] = tstar[k[j] - 1]
    out[k == 0] = tstar[0]
    out[k == R] = tstar[R - 1]
    temp = np.nonzero(~ints & (k != 0) & (k != R))[0]
    if len(temp):
        temp1 = stats.norm.ppf(alpha[temp])
        temp2 = stats.norm.ppf(k[temp] / (R + 1))
        temp3 = stats.norm.ppf((k[temp] + 1) / (R + 1))
        tk = tstar[k[temp] - 1]
        tk1 = tstar[k[temp]]
        out[temp] = tk + (temp1 - temp2) / (temp3 - temp2) * (tk1 - tk)
    return np.column_stack([np.round(rk, 2), out])


def boot_ci(t0, t, conf=0.95, type="perc"):
    """boot::boot.ci for one statistic; returns the interval matrix row of
    the requested type (perc/basic: 5 values, norm: 3 values) or None"""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return None
    mt = np.mean(t)
    eps = min(1e-8, mt / 1e6)
    if np.all(np.abs(t - mt) < eps):
        print(f"All values of t are equal to  {mt} \n Cannot calculate confidence intervals")
        return None
    if type == "perc":
        alpha = (1 + np.array([-conf, conf])) / 2
        qq = norm_inter(t, alpha)
        return np.concatenate([[conf], qq[:, 0], qq[:, 1]])
    if type == "basic":
        alpha = (1 + np.array([conf, -conf])) / 2
        qq = norm_inter(t, alpha)
        return np.concatenate([[conf], qq[:, 0], 2 * t0 - qq[:, 1]])
    if type == "norm":
        var_t0 = np.var(t, ddof=1)
        t0b = 2 * t0 - np.mean(t)
        merr = math.sqrt(var_t0) * stats.norm.ppf((1 + conf) / 2)
        return np.array([conf, t0b - merr, t0b + merr])
    raise ValueError("'arg' should be one of \"perc\", \"norm\", \"basic\"")


def confint_bootMer(object, parm=None, level=0.95, type="perc"):
    if type not in ("perc", "norm", "basic"):
        raise ValueError("'arg' should be one of \"perc\", \"norm\", \"basic\"")
    blen = {"norm": 3, "basic": 5, "perc": 5}[type]
    nt = len(object.t0)
    parm = list(range(nt)) if parm is None else list(np.atleast_1d(parm))
    rows = []
    for i in parm:
        val = boot_ci(object.t0.values[i], object.t[:, i], conf=level, type=type)
        rows.append(np.full(blen, np.nan) if val is None else val)
    btab = np.vstack(rows)[:, blen - 2:blen] if rows else np.zeros((0, 2))
    a = (1 - level) / 2
    a = np.array([a, 1 - a])
    from .profile import _format_perc
    idx = list(object.t0.index[parm]) if len(parm) else []
    return pd.DataFrame(btab, index=idx, columns=_format_perc(a, 3))
