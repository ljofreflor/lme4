"""Refit a [g]lmer model with all available optimizers: transcription of
lme4/R/allFit.R (``allFit``, ``print.allFit``, ``summary.allFit``).

Optimizers whose R packages are unavailable in Python (``nmkbw`` from
dfoptim, ``optimx``) are dropped, as R's allFit() does when those packages
are not installed.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .control import glmerControl, lmerControl
from .rlang import deparse

meth_tab_0 = pd.DataFrame(
    {"optimizer": ["bobyqa", "Nelder_Mead", "nlminbwrap", "nmkbw", "optimx",
                   "nloptwrap", "nloptwrap"],
     "method": ["", "", "", "", "L-BFGS-B", "NLOPT_LN_NELDERMEAD", "NLOPT_LN_BOBYQA"]})

# hardcoded list (incomplete?) of allowable *control* options by optimizer
opt_ctrls = dict(bobyqa=["npt", "rhobeg", "rhoend", "iprint", "maxfun"],
                 Nelder_Mead=["iprint", "maxfun", "FtolAbs", "FtolRel", "XtolRel", "MinfMax",
                              "xst", "xt", "verbose", "warnOnly"],
                 nlminbwrap=["eval.max", "iter.max", "trace", "abs.tol", "rel.tol", "x.tol",
                             "xf.tol", "step.min", "step.max", "sing.tol", "scale.init",
                             "diff.g"],
                 optimx=["trace", "fnscale", "parscale", "ndeps", "maxit", "abstol", "reltol",
                         "method"],
                 nloptwrap=["minf_max", "ftol_rel", "ftol_abs", "xtol_rel", "xtol_abs",
                            "maxeval", "maxtime", "algorithm"],
                 nmkbw=["tol", "maxfeval", "regsimp", "maximize", "restarts.max", "trace"])

# name of 'max function evaluations' for each optimizer
maxfun_arg = dict(bobyqa="maxfun", Nelder_Mead="maxfun", nlminbwrap="eval.max",
                  optimx="maxit", nloptwrap="maxeval", nmkbw="maxfeval")

_UNAVAILABLE = ("nmkbw", "optimx")   # dfoptim / optimx are R-only


class AllFitError:
    """an error caught while refitting (R's condition object)"""

    def __init__(self, exc):
        self.exc = exc
        self.optCtrl = None
        self.time = None

    def __repr__(self):
        return f"<simpleError: {self.exc}>"


class AllFit(dict):
    """R's ``allFit`` object: dict optimizer name -> fitted model (or
    :class:`AllFitError`), with attributes ``fit`` (original model) and ``data``"""

    fit = None
    data = None

    def summary(self):
        return summary_allFit(self)

    def __str__(self):
        out = ["original model:"]
        f = self.fit
        ff = deparse(f.formula().expr)
        width = 80
        if len(ff) > width:
            ff = ff[:width - 3] + "..."
        out.append(ff + " ")
        dname = f.call.get("data")
        out.append(f"data:  {dname if dname is not None else 'NULL'} ")
        names = ", ".join(self.keys())
        if len(names) > width:
            names = names[:width - 3] + "..."
        out.append(f"optimizers ({len(self)}): {names} ")
        bad = [k for k, v in self.items() if isinstance(v, AllFitError)]
        if bad:
            out.append(f"{len(bad)} optimizer(s) failed")
        out.append("differences in negative log-likelihoods:")
        nll = np.array([-v.logLik().value for v in self.values()
                        if not isinstance(v, AllFitError)])
        sd = np.std(nll, ddof=1) if len(nll) > 1 else np.nan
        out.append(f"max= {_signif(np.max(nll - np.min(nll)), 3):.3g} ; std dev= {_signif(sd, 3):.3g} ")
        return "\n".join(out)

    def __repr__(self):
        return self.__str__()


def _signif(x, d):
    if not np.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{d}g}")


def _flatten_control(ctrl):
    """unpack a nested control object back into a flat argument list"""
    out = {k: v for k, v in ctrl.items() if k not in ("checkControl", "checkConv")}
    out.update(ctrl.get("checkControl") or {})
    out.update(ctrl.get("checkConv") or {})
    return out


def allFit(object, meth_tab=None, data=None, verbose=True, show_meth_tab=False, maxfun=1e5,
           parallel="no", ncpus=1, cl=None, catch_errs=True, start_from_mle=True):
    """Attempt to re-fit a [g]lmer model with a range of optimizers.

    Returns an :class:`AllFit` (dict of fitted models keyed by
    ``"optimizer[.method]"``)."""
    if meth_tab is None:
        meth_tab = meth_tab_0
    meth_tab = pd.DataFrame(meth_tab)
    meth_tab = meth_tab[~meth_tab["optimizer"].isin(_UNAVAILABLE)].reset_index(drop=True)
    if show_meth_tab:
        return meth_tab
    if not (meth_tab.shape[0] >= 1 and meth_tab.shape[1] >= 2):
        raise ValueError("invalid 'meth_tab'")
    if parallel not in ("no", "multicore", "snow", "future"):
        raise ValueError("'parallel' should be one of 'no', 'multicore', 'snow', 'future'")
    optimizer = list(meth_tab["optimizer"])
    method = list(meth_tab["method"])
    fit_names = [(o + "." + m) if m else o for o, m in zip(optimizer, method)]
    isGLMM = object.isGLMM()

    def ffun(i):
        if verbose:
            print(fit_names[i], ": ", end="")
        ctrl = object.call.get("control")
        if ctrl is None:
            ctrl = dict(optimizer=optimizer[i])
        else:
            ctrl = _flatten_control(ctrl)
            ctrl["optimizer"] = optimizer[i]

        def mkOptCtrl(**x):
            cc = dict(ctrl.get("optCtrl") or {})
            cc[maxfun_arg[optimizer[i]]] = maxfun
            cc.update(x)
            return cc

        def sanitize(x, okvals):
            if x is None:
                return None
            if okvals is None:
                return x
            return {k: v for k, v in x.items() if k in okvals}

        if optimizer[i] == "optimx":
            ctrl["optCtrl"] = mkOptCtrl(method=method[i])
        elif optimizer[i] == "nloptwrap":
            ctrl["optCtrl"] = mkOptCtrl(algorithm=method[i])
        else:
            ctrl["optCtrl"] = mkOptCtrl()
        ctrl["optCtrl"] = sanitize(ctrl["optCtrl"], opt_ctrls.get(optimizer[i]))
        ctrl = (glmerControl if isGLMM else lmerControl)(**ctrl)
        kw = dict(control=ctrl)
        if start_from_mle:
            if isGLMM:
                pars = dict(theta=np.asarray(object.theta, float),
                            fixef=np.asarray(object.beta, float))
            else:
                pars = dict(theta=np.asarray(object.theta, float))
                if object.isNLMM():
                    warnings.warn("results are not guaranteed when using nlmer")
            kw["start"] = pars
        if data is not None:
            kw["data"] = data
        t0 = time.perf_counter()
        if catch_errs:
            try:
                rr = object.update(**kw)
            except Exception as e:      # tryCatch(eval(fit_call), error = function(e) e)
                rr = AllFitError(e)
        else:
            rr = object.update(**kw)
        rr.optCtrl = ctrl["optCtrl"]   # contains crucial info here
        rr.time = time.perf_counter() - t0
        if verbose:
            print("[failed]" if isinstance(rr, AllFitError) else "[OK]")
        return rr

    if parallel == "no":
        res = [ffun(i) for i in range(len(fit_names))]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, int(ncpus))) as ex:
            res = list(ex.map(ffun, range(len(fit_names))))
    out = AllFit(zip(fit_names, res))
    out.fit = object
    out.data = data
    return out


class SummaryAllFit(dict):
    """R's ``summary.allFit`` list: which_OK, msgs, fixef, llik, sdcor, theta,
    times, feval"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def summary_allFit(object: AllFit):
    which_OK = pd.Series({k: not isinstance(v, AllFitError) for k, v in object.items()})
    objOK = {k: v for k, v in object.items() if which_OK[k]}
    msgs = {k: (v.optinfo["conv"]["lme4"] or {}).get("messages") for k, v in objOK.items()}
    fixef = pd.DataFrame({k: v.fixef() for k, v in objOK.items()}).T
    llik = pd.Series({k: v.logLik().value for k, v in objOK.items()})
    times = pd.DataFrame({"elapsed": {k: v.time for k, v in objOK.items()}})
    feval = pd.Series({k: float(v.optinfo.get("feval", np.nan)) if v.optinfo.get("feval")
                       is not None else np.nan for k, v in objOK.items()})
    sdcor = pd.DataFrame({k: v.VarCorr().as_data_frame()["sdcor"].values
                          for k, v in objOK.items()}).T
    theta = pd.DataFrame({k: v.getME("theta") for k, v in objOK.items()}).T
    first = next(iter(objOK.values()))
    cnm = []
    for c, (g, cn) in zip(first.getReCovs(), first.cnms):
        vcomp, ccomp = c.getVCNames(cn, g, None, True)
        cnm += list(vcomp) + list(ccomp)
    if first.sigma() != 1:
        cnm.append("sigma")
    sdcor.columns = cnm
    return SummaryAllFit(which_OK=which_OK, msgs=msgs, fixef=fixef, llik=llik, sdcor=sdcor,
                         theta=theta, times=times, feval=feval)
