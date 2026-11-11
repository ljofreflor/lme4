"""Transcription of lmer(), glmer() (lme4/R/lmer.R) and mkMerMod()
(lme4/R/utilities.R)."""
from __future__ import annotations

import inspect
import warnings

import numpy as np

from .control import lmerControl, glmerControl, lmerControlObj, glmerControlObj, merControl
from .covariance import upReCovs
from .family import get_family, hasNoScale
from .modular import (lFormula, glFormula, mkLmerDevfun, optimizeLmer, mkGlmerDevfun,
                      updateGlmerDevfun, optimizeGlmer, getStart, updateStart)
from .modelframe import Factor
from .modules import lmerResp, glmResp, nlsResp
from .optimizers import checkConv, OptResult
from .rlang import Formula, as_formula
from .symbolic import to_formula


def _caller_env(depth=2):
    fr = inspect.currentframe()
    for _ in range(depth):
        fr = fr.f_back
    env = dict(fr.f_globals)
    env.update(fr.f_locals)
    return env


def _name_of(obj, env):
    if obj is None:
        return None
    for k, v in env.items():
        if v is obj and not k.startswith("_"):
            return k
    return None


class Call(dict):
    """the matched call (R's ``mc``), used for printing and update()"""

    def __init__(self, fun, **kw):
        super().__init__(**kw)
        self.fun = fun


def lmer(formula, data=None, REML=True, control=None, start=None, verbose=0, subset=None,
         weights=None, na_action="na.omit", offset=None, contrasts=None, devFunOnly=False,
         _env=None):
    """Fit a linear mixed-effects model (lme4::lmer).

    ``formula`` may be an R-style string (``"Reaction ~ Days + (Days | Subject)"``),
    a :class:`~lme4.Formula`, or a symbolic formula built with ``==``::

        Reaction, Days, Subject = V("Reaction Days Subject")
        lmer(Reaction == Days + (Days | Subject), data=sleepstudy)
    """
    env = _env if _env is not None else _caller_env()
    formula = to_formula(formula, env)
    if control is None:
        control = lmerControl()
    elif not isinstance(control, lmerControlObj):
        if not isinstance(control, dict):
            raise TypeError("'control' is not a list; use lmerControl()")
        warnings.warn("passing control as list is deprecated: please use lmerControl() instead")
        control = lmerControl(**control)
    mcout = Call("lmer", formula=formula, data=_name_of(data, env), REML=REML,
                 weights=weights if isinstance(weights, str) else None,
                 offset=offset if isinstance(offset, str) else None,
                 subset=subset if isinstance(subset, str) else None, control=control,
                 verbose=verbose, contrasts=contrasts, na_action=na_action, start=start,
                 _data=data)
    lmod = lFormula(formula, data, REML=REML, subset=subset, weights=weights,
                    na_action=na_action, offset=offset, contrasts=contrasts, control=control)
    mcout["formula"] = lmod["formula"]
    y = lmod["fr"].response()
    if y is not None and np.ndim(y) == 2 and np.shape(y)[1] > 1:
        raise ValueError("can't handle matrix-valued responses: consider using refit()")
    devfun = mkLmerDevfun(lmod["fr"], lmod["X"], lmod["reTrms"], REML=REML, start=start,
                          verbose=verbose, control=control, Xattrs=lmod["Xattrs"])
    if devFunOnly:
        return devfun
    rho = devfun.rho
    nobs = lmod["fr"].nrow
    npar = len(rho.lower)
    cc_ = control["checkConv"]
    calc_derivs = control["calc_derivs"]
    if calc_derivs is None:
        calc_derivs = nobs < cc_["check_conv_nobsmax"] and npar < cc_["check_conv_nparmax"]
    if control["optimizer"] == "none":
        raise ValueError('deprecated use of optimizer=="none"; use NULL instead')
    if control["optimizer"] is None or (isinstance(control["optimizer"], (list, tuple)) and
                                         len(control["optimizer"]) == 0):
        s = getStart(start, rho, 0)
        opt = OptResult(par=s, fval=devfun(s), conv=1000, message="no optimization")
    else:
        opt = optimizeLmer(devfun, optimizer=control["optimizer"],
                           restart_edge=control["restart_edge"],
                           boundary_tol=control["boundary_tol"], control=control["optCtrl"],
                           verbose=verbose, start=start, calc_derivs=calc_derivs,
                           force_calc_derivs=bool(control["calc_derivs"]),
                           use_last_params=control["use_last_params"])
    cc = checkConv(opt.attrs.get("derivs"), opt["par"], ctrl=cc_, lbound=rho.lower,
                   ubound=rho.upper, nobs=nobs, ndim=npar)
    return mkMerMod(rho, opt, lmod["reTrms"], fr=lmod["fr"], mc=mcout, lme4conv=cc,
                    wmsgs=lmod["wmsgs"])


def glmer(formula, data=None, family="gaussian", control=None, start=None, verbose=0, nAGQ=1,
          subset=None, weights=None, na_action="na.omit", offset=None, contrasts=None,
          mustart=None, etastart=None, devFunOnly=False, _env=None):
    """Fit a generalized linear mixed-effects model (lme4::glmer)."""
    env = _env if _env is not None else _caller_env()
    formula = to_formula(formula, env)
    if control is None:
        control = glmerControl()
    elif not isinstance(control, glmerControlObj):
        if isinstance(control, lmerControlObj):
            warnings.warn("please use glmerControl() instead of lmerControl()")
            kw = {k: v for k, v in control.items() if k not in ("checkConv", "checkControl")}
            kw.update(control["checkControl"])
            kw.update(control["checkConv"])
            kw.pop("restart_edge", None)
            control = glmerControl(**kw)
        elif isinstance(control, dict):
            warnings.warn("Use control=glmerControl(..) instead of passing a list")
            control = glmerControl(**control)
        else:
            raise TypeError("'control' is not a list; use glmerControl()")
    family = get_family(family)
    if family.is_gaussian_identity():
        warnings.warn("calling glmer() with family=gaussian (identity link) as a shortcut to "
                      "lmer() is deprecated; please call lmer() directly")
        return lmer(formula, data, control=None, start=start, verbose=verbose, subset=subset,
                    weights=weights, na_action=na_action, offset=offset, contrasts=contrasts,
                    devFunOnly=devFunOnly, _env=env)
    mcout = Call("glmer", formula=formula, data=_name_of(data, env), family=family,
                 weights=weights if isinstance(weights, str) else None,
                 offset=offset if isinstance(offset, str) else None,
                 subset=subset if isinstance(subset, str) else None, control=control,
                 nAGQ=nAGQ, verbose=verbose, contrasts=contrasts, na_action=na_action,
                 start=start, _data=data)
    glmod = glFormula(formula, data, family=family, subset=subset, weights=weights,
                      na_action=na_action, offset=offset, contrasts=contrasts, start=start,
                      mustart=mustart, etastart=etastart, control=control)
    mcout["formula"] = glmod["formula"]
    y = glmod["fr"].response()
    if y is not None and not isinstance(y, Factor) and np.ndim(y) == 2 and (
            (family.base != "binomial" and np.shape(y)[1] > 1) or np.shape(y)[1] > 2):
        raise ValueError("can't handle matrix-valued responses: consider using refit()")
    nAGQinit = 0 if control["nAGQ0initStep"] else nAGQ
    devfun = mkGlmerDevfun(glmod["fr"], glmod["X"], glmod["reTrms"], family, nAGQ=nAGQinit,
                           verbose=verbose, control=control, Xattrs=glmod["Xattrs"])
    if devFunOnly and nAGQ == 0:
        return devfun
    rho = devfun.rho
    opt = None
    if nAGQinit == 0:
        start0 = start
        if nAGQ > 0 and isinstance(start, dict):
            start0 = {k: v for k, v in start.items() if k not in ("fixef", "beta")}
        opt = optimizeGlmer(devfun, optimizer=control["optimizer"][0],
                            restart_edge=control["restart_edge"] if nAGQ == 0 else False,
                            boundary_tol=control["boundary_tol"] if nAGQ == 0 else 0,
                            control=control["optCtrl"], start=start0, nAGQ=0,
                            verbose=verbose, calc_derivs=False)
        if nAGQ > 0:
            devfun = updateGlmerDevfun(devfun, glmod["reTrms"], nAGQ=nAGQ)
            if devFunOnly:
                return devfun
            start = updateStart(start, opt["par"])
    nobs = glmod["fr"].nrow
    npar = len(rho.lower)
    cc_ = control["checkConv"]
    calc_derivs = control["calc_derivs"]
    if calc_derivs is None:
        calc_derivs = nobs < cc_["check_conv_nobsmax"] and npar < cc_["check_conv_nparmax"]
    if nAGQ > 0:
        opt = optimizeGlmer(devfun, optimizer=control["optimizer"][1],
                            restart_edge=control["restart_edge"],
                            boundary_tol=control["boundary_tol"], control=control["optCtrl"],
                            start=start, nAGQ=nAGQ, verbose=verbose, calc_derivs=calc_derivs,
                            use_last_params=control["use_last_params"])
    cc = None
    if calc_derivs:
        cc = checkConv(opt.attrs.get("derivs"), opt["par"], ctrl=cc_, lbound=rho.lower,
                       ubound=rho.upper, nobs=nobs, ndim=npar)
    return mkMerMod(rho, opt, glmod["reTrms"], fr=glmod["fr"], mc=mcout, lme4conv=cc,
                    wmsgs=glmod["wmsgs"])


# ---------------------------------------------------------------------------
# mkMerMod (lme4/R/utilities.R)
# ---------------------------------------------------------------------------

def _optinfo(opt, lme4conv=None):
    from .optimizers import getConv, getMsg
    return dict(optimizer=opt.attrs.get("optimizer") if hasattr(opt, "attrs") else None,
                control=opt.attrs.get("control") if hasattr(opt, "attrs") else None,
                derivs=opt.attrs.get("derivs") if hasattr(opt, "attrs") else None,
                conv=dict(opt=getConv(opt), lme4=lme4conv or {}),
                feval=opt.get("feval", np.nan), message=getMsg(opt),
                warnings=opt.attrs.get("warnings", []) if hasattr(opt, "attrs") else [],
                val=np.asarray(opt["par"], float))


def mkMerMod(rho, opt, reTrms, fr, mc=None, lme4conv=None, wmsgs=None):
    from .mermod import lmerMod, glmerMod, nlmerMod
    pp, resp = rho.pp, rho.resp
    rcl = resp.cls
    n = pp.V.shape[0]
    p = pp.V.shape[1]
    isGLMM = rcl == "glmResp"
    useSc = not (isGLMM and hasNoScale(resp.family))
    dims = dict(N=pp.X.shape[0], n=n, p=p, nmp=n - p, q=pp.Zt.shape[0], nth=len(pp.theta),
                nAGQ=getattr(rho, "nAGQ", None), compDev=getattr(rho, "compDev", None),
                useSc=int(useSc), reTrms=len(reTrms["cnms"]), spFe=0,
                REML=resp.REML if rcl == "lmerResp" else 0, GLMM=int(isGLMM),
                NLMM=int(rcl == "nlsResp"), npar=len(reTrms["lower"]),
                dispProfile=getattr(rho, "dispProfile", None),
                maxPhiIter=getattr(rho, "maxPhiIter", None), qEff=getattr(rho, "qEff", None))
    fac = float(rcl != "nlsResp")
    trivial_y = len(resp.y) == 0
    if trivial_y:
        sqrLenU = wrss = pwrss = np.nan
    else:
        sqrLenU = pp.sqrL(fac)
        wrss = resp.wrss()
        pwrss = wrss + sqrLenU
    beta = np.asarray(pp.beta(fac))
    hasFreeDisp = isGLMM and not hasNoScale(resp.family)
    dispProfile = getattr(rho, "dispProfile", True)
    dispProfile = True if dispProfile is None else dispProfile
    sigmaML = resp.phi() if (hasFreeDisp and not trivial_y and dispProfile) else pwrss / n
    if rcl != "lmerResp":
        pars = np.asarray(opt["par"], float)
        nAGQ = dims["nAGQ"] or 0
        expected = dims["npar"] + (dims["p"] if nAGQ > 0 else 0)
        if not np.any(np.isnan(pars)) and len(pars) != expected:
            raise ValueError(f"unexpected parameter vector length: length(pars)={len(pars)}, "
                             f"npar={dims['npar']}, nAGQ={nAGQ}; (expected length(pars)==npar iff nAGQ==0)")
        if nAGQ > 0:
            beta = pars[dims["npar"]:]
    REMLv = resp.REML if rcl == "lmerResp" else 0
    cmp = dict(ldL2=pp.ldL2(), ldRX2=pp.ldRX2(), wrss=wrss, ussq=sqrLenU, pwrss=pwrss,
               drsum=resp.resDev() if (rcl == "glmResp" and not trivial_y) else np.nan,
               REML=opt["fval"] if (rcl == "lmerResp" and REMLv != 0 and not trivial_y) else np.nan,
               dev=np.nan if ((rcl == "lmerResp" and REMLv != 0) or trivial_y) else opt["fval"],
               sigmaML=np.sqrt(np.nan if (not useSc or trivial_y) else sigmaML),
               sigmaREML=np.sqrt(np.nan if (rcl != "lmerResp" or trivial_y) else
                                 sigmaML * (dims["n"] / dims["nmp"])),
               tolPwrss=getattr(rho, "tolPwrss", np.nan))
    cls = {"lmerResp": lmerMod, "glmResp": glmerMod, "nlsResp": nlmerMod}[rcl]
    ans = cls(call=mc, frame=fr, flist=reTrms["flist"], flist_assign=reTrms["flist_assign"],
              cnms=reTrms["cnms"], Gp=reTrms["Gp"], theta=np.asarray(pp.theta).copy(), beta=beta,
              u=np.full(pp.Zt.shape[0], np.nan) if trivial_y else np.asarray(pp.u(fac)),
              lower=np.asarray(reTrms["lower"], float), devcomp=dict(cmp=cmp, dims=dims),
              pp=pp, resp=resp, optinfo=_optinfo(opt, lme4conv))
    ans.upper = np.asarray(reTrms.get("upper", np.full(len(reTrms["lower"]), np.inf)), float)
    ans.reCovs = upReCovs(reTrms["reCovs"], np.asarray(pp.theta))
    ans.wmsgs = wmsgs or {}
    return ans
