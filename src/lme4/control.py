"""Transcription of lme4/R/lmerControl.R."""
from __future__ import annotations

import copy
import warnings

from .covariance import getSingTol

_RANKX = ["message+drop.cols", "silent.drop.cols", "warn+drop.cols", "stop.deficient", "ignore"]
_SCALEX = ["warning", "stop", "silent.rescale", "message+rescale", "warn+rescale", "ignore"]


def _makeCC(action, tol, relTol=..., **kw):
    if not isinstance(action, str):
        raise TypeError("action must be a string")
    if tol < 0:
        raise ValueError("'tol' must be number >= 0")
    out = {"action": action, "tol": tol}
    if relTol is not ...:
        out["relTol"] = relTol
    out.update(kw)
    return out


class merControl(dict):
    """dict with attribute access; class name mirrors R's control classes"""
    kind = "merControl"

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def copy(self):
        return copy.deepcopy(self)


class lmerControlObj(merControl):
    kind = "lmerControl"


class glmerControlObj(merControl):
    kind = "glmerControl"


class nlmerControlObj(merControl):
    kind = "nlmerControl"


def _conv_opt(val, default):
    if isinstance(val, str):
        d = dict(default)
        d["action"] = val
        return d
    if not isinstance(val, dict) or not isinstance(val.get("action"), str) or "tol" not in val:
        raise ValueError("check.conv* options must be a dict with 'action' and 'tol'")
    return val


def merControl_(optimizer="nloptwrap", restart_edge=True, boundary_tol=1e-5, calc_derivs=None,
                use_last_params=False, sparseX=False, standardize_X=False, autoscale=None,
                check_nobs_vs_rankZ="ignore", check_nobs_vs_nlev="stop",
                check_nlev_gtreq_5="ignore", check_nlev_gtr_1="stop", check_nobs_vs_nRE="stop",
                check_rankX="message+drop.cols", check_scaleX="warning",
                check_formula_LHS="stop", check_conv_nobsmax=1e4, check_conv_nparmax=10,
                check_conv_grad=None, check_conv_singular=None, check_conv_hess=None,
                optCtrl=None, mod_type="lmer", **glmer_extra):
    optCtrl = dict(optCtrl or {})
    if mod_type == "glmer" and (isinstance(optimizer, str) or callable(optimizer)):
        optimizer = [optimizer, optimizer]
    if check_rankX not in _RANKX:
        raise ValueError(f"'check.rankX' should be one of {_RANKX}")
    if check_scaleX not in _SCALEX:
        raise ValueError(f"'check.scaleX' should be one of {_SCALEX}")
    cg = _conv_opt(check_conv_grad, _makeCC("warning", tol=2e-3, relTol=None)) \
        if check_conv_grad is not None else _makeCC("warning", tol=2e-3, relTol=None)
    cs = _conv_opt(check_conv_singular, _makeCC("message", tol=getSingTol())) \
        if check_conv_singular is not None else _makeCC("message", tol=getSingTol())
    ch = _conv_opt(check_conv_hess, _makeCC("warning", tol=1e-6)) \
        if check_conv_hess is not None else _makeCC("warning", tol=1e-6)
    if mod_type == "glmer" and use_last_params and calc_derivs:
        warnings.warn("using 'use.last.params'=TRUE and 'calc.derivs'=TRUE with 'glmer' "
                      "will not give backward-compatible results")
    cls = lmerControlObj if mod_type == "lmer" else glmerControlObj
    ret = cls(optimizer=optimizer, restart_edge=restart_edge, boundary_tol=boundary_tol,
              calc_derivs=calc_derivs, use_last_params=use_last_params,
              checkControl=dict(autoscale=autoscale,
                                check_nobs_vs_rankZ=check_nobs_vs_rankZ,
                                check_nobs_vs_nlev=check_nobs_vs_nlev,
                                check_nlev_gtreq_5=check_nlev_gtreq_5,
                                check_nlev_gtr_1=check_nlev_gtr_1,
                                check_nobs_vs_nRE=check_nobs_vs_nRE,
                                check_rankX=check_rankX,
                                check_scaleX=check_scaleX,
                                check_formula_LHS=check_formula_LHS),
              checkConv=dict(check_conv_nobsmax=check_conv_nobsmax,
                             check_conv_nparmax=check_conv_nparmax,
                             check_conv_grad=cg, check_conv_singular=cs,
                             check_conv_hess=ch),
              optCtrl=optCtrl)
    if mod_type == "glmer":
        g = dict(tolPwrss=1e-7, compDev=True, nAGQ0initStep=True,
                 check_response_not_const="stop", disp_method="moment", maxPhiIter=100,
                 disp_dof_correction=True)
        g.update(glmer_extra)
        if g["disp_method"] not in ("moment", "old/buggy"):
            raise ValueError("'disp_method' should be one of 'moment', 'old/buggy'")
        if int(g["maxPhiIter"]) < 1:
            raise ValueError("'maxPhiIter' must be a single positive integer")
        g["maxPhiIter"] = int(g["maxPhiIter"])
        if not isinstance(g["disp_dof_correction"], bool):
            raise ValueError("'disp_dof_correction' must be TRUE or FALSE")
        ret.update(tolPwrss=g["tolPwrss"], compDev=g["compDev"], nAGQ0initStep=g["nAGQ0initStep"],
                   disp_method=g["disp_method"], maxPhiIter=g["maxPhiIter"],
                   disp_dof_correction=g["disp_dof_correction"])
        ret["checkControl"]["check_response_not_const"] = g["check_response_not_const"]
    elif glmer_extra:
        raise TypeError(f"unexpected arguments: {list(glmer_extra)}")
    return ret


def lmerControl(optimizer="nloptwrap", restart_edge=True, **kw):
    """lme4::lmerControl (argument names use '_' instead of '.')"""
    return merControl_(optimizer=optimizer, restart_edge=restart_edge, mod_type="lmer", **kw)


def glmerControl(optimizer=("bobyqa", "Nelder_Mead"), restart_edge=False, check_conv_nparmax=20,
                 **kw):
    """lme4::glmerControl"""
    if isinstance(optimizer, tuple):
        optimizer = list(optimizer)
    return merControl_(optimizer=optimizer, restart_edge=restart_edge,
                       check_conv_nparmax=check_conv_nparmax, mod_type="glmer", **kw)


def nlmerControl(optimizer="Nelder_Mead", tolPwrss=1e-10, optCtrl=None):
    if isinstance(optimizer, str) or callable(optimizer):
        optimizer = [optimizer, optimizer]
    return nlmerControlObj(optimizer=list(optimizer), tolPwrss=tolPwrss, optCtrl=dict(optCtrl or {}))
