"""Optimizers and convergence checks.

Transcriptions of lme4/R/optimizer.R (Nelder_Mead), the optimizer wrappers
of lme4/R/utilities.R (nloptwrap, nlminbwrap), minqa::bobyqa's R wrapper,
optwrap()/getOptfun() from lme4/R/lmer.R, deriv12() (deriv.R),
checkConv()/checkHess() (checkConv.R) and GHrule() (GHrule.R).
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from ._backend import cpp
from .covariance import getSingTol


class ConvergenceWarning(UserWarning):
    pass


# ---------------------------------------------------------------------------
# Nelder-Mead (lme4/R/optimizer.R)
# ---------------------------------------------------------------------------

def Nelder_Mead(fn, par, lower=None, upper=None, control=None):
    par = np.asarray(par, float)
    n = len(par)
    lower = np.full(n, -np.inf) if lower is None else np.broadcast_to(np.asarray(lower, float), (n,)).copy()
    upper = np.full(n, np.inf) if upper is None else np.broadcast_to(np.asarray(upper, float), (n,)).copy()
    control = dict(control or {})
    xst = control.pop("xst", None)
    xst = np.full(n, 0.02) if xst is None else np.broadcast_to(np.asarray(xst, float), (n,)).copy()
    xt = control.pop("xt", None)
    xt = xst * 5e-4 if xt is None else np.broadcast_to(np.asarray(xt, float), (n,)).copy()
    verbose = control.pop("verbose", 0) or 0
    if control.get("iprint") is None:
        control["iprint"] = {0: 0, 1: 20, 2: 10, 3: 1}[int(min(float(verbose), 3))]
    if np.any(xst == 0):
        raise ValueError("all(xst != 0) is not TRUE")
    cc = dict(iprint=0, maxfun=10000, FtolAbs=1e-5, FtolRel=1e-15, XtolRel=1e-7,
              MinfMax=-np.finfo(float).max, warnOnly=False)
    unused = set(control) - set(cc)
    if unused:
        warnings.warn("unused control arguments ignored")
    cc.update({k: v for k, v in control.items() if k in cc})
    nM = cpp.NelderMead(lower, upper, xst, par, xt)
    nM.setFtolAbs(cc["FtolAbs"])
    nM.setFtolRel(cc["FtolRel"])
    nM.setIprint(int(cc["iprint"]))
    nM.setMaxeval(int(cc["maxfun"]))
    nM.setMinfMax(cc["MinfMax"])
    it = 0
    while True:
        it += 1
        nMres = nM.newf(float(fn(np.asarray(nM.xeval()).copy())))
        if nMres != 0:
            break
    cmsg = "reached max evaluations"
    if nMres == -4:
        cmsg = f"failure to converge in {cc['maxfun']} evaluations"
        warnings.warn(cmsg, ConvergenceWarning)
        nMres = 4
    msgvec = ["nm_forced", "cannot generate a feasible simplex", "initial x is not feasible",
              "active", "objective function went below allowed minimum",
              "objective function values converged to within tolerance",
              "parameter values converged to within tolerance", cmsg]
    if nMres < 0:
        if cc["warnOnly"]:
            warnings.warn(msgvec[nMres + 3])
        else:
            raise RuntimeError(msgvec[nMres + 3])
    return dict(fval=nM.value(), par=np.asarray(nM.xpos()), convergence=min(0, nMres),
                NM_result=nMres, message=msgvec[nMres + 3],
                control=dict(cc, xst=xst, xt=xt), feval=it)


# ---------------------------------------------------------------------------
# minqa::bobyqa
# ---------------------------------------------------------------------------

def bobyqa(par, fn, lower=-np.inf, upper=np.inf, control=None):
    """minqa::bobyqa (Powell's BOBYQA, Fortran transcribed to C++)"""
    par = np.asarray(par, float).copy()
    n = len(par)
    control = dict(control or {})
    cc = dict(npt=min(n + 2, 2 * n), rhobeg=None, rhoend=None, iprint=0, maxfun=10000,
              obstop=True, force_start=False)
    if "force.start" in control:
        control["force_start"] = control.pop("force.start")
    if set(control) - set(cc):
        warnings.warn("unused control arguments ignored")
    cc.update({k: v for k, v in control.items() if k in cc})
    cc["npt"] = int(max(n + 2, min(cc["npt"], ((n + 1) * (n + 2)) // 2)))
    if cc["npt"] > 2 * n + 1:
        warnings.warn("Setting npt > 2 * length(par) + 1 is not recommended.")
    if cc["rhobeg"] is None:
        cc["rhobeg"] = min(0.95, 0.2 * np.max(np.abs(par)))
    if cc["rhoend"] is None:
        cc["rhoend"] = 1.0e-6 * cc["rhobeg"]
    if not (0 < cc["rhoend"] <= cc["rhobeg"]):
        raise ValueError("0 < ctrl$rhoend && ctrl$rhoend <= ctrl$rhobeg is not TRUE")
    if cc["maxfun"] < 10 * n ** 2:
        warnings.warn("maxfun < 10 * length(par)^2 is not recommended.")
    checkObj = fn(par.copy())
    if not np.isscalar(checkObj) and np.size(checkObj) != 1:
        raise ValueError("Objective function must return a single numeric value.")
    lower = np.broadcast_to(np.asarray(lower, float), (n,)).copy()
    upper = np.broadcast_to(np.asarray(upper, float), (n,)).copy()
    if not np.all(lower < upper):
        raise ValueError("all(lower < upper) is not TRUE")
    if np.any((par < lower) | (par > upper)):
        if cc["obstop"]:
            raise ValueError("Starting values violate bounds")
        par = np.maximum(lower, np.maximum(par, upper))
        warnings.warn("Some parameters adjusted to nearest bound")
    rng = upper - lower
    if np.any(rng < 2 * cc["rhobeg"]):
        warnings.warn("All upper - lower must be >= 2*rhobeg. Changing rhobeg")
        cc["rhobeg"] = 0.2 * rng.min()
    if np.all(np.isfinite(upper)) and np.all(np.isfinite(lower)) and not cc["force_start"]:
        i = rng < cc["rhobeg"]
        if np.any(i):
            par[i] = lower[i] + cc["rhobeg"]
            warnings.warn("Some parameters adjusted away from lower bound")
        i = rng < cc["rhobeg"]
        if np.any(i):
            par[i] = upper[i] - cc["rhobeg"]
            warnings.warn("Some parameters adjusted away from upper bound")
    if not hasattr(cpp, "bobyqa"):
        raise RuntimeError("the C++ backend was built without bobyqa")
    res = cpp.bobyqa(lambda x: float(fn(np.asarray(x, float).copy())), par, lower, upper,
                     float(cc["rhobeg"]), float(cc["rhoend"]), int(cc["iprint"]),
                     int(cc["maxfun"]), int(cc["npt"]))
    if isinstance(res, dict):
        rpar, fval, feval, ierr = res["par"], res["fval"], res["feval"], res["ierr"]
    else:
        rpar, fval, feval, ierr = res
    msg = "Normal exit from bobyqa"
    ierr = int(ierr)
    if ierr > 0:
        ierr, msg = {10: (2, "bobyqa -- NPT is not in the required interval"),
                     320: (5, "bobyqa detected too much cancellation in denominator"),
                     390: (1, "bobyqa -- maximum number of function evaluations exceeded"),
                     430: (3, "bobyqa -- a trust region step failed to reduce q"),
                     20: (4, "bobyqa -- one of the box constraint ranges is too small (< 2*RHOBEG)")
                     }.get(ierr, (ierr, f"bobyqa error {ierr}"))
    return dict(par=np.asarray(rpar, float), fval=float(fval), feval=int(feval), ierr=ierr, msg=msg)


# ---------------------------------------------------------------------------
# nloptwrap / nlminbwrap (lme4/R/utilities.R)
# ---------------------------------------------------------------------------

_NLOPT_DEFAULT = dict(algorithm="NLOPT_LN_BOBYQA", xtol_abs=1e-8, ftol_abs=1e-8, maxeval=1e5)

_NLOPT_MSG = {1: "NLOPT_SUCCESS: Generic success return value.",
              2: "NLOPT_STOPVAL_REACHED: Optimization stopped because stopval (above) was reached.",
              3: "NLOPT_FTOL_REACHED: Optimization stopped because ftol_rel or ftol_abs (above) was reached.",
              4: "NLOPT_XTOL_REACHED: Optimization stopped because xtol_rel or xtol_abs (above) was reached.",
              5: "NLOPT_MAXEVAL_REACHED: Optimization stopped because maxeval (above) was reached.",
              6: "NLOPT_MAXTIME_REACHED: Optimization stopped because maxtime (above) was reached.",
              -1: "NLOPT_FAILURE: Generic failure code.",
              -2: "NLOPT_INVALID_ARGS: Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).",
              -3: "NLOPT_OUT_OF_MEMORY: Ran out of memory.",
              -4: "NLOPT_ROUNDOFF_LIMITED: Roundoff errors led to a breakdown of the optimization algorithm.",
              -5: "NLOPT_FORCED_STOP: Halted because of a forced termination."}


def nloptwrap(par, fn, lower, upper, control=None):
    import nlopt
    control = dict(control or {})
    for k, v in _NLOPT_DEFAULT.items():
        control.setdefault(k, v)
    alg = control.pop("algorithm")
    algname = alg[6:] if alg.startswith("NLOPT_") else alg
    par = np.asarray(par, float)
    n = len(par)
    opt = nlopt.opt(getattr(nlopt, algname), n)
    neval = [0]
    best = [np.inf, par.copy()]

    def f(x, grad):
        neval[0] += 1
        v = float(fn(np.asarray(x, float).copy()))
        if v < best[0]:
            best[0], best[1] = v, np.asarray(x).copy()
        return v
    opt.set_min_objective(f)
    lb = np.broadcast_to(np.asarray(lower, float), (n,))
    ub = np.broadcast_to(np.asarray(upper, float), (n,))
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    print_level = control.pop("print_level", 0)
    for k, v in control.items():
        if k == "xtol_abs":
            opt.set_xtol_abs(np.broadcast_to(np.asarray(v, float), (n,)).copy())
        elif k == "ftol_abs":
            opt.set_ftol_abs(float(v))
        elif k == "xtol_rel":
            opt.set_xtol_rel(float(v))
        elif k == "ftol_rel":
            opt.set_ftol_rel(float(v))
        elif k == "maxeval":
            opt.set_maxeval(int(v))
        elif k == "minf_max" or k == "stopval":
            opt.set_stopval(float(v))
        elif k == "maxtime":
            opt.set_maxtime(float(v))
        else:
            warnings.warn(f"nloptwrap: ignoring unknown control option {k!r}")
    try:
        x = opt.optimize(np.clip(par, lb, ub))
        status = opt.last_optimize_result()
        fval = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        x, fval, status = best[1], best[0], -4
    except Exception:
        x, fval, status = best[1], best[0], -1
    conv = status if (status < 0 or status == 5) else 0
    return dict(par=np.asarray(x), fval=float(fval), feval=neval[0], conv=conv,
                message=_NLOPT_MSG.get(status, str(status)))


def nlminbwrap(par, fn, lower, upper, control=None):
    """Approximation of nlminbwrap: R's nlminb (PORT library) is not available
    in Python; scipy's L-BFGS-B with finite differences is used instead."""
    from scipy.optimize import minimize
    control = dict(control or {})
    maxfun = control.pop("maxfun", control.pop("eval.max", 200))
    n = len(par)
    bounds = list(zip(np.broadcast_to(lower, (n,)), np.broadcast_to(upper, (n,))))
    bounds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b) for a, b in bounds]
    r = minimize(lambda x: float(fn(x)), np.asarray(par, float), method="L-BFGS-B", bounds=bounds,
                 options=dict(maxfun=maxfun))
    return dict(par=r.x, fval=float(r.fun), conv=0 if r.success else 1, message=str(r.message))


_OPTIMIZERS = {"Nelder_Mead": Nelder_Mead, "bobyqa": None, "nloptwrap": nloptwrap,
               "nlminbwrap": nlminbwrap}


def getOptfun(optimizer):
    if callable(optimizer):
        return optimizer
    if optimizer == "bobyqa":
        return lambda fn, par, lower, upper, control: bobyqa(par, fn, lower, upper, control)
    if optimizer == "Nelder_Mead":
        return lambda fn, par, lower, upper, control: Nelder_Mead(fn, par, lower, upper, control)
    if optimizer in ("nloptwrap", "nlminbwrap"):
        f = _OPTIMIZERS[optimizer]
        return lambda fn, par, lower, upper, control: f(par, fn, lower, upper, control)
    if optimizer == "optimx":
        raise NotImplementedError("optimizer='optimx' is not available in Python")
    raise ValueError(f"couldn't find optimizer function {optimizer}")


def getConv(x):
    if x.get("conv") is not None:
        return x["conv"]
    return x.get("convergence")


def getMsg(x):
    if x.get("msg") is not None:
        return x["msg"]
    if x.get("message") is not None:
        return x["message"]
    return ""


class OptResult(dict):
    """result of optwrap(); R attributes stored in .attrs"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attrs = {}

    @property
    def par(self):
        return self["par"]

    @property
    def fval(self):
        return self.get("fval", self.get("value"))


def optwrap(optimizer, fn, par, lower=-np.inf, upper=np.inf, control=None, adj=False,
            calc_derivs=True, force_calc_derivs=False, use_last_params=False, verbose=0, rho=None):
    """lme4:::optwrap"""
    calc_derivs = True if calc_derivs is None else calc_derivs
    optfun = getOptfun(optimizer)
    optName = optimizer if isinstance(optimizer, str) else getattr(optimizer, "__name__", "optimizer")
    par = np.array(par, float, copy=True)
    lower = np.broadcast_to(np.asarray(lower, float), par.shape).copy()
    upper = np.broadcast_to(np.asarray(upper, float), par.shape).copy()
    control = dict(control or {})
    if adj:
        if optName == "bobyqa":
            if not isinstance(control.get("rhobeg"), (int, float)):
                control["rhobeg"] = 0.0002
            if not isinstance(control.get("rhoend"), (int, float)):
                control["rhoend"] = 2e-7
        elif optName == "Nelder_Mead":
            if control.get("xst") is None:
                betaSD = np.sqrt(np.diag(rho.pp.unsc()))
                xst0 = np.full(len(par) - len(betaSD), 0.1)
                xst1 = np.minimum(betaSD, 10)
                control["xst"] = 0.2 * np.concatenate([xst0, xst1])
            if control.get("xt") is None:
                control["xt"] = np.asarray(control["xst"]) * 5e-4
    if optName == "bobyqa":
        if np.all(par == 0):
            par[:] = 0.001
        if not isinstance(control.get("iprint"), (int, float)):
            control["iprint"] = min(int(verbose), 3)
    elif optName == "Nelder_Mead":
        control["verbose"] = verbose
    elif optName == "nloptwrap":
        control["print_level"] = min(int(verbose), 3)
    elif verbose:
        warnings.warn(f"'verbose' not yet passed to optimizer '{optName}'; consider fixing optwrap()")
    last_par = [None]
    if use_last_params:
        def fn_opt(p):
            last_par[0] = np.array(p, float)
            return fn(p)
    else:
        fn_opt = fn
    curWarnings = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        opt = OptResult(optfun(fn_opt, par, lower, upper, control))
    for w in wlist:
        curWarnings.append(str(w.message))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if optName == "bobyqa":
        opt["convergence"] = opt["ierr"]
    elif optName == "Nelder_Mead":
        if opt["NM_result"] == 4:
            opt["convergence"] = 4
    if "fval" not in opt and "value" in opt:
        opt["fval"] = opt["value"]
    optconv = getConv(opt)
    if optconv is not None and optconv != 0:
        wmsg = f"convergence code {optconv} from {optName}"
        if getMsg(opt):
            wmsg += ": " + getMsg(opt)
        warnings.warn(wmsg, ConvergenceWarning)
        curWarnings.append(wmsg)
    singular = min(np.min(opt["par"] - lower), np.min(upper - opt["par"])) < getSingTol()
    derivs = None
    if force_calc_derivs or (calc_derivs and not singular):
        derivs = deriv12(fn, opt["par"], fx=opt.get("value", opt["fval"]))
        if use_last_params and last_par[0] is not None:
            fn(last_par[0])
    if not use_last_params:
        fn(opt["par"])
    opt.attrs = dict(optimizer=optimizer, control=control, warnings=curWarnings,
                     derivs=derivs, last_par=last_par[0])
    return opt


# ---------------------------------------------------------------------------
# deriv12 (lme4/R/deriv.R)
# ---------------------------------------------------------------------------

def deriv12(fun, x, delta=1e-4, fx=None, lower=None, upper=None):
    x = np.asarray(x, float)
    nx = len(x)
    if fx is None:
        fx = fun(x)
    H = np.full((nx, nx), np.nan)
    g = np.zeros(nx)
    xadd = x + delta
    xsub = x - delta
    hasUB = upper is not None and np.any(~np.isnan(upper))
    hasLB = lower is not None and np.any(~np.isnan(lower))
    udelta = ldelta = None
    if hasUB:
        active = ~np.isnan(upper) & (xadd > upper)
        hasUB = bool(np.any(active))
        if hasUB:
            udelta = np.where(active, upper - x, delta)
            xadd[active] = upper[active]
    if hasLB:
        active = ~np.isnan(lower) & (xsub < lower)
        hasLB = bool(np.any(active))
        if hasLB:
            ldelta = np.where(active, x - lower, delta)
            xsub[active] = lower[active]

    def spos(pos_vals):
        b = x.copy()
        for mod, pos in pos_vals:
            b[pos] = mod[pos]
        return b
    for j in range(nx):
        fadd = fun(spos([(xadd, j)]))
        fsub = fun(spos([(xsub, j)]))
        udj = udelta[j] if hasUB else delta
        ldj = ldelta[j] if hasLB else delta
        H[j, j] = fadd / udj ** 2 - 2 * fx / (udj * ldj) + fsub / ldj ** 2
        g[j] = (fadd - fsub) / (udj + ldj)
        for i in range(j):
            udi = udelta[i] if hasUB else delta
            ldi = ldelta[i] if hasLB else delta
            H[i, j] = H[j, i] = (fun(spos([(xadd, i), (xadd, j)])) / (udi + udj) ** 2
                                 - fun(spos([(xadd, i), (xsub, j)])) / (udi + ldj) ** 2
                                 - fun(spos([(xsub, i), (xadd, j)])) / (ldi + udj) ** 2
                                 + fun(spos([(xsub, i), (xsub, j)])) / (ldi + ldj) ** 2)
    return dict(gradient=g, Hessian=H)


# ---------------------------------------------------------------------------
# checkConv (lme4/R/checkConv.R)
# ---------------------------------------------------------------------------

help_str = "\n  See ?lme4::convergence and ?lme4::troubleshooting."


def _act(cc, wstr, cstr):
    if cc == "message":
        print(wstr, file=sys.stderr)
    elif cc == "warning":
        warnings.warn(wstr, ConvergenceWarning)
    elif cc == "stop":
        raise RuntimeError(wstr)
    else:
        raise ValueError(f"unknown check level for '{cstr}'")


def checkConv(derivs, coefs, ctrl, lbound, ubound=None, debug=False, nobs=None, ndim=None):
    res = {}
    lbound = np.asarray(lbound, float)
    ntheta = len(lbound)
    ubound = np.full(ntheta, np.inf) if ubound is None else np.asarray(ubound, float)
    coefs = np.asarray(coefs, float)
    ccl = ctrl["check_conv_singular"]
    cc = ccl["action"]
    th = coefs[:ntheta]
    is_singular = min(np.min(th - lbound), np.min(ubound - th)) < ccl["tol"]
    if cc != "ignore" and is_singular:
        wstr = "boundary (singular) fit: see help('isSingular')"
        res.setdefault("messages", []).append(wstr)
        _act(cc, wstr, "check.conv.singular")
    if is_singular:
        return res
    if derivs is None or (nobs is not None and nobs > ctrl["check_conv_nobsmax"]) or \
            (ndim is not None and ndim > ctrl["check_conv_nparmax"]):
        return None
    grad = np.asarray(derivs["gradient"], float)
    Hess = np.asarray(derivs["Hessian"], float)
    if np.any(np.isnan(grad)):
        return dict(code=-5, messages=["Gradient contains NAs"])
    ccl = ctrl["check_conv_grad"]
    cc = ccl["action"]
    wstr = None
    if cc != "ignore":
        try:
            R = np.linalg.cholesky(Hess).T
            scgrad = np.linalg.solve(R, grad)
            bad = np.any(np.isnan(scgrad))
        except np.linalg.LinAlgError:
            bad = True
        if bad:
            wstr = "unable to evaluate scaled gradient"
            res["code"] = -1
        else:
            mingrad = np.minimum(np.abs(scgrad), np.abs(grad))
            maxmingrad = mingrad.max()
            if maxmingrad > ccl["tol"]:
                w = int(np.argmax(mingrad)) + 1
                res["code"] = -1
                wstr = (f"Model failed to converge with max|grad| = {maxmingrad:g} "
                        f"(tol = {ccl['tol']:g}, component {w})" + help_str)
        if wstr is not None:
            res["messages"] = [wstr]
            _act(cc, wstr, "check.conv.grad")
        if ccl.get("relTol") is not None:
            mrg = np.max(np.abs(grad / coefs))
            if mrg > ccl["relTol"]:
                res["code"] = -2
                wstr = f"Model failed to converge with max|relative grad| = {mrg:g} (tol = {ccl['relTol']:g})" + help_str
                res["messages"] = [wstr]
                _act(cc, wstr, "check.conv.grad")
    ccl = ctrl["check_conv_hess"]
    cc = ccl["action"]
    if cc != "ignore":
        if len(coefs) > ntheta:
            H_beta = Hess[ntheta:, ntheta:]
            rh = checkHess(H_beta, ccl["tol"], "fixed-effect")
            if any(c != 0 for c in rh["code"]):
                res["code"] = rh["code"]
                res.setdefault("messages", []).extend(rh["messages"])
                _act(cc, ";".join(rh["messages"]), "check.conv.hess")
        rh = checkHess(Hess, ccl["tol"])
        if any(c != 0 for c in rh["code"]):
            res["code"] = rh["code"]
            res.setdefault("messages", []).extend(rh["messages"])
            _act(cc, ";".join(rh["messages"]), "check.conv.hess")
    return res


def checkHess(H, tol, hesstype=""):
    res = dict(code=[], messages=[])
    try:
        evd = np.linalg.eigvalsh(H)[::-1]
        if np.any(~np.isfinite(evd)):
            raise np.linalg.LinAlgError
    except (np.linalg.LinAlgError, ValueError):
        res["code"] = [-6]
        res["messages"] = ["Problem with Hessian check (infinite or missing values?)"]
        return res
    negative = int(np.sum(evd < -tol))
    if negative:
        res["code"] = [-3]
        res["messages"] = [f"Model failed to converge: degenerate {hesstype} Hessian with "
                           f"{negative} negative eigenvalues" + help_str]
    else:
        zero = int(np.sum(np.abs(evd) < tol))
        try:
            np.linalg.cholesky(H)
            chol_fail = False
        except np.linalg.LinAlgError:
            chol_fail = True
        if zero or chol_fail:
            res["code"] = [-4]
            res["messages"] = [f"{hesstype} Hessian is numerically singular: parameters are not uniquely determined"]
        else:
            res["cond.H"] = evd.max() / evd.min()
            if evd.max() * tol > 1:
                res["code"].append(2)
                res["messages"].append("Model is nearly unidentifiable: very large eigenvalue\n - Rescale variables?")
            if evd.min() / evd.max() < tol:
                res["code"].append(3)
                if 5 not in res["code"]:
                    res["messages"].append("Model is nearly unidentifiable: large eigenvalue ratio\n - Rescale variables?")
    if not res["code"]:
        res["code"] = [0]
    return res


# ---------------------------------------------------------------------------
# GHrule (lme4/R/GHrule.R)
# ---------------------------------------------------------------------------

_FGQ = None


def GHrule(ord_, asMatrix=True):
    """Gauss-Hermite quadrature rule: columns z, w, ldnorm"""
    global _FGQ
    ord_ = int(ord_)
    if not (0 <= ord_ < 101):
        raise ValueError("ord must be in 0..100")
    if ord_ == 0:
        return np.zeros((0, 3))
    if _FGQ is None:
        _FGQ = json.loads((Path(__file__).parent / "_fgq_rules.json").read_text())
    r = _FGQ[ord_ - 1]
    z = np.asarray(r["z"], float)
    w = np.asarray(r["w"], float)
    z = (z - z[::-1]) / 2
    w = (w + w[::-1]) / 2
    ldnorm = -0.5 * np.log(2 * np.pi) - 0.5 * z * z
    return np.column_stack([z, w, ldnorm])
