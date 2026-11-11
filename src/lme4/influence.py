"""Influence diagnostics for merMod objects: transcription of the
``influence.merMod``, ``dfbeta``/``dfbetas``/``cooks.distance`` methods for
``influence.merMod``, ``cooks.distance.merMod``, ``rstudent.merMod``,
``rstudent.glmerMod`` and ``rstandard.glmerMod`` from lme4/R/methods.R.

``influence()`` refits the model with each case (or each level of
``groups``) deleted, via ``update(model, subset = ...)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .control import glmerControl, lmerControl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _naresid(model, x):
    """naresid(): pad with NA for observations dropped by na.exclude"""
    x = pd.Series(np.asarray(x, float), index=model.frame.row_names)
    na = model.frame.attrs.get("na.action")
    if not na or na.get("type") != "na.exclude":
        return x
    data = model.call.get("_data")
    if data is None:
        return x
    full = pd.Series(np.nan, index=data.index)
    full.loc[x.index] = x.values
    return full


def _hat(model):
    with warnings.catch_warnings():
        if model.isGLMM():
            warnings.simplefilter("ignore")
        h = model.hatvalues()
    if model.isGLMM():
        warnings.warn("the hat matrix may not make sense for GLMMs")
    return np.asarray(h, float)


def _vcov(x):
    return x.vcov()


# ---------------------------------------------------------------------------
# influence.merMod
# ---------------------------------------------------------------------------

_IPOS = {"fixed": 0, "fixed.sub": 1, "var.cov": 2, "var.cov.sub": 3, "vcov": 4, "vcov.sub": 5}


class InfluenceMerMod(dict):
    """R's ``influence.merMod`` list (an ordered dict); element positions
    follow R (fixed.effects, fixed.effects[-g], var.cov.comps,
    var.cov.comps[-g], vcov, vcov[-g], groups, deleted, converged,
    function.evals)"""

    def _pos(self, i):
        return list(self.values())[i]

    def dfbeta(self, which="fixed"):
        return dfbeta_influence_merMod(self, which)

    def dfbetas(self):
        return dfbetas_influence_merMod(self)

    def cooks_distance(self):
        return cooks_distance_influence_merMod(self)


def influence_merMod(model, groups=None, data=None, maxfun=1000, do_coef=True, start=None,
                     parallel="no", ncpus=1, cl=None, **kw):
    """Case-deletion influence diagnostics (influence.merMod): refit the
    model deleting each case (or each level of ``groups``) in turn."""
    if parallel not in ("no", "multicore", "snow", "future"):
        raise ValueError("'parallel' should be one of 'no', 'multicore', 'snow', 'future'")
    if kw:
        warnings.warn("disregarded additional arguments")
    if not do_coef:
        # simple/quick/trivial results
        return InfluenceMerMod(hat=_naresid(model, _hat(model)))
    if data is None:
        data = model.call.get("_data")
        if data is None:
            raise ValueError("model did not use the data argument")
    data = pd.DataFrame(data).copy()
    if groups is None:
        groups = ".case"
        data[".case"] = [str(i) for i in data.index]
    elif not isinstance(groups, str):
        groups = list(groups)
        if len(groups) > 1:
            # NB: R pastes *all* columns of the row (apply(data, 1, paste0, ...));
            #     here only the grouping columns are pasted, as intended
            del_var = ".".join(groups)
            data[del_var] = data[groups].astype(str).agg(".".join, axis=1)
            groups = del_var
        else:
            groups = groups[0]
    gvals = data[groups]
    gvals = np.asarray(gvals.astype(object) if hasattr(gvals, "astype") else gvals, dtype=object)
    unique_del = list(pd.unique(gvals))
    if start is not None:
        par = start
    else:
        par = dict(theta=np.asarray(model.theta, float))
        if model.isGLMM() and (model.devcomp["dims"]["nAGQ"] or 0) > 0:
            par["fixef"] = np.asarray(model.beta, float)
    fixed = model.fixef()
    Vs = model.VarCorr()
    items = Vs.items_list()
    nms = [k for k, _ in items]
    sep = ":"
    if len(nms) == 1:
        nms = [""]
        sep = ""
    vc_names = ["sigma^2"]
    vc = [model.getME("sigma") ** 2]
    for i, (_, V) in enumerate(items):
        cn = list(V.cnms)
        nc = len(cn)
        for b in range(nc):           # lower.tri(diag=TRUE), column-major
            for a in range(b, nc):
                nm = f"v[{cn[a]}]" if a == b else f"C[{cn[a]},{cn[b]}]"
                vc_names.append(nms[i] + sep + nm)
                vc.append(V.cov[a, b])
    vc = pd.Series(vc, index=vc_names)
    if model.isGLMM():
        control = glmerControl(optCtrl=dict(maxfun=maxfun))
    else:
        control = lmerControl(optCtrl=dict(maxfun=maxfun))

    def deleteGroup(d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod_1 = model.update(data=data, subset=(gvals != d), start=par, control=control)
        opt = mod_1.optinfo
        feval = opt.get("feval", np.nan)
        converged = opt["conv"]["opt"] == 0 and len(opt.get("warnings") or []) == 0
        fixed_1 = mod_1.fixef()
        vc_0 = [mod_1.getME("sigma") ** 2]
        for _, V in mod_1.VarCorr().items_list():
            nc = V.cov.shape[0]
            for b in range(nc):
                for a in range(b, nc):
                    vc_0.append(V.cov[a, b])
        return dict(fixed_1=fixed_1.values, vc_1=np.asarray(vc_0, float),
                    vcov_1=_vcov(mod_1), converged=bool(converged),
                    feval=np.nan if feval is None else feval)

    if parallel == "no":
        result = [deleteGroup(d) for d in unique_del]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, int(ncpus))) as ex:
            result = list(ex.map(deleteGroup, unique_del))
    rn = [str(u) for u in unique_del]
    fixed_1 = pd.DataFrame(np.vstack([r["fixed_1"] for r in result]), index=rn,
                           columns=fixed.index)
    vc_1 = pd.DataFrame(np.vstack([r["vc_1"] for r in result]), index=rn, columns=vc.index)
    feval = pd.Series([r["feval"] for r in result], index=rn, dtype=float)
    converged = pd.Series([r["converged"] for r in result], index=rn)
    vcov_1 = {k: r["vcov_1"] for k, r in zip(rn, result)}
    if groups == ".case":
        groups = "case"
    left, right = "[-", "]"
    out = InfluenceMerMod()
    out["fixed.effects"] = fixed
    out[f"fixed.effects{left}{groups}{right}"] = fixed_1
    out["var.cov.comps"] = vc
    out[f"var.cov.comps{left}{groups}{right}"] = vc_1
    out["vcov"] = _vcov(model)
    out[f"vcov{left}{groups}{right}"] = vcov_1
    out["groups"] = groups
    out["deleted"] = unique_del
    out["converged"] = converged
    out["function.evals"] = feval
    return out


def influence(model, *args, **kw):
    return influence_merMod(model, *args, **kw)


# ---------------------------------------------------------------------------
# dfbeta / dfbetas / cooks.distance for influence.merMod
# ---------------------------------------------------------------------------

def dfbeta_influence_merMod(model: InfluenceMerMod, which="fixed"):
    if which not in ("fixed", "var.cov"):
        raise ValueError("'which' should be one of 'fixed', 'var.cov'")
    b = model._pos(_IPOS[which + ".sub"])
    b0 = model._pos(_IPOS[which])
    return b - np.asarray(b0, float)[None, :]


def dfbetas_influence_merMod(model: InfluenceMerMod):
    vList = model._pos(_IPOS["vcov.sub"])
    vmat = np.vstack([np.sqrt(np.diag(np.asarray(v, float))) for v in vList.values()])
    return dfbeta_influence_merMod(model) / vmat


def cooks_distance_influence_merMod(model: InfluenceMerMod):
    db = dfbeta_influence_merMod(model)
    n, p = db.shape
    vcov_inv = (n - p) / (n * p) * np.linalg.inv(np.asarray(model["vcov"], float))
    D = np.asarray(db, float)
    d = np.einsum("ij,jk,ik->i", D, vcov_inv, D)
    return pd.Series(d, index=db.index)


# ---------------------------------------------------------------------------
# cooks.distance / rstudent / rstandard for merMod
# ---------------------------------------------------------------------------

def cooks_distance_merMod(model):
    p = np.linalg.matrix_rank(model.getME("X"))
    hat = _hat(model)
    dispersion = model.sigma() ** 2
    res = np.asarray(model.residuals(type="pearson"), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (res / (1 - hat)) ** 2 * hat / (dispersion * p)
    res[np.isinf(res)] = np.nan
    return _naresid(model, res)


def rstudent_merMod(model):
    """rstudent.merMod (lmerMod) / rstudent.glmerMod"""
    if model.isGLMM():
        return rstudent_glmerMod(model)
    r = np.asarray(model.residuals(type="deviance"), float)
    hat = _hat(model)
    pr = np.asarray(model.residuals(type="pearson"), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sign(r) * np.sqrt(r ** 2 + (hat * pr ** 2) / (1 - hat))
    r[np.isinf(r)] = np.nan
    return _naresid(model, r / model.sigma())


def rstandard_glmerMod(model, infl=None, type="pearson"):
    if type not in ("pearson", "deviance"):
        raise ValueError("'type' should be one of 'pearson', 'deviance'")
    res = np.asarray(model.residuals(type=type), float)
    hat = _hat(model) if infl is None else np.asarray(infl["hat"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = res / (model.sigma() * np.sqrt(1 - hat))
    res[np.isinf(res)] = np.nan
    return _naresid(model, res)


def rstandard(model, infl=None, type="pearson"):
    if not model.isGLMM():
        raise TypeError("no applicable method for 'rstandard' applied to an object of class "
                        f"\"{type(model).__name__}\"")
    return rstandard_glmerMod(model, infl=infl, type=type)


def rstudent_glmerMod(model, infl=None):
    useSc = model.devcomp["dims"]["useSc"]
    if useSc:
        return rstandard_glmerMod(model, infl, type="pearson")
    hat = _hat(model) if infl is None else np.asarray(infl["hat"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.asarray(model.residuals(type="pearson"), float) / (model.sigma() * np.sqrt(1 - hat))
    return _naresid(model, res)


def rstudent(model, infl=None):
    if model.isGLMM():
        return rstudent_glmerMod(model, infl)
    return rstudent_merMod(model)


def cooks_distance(model):
    """dispatch: influence.merMod object or merMod"""
    if isinstance(model, InfluenceMerMod):
        return cooks_distance_influence_merMod(model)
    return cooks_distance_merMod(model)


def dfbeta(model, which="fixed"):
    if not isinstance(model, InfluenceMerMod):
        model = influence_merMod(model)
    return dfbeta_influence_merMod(model, which)


def dfbetas(model):
    if not isinstance(model, InfluenceMerMod):
        model = influence_merMod(model)
    return dfbetas_influence_merMod(model)
