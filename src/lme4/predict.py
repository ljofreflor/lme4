"""Transcription of lme4/R/predict.R: ``predict.merMod``, ``mkNewReTrms``,
``levelfun``, ``get.orig.levs``, ``setParams``, ``simulate.merMod`` /
``.simulateFun`` and the family simulation functions.

Conventions
-----------
* ``re_form``: ``None`` is R's ``NULL`` (condition on all random effects);
  the string ``"NA"``, ``numpy.nan``, ``pandas.NA`` or ``False`` is R's ``NA``
  (no random effects); otherwise a formula (string such as ``"~(1|Subject)"``
  or :class:`lme4.rlang.Formula`).  ``"~0"`` is equivalent to ``NA``.
* ``na_action``: one of ``"na.pass"`` (default), ``"na.omit"``,
  ``"na.exclude"``, ``"na.fail"``.
* random numbers come from a :class:`numpy.random.Generator`; ``seed`` may be
  ``None``, an integer (passed to :func:`numpy.random.default_rng`), or a
  ``Generator``.  The streams therefore differ from R's.
"""
from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import options
from .covariance import upReTrms, mkMkTheta
from .family import isNBfamily
from .modelframe import (Factor, ModelFrame, EvalContext, as_factor, data_to_dict, eval_expr,
                         model_frame, model_matrix, terms, _fmt_level)
from .reformulas import (findbars_x, findbars, lme4_reOnly, lme4_specials, mkReTrms, noSpecials,
                         sub_specials, sumTerms, makeOp)
from .rlang import Call, Const, Formula, Sym, all_vars, deparse, is_call

__all__ = ["predict_merMod", "simulate_merMod", "mkNewReTrms", "levelfun", "get_orig_levs",
           "setParams", "isRE", "simfunList"]


class _Missing:
    def __repr__(self):
        return "<missing>"


_MISSING = _Missing()


class _NAType:
    """R's ``NA`` as a value of ``re.form``"""

    def __repr__(self):
        return "NA"


NA = _NAType()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_na_value(x):
    if x is NA or x is False:
        return True
    if isinstance(x, str):
        return x.strip() == "NA"
    try:
        return x is pd.NA or (isinstance(x, float) and math.isnan(x))
    except TypeError:
        return False


def _norm_reform(re_form, object):
    """translate the Python spelling of ``re.form`` into None / NA / Formula"""
    if re_form is None:
        return None
    if _is_na_value(re_form):
        return NA
    from .symbolic import to_formula
    env = object.call["formula"].env if object.call and object.call.get("formula") is not None else None
    f = to_formula(re_form, env)
    return f


def isRE(re_form):
    """TRUE unless re.form is NA or ~0"""
    if re_form is None:
        return True
    if re_form is NA:
        return False
    if isinstance(re_form, Formula):
        if re_form.lhs is None and isinstance(re_form.rhs, Const) and re_form.rhs.value == 0:
            return False
        return True
    return not _is_na_value(re_form)


def _na_type(na_action):
    """class of attr(na.action(NA), "na.action") as used by napredict"""
    if na_action in ("na.exclude",):
        return "na.exclude"
    if na_action in ("na.omit",):
        return "na.omit"
    return None


def _napredict(na_info, x, index=None):
    """stats::napredict: re-insert NAs for omitted rows when na.action is na.exclude.
    ``x`` is a 1-d or 2-d array; returns (x, index)"""
    if na_info is None or na_info.get("type") != "na.exclude" or len(na_info["omitted"]) == 0:
        return x, index
    omit = np.asarray(na_info["omitted"], int)
    x = np.asarray(x)
    n = x.shape[0] + len(omit)
    keep = np.ones(n, bool)
    keep[omit] = False
    if x.dtype.kind in "biu":
        x = x.astype(float)
    shape = (n,) + x.shape[1:]
    out = np.full(shape, np.nan, dtype=float if x.dtype.kind != "O" else object)
    out[keep] = x
    if index is not None:
        idx = np.empty(n, dtype=object)
        idx[keep] = np.asarray(index, dtype=object)
        idx[omit] = np.asarray(na_info["names"], dtype=object)
        index = idx
    return out, index


def _as_dataframe(newdata):
    if newdata is None or isinstance(newdata, pd.DataFrame):
        return newdata
    if isinstance(newdata, ModelFrame):
        return newdata.to_pandas()
    return pd.DataFrame(newdata)


def _col_strings(v):
    """as.character() of a data column; None for NA"""
    if isinstance(v, Factor):
        return v.as_strings()
    if isinstance(v, pd.Series):
        v = v.array
    if isinstance(v, pd.Categorical):
        return as_factor(v).as_strings()
    a = np.asarray(v, dtype=object).ravel()
    return np.array([None if (x is None or (not isinstance(x, str) and pd.isna(x))) else _fmt_level(x)
                     for x in a.tolist()], dtype=object)


def _relevel(v, levels, name, contrasts=None):
    """``factor(x, levels = xlev[[nm]])`` as done by model.frame(xlev=); values that
    are not among the levels raise the same error as R"""
    s = _col_strings(v)
    key = {l: i for i, l in enumerate(levels)}
    codes = np.array([-1 if x is None else key.get(x, -2) for x in s.tolist()], dtype=np.int64)
    if np.any(codes == -2):
        bad = list(dict.fromkeys(x for x, c in zip(s.tolist(), codes) if c == -2))
        raise ValueError(f"factor {name} has new level{'s' if len(bad) > 1 else ''} {', '.join(bad)}")
    ordered = v.ordered if isinstance(v, Factor) else False
    return Factor(codes, list(levels), ordered, contrasts)


def _response_key(object):
    tt = object.frame.terms
    if tt is not None and tt.response:
        return tt.varnames[0]
    return None


def _fixed_terms(object):
    tt = object.frame.terms.extra.get("fixed.terms")
    if tt is None:
        from .reformulas import nobars
        tt = terms(nobars(object.formula()))
    return tt


def _expand_dot(rhs, cols, resp_vars):
    """substitute the '.' of a formula with the data columns (as terms.formula(data=))"""
    if isinstance(rhs, Sym) and rhs.name == ".":
        syms = [Sym(c) for c in cols if c not in resp_vars]
        return sumTerms(syms) if syms else Const(1.0)
    if isinstance(rhs, Call):
        return Call(rhs.fn, tuple(_expand_dot(a, cols, resp_vars) for a in rhs.args), rhs.names)
    return rhs


def _rhs_terms(full_tt, predvars, newdata=None):
    """delete.response(tt) with predvars carried over by variable name;
    returns (rhs Formula, {varname: predvar})"""
    f = full_tt.formula
    rhs = f.rhs
    if newdata is not None:
        resp_vars = set(all_vars(f.lhs)) if f.lhs is not None else set()
        rhs = _expand_dot(rhs, [str(c) for c in newdata.columns], resp_vars)
    pv = {}
    if predvars is not None:
        for k, p in zip(full_tt.varnames, predvars):
            pv[k] = p
    return Formula(Call(Sym("~"), (rhs,)), f.env), pv


def _frame_newdata(form, newdata, pvmap, na_action, xlev, contrasts=None, env=None):
    """model.frame(form, newdata, na.action=, xlev=) using the stored predvars"""
    tt = terms(form, data_columns=[str(c) for c in newdata.columns])
    predvars = [pvmap.get(k, v) for k, v in zip(tt.varnames, tt.variables)]
    fr = model_frame(form, newdata, na_action=na_action, drop_unused_levels=False,
                     predvars=predvars, env=env if env is not None else form.env)
    contrasts = contrasts or {}
    if xlev:
        for nm, lev in xlev.items():
            if nm in fr.columns:
                fr.columns[nm] = _relevel(fr.columns[nm], lev, nm, contrasts.get(nm))
    return fr


# ---------------------------------------------------------------------------
# get.orig.levs
# ---------------------------------------------------------------------------

def get_orig_levs(object, FUN="levels", newdata=None, fixed_only=False, random_only=False,
                  sparse=False):
    """lme4:::get.orig.levs.  ``FUN`` is ``"levels"`` or ``"contrasts"`` (the
    latter returns the contrasts specification stored with each factor)."""
    fr = object.frame
    cols = dict(fr.columns)
    if fixed_only and not random_only:
        vars_ = fr.terms.extra.get("varnames.fixed")
        if vars_:
            cols = {k: cols[k] for k in vars_ if k in cols}
    rk = _response_key(object)
    isFac = {k: isinstance(v, Factor) and k != rk for k, v in cols.items()}
    if not any(isFac.values()):
        return None
    if FUN == "levels":
        orig = {k: list(cols[k].levels) for k in cols if isFac[k]}
    else:
        orig = {k: cols[k].contrasts for k in cols if isFac[k]}
    if newdata is not None and FUN == "levels":
        for n in list(orig):
            if n in newdata.columns:
                seen = orig[n]
                add = [x for x in dict.fromkeys(_col_strings(newdata[n]).tolist())
                       if x is not None and x not in seen]
                orig[n] = seen + add
    return orig


# ---------------------------------------------------------------------------
# setParams
# ---------------------------------------------------------------------------

def setParams(object, params, inplace=False, subset=False, requireSigma=False):
    """Force new parameters (beta, theta or par, sigma) into a (copy of a) merMod"""
    pNames = ["beta", "theta", "par"]
    if object.devcomp["dims"]["useSc"]:
        pNames.append("sigma")
        if requireSigma and params.get("sigma") is None:
            raise ValueError("must specify sigma when using a family with an estimated "
                             "scale/dispersion parameter")
    if not isinstance(params, dict) or set(params) - set(pNames):
        raise ValueError("params should be specifed as a list with elements from {" +
                         ", ".join(f"'{p}'" for p in pNames) + "}")
    if not subset and (params.get("beta") is None or
                       (params.get("theta") is None and params.get("par") is None)):
        warnings.warn("some parameters not specified in setParams()")
    nbeta = len(object.beta)
    ntheta = len(object.theta)
    npar = object.getParLength()

    def lengthTest(nm, expected):
        x = params.get(nm)
        if x is not None:
            n = 1 if np.ndim(x) == 0 else len(x)
            if n != expected:
                raise ValueError(f"length mismatch in {nm} ({n} != {expected})")
        return x

    beta = lengthTest("beta", nbeta)
    if params.get("theta") is not None and params.get("par") is not None:
        raise ValueError("should specify exactly one of 'theta' and 'par'")
    theta = lengthTest("theta", ntheta)
    par = lengthTest("par", npar)

    def matchNames(x, tn, vecname):
        if x is None:
            return None
        if isinstance(x, dict):
            x = pd.Series(x)
        if isinstance(x, pd.Series) and not isinstance(x.index, pd.RangeIndex):
            pn = [str(i) for i in x.index]
            if set(pn) != set(tn):
                raise ValueError(f"mismatch between '{vecname}' parameter vector names and "
                                 f"internal names ({','.join(tn)})")
            x = pd.Series(np.asarray(x, float), index=pn)[list(tn)]
        return np.atleast_1d(np.asarray(x, float))

    theta = matchNames(theta, object.getThetaNames(), "theta")
    beta = matchNames(beta, object.colnames_X, "beta")
    if par is not None:
        par = np.atleast_1d(np.asarray(par, float))
    sigma = params.get("sigma")
    if inplace:
        raise NotImplementedError("modification in place (copy=FALSE) not yet implemented")
    newObj = copy.copy(object)
    newObj.pp = object.pp.copy()
    newObj.resp = object.resp.copy()
    newObj.devcomp = {"cmp": dict(object.devcomp["cmp"]), "dims": dict(object.devcomp["dims"])}
    if beta is not None:
        newObj.pp.setBeta0(np.asarray(beta, float))
        newObj.beta = np.asarray(beta, float)
    if par is not None:
        theta = mkMkTheta(object.getReCovs())(par)
    if theta is not None:
        newObj.theta = np.asarray(theta, float)
        newObj.pp.setTheta(np.asarray(theta, float))
    if sigma is not None:
        snm = "sigmaREML" if object.devcomp["dims"]["REML"] else "sigmaML"
        newObj.devcomp["cmp"][snm] = float(np.asarray(sigma, float).ravel()[0])
    return newObj


# ---------------------------------------------------------------------------
# mkNewReTrms / levelfun
# ---------------------------------------------------------------------------

def _random_terms(object):
    """delete.response(terms(object, random.only=TRUE)) as (Formula, predvars map)"""
    f = object.call["formula"]
    fr_form_ = noSpecials(f, delete=False, specials=lme4_specials)
    rr = lme4_reOnly(fr_form_)
    rhs = sub_specials(noSpecials(rr.rhs, delete=False, specials=lme4_specials),
                       specials=("|", "||"), keep_args=(2, 2))
    full = Formula(Call(Sym("~"), (fr_form_.lhs, rhs)) if fr_form_.lhs is not None
                   else Call(Sym("~"), (rhs,)), f.env)
    ft = terms(full)
    pv = object.frame.terms.extra.get("predvars.random")
    pvmap = dict(zip(ft.varnames, pv)) if pv is not None and len(pv) == len(ft.varnames) else {}
    return Formula(Call(Sym("~"), (rhs,)), f.env), pvmap, ft


def _orig_data_rows(object):
    from .modular import _data_rows
    data = object.call.get("_data") if object.call else None
    if data is None:
        return None
    try:
        return _data_rows(data, object.frame)
    except Exception:
        return None


def mkNewReTrms(object, newdata, re_form=None, na_action="na.pass", allow_new_levels=False):
    """Make new random-effect terms from a fitted model and (new) data,
    possibly omitting some random-effect terms"""
    fixed_na_action = None
    if re_form is None:
        re_form = lme4_reOnly(object.formula())
    extra = None
    newdata = _as_dataframe(newdata)
    if newdata is None:
        rfd = object.frame
        fixed_na_action = rfd.attrs.get("na.action")
        extra = _orig_data_rows(object)
    else:
        if na_action not in (None, "na.pass"):
            fixed_tt = _fixed_terms(object)
            form, pvmap = _rhs_terms(fixed_tt, object.frame.terms.extra.get("predvars.fixed"), newdata)
            mfnew = _frame_newdata(form, newdata, pvmap, na_action, None)
            fixed_na_action = mfnew.attrs.get("na.action")
        newdata_NA = newdata
        if fixed_na_action is not None and len(fixed_na_action["omitted"]) > 0:
            keep = np.ones(len(newdata), bool)
            keep[fixed_na_action["omitted"]] = False
            newdata_NA = newdata.iloc[np.nonzero(keep)[0]]
        tt_form, pvmap, _ = _random_terms(object)
        orig_random_levs = get_orig_levs(object, random_only=True, newdata=newdata_NA)
        orig_random_cntr = get_orig_levs(object, FUN="contrasts", random_only=True) or {}
        if isinstance(re_form, Formula):
            newdata_NA = newdata_NA.copy()
            rvars = set(all_vars(re_form.expr))
            tt0 = terms(tt_form)
            for k, v in zip(tt0.varnames, tt0.variables):
                pvi = pvmap.get(k, v)
                for mv in all_vars(pvi):
                    if mv not in rvars:
                        newdata_NA[mv] = np.nan
        rfd = _frame_newdata(tt_form, newdata_NA, pvmap, "na.pass", orig_random_levs)
        termvars = []
        for b in findbars(re_form):
            for v in all_vars(b[1]):
                if v not in termvars:
                    termvars.append(v)
        for fn in orig_random_cntr:
            if fn in termvars and fn in rfd.columns:
                col = rfd.columns[fn]
                if not isinstance(col, Factor):
                    col = as_factor(col)
                rfd.columns[fn] = Factor(col.codes, col.levels, col.ordered, orig_random_cntr[fn])
        if fixed_na_action is not None:
            rfd.attrs["na.action"] = fixed_na_action
    out_b = None
    if isinstance(re_form, Formula):
        bb1 = findbars_x(re_form, specials=lme4_specials, default_special="us", target="|",
                         expand_doublevert_method=options.get("lme4.doublevert.default"))
        bb0 = [b[1] for b in bb1]
        reTrms = mkReTrms(bb0, rfd, calc_lambdat=False, env=re_form.env, extra=extra)
        bb1 = [bb1[i] for i in reTrms["ord"]]
        reTrms = upReTrms(reTrms, bb1)
        Lt = reTrms["Lambdat"].copy()
        Lind = np.asarray(reTrms["Lind"], int)
        Lt.data = np.asarray(object.theta, float)[Lind - 1]
        reTrms["Lambdat"] = Lt
        if not allow_new_levels and any(np.any(f.isna()) for f in reTrms["flist"].values()):
            raise ValueError("NAs are not allowed in prediction data for grouping variables "
                             "unless allow.new.levels is TRUE")
        re = object.ranef(condVar=False)
        ns_re = list(re)
        Rcnms = reTrms["cnms"]
        if not all(g in ns_re for g, _ in Rcnms):
            raise ValueError("grouping factors specified in re.form that were not present in original model")
        new_levels = {k: list(f.droplevels().levels) for k, f in reTrms["flist"].items()}
        re_x = {k: levelfun(re[k], new_levels[k], allow_new_levels=allow_new_levels)
                for k in new_levels}
        hacked = [False]

        def get_re(rname, cnms):
            x = re_x[rname]
            nms = [str(c) for c in x.columns]
            if list(cnms) == ["(Intercept)"] and len(nms) == 1 and nms[0].startswith("s(") \
                    and nms[0].endswith(")"):
                hacked[0] = True
                cnms = nms
            miss = [c for c in cnms if c not in nms]
            if miss:
                raise ValueError("random effects specified in re.form that were not present in "
                                 "original model " + ", ".join(miss))
            # rows aligned with the levels of the new grouping factor (= rows of Zt)
            x = x.reindex([l for l in new_levels[rname]])
            return np.asarray(x[list(cnms)], float).ravel()   # t(.) then unlist: row-major

        out_b = np.concatenate([get_re(r, c) for r, c in Rcnms]) if Rcnms else np.zeros(0)
        if hacked[0]:
            warnings.warn("modified RE names for gamm4 prediction")
    return {"Zt": reTrms["Zt"], "b": out_b, "Lambdat": reTrms["Lambdat"],
            "flist": reTrms["flist"], "na.action": fixed_na_action}


def levelfun(x, nl_n, allow_new_levels=False):
    """x: data frame of random effects (rows = levels); nl_n: the new levels"""
    rn = [str(i) for i in x.index]
    x = x.copy()
    x.index = rn
    rset = set(rn)
    new_levels = [l for l in dict.fromkeys(nl_n) if l not in rset]
    if new_levels:
        if not allow_new_levels:
            max_err_len = 60
            err = ", ".join(str(l) for l in new_levels)
            if len(err) > max_err_len:
                import re as _re
                err = _re.sub(r",[^,]*$", ", ...", err[:max_err_len])
            raise ValueError("new levels detected in newdata: " + err)
        comb = list(dict.fromkeys(list(nl_n) + rn))
        newx = pd.DataFrame(0.0, index=comb, columns=x.columns)
        newx.loc[rn, :] = x.values
        x = newx
    nset = set(nl_n)
    r_inn = [r in nset for r in x.index]
    if not all(r_inn):
        x = x.loc[r_inn]
    return x


# ---------------------------------------------------------------------------
# predict.merMod
# ---------------------------------------------------------------------------

def _getData(object):
    data = object.call.get("_data") if object.call else None
    if isinstance(data, pd.DataFrame):
        return data
    if data is not None:
        try:
            return pd.DataFrame(data)
        except Exception:
            pass
    return object.frame.to_pandas()


def _newdata_X(object, newdata, na_action):
    """fixed-effects model matrix and offset for new data"""
    fixed_tt = _fixed_terms(object)
    form, pvmap = _rhs_terms(fixed_tt, object.frame.terms.extra.get("predvars.fixed"), newdata)
    orig_fixed_levs = get_orig_levs(object, fixed_only=True, newdata=newdata)
    contrasts = object.pp.Xattrs.get("contrasts") or {}
    orig_cntr = get_orig_levs(object, FUN="contrasts", fixed_only=True) or {}
    mfnew = _frame_newdata(form, newdata, pvmap, na_action, orig_fixed_levs, contrasts=orig_cntr)
    tt = mfnew.terms
    X, cn, _, _ = model_matrix(tt, mfnew, contrasts=contrasts)
    # offset terms of the (fixed) formula, evaluated in newdata
    offset = 0.0
    if fixed_tt.offset:
        ctx = EvalContext(data_to_dict(newdata), fixed_tt.formula.env, len(newdata))
        for i in fixed_tt.offset:
            offset = offset + np.asarray(eval_expr(fixed_tt.variables[i], ctx), float)
    fit_na_action = mfnew.attrs.get("na.action")
    if fit_na_action is not None and np.ndim(offset) > 0 and len(offset) != X.shape[0]:
        offset = np.delete(offset, fit_na_action["omitted"])
    # drop the columns dropped (rank deficiency) at fit time
    target = object.colnames_X
    dropped = object.pp.Xattrs.get("col.dropped")
    if dropped:
        cn_keep = [j for j, c in enumerate(cn) if c not in dropped]
        X = X[:, cn_keep]
        cn = [cn[j] for j in cn_keep]
    if cn != target:
        idx = {c: j for j, c in enumerate(cn)}
        miss = [c for c in target if c not in idx]
        if miss or len(cn) != len(target):
            raise ValueError("non-conformable arguments: the model matrix for newdata has columns "
                             f"{cn}, the fitted model {target}")
        X = X[:, [idx[c] for c in target]]
    return X, offset, fit_na_action, mfnew.row_names


def predict_merMod(object, newdata=None, newparams=None, re_form=None, random_only=False,
                   terms=None, type="link", allow_new_levels=False, na_action=_MISSING,
                   se_fit=False, **kw):
    """Predictions from a merMod object (predict.merMod).

    Returns a :class:`pandas.Series` named by the rows of ``newdata`` (or of the
    model frame)."""
    for k in ("ReForm", "REForm", "REform"):
        if k in kw:
            raise ValueError("synonyms 'ReForm', 'REForm', 'REform' are deprecated: please use "
                             "'re.form' instead")
    if kw:
        warnings.warn("unused arguments ignored")
    if type not in ("link", "response"):
        raise ValueError("'arg' should be one of \"link\", \"response\"")
    if terms is not None:
        raise NotImplementedError("terms functionality for predict not yet implemented")
    if se_fit:
        raise NotImplementedError("se.fit is not implemented")
    na_missing = na_action is _MISSING
    if na_missing:
        na_action = "na.pass"
    re_form = _norm_reform(re_form, object)
    if newparams is not None:
        object = setParams(object, newparams)
    newdata = _as_dataframe(newdata)
    if newdata is None and re_form is None and newparams is None and not random_only:
        rn = object.frame.row_names
        if object.isLMM() or object.isNLMM():
            pred = np.asarray(object.resp.mu, float)
            ok = ~np.isnan(pred)
            pred, index = pred[ok], rn[ok]
        else:
            pred = np.asarray(object.resp.mu if type == "response" else object.resp.eta, float)
            index = rn
        fit_na_action = None
    else:
        fit_na_action = object.frame.attrs.get("na.action")
        nobs = object.frame.nrow if newdata is None else len(newdata)
        pred = np.zeros(nobs)
        index = object.frame.row_names if newdata is None else np.asarray(newdata.index)
        if not random_only:
            if newdata is None:
                X = np.asarray(object.pp.X)
                offset = object.frame.offset()
                if offset is None:
                    offset = 0.0
                pred = X @ np.asarray(object.beta, float)
            else:
                X, offset, fit_na_action, index = _newdata_X(object, newdata, na_action)
                pred = X @ object.fixef().values
            pred = pred + offset
        if isRE(re_form):
            if re_form is None:
                re_form = lme4_reOnly(object.formula())
            rfd = _getData(object) if newdata is None else newdata
            newRE = mkNewReTrms(object, rfd, re_form, na_action=na_action,
                                allow_new_levels=allow_new_levels)
            REvals = np.asarray(sp.csc_matrix(newRE["Zt"]).T @ newRE["b"]).ravel()
            if len(pred) != len(REvals):
                fna = fit_na_action
                if fna is not None and fna.get("type") not in ("na.omit", "na.exclude") \
                        and len(fna["omitted"]) > 0:
                    raise ValueError("fixed/RE pred length mismatch")
                if fna is not None:
                    REvals = np.delete(REvals, fna["omitted"])
            pred = pred + REvals
            if random_only:
                fit_na_action = newRE["na.action"]
        if object.isGLMM() and type == "response":
            pred = object.resp.family.linkinv(pred)
    if newdata is None:
        fit_na_action = object.frame.attrs.get("na.action")
        if not na_missing and fit_na_action is not None:
            fit_na_action = dict(fit_na_action, type=_na_type(na_action))
    pred, index = _napredict(fit_na_action, np.asarray(pred, float), index)
    if index is None or len(index) != len(pred):
        index = np.arange(1, len(pred) + 1)
    return pd.Series(np.asarray(pred, float), index=index)


# ---------------------------------------------------------------------------
# family simulation functions (modified from stats/family.R)
# ---------------------------------------------------------------------------

def _wts_rep(wts, n):
    w = np.asarray(wts, float).ravel()
    return np.resize(w, n) if len(w) != n else w


def gaussian_simfun(object, nsim, ftd, wts, rng):
    if np.any(np.asarray(wts) != 1):
        warnings.warn("ignoring prior weights")
    ftd = np.asarray(ftd, float)
    return rng.normal(np.resize(ftd, nsim * len(ftd)), object.sigma())


def binomial_simfun(object, nsim, ftd, wts, rng):
    ftd = np.asarray(ftd, float)
    n = len(ftd)
    ntot = n * nsim
    w = np.asarray(wts, float)
    if np.any(w % 1 != 0):
        raise ValueError("cannot simulate from non-integer prior.weights")
    y = object.frame.response() if object.frame is not None else None
    if isinstance(y, Factor):
        draws = rng.binomial(1, np.resize(ftd, ntot))
        yy = Factor(draws.astype(np.int64), y.levels)
        return [yy.subset(np.arange(i * n, (i + 1) * n)) for i in range(nsim)]
    if y is not None and np.ndim(y) == 2 and np.shape(y)[1] == 2:
        cn = getattr(y, "colnames", None) or ["", ""]
        wr = _wts_rep(w, n).astype(np.int64)
        out = []
        for _ in range(nsim):
            Y = rng.binomial(wr, ftd).astype(float)
            out.append((np.column_stack([Y, wr - Y]), list(cn)))
        return out
    wr = _wts_rep(w, ntot)
    return rng.binomial(wr.astype(np.int64), np.resize(ftd, ntot)) / wr


def poisson_simfun(object, nsim, ftd, wts, rng):
    wts = object.weights()
    if np.any(np.asarray(wts) != 1):
        warnings.warn("ignoring prior weights")
    ftd = np.asarray(ftd, float)
    return rng.poisson(np.resize(ftd, nsim * len(ftd))).astype(float)


def Gamma_simfun(object, nsim, ftd, wts, rng):
    if np.any(np.asarray(wts) != 1):
        print("using weights to scale shape parameter")
    ftd = np.resize(np.asarray(ftd, float), nsim * len(ftd))
    shape = 1 / object.sigma() ** 2 * _wts_rep(wts, len(ftd))
    return rng.gamma(shape, ftd / shape)   # rate = shape/ftd


def inverse_gaussian_simfun(object, nsim, ftd, wts, rng):
    if np.any(np.asarray(wts) != 1):
        print("using weights as inverse variances")
    ftd = np.resize(np.asarray(ftd, float), nsim * len(ftd))
    shape = _wts_rep(wts, len(ftd)) / object.sigma()
    return rng.wald(ftd, shape)


def negative_binomial_simfun(object, nsim, ftd, wts, rng):
    if np.any(np.asarray(wts) != 1):
        warnings.warn("ignoring prior weights")
    theta = object.resp.family.theta
    ftd = np.resize(np.asarray(ftd, float), nsim * len(ftd))
    return rng.negative_binomial(theta, theta / (theta + ftd)).astype(float)


simfunList = {"gaussian": gaussian_simfun, "binomial": binomial_simfun,
              "poisson": poisson_simfun, "Gamma": Gamma_simfun,
              "negative.binomial": negative_binomial_simfun,
              "inverse.gaussian": inverse_gaussian_simfun}


# ---------------------------------------------------------------------------
# simulate.merMod
# ---------------------------------------------------------------------------

def _get_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _complement_reform(full, re_form):
    """update.formula(compReForm, . ~ . - <re.form terms>)[-2]"""
    rr = lme4_reOnly(re_form)
    drop = {deparse(t) for t in findbars_x(rr, expand_doublevert_method="split")}
    keep = [b for b in findbars_x(full, expand_doublevert_method="split") if deparse(b) not in drop]
    if not keep:
        return Formula(Call(Sym("~"), (Const(1.0),)), full.env)
    return Formula(Call(Sym("~"), (sumTerms([makeOp(b, Sym("(")) for b in keep]),)), full.env)


def simulate_merMod(object, nsim=1, seed=None, use_u=_MISSING, re_form=_MISSING, newdata=None,
                    newparams=None, family=None, cluster_rand=None, allow_new_levels=False,
                    na_action=_MISSING, cond_sim=True, weights=None, offset=None, **kw):
    """simulate.merMod / .simulateFun.

    ``re_form`` defaults to R's ``NA`` (unconditional simulation); ``re_form=None``
    is treated as *not specified* (use ``use_u=True`` for R's ``re.form=NULL``).
    ``cluster_rand(rng, size)`` replaces R's ``cluster.rand`` (default: standard normal).
    Returns a DataFrame with columns ``sim_1``, ..., ``sim_nsim`` (for a two-column
    binomial response, MultiIndex columns ``(sim_k, colname)``); ``.attrs`` holds
    ``na.action`` and ``seed`` (the generator state before simulating)."""
    if kw:
        warnings.warn("unused arguments ignored")
    nsim = int(np.atleast_1d(nsim)[0])
    if nsim <= 0:
        raise ValueError("(nsim <- as.integer(nsim[1])) > 0 is not TRUE")
    newdata = _as_dataframe(newdata)
    nullWts = False
    if weights is None:
        if newdata is None:
            weights = object.weights()
        else:
            nullWts = True
            weights = np.ones(len(newdata))
    if newparams is not None:
        object = setParams(object, newparams)
    if re_form is None:
        re_form = _MISSING
    if use_u is not _MISSING and use_u is not None:
        if re_form is not _MISSING:
            if use_u:
                raise ValueError("should specify only one of 'use.u' and 're.form'")
        else:
            re_form = None if use_u else Formula(Call(Sym("~"), (Const(0.0),)))
    if re_form is _MISSING:
        re_form = NA
    elif re_form is not None:
        re_form = _norm_reform(re_form, object)
    if re_form is None:
        re_form = lme4_reOnly(object.formula())
    rng = _get_rng(seed)
    RNGstate = copy.deepcopy(rng.bit_generator.state)
    sigma = object.sigma()
    etapred = np.asarray(predict_merMod(object, newdata=newdata, re_form=re_form, type="link",
                                        na_action="na.omit",
                                        allow_new_levels=allow_new_levels), float)
    n = len(etapred)
    compReForm = lme4_reOnly(object.formula())
    if isRE(re_form):
        compReForm = _complement_reform(compReForm, re_form)
    na_act = "na.pass" if na_action is _MISSING else na_action
    if findbars(compReForm):
        newRE = mkNewReTrms(object, newdata, compReForm, na_action=na_act,
                            allow_new_levels=allow_new_levels)
        U = (sp.csc_matrix(newRE["Lambdat"]) @ sp.csc_matrix(newRE["Zt"])).T.tocsr()
        q = U.shape[1]
        if cluster_rand is None:
            u = rng.standard_normal(q * nsim)
        else:
            u = np.asarray(cluster_rand(rng, q * nsim), float)
        sim_reff = np.asarray(U @ u.reshape(q, nsim, order="F"))
    else:
        sim_reff = 0.0
    if object.isLMM():
        resid = rng.standard_normal(n * nsim).reshape(n, nsim, order="F") if cond_sim else 0.0
        val = etapred[:, None] + sigma * (sim_reff + resid)
        val = np.broadcast_to(val, (n, nsim))
        is_list = False
    elif object.isGLMM():
        etasim = etapred[:, None] + sim_reff
        etasim = np.broadcast_to(etasim, (n, nsim))
        fam = object.resp.family
        famname = "negative.binomial" if isNBfamily(fam.family) else fam.family
        musim = fam.linkinv(np.asarray(etasim).ravel(order="F"))
        r = object.frame.response()
        if famname == "binomial" and isinstance(r, np.ndarray) and r.ndim == 2:
            if nullWts:
                weights = np.asarray(r, float).sum(axis=1)
        if weights is None:
            weights = np.ones(n)
        sfun = simfunList.get(famname)
        if sfun is None:
            raise ValueError(f"simulation not implemented for family '{famname}'")
        if cond_sim:
            res = sfun(object, 1, np.resize(musim, n * nsim), weights, rng)
        else:
            res = np.resize(musim, n * nsim)
        is_list = True
        if famname == "binomial" and isinstance(r, np.ndarray) and r.ndim == 2 and cond_sim:
            M, cn = res[0]
            val = [(M[i * n:(i + 1) * n], cn) for i in range(nsim)]
            kind = "matrix"
        elif famname == "binomial" and isinstance(res, list) and isinstance(res[0], Factor):
            yy = res[0]
            val = [yy.subset(np.arange(i * n, (i + 1) * n)) for i in range(nsim)]
            kind = "factor"
        else:
            res = np.asarray(res, float)
            val = [res[i * n:(i + 1) * n] for i in range(nsim)]
            kind = "vector"
    else:
        raise NotImplementedError("simulate method for NLMMs not yet implemented")

    names = [f"sim_{i + 1}" for i in range(nsim)]
    # row names
    if newdata is not None:
        nm = np.asarray(newdata.index) if len(newdata) == n else np.arange(1, n + 1)
    else:
        nm = object.frame.row_names if object.frame.nrow == n else np.arange(1, n + 1)
    fit_na_action = object.frame.attrs.get("na.action")
    if na_action is not _MISSING and fit_na_action is not None:
        fit_na_action = dict(fit_na_action, type=_na_type(na_action))
    idx_full = None
    if not is_list:
        data, idx_full = _napredict(fit_na_action, np.asarray(val), nm)
        out = pd.DataFrame(data, columns=names)
    elif kind == "matrix":
        cols = {}
        for nme, (M, cn) in zip(names, val):
            Mx, idx_full = _napredict(fit_na_action, M, nm)
            for j in range(2):
                cols[(nme, cn[j])] = Mx[:, j]
        out = pd.DataFrame(cols)
        out.columns = pd.MultiIndex.from_tuples(list(cols))
    elif kind == "factor":
        cols = {}
        for nme, f in zip(names, val):
            codes, idx_full = _napredict(fit_na_action, f.codes.astype(float), nm)
            codes = np.where(np.isnan(codes), -1, codes).astype(np.int64)
            cols[nme] = pd.Categorical.from_codes(codes, categories=f.levels)
        out = pd.DataFrame(cols)
    else:
        cols = {}
        for nme, v in zip(names, val):
            cols[nme], idx_full = _napredict(fit_na_action, v, nm)
        out = pd.DataFrame(cols)
    if idx_full is not None and len(idx_full) == len(out):
        out.index = idx_full
    out.attrs["na.action"] = fit_na_action
    out.attrs["seed"] = RNGstate
    return out
