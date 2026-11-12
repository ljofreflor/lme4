"""drop1.merMod (lme4/R/lmer.R) and stats::drop.scope."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def drop_scope(term_labels):
    """stats::drop.scope: terms that can be dropped without violating marginality
    (a term is droppable if it is not contained in any other term)"""
    sets = [frozenset(t.split(":")) for t in term_labels]
    out = []
    for i, s in enumerate(sets):
        if not any(i != j and s < o for j, o in enumerate(sets)):
            out.append(term_labels[i])
    return out


def drop1_merMod(obj, scope=None, test="none", k=2, sumFun=None, scale=0, **kwargs):
    """Single-term deletions of fixed-effect terms (drop1.merMod).
    test: "none", "Chisq" or "user".  Models are refitted by maximum likelihood.

    With test="user", ``sumFun(object, objectDrop=None, scale=, k=, **kwargs)``
    is called once without ``objectDrop`` (it must then return a pandas Series
    with the right length and names) and once per dropped term with the full
    and the reduced model; the rows are returned as a DataFrame."""
    if test not in ("none", "Chisq", "user"):
        raise ValueError("test must be 'none', 'Chisq' or 'user'")
    if (test == "user") != (sumFun is not None):
        raise ValueError("'sumFun' must be specified if (and only if) test==\"user\"")
    tt = obj.frame.terms.extra.get("fixed.terms")
    tl = list(tt.term_labels) if tt is not None else []
    if scope is None:
        scope = drop_scope(tl)
    else:
        scope = [scope] if isinstance(scope, str) else list(scope)
        if not all(s in tl for s in scope):
            raise ValueError("scope is not a subset of term labels")
    n0 = obj.nobs()
    if test == "user":
        ss = pd.Series(sumFun(obj, scale=scale, k=k, **kwargs))
        rows = [ss.values]
        for term in scope:
            nfit = obj.update(f". ~ . - {term}")
            rows.append(pd.Series(sumFun(obj, nfit, scale=scale, k=k, **kwargs)).values)
            if nfit.nobs() != n0:
                raise ValueError("number of rows in use has changed: remove missing values?")
        aod = pd.DataFrame(np.array(rows, float), index=["<none>"] + scope, columns=ss.index)
        aod.attrs["heading"] = ["Single term deletions", "", "Model:", str(obj.formula())]
        method = getattr(sumFun, "method", None)
        if method:
            aod.attrs["heading"] = [method] + aod.attrs["heading"]
        return aod
    df0, aic0 = obj.extractAIC(k=k)
    rows = [(df0, aic0)]
    for term in scope:
        nfit = obj.update(f". ~ . - {term}")
        if nfit.nobs() != n0:
            raise ValueError("number of rows in use has changed: remove missing values?")
        rows.append(nfit.extractAIC(k=k))
    ans = np.array(rows, float)
    dfs = ans[0, 0] - ans[:, 0]
    dfs[0] = np.nan
    aod = pd.DataFrame({"npar": dfs, "AIC": ans[:, 1]}, index=["<none>"] + scope)
    if test == "Chisq":
        dev = ans[:, 1] - k * ans[:, 0]
        dev = dev - dev[0]
        dev[0] = np.nan
        p = np.full(len(dev), np.nan)
        ok = ~np.isnan(dev) & (dfs > 0)
        p[ok] = stats.chi2.sf(dev[ok], dfs[ok])
        aod["LRT"] = dev
        aod["Pr(Chi)"] = p
    aod.attrs["heading"] = ["Single term deletions", "", "Model:", str(obj.formula())]
    return aod
