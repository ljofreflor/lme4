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


def drop1_merMod(obj, scope=None, test="none", k=2):
    """Single-term deletions of fixed-effect terms (drop1.merMod).
    test: "none" or "Chisq".  Models are refitted by maximum likelihood."""
    if test not in ("none", "Chisq"):
        raise ValueError("test must be 'none' or 'Chisq'")
    tt = obj.frame.terms.extra.get("fixed.terms")
    tl = list(tt.term_labels) if tt is not None else []
    if scope is None:
        scope = drop_scope(tl)
    else:
        scope = [scope] if isinstance(scope, str) else list(scope)
        if not all(s in tl for s in scope):
            raise ValueError("scope is not a subset of term labels")
    df0, aic0 = obj.extractAIC(k=k)
    rows = [(df0, aic0)]
    n0 = obj.nobs()
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
