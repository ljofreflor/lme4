"""Transcription of lme4/R/VarCorr.R and reformulas' formatVC()."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import CovarianceUS
from .rformat import r_format, format_char_matrix


class VCmat:
    """one random-effects covariance matrix (R's ``vcmat_*`` class)"""

    def __init__(self, cov, cnms, kind):
        self.cov = np.asarray(cov, float)
        self.cnms = list(cnms)
        self.kind = kind                      # us, diag, cs, homcs, ar1, hetar1
        sd = np.sqrt(np.diag(self.cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            cor = self.cov / sd[:, None] / sd[None, :]
        np.fill_diagonal(cor, 1.0)
        self.stddev = sd
        self.correlation = cor

    def to_frame(self):
        return pd.DataFrame(self.cov, index=self.cnms, columns=self.cnms)

    def __array__(self, dtype=None, copy=None):
        return self.cov.astype(dtype) if dtype else self.cov

    def __repr__(self):
        return repr(self.to_frame())


class VarCorr(dict):
    """R's VarCorr.merMod: dict name -> VCmat, with attributes sc and useSc"""

    def __init__(self, items, sc, useSc):
        super().__init__()
        self._items = items
        for k, v in items:
            super().__setitem__(k, v)
        self.sc = sc
        self.useSc = useSc

    def items_list(self):
        return list(self._items)

    def as_data_frame(self, order="cov.last"):
        rows = []
        for grp, S in self._items:
            nc = len(S.cnms)
            if order == "cov.last":
                for i in range(nc):
                    rows.append((grp, S.cnms[i], None, S.cov[i, i], S.stddev[i]))
                for j in range(nc):
                    for i in range(j + 1, nc):
                        rows.append((grp, S.cnms[i], S.cnms[j], S.cov[i, j], S.correlation[i, j]))
            else:
                for j in range(nc):
                    for i in range(j, nc):
                        if i == j:
                            rows.append((grp, S.cnms[i], None, S.cov[i, i], S.stddev[i]))
                        else:
                            rows.append((grp, S.cnms[j], S.cnms[i], S.cov[i, j], S.correlation[i, j]))
        if self.useSc:
            rows.append(("Residual", None, None, self.sc ** 2, self.sc))
        return pd.DataFrame(rows, columns=["grp", "var1", "var2", "vcov", "sdcor"])

    def format(self, digits=5, comp=("Std.Dev.",), corr_digits=None, maxdim=10):
        return formatVC(self, digits=digits, comp=comp, corr_digits=corr_digits, maxdim=maxdim)

    def __repr__(self):
        return self.format(digits=5)

    __str__ = __repr__


def mkVarCorr(sc, cnms, theta=None, reCovs=None, relReCovs=True):
    """cnms: list of (group name, column names)"""
    nc = [len(c) for _, c in cnms]
    if reCovs is None:
        reCovs, pos = [], 0
        for k in nc:
            m = (k * (k + 1)) // 2
            reCovs.append(CovarianceUS(k, par=theta[pos:pos + m]))
            pos += m
    items = []
    nms = [g for g, _ in cnms]
    # make.names(unique=TRUE) for duplicated group names
    seen = {}
    out_names = []
    for n in nms:
        if nms.count(n) > 1:
            k = seen.get(n, 0)
            out_names.append(n if k == 0 else f"{n}.{k}")
            seen[n] = k + 1
        else:
            out_names.append(n)
    for (g, cn), obj, nm in zip(cnms, reCovs, out_names):
        L = obj.getLambda()
        S = sc * sc * (L @ L.T) if relReCovs else L @ L.T
        typ = obj.kind
        if typ == "cs" and obj.hom:
            typ = "homcs"
        if typ == "ar1" and not obj.hom:
            typ = "hetar1"
        items.append((nm, VCmat(S, cn, typ)))
    return items


def _get_sd(S: VCmat):
    if S.kind in ("ar1", "homcs", "homdiag"):
        return S.stddev[:1]
    return S.stddev


def _format_corr(S: VCmat, maxdim, digits):
    nc = len(S.cnms)
    if S.kind in ("diag", "homdiag"):
        return [[""]]
    if S.kind in ("ar1", "hetar1", "cs", "homcs"):
        tag = "ar1" if "ar1" in S.kind else "cs"
        x = S.correlation
        v = x[0, 0] if x.size == 1 else x[1, 0]
        return [[f"{np.round(v, digits):.{digits}f} ({tag})"]]
    if nc == 0:
        return [[""]]
    newdim = min(maxdim, nc)
    x = S.correlation[:newdim, :newdim]
    cc = r_format(np.round(x, digits).T.ravel(), digits=7, nsmall=digits)
    cc = np.array([s.strip() for s in cc], dtype=object).reshape(newdim, newdim).T
    # R: format() on a matrix pads to a common width
    w = max(len(s) for s in cc.ravel())
    cc = np.array([[s.rjust(w) for s in row] for row in cc], dtype=object)
    for i in range(newdim):
        for j in range(newdim):
            if j >= i:
                cc[i, j] = ""
    rows = [list(r) for r in cc]
    if nc > maxdim:
        rows.append(["..."] * newdim)
    return rows


def formatVC(varcor: VarCorr, digits=5, corr_digits=None, maxdim=10, comp=("Std.Dev.",),
             useScale=None):
    """reformulas::formatVC -> (header, rows) and printable text"""
    if corr_digits is None:
        corr_digits = max(2, digits - 2)
    if isinstance(comp, str):
        comp = (comp,)
    use_c = [c for c in ("Variance", "Std.Dev.") if any(c.startswith(x) for x in comp)]
    if not use_c:
        raise ValueError("Must report either standard deviations or variances")
    useScale = varcor.useSc if useScale is None else useScale
    termnames = [k for k, _ in varcor.items_list()]
    reStdDev = [(_get_sd(S), S.cnms if len(_get_sd(S)) == len(S.cnms) else S.cnms[:1])
                for _, S in varcor.items_list()]
    corr_out = [_format_corr(S, maxdim, corr_digits) for _, S in varcor.items_list()]
    if useScale:
        reStdDev.append((np.array([varcor.sc]), [""]))
        termnames.append("Residual")
        corr_out.append([[""]])
    allsd = np.concatenate([sd for sd, _ in reStdDev]) if reStdDev else np.zeros(0)
    cols = {}
    if "Variance" in use_c:
        cols["Variance"] = r_format(allsd ** 2, digits=digits)
    if "Std.Dev." in use_c:
        cols["Std.Dev."] = r_format(allsd, digits=digits)
    names_all = [n for _, nms in reStdDev for n in nms]
    # assemble
    blocks = []
    pos = 0
    for t, (sd, nms), cor in zip(termnames, reStdDev, corr_out):
        k = len(sd)
        sv = [[names_all[pos + i]] + [cols[c][pos + i] for c in use_c] for i in range(k)]
        if k > maxdim:
            sv = sv[:maxdim] + [["..."] * (1 + len(use_c))]
        pos += k
        blocks.append((t, sv, cor))
    max_corr_cols = max(len(c[0]) if c else 0 for _, _, c in blocks)
    rows = []
    for t, sv, cor in blocks:
        nrows = max(len(sv), len(cor))
        for i in range(nrows):
            g = t if i == 0 else ""
            s = sv[i] if i < len(sv) else [""] * (1 + len(use_c))
            c = cor[i] if i < len(cor) else []
            c = list(c) + [""] * (max_corr_cols - len(c))
            rows.append([g] + s + c)
    colnames = ["Groups", "Name"] + use_c
    all_corr_empty = all(all(x == "" for x in r[len(colnames):]) for r in rows)
    if all_corr_empty:
        rows = [r[:len(colnames)] for r in rows]
    else:
        colnames = colnames + ["Corr"] + [""] * (max_corr_cols - 1)
    return format_char_matrix(rows, colnames, [""] * len(rows), right=False)
