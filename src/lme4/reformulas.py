"""Transcription of the formula utilities of the R package *reformulas*
(findbars_x, nobars, subbars, expandDoubleVerts, noSpecials, reOnly,
mkReTrms, mkBlist, ...) used by lme4 >= 2.0.

Formulas are R language objects (see :mod:`lme4.rlang`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .rlang import Call, Const, Formula, Sym, NULL, deparse, head, is_call, safe_length, all_names
from .modelframe import (Factor, ModelFrame, EvalContext, as_factor, eval_expr, model_matrix,
                         terms, interaction_factor)

lme4_specials = ["us", "diag", "cs", "ar1"]
_valid_covstruct = ["diag", "us", "cs", "ar1", "ou", "exp", "gau", "mat", "toep", "rr", "homdiag"]


def findReTrmClasses():
    return _valid_covstruct + ["s"]


def makeOp(x, y=None, op=None):
    """reformulas::makeOp"""
    if op is None or y is None:
        opx = y if op is None else op
        return Call(opx, (x,))
    return Call(op, (x, y))


def sumTerms(termList):
    out = termList[0]
    for t in termList[1:]:
        out = Call(Sym("+"), (out, t))
    return out


def RHSForm(form):
    if isinstance(form, Formula):
        return form.rhs
    return form[safe_length(form) - 1]


def anyBars(term):
    nm = all_names(term)
    return "|" in nm or "||" in nm


def isBar(term):
    return is_call(term, ("|", "||"))


def isAnyArgBar(term):
    if not (is_call(term, "~") or is_call(term, "(")):
        if isinstance(term, Call):
            for a in (term.fn,) + term.args:
                if isBar(a):
                    return True
    return False


def nobars_(term):
    if not anyBars(term):
        return term
    if isBar(term):
        return None
    if isAnyArgBar(term):
        return None
    if safe_length(term) == 2:
        nb = nobars_(term[1])
        if nb is None:
            return None
        return term.replace(1, nb)
    nb2 = nobars_(term[1])
    nb3 = nobars_(term[2])
    if nb2 is None:
        return nb3
    if nb3 is None:
        return nb2
    return term.replace(1, nb2).replace(2, nb3)


def nobars(term):
    """remove random-effect terms; formula in, formula out"""
    if isinstance(term, Formula):
        nb = nobars_(term.expr)
        if nb is None or not is_call(nb, "~"):
            # e.g. y ~ (1|g)  ->  y ~ 1
            if term.lhs is not None:
                return Formula(Call(Sym("~"), (term.lhs, Const(1.0))), term.env)
            return Formula(Call(Sym("~"), (Const(1.0),)), term.env)
        if len(nb.args) == 1 and term.lhs is not None and nb.args[0] == term.lhs:
            return Formula(Call(Sym("~"), (term.lhs, Const(1.0))), term.env)
        return Formula(nb, term.env)
    nb = nobars_(term)
    return Const(1.0) if nb is None else nb


def expandDoubleVert(term):
    """reformulas:::expandDoubleVert (used by findbars_x with method='split')"""
    tt = terms(Formula(Call(Sym("~"), (term[1],))))
    from .rlang import parse
    newtrms = [sumTerms([Const(0.0), parse(t)]) for t in tt.term_labels]
    if tt.intercept != 0:
        newtrms = [Const(1.0)] + newtrms
    return [makeOp(makeOp(t, term[2], Sym("|")), Sym("(")) for t in newtrms]


def expandDoubleVerts(term):
    """reformulas::expandDoubleVerts"""
    def edv(term):
        tt = terms(Formula(Call(Sym("~"), (term[1],))))
        from .rlang import parse
        newtrms = ["0+" + t for t in tt.term_labels]
        if tt.intercept != 0:
            newtrms = ["1"] + newtrms
        s = "~(" + ")+(".join(f"{trm}|{deparse(term[2])}" for trm in newtrms) + ")"
        return parse(s).args[0]

    if isinstance(term, Formula):
        return Formula(expandDoubleVerts(term.expr), term.env)
    if not isinstance(term, Sym) and isinstance(term, Call):
        if is_call(term, "("):
            term = term.replace(1, expandDoubleVerts(term[1]))
        if is_call(term, "||"):
            return edv(term)
        term = term.replace(1, expandDoubleVerts(term[1]))
        if safe_length(term) == 3:
            term = term.replace(2, expandDoubleVerts(term[2]))
    return term


def expandGrpVar(f):
    tt = terms(Formula(Call(Sym("~"), (f,))))
    tl = list(tt.term_labels)
    from .rlang import parse

    def inForm(form, value):
        if form == value:
            return True
        if isinstance(form, Call):
            return any(inForm(a, value) for a in (form.fn,) + form.args)
        return False
    if inForm(f, Sym("/")):
        tl = [":".join(reversed(x.split(":"))) for x in tl]
        tl = list(reversed(tl))
    return [parse(x) for x in tl]


def anySpecial(term, specials):
    found = [False]

    def as2(expr):
        if safe_length(expr) == 1 or found[0]:
            return
        for ss in specials:
            if head(expr) == Sym(ss):
                found[0] = True
                return
        for a in expr.args:
            as2(a)
    as2(term)
    return found[0]


def esfun(x):
    if safe_length(x) == 1 or not anySpecial(x, ["|"]):
        return x
    if safe_length(x) == 2:
        inner = esfun(x[1])
        inner = inner if isinstance(inner, list) else [inner]
        return [makeOp(i, x.fn) for i in inner]
    if safe_length(x) == 3:
        if is_call(x, "|"):
            return [makeOp(x[1], g, Sym("|")) for g in expandGrpVar(x[2])]
        return x
    return x


def expandAllGrpVar(bb):
    if not isinstance(bb, list):
        bb = [bb]
    out = []
    for b in bb:
        r = esfun(b)
        if isinstance(r, list):
            out.extend(r)
        else:
            out.append(r)
    return out


def findbars_x(term, specials=(), default_special="us", target="|",
               expand_doublevert_method="diag_special"):
    """reformulas::findbars_x -- returns a list of (possibly special-wrapped) bar terms"""
    if isinstance(term, Formula):
        term = term.rhs
    elif is_call(term, "~") and safe_length(term) == 3:
        term = term[2]
    ds = Sym(default_special) if default_special is not None else None
    specials = list(specials)

    def has_bars(t):
        return is_call(t[1], ("|", "||"))

    def fbx(t):
        if isinstance(t, Sym) or not isinstance(t, Call):
            return []
        fn = t.fn.name if isinstance(t.fn, Sym) else None
        if fn in specials and (fn == "s" or target != "|" or has_bars(t)):
            return [t]
        if fn == target:
            if ds is None:
                return [t]
            return [makeOp(t, ds)]
        if fn == "||":
            if expand_doublevert_method == "diag_special":
                return [makeOp(makeOp(t[1], t[2], Sym("|")), Sym("diag"))]
            out = []
            for x in expandDoubleVert(t):
                out.extend(fbx(x))
            return out
        if fn == "(":
            return fbx(t[1])
        if safe_length(t) == 2:
            return fbx(t[1])
        return fbx(t[1]) + fbx(t[2])
    return expandAllGrpVar(fbx(term))


def findbars(term):
    return findbars_x(term, default_special=None, expand_doublevert_method="split")


def isSpecial(term, specials):
    return isinstance(term, Call) and isinstance(term.fn, Sym) and term.fn.name in specials


def noSpecials_(term, delete=True, specials=None):
    specials = findReTrmClasses() if specials is None else specials
    if not anySpecial(term, specials):
        return term
    if safe_length(term) == 1:
        return term
    if isSpecial(term, specials):
        if delete:
            return None
        return Call(Sym("("), (term[1],))
    nb2 = noSpecials_(term[1], delete, specials)
    nb3 = noSpecials_(term[2], delete, specials) if safe_length(term) == 3 else None
    if nb2 is None:
        return nb3
    if nb3 is None:
        if safe_length(term) == 2 and is_call(term, "~"):
            return term.replace(1, nb2)
        return nb2
    return term.replace(1, nb2).replace(2, nb3)


def noSpecials(term, delete=True, specials=None):
    if isinstance(term, Formula):
        ns = noSpecials_(term.expr, delete, specials)
        if ns is None:
            return Formula(Call(Sym("~"), (Const(1.0),)), term.env)
        if not is_call(ns, "~"):
            return Formula(Call(Sym("~"), (ns, Const(1.0))), term.env)
        return Formula(ns, term.env)
    ns = noSpecials_(term, delete, specials)
    return ns


def sub_specials(term, specials=("|", "||", "s"), keep_args=(2, 2, None)):
    if isinstance(term, Formula):
        return Formula(sub_specials(term.expr, specials, keep_args), term.env)
    if isinstance(term, Sym) or not isinstance(term, Call):
        return term
    for sname, ka in zip(specials, keep_args):
        if is_call(term, sname):
            if ka is None:
                args = tuple(a for a, n in zip(term.args, term.names) if n is None)
            else:
                args = term.args[:ka]
            term = Call(Sym("+"), args)
    args = tuple(sub_specials(a, specials, keep_args) for a in term.args)
    return Call(term.fn, args, term.names)


def subbars(term):
    return sub_specials(term, specials=("|", "||"), keep_args=(2, 2))


def reOnly(f: Formula, response=False, bracket=True, doublevert_split=True,
           specials=(), default_special=None):
    """lme4's my_reOnly / lme4_reOnly"""
    flen = safe_length(f.expr)
    if bracket:
        xdv = "split" if doublevert_split else "diag_special"
        fb = findbars_x(f, expand_doublevert_method=xdv, specials=specials,
                        default_special=default_special)
        fl = [makeOp(x, Sym("(")) for x in fb]
    else:
        fl = [f.rhs]
    s = sumTerms(fl)
    if response and flen == 3:
        return Formula(Call(Sym("~"), (f.lhs, s)), f.env)
    return Formula(Call(Sym("~"), (s,)), f.env)


def lme4_reOnly(form, **kw):
    return reOnly(form, specials=lme4_specials, doublevert_split=True, **kw)


def barnames(bars):
    return [deparse(b[2]) for b in bars]


# ---------------------------------------------------------------------------
# mkBlist / mkReTrms
# ---------------------------------------------------------------------------

def _replace_colon(term):
    """replaceTerm(x, `:`, `%i%`)"""
    if isinstance(term, Call):
        fn = Sym("%i%") if term.fn == Sym(":") else term.fn
        return Call(fn, tuple(_replace_colon(a) for a in term.args), term.names)
    return term


class _FrameLookup(dict):
    """columns of the model frame, falling back to the original data rows"""


def frame_context(fr: ModelFrame, env=None, extra=None):
    data = dict(fr.columns)
    if extra:
        for k, v in extra.items():
            data.setdefault(k, v)
    return EvalContext(data, env or {}, fr.nrow)


def eval_in_frame(expr, fr: ModelFrame, env=None, extra=None):
    """evaluate an expression in the model frame; whole sub-expressions that
    are columns of the frame (e.g. ``log(x)``) are taken from it directly"""
    key = deparse(expr)
    if key in fr.columns:
        return fr.columns[key]
    return eval_expr(expr, frame_context(fr, env, extra))


def factorize(form, fr: ModelFrame, char_only=False):
    """lme4/reformulas factorize(): convert variables to factors"""
    from .rlang import all_vars
    rhs = form.rhs if isinstance(form, Formula) else form
    for v in all_vars(rhs):
        if v in fr.columns:
            cur = fr.columns[v]
            if not isinstance(cur, Factor):
                a = np.asarray(cur)
                if (not char_only) or a.dtype.kind in "OUS":
                    if a.ndim == 1:
                        fr.columns[v] = as_factor(a)
    return fr


def makeFac(x, char_only=False):
    if isinstance(x, Factor):
        return x
    a = np.asarray(x)
    if (not char_only) or a.dtype.kind in "OUS":
        return as_factor(a)
    return x


def mkBlist(x, frloc: ModelFrame, drop_unused_levels=True, env=None, extra=None):
    import copy
    fr2 = ModelFrame(dict(frloc.columns), frloc.terms, frloc.row_names, frloc.attrs)
    # R: factorize(x, frloc) uses all.vars(RHSForm(x)), i.e. the grouping part x[[3]]
    factorize(Formula(Call(Sym("~"), (x[2],))), fr2)
    ff0 = _replace_colon(x[2])
    try:
        ff = makeFac(eval_in_frame(ff0, fr2, env, extra))
    except Exception as e:
        raise ValueError(f"couldn't evaluate grouping factor {deparse(x[2])} within model frame: {e}")
    if np.all(ff.isna()):
        raise ValueError(f"Invalid grouping factor specification, {deparse(x[2])}")
    if drop_unused_levels:
        ff = ff.droplevels()
    nl = ff.nlevels
    # model matrix for ~ x[[2]] evaluated in the frame
    tt = terms(Formula(Call(Sym("~"), (x[1],))))
    cols = {}
    for key, v in zip(tt.varnames, tt.variables):
        if key in fr2.columns:
            cols[key] = fr2.columns[key]
        else:
            cols[key] = eval_in_frame(v, fr2, env, extra)
    mfr = ModelFrame(cols, tt, fr2.row_names)
    mm, cnames, _, _ = model_matrix(tt, mfr)
    n = fr2.nrow
    nc = mm.shape[1]
    # Khatri-Rao product of fac2sparse(ff) and t(mm): rows level*nc + j
    codes = ff.codes
    ok = codes >= 0
    rows = (codes[:, None] * nc + np.arange(nc)[None, :])[ok].ravel()
    colidx = np.repeat(np.arange(n)[ok], nc)
    vals = mm[ok].ravel()
    sm = sp.csc_matrix((vals, (rows, colidx)), shape=(nl * nc, n))
    sm.sum_duplicates()
    sm.sort_indices()
    rownames = [lev for lev in ff.levels for _ in range(nc)]
    return {"ff": ff, "sm": sm, "nl": nl, "cnms": cnames, "rownames": rownames}


def _drop0(m: sp.csc_matrix):
    m = m.tocsc(copy=True)
    m.eliminate_zeros()
    m.sort_indices()
    return m


def mkReTrms(bars, fr: ModelFrame, drop_unused_levels=True, reorder_terms=True,
             calc_lambdat=True, env=None, extra=None):
    """reformulas::mkReTrms"""
    if not bars:
        raise ValueError("No random effects terms specified in formula")
    names = barnames(bars)
    term_names = [deparse(b) for b in bars]
    blist = [mkBlist(b, fr, drop_unused_levels, env, extra) for b in bars]
    nl = np.array([b["nl"] for b in blist])
    ord_ = list(range(len(nl)))
    if reorder_terms and np.any(np.diff(nl) > 0):
        ord_ = list(reversed(np.argsort(nl, kind="stable").tolist()))
        blist = [blist[i] for i in ord_]
        nl = nl[ord_]
        term_names = [term_names[i] for i in ord_]
        names = [names[i] for i in ord_]
    Ztlist = {tn: b["sm"] for tn, b in zip(term_names, blist)}
    Ztlist_l = [b["sm"] for b in blist]
    Zt = sp.vstack(Ztlist_l, format="csc")
    q = Zt.shape[0]
    cnms = [b["cnms"] for b in blist]
    nc = np.array([len(c) for c in cnms])
    nb = nc * nl
    if nb.sum() != q:
        raise ValueError(f"total number of RE ({nb.sum()}) not equal to nrow(Zt) ({q})")
    boff = np.concatenate([[0], np.cumsum(nb)])
    out: Dict[str, Any] = {}
    if calc_lambdat:
        nth = (nc * (nc + 1)) // 2
        thoff = np.concatenate([[0], np.cumsum(nth)])
        I, J, X = [], [], []
        for i in range(len(blist)):
            mm = np.arange(1, nb[i] + 1).reshape(-1, nc[i])   # byrow
            ii, jj = np.tril_indices(nc[i])
            # R: lower.tri row/col in column-major order
            order = np.lexsort((ii, jj))
            ii, jj = ii[order], jj[order]
            for k, (a, b) in enumerate(zip(ii, jj)):
                I.append(mm[:, a] + boff[i])
                J.append(mm[:, b] + boff[i])
                X.append(np.full(nl[i], k + 1 + thoff[i], dtype=float))
        I, J, X = (np.concatenate(v) for v in (I, J, X))
        Lambda = sp.csc_matrix((X, (I - 1, J - 1)), shape=(q, q))
        Lambdat = Lambda.T.tocsc()
        Lambdat.sort_indices()
        Lind = Lambdat.data.astype(np.int32)
        theta = np.zeros(nth.sum())
        lower = np.full(len(theta), -np.inf)
        lower[np.unique(Lambdat.diagonal().astype(int)) - 1] = 0.0
        theta[:] = np.isfinite(lower).astype(float)
        Lambdat.data = theta[Lind - 1]
        out.update(theta=theta, Lind=Lind, lower=lower, Lambdat=Lambdat)
    else:
        out.update(theta=None, Lind=None, lower=None, Lambdat=None)
    out["Zt"] = _drop0(Zt)
    out["Gp"] = np.concatenate([[0], np.cumsum(nb)]).astype(int)
    # factor list
    fl_names, fl = [], []
    for nm_, b in zip(names, blist):
        fl_names.append(nm_)
        fl.append(b["ff"])
    ufn = list(dict.fromkeys(fl_names))
    if len(ufn) < len(fl_names):
        flist = {u: fl[fl_names.index(u)] for u in ufn}
        asgn = [ufn.index(n_) + 1 for n_ in fl_names]
    else:
        flist = dict(zip(fl_names, fl))
        asgn = list(range(1, len(fl) + 1))
    out["flist"] = flist
    out["flist_assign"] = np.array(asgn, dtype=int)
    out["cnms"] = list(zip(names, cnms))      # list of (group name, colnames) pairs
    out["Ztlist"] = Ztlist
    out["nl"] = nl
    out["ord"] = ord_
    out["Zt_rownames"] = [rn for b in blist for rn in b["rownames"]]
    return out
