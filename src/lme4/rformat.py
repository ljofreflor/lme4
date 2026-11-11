"""Number formatting mimicking R's format(), print.default() and
printCoefmat(), so that printed summaries match lme4's output."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


def _sig_decimals(x, digits):
    """number of decimals R's formatReal needs to show x with <= digits sig. digits"""
    if x == 0 or not math.isfinite(x):
        return 0, 0
    ax = abs(x)
    e = math.floor(math.log10(ax))
    # value rounded to `digits` significant digits
    r = float(f"{ax:.{digits - 1}e}")
    e = math.floor(math.log10(r)) if r > 0 else e
    nsig = digits
    for k in range(1, digits + 1):
        if float(f"{ax:.{k - 1}e}") == r:
            nsig = k
            break
    return max(0, nsig - 1 - e), e


def r_format(values, digits=7, nsmall=0, width=None, justify="right"):
    """R's format() for a numeric vector (common number of decimals)"""
    vals = np.asarray(values, float).ravel()
    fin = [v for v in vals if math.isfinite(v)]
    rgt = 0
    for v in fin:
        d, _ = _sig_decimals(v, digits)
        rgt = max(rgt, d)
    rgt = max(rgt, nsmall)
    # R switches to scientific when fixed width exceeds sci width by > scipen(0)
    out = []
    use_sci = False
    if fin:
        mx = max(abs(v) for v in fin)
        mn = min((abs(v) for v in fin if v != 0), default=0)
        fixed_w = len(f"{mx:.{rgt}f}") + (1 if any(v < 0 for v in fin) else 0)
        if mn != 0:
            sig = max(_sig_decimals(v, digits)[0] + _sig_decimals(v, digits)[1] + 1 for v in fin if v != 0)
            sig = max(1, min(digits, sig))
            sci_w = len(f"{mx:.{sig - 1}e}") + (1 if any(v < 0 for v in fin) else 0)
            if fixed_w > sci_w + 0 and (mx >= 1e15 or mn < 1e-4):
                use_sci = True
    for v in vals:
        if v == 0:
            v = 0.0                    # R prints negative zero as 0
        if math.isnan(v):
            out.append("NA")
        elif math.isinf(v):
            out.append("Inf" if v > 0 else "-Inf")
        elif use_sci:
            d = max(1, min(digits, 15))
            s = f"{v:.{max(0, _sig_count(v, digits) - 1)}e}"
            mant, ex = s.split("e")
            out.append(f"{mant}e{int(ex):+03d}")
        else:
            out.append(f"{v:.{rgt}f}")
    w = max((len(s) for s in out), default=0)
    if width:
        w = max(w, width)
    return [s.rjust(w) if justify == "right" else s.ljust(w) for s in out]


def _sig_count(v, digits):
    ax = abs(v)
    r = float(f"{ax:.{digits - 1}e}")
    for k in range(1, digits + 1):
        if float(f"{ax:.{k - 1}e}") == r:
            return k
    return digits


def r_round(x, digits=0):
    """R's round() (IEC 60559: round half to even on the decimal representation)"""
    return np.round(np.asarray(x, float), digits)


def print_named_vector(values: Sequence[str], names: Sequence[str], right=True, gap=1):
    """print.default() for a named character vector (quote=FALSE)"""
    W = max([len(v) for v in values] + [len(n) for n in names] + [0])
    w = [W] * len(values)        # R uses a common column width for named vectors
    sep = " " * gap
    if right:
        hdr = sep.join(n.rjust(k) for n, k in zip(names, w))
        row = sep.join(v.rjust(k) for v, k in zip(values, w))
    else:
        hdr = sep.join(n.ljust(k) for n, k in zip(names, w))
        row = sep.join(v.ljust(k) for v, k in zip(values, w))
    return hdr + " \n" + row + " "


def format_char_matrix(rows: List[List[str]], colnames: Sequence[str], rownames: Sequence[str],
                       right=False):
    """print.default() for a character matrix with quote=FALSE"""
    ncol = len(colnames)
    rw = max([len(r) for r in rownames] + [0])
    cw = [max([len(colnames[j])] + [len(r[j]) for r in rows]) for j in range(ncol)]
    just = (lambda s, k: s.rjust(k)) if right else (lambda s, k: s.ljust(k))
    lines = [" " * rw + "".join(" " + just(colnames[j], cw[j]) for j in range(ncol))]
    for rn, r in zip(rownames, rows):
        lines.append(rn.ljust(rw) + "".join(" " + just(r[j], cw[j]) for j in range(ncol)))
    return "\n".join(lines)


def format_pval(pv, digits, eps=np.finfo(float).eps):
    out = []
    for p in pv:
        if np.isnan(p):
            out.append("NA")
        elif p < eps:
            out.append("<")            # filled in below
        else:
            out.append(None)
    # R: values with exponent >= -3 are formatted together in fixed notation,
    # the others together in scientific notation
    idx = [i for i, o in enumerate(out) if o is None]
    def _expo(v):
        return math.floor(math.log10(v if v > 0 else 1e-50))
    fixp = [i for i in idx if _expo(pv[i]) >= -3 or (_expo(pv[i]) == -4 and digits > 1)]
    scip = [i for i in idx if i not in fixp]
    if fixp:
        for i, s in zip(fixp, r_format([pv[i] for i in fixp], digits=digits)):
            out[i] = s.strip()
    if scip:
        for i, s in zip(scip, _format_sci([pv[i] for i in scip], digits)):
            out[i] = s
    is0 = [i for i, o in enumerate(out) if o == "<"]
    if is0:
        d = max(1, digits - 2)
        rr = [out[i] for i in idx]
        if rr:
            nc = max(len(x) for x in rr)
            if d > 1 and d + 6 > nc:
                d = max(1, nc - 7)
            sep = "" if (d == 1 and nc <= 6) else " "
        else:
            sep = "" if d == 1 else " "
        for i in is0:
            out[i] = "<" + sep + _fmt_g(eps, d)
    return out


def _format_sci(vals, digits):
    """R's format() in scientific notation with a common number of significant digits"""
    nsig = max(_sig_count(v, digits) for v in vals)
    out = []
    for v in vals:
        s = f"{v:.{nsig - 1}e}"
        mant, ex = s.split("e")
        out.append(f"{mant}e{int(ex):+03d}")
    return out


def _fmt_g(x, digits):
    s = f"{x:.{digits - 1}e}"
    mant, ex = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(ex):+03d}"


def signif_stars(p):
    if np.isnan(p):
        return ""
    for cut, s in ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, "."), (1.0, " ")):
        if p <= cut:
            return s
    return " "


def printCoefmat(coefs, colnames, rownames, digits=4, signif_stars_=True, has_p=None):
    """stats::printCoefmat for an Estimate / Std. Error / t|z value [/ p] table"""
    coefs = np.asarray(coefs, float)
    nr, nc = coefs.shape
    has_p = (nc == 4) if has_p is None else has_p
    cols = [[""] * nr for _ in range(nc)]
    cs = coefs[:, :2]
    acs = np.abs(cs)
    ok = np.isfinite(acs) & (acs != 0)
    if ok.any():
        digmin = 1 + math.floor(math.log10(acs[ok].min()))
        dd = max(1, digits - digmin)
        f = r_format(np.round(cs, dd).T.ravel(), digits=digits)
        for j in range(2):
            for i in range(nr):
                cols[j][i] = f[j * nr + i]
    dig_tst = max(1, min(5, digits - 1))
    tv = coefs[:, 2]
    ft = r_format(np.round(tv, dig_tst), digits=digits)
    cols[2] = ft
    if has_p:
        pv = coefs[:, 3]
        fp = format_pval(pv, dig_tst)
        w = max(len(s) for s in fp)
        cols[3] = [s.rjust(w) for s in fp]
    rows = [[cols[j][i] for j in range(nc)] for i in range(nr)]
    show_stars = has_p and signif_stars_ and np.any(coefs[:, 3] < 0.1)
    if show_stars:
        stars = [signif_stars(coefs[i, 3]) for i in range(nr)]
        w = max(len(s) for s in stars)
        for i in range(nr):
            rows[i].append(stars[i].ljust(w))      # format() of a character vector: left-justified
        colnames = list(colnames) + [""]
    txt = format_char_matrix(rows, colnames, rownames, right=True)
    if show_stars:
        txt += ("\n---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ "
                "0.05 ‘.’ 0.1 ‘ ’ 1")
    return txt


def abbreviate(s, minlength=4):
    """R's abbreviate() (simplified: drops lower-case vowels from the end, then
    lower-case letters, until the string is at most minlength characters)"""
    s = s.lstrip(" ")
    if len(s) <= minlength:
        return s
    # characters starting a word (first char, or preceded by a space) are protected
    chars = [(c, i == 0 or s[i - 1] == " ") for i, c in enumerate(s)]
    # 1. drop the other spaces
    if len(chars) > minlength:
        chars = [x for x in chars if x[0] != " "]
    # 2. lower-case vowels, then 3. lower-case letters, from the right
    for test in (lambda c: c in "aeiou", lambda c: c.islower(), lambda c: True):
        i = len(chars) - 1
        while len(chars) > minlength and i > 0:
            c, first = chars[i]
            if test(c) and not first:
                del chars[i]
            i -= 1
    return "".join(c for c, _ in chars)


def abbreviate_vec(names, minlength=4):
    """R's abbreviate() on a vector (method "left.kept"): every member of a
    group of duplicated abbreviations is re-abbreviated with minlength + 1"""
    names = list(names)
    x = list(names)
    dup2 = [True] * len(names)
    ml = minlength
    while True:
        for i, d in enumerate(dup2):
            if d:
                x[i] = abbreviate(names[i], ml)
        seen, dups = set(), set()
        for v in x:
            if v in seen:
                dups.add(v)
            seen.add(v)
        if not dups or ml > max(len(n) for n in names):
            return x
        ml += 1
        dup2 = [v in dups for v in x]
