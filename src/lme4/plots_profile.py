"""Plots of mixed-model profiles (R class 'thpr'): matplotlib transcriptions
of ``xyplot.thpr``, ``densityplot.thpr`` and ``splom.thpr`` together with
their helpers ``dens()``, ``cont()``, ``ad()``, ``tauij()``, ``sacos()`` and
``stripExpr()`` (lme4/R/profile.R).

lattice is replaced by matplotlib; the *data* drawn in each panel are
computed exactly as in R (the helpers that compute them are exported so that
they can be checked against R), while the visual details (fonts, padding,
panel order) follow matplotlib conventions.  Panels are drawn left to right,
top to bottom (lattice's ``as.table = TRUE``).

``splines::periodicSpline`` (order 4, knots at the data, period 2*pi) is the
unique C2 periodic cubic interpolant, i.e. ``scipy``'s ``CubicSpline`` with
``bc_type="periodic"``.
"""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline

from .profile import thpr, polySpline, interpSpline, predy, get_which

__all__ = ["xyplot_thpr", "xyplot_thpr_data", "densityplot_thpr", "dens", "splom_thpr",
           "splom_traces", "cont", "stripExpr"]

_VARPAR = re.compile(r"^(\.sig[0-9]+|.sigma|sd_|cor_)")


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _par_names(x):
    """levels(x[[".par"]]) (the profiled parameters, in profile order)"""
    return list(x.forward.keys())


def _is_spline(sp):
    return isinstance(sp, polySpline)


# ---------------------------------------------------------------------------
# strip labels (stripExpr)
# ---------------------------------------------------------------------------

def stripExpr(nms):
    """panel labels: '.sigma' -> sigma, '.sig01' -> sigma_1, '.lsig01' ->
    log(sigma_1), '.sigsq01' -> sigma_1^2 (as mathtext)"""
    out = []
    for nm in nms:
        if nm == ".sigma":
            out.append(r"$\sigma$")
        elif nm == ".lsigma":
            out.append(r"$\log(\sigma)$")
        elif nm == ".sigmasq":
            out.append(r"$\sigma^2$")
        elif re.match(r"^\.sig[0-9]+", nm):
            out.append(r"$\sigma_{%d}$" % int(nm[4:]))
        elif re.match(r"^\.lsig[0-9]+", nm):
            out.append(r"$\log(\sigma_{%d})$" % int(nm[5:]))
        elif re.match(r"^\.sigsq[0-9]+", nm):
            out.append(r"${\sigma_{%d}}^2$" % int(nm[6:]))
        else:
            out.append(nm)
    return out


def _layout(n, layout):
    if layout is not None:
        ncol, nrow = layout
        return int(nrow), int(ncol)
    ncol = n if n <= 3 else int(math.ceil(math.sqrt(n)))
    return int(math.ceil(n / ncol)), ncol


def _extend(lim, prop=0.07):
    lo, hi = float(np.nanmin(lim)), float(np.nanmax(lim))
    d = hi - lo
    if d == 0:
        d = abs(lo) if lo != 0 else 1.0
    return lo - prop * d, hi + prop * d


def _panels(n, layout, figsize, sharey):
    plt = _plt()
    nrow, ncol = _layout(n, layout)
    if figsize is None:
        figsize = (3.2 * ncol, 3.0 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, sharey=sharey)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# ---------------------------------------------------------------------------
# xyplot.thpr: profile zeta plots
# ---------------------------------------------------------------------------

def _levels_conf(levels, conf, df):
    conf = np.atleast_1d(np.asarray(conf, float))
    if np.any(~np.isfinite(conf) | (conf <= 0) | (conf >= 1)):
        raise ValueError("values of 'conf' must be strictly between 0 and 1")
    if levels is None:
        levels = np.sqrt(stats.chi2.ppf(np.clip(conf, 0, 1), df=df))
    levels = np.atleast_1d(np.asarray(levels, float))
    return np.sort(levels[np.isfinite(levels) & (levels > 0)])


def _which_names(x, which):
    nms = _par_names(x)
    nptot = len(nms)
    if nptot < 1:
        raise ValueError("no profiled parameters")
    nvp = sum(1 for n in nms if _VARPAR.search(n))
    w = get_which(which, nvp, nptot, nms)          # 1-based
    return [nms[k - 1] for k in w]


def xyplot_thpr_data(x, levels=None, conf=(0.50, 0.80, 0.90, 0.95, 0.99), absVal=False,
                     which=None):
    """The data of ``xyplot.thpr``: returns ``(fr, spl)`` where ``fr`` has
    columns 'zeta', 'pval' (the parameter value at which the profile reaches
    each zeta = +/- sqrt(qchisq(conf, 1))) and 'pnm', and ``spl`` maps each
    parameter to its forward spline (or, for a failed spline, to the raw
    profile data as a DataFrame with columns x, y)."""
    lev = _levels_conf(levels, conf, 1)
    names = _which_names(x, which)
    spl, bspl = {}, {}
    for n in names:
        s = x.forward.get(n)
        if _is_spline(s):
            spl[n], bspl[n] = s, x.backward.get(n)
        else:  # replace by the raw profile data
            r = x.frame.loc[x.frame[".par"] == n]
            d = pd.DataFrame({"y": r[".zeta"].values.astype(float),
                              "x": r[n].values.astype(float)})
            spl[n] = d
            bspl[n] = pd.DataFrame({"x": d["y"].values, "y": d["x"].values})
    zeta = np.concatenate([-lev[::-1], [0.0], lev])

    def mypred(bs, z):  # linear approximation if the backspline does not work
        if _is_spline(bs):
            return predy(bs, z)
        if isinstance(bs, pd.DataFrame):
            xx, yy = bs["x"].values, bs["y"].values
            o = np.argsort(xx, kind="stable")
            xx, yy = xx[o], yy[o]
            out = np.interp(z, xx, yy)
            out[(z < xx[0]) | (z > xx[-1])] = np.nan       # approx(): rule = 1
            return out
        return np.full(len(z), np.nan)

    fr = pd.DataFrame({"zeta": np.tile(zeta, len(spl)),
                       "pval": np.concatenate([mypred(bspl[n], zeta) for n in spl]),
                       "pnm": pd.Categorical(np.repeat(list(spl), len(zeta)),
                                             categories=list(spl))})
    for i in np.flatnonzero(fr["pval"].isna().values):
        fr.loc[i, "zeta"] = 0.0
        cur = spl[fr["pnm"].iloc[i]]
        if _is_spline(cur):
            fr.loc[i, "pval"] = float(np.min(cur.knots))
    if absVal:
        fr["zeta"] = np.abs(fr["zeta"].values)
    return fr, spl


def xyplot_thpr(x, levels=None, conf=(0.50, 0.80, 0.90, 0.95, 0.99), absVal=False,
                which=None, layout=None, figsize=None, main=None, sharey=True):
    """Profile zeta plots (``xyplot(<thpr>)``): the signed square root of the
    likelihood-ratio statistic, zeta, as a function of each parameter.  The
    vertical segments mark the parameter values at which |zeta| reaches the
    ``conf`` confidence levels (``sqrt(qchisq(conf, 1))``); with
    ``absVal=True`` |zeta| is plotted.  Returns the matplotlib Figure."""
    fr, spl = xyplot_thpr_data(x, levels=levels, conf=conf, absVal=absVal, which=which)
    names = list(spl)
    fig, axes = _panels(len(names), layout, figsize, sharey)
    labels = stripExpr(names)
    ylim = _extend(fr["zeta"].values)
    for k, (ax, n) in enumerate(zip(axes, names)):
        d = fr.loc[fr["pnm"] == n]
        xv, yv = d["pval"].values, d["zeta"].values.copy()
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        ax.vlines(xv, np.minimum(yv, 0), np.maximum(yv, 0), color="C0", lw=1)
        if absVal:
            yv[yv == 0] = np.nan
            ax.hlines(yv, np.minimum(xv, xv[::-1]), np.maximum(xv, xv[::-1]), color="C0", lw=1)
        else:
            ax.axhline(0, color="C0", lw=1)
        s = spl[n]
        if not _is_spline(s):
            yy = np.abs(s["y"].values) if absVal else s["y"].values
            ax.plot(s["x"].values, yy, color="C0")
            ax.plot(s["x"].values, yy, "+", color="C0")
            warnings.warn(f"bad profile for variable {k + 1}: using linear interpolation")
            ax.set_xlim(_extend(np.concatenate([xv, s["x"].values])))
        else:
            lims = _extend(xv)
            kn = s.knots
            g = np.linspace(max(lims[0], kn.min()), min(lims[1], kn.max()), 101)
            yy = s.predict(g)
            ax.plot(g, np.abs(yy) if absVal else yy, color="C0")
            ax.set_xlim(lims)
        ax.set_ylim(ylim)
        ax.set_title(labels[k], fontsize=11, backgroundcolor="#ffe5cc")
        if k % axes[0].get_subplotspec().get_gridspec().ncols == 0:
            ax.set_ylabel(r"$|\zeta|$" if absVal else r"$\zeta$")
    if main:
        fig.suptitle(main)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# densityplot.thpr
# ---------------------------------------------------------------------------

def dens(pr, npts=201, upper=0.999):
    """Approximate densities from a profile (``lme4:::dens``): for each
    parameter p, the density of p implied by treating zeta(p) as standard
    normal, dnorm(zeta(p)) * zeta'(p), on an ``npts`` grid spanning the
    ``(1-upper, upper)`` quantiles.  Returns a DataFrame (pval, pnm, density)."""
    npts = int(npts)
    if not isinstance(pr, thpr) or npts <= 0 or not (0.5 < upper < 1):
        raise ValueError("invalid arguments to dens()")
    spl, bspl = pr.forward, pr.backward
    vNms = list(spl)
    if vNms != list(bspl):
        raise ValueError("forward and backward splines do not match")
    bad_vars = []
    zeta = np.array([-1.0, 1.0]) * stats.norm.ppf(upper)
    rng = {}
    for nm in vNms:
        b = bspl[nm]
        if not _is_spline(b):
            rng[nm] = np.full(npts, np.nan)
            bad_vars.append(nm)
            continue
        rng0 = predy(b, zeta)
        if np.isnan(rng0[0]):
            rng0[0] = 0.0
        if np.isnan(rng0[1]):
            # try harder to pick an upper bound
            for upp in 1 - 10 ** np.linspace(-4, -1, 21):
                rng0[1] = predy(b, stats.norm.ppf(upp))[0]
                if not np.isnan(rng0[1]):
                    break
            if np.isnan(rng0[1]):
                warnings.warn("can't find an upper bound for the profile")
                bad_vars.append(nm)
                rng[nm] = None     # R: 'next' leaves the list element NULL
                continue
        rng[nm] = np.linspace(rng0[0], rng0[1], npts)
    keep = [nm for nm in vNms if rng[nm] is not None]
    pval = np.concatenate([rng[nm] for nm in keep]) if keep else np.zeros(0)
    dd = []
    for nm in keep:
        s = spl[nm]
        if not _is_spline(s):
            dd.append(np.full(npts, np.nan))
        else:
            zz = predy(s, rng[nm])
            dd.append(stats.norm.pdf(zz) * s.predict(rng[nm], deriv=1))
    fr = pd.DataFrame({"pval": pval,
                       "pnm": pd.Categorical(np.repeat(keep, npts), categories=vNms),
                       "density": np.concatenate(dd) if dd else np.zeros(0)})
    if bad_vars:
        warnings.warn("unreliable profiles for some variables skipped: " + ", ".join(bad_vars))
    return fr


def densityplot_thpr(x, npts=201, upper=0.999, layout=None, figsize=None, main=None):
    """``densityplot(<thpr>)``: the densities from :func:`dens`, one panel per
    parameter with free x and y scales.  Returns the matplotlib Figure."""
    dd = dens(x, npts=npts, upper=upper)
    names = list(x.forward)
    fig, axes = _panels(len(names), layout, figsize, sharey=False)
    labels = stripExpr(names)
    for k, (ax, n) in enumerate(zip(axes, names)):
        d = dd.loc[dd["pnm"] == n]
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        ax.plot(d["pval"].values, d["density"].values, color="C0")
        ax.set_title(labels[k], fontsize=11, backgroundcolor="#ffe5cc")
        if k % axes[0].get_subplotspec().get_gridspec().ncols == 0:
            ax.set_ylabel("density")
    if main:
        fig.suptitle(main)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# splom.thpr: profile pairs plots
# ---------------------------------------------------------------------------

def ad(xc, yc):
    """convert x-cosine and y-cosine to average and (positive) difference"""
    xc, yc = np.broadcast_arrays(np.asarray(xc, float), np.asarray(yc, float))
    a = (xc + yc) / 2
    d = xc - yc
    return np.column_stack([np.sign(d) * a, np.abs(d)])


def tauij(xy, lev):
    """convert d versus a (x = a, y = d) and level to a matrix of tau_i, tau_j"""
    x, y = xy
    return lev * np.cos(x[:, None] + np.outer(y / 2, [-1.0, 1.0]))


def sacos(x):
    """acos(x), careful of boundary conditions"""
    return np.arccos(np.maximum(-0.999, np.minimum(0.999, np.asarray(x, float))))


def _periodic_predict(x, y, nseg, period=2 * np.pi):
    """predict(periodicSpline(x, y, period = period), nseg = nseg)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    if len(np.unique(x)) != len(x) or np.any(np.diff(x) <= 0):
        raise ValueError("values of 'x' must be distinct")
    if x[-1] - x[0] >= period:
        raise ValueError("the range of 'x' values exceeds one period")
    cs = CubicSpline(np.append(x, x[0] + period), np.append(y, y[0]), bc_type="periodic")
    xx = np.linspace(x[0], x[0] + period, nseg + 1)
    return xx, cs(xx)


def _predict_grid(sp, nseg=50):
    """predict(<polySpline>) without x: nseg+1 points spanning the knots"""
    g = np.linspace(sp.knots[0], sp.knots[-1], nseg + 1)
    return g, sp.predict(g)


def cont(sij, sji, levels, nseg=101):
    """Contours of the 2-d profile (``lme4:::cont``).  Returns a dict with
    'tki', 'tkj' (the tau-scale predictions of i on j and of j on i at
    ``c(-rev(levels), 0, levels)``, each as (x, y)) and 'pts', an array of
    shape (len(levels), nseg + 1, 2) with the points on the contours."""
    levels = np.asarray(levels, float)
    ada = np.zeros((len(levels), 2, 4))
    ada[:, :, 0] = ad(0, sacos(predy(sij, levels) / levels))
    ada[:, :, 1] = ad(sacos(predy(sji, levels) / levels), 0)
    ada[:, :, 2] = ad(np.pi, sacos(predy(sij, -levels) / levels))
    ada[:, :, 3] = ad(sacos(predy(sji, -levels) / levels), np.pi)
    pts = np.zeros((len(levels), nseg + 1, 2))
    for i in range(len(levels)):
        pts[i] = tauij(_periodic_predict(ada[i, 0, :], ada[i, 1, :], nseg), levels[i])
    levs = np.concatenate([-levels[::-1], [0.0], levels])
    return {"tki": (levs, sij.predict(levs)), "tkj": (levs, sji.predict(levs)), "pts": pts}


def splom_traces(x, levels=None, conf=(0.50, 0.80, 0.90, 0.95, 0.99), which=None):
    """The computations of ``splom.thpr``: returns a dict with 'names',
    'levels', 'mlev', 'spl', 'bsp', 'pfr' (DataFrame of the parameter ranges
    at zeta = -/+ max(levels)) and 'traces' ({(j, i): {'sij', 'sji', 'll'}}
    for 0-based j > i)."""
    lev = _levels_conf(levels, conf, 2)
    fr0 = x.frame.dropna()
    # singular-fit check (getSingTol() = 1e-4)
    tol = 1e-4
    zero_zeta = fr0[".zeta"].values == 0
    singfit = any(np.any(zero_zeta & (np.abs(fr0[c].values.astype(float)) <= tol))
                  for c in fr0.columns if re.match(r"^(\.sig[0-9]|sd_)", c))
    if singfit:
        warnings.warn("splom is unreliable for singular fits")
    names = _which_names(x, which)
    ok = [_is_spline(x.forward.get(n)) and _is_spline(x.backward.get(n)) for n in names]
    if not all(ok):
        warnings.warn("Dropping parameters from splom due to failed spline interpolation: "
                      + ", ".join(n for n, o in zip(names, ok) if not o))
        names = [n for n, o in zip(names, ok) if o]
    if len(names) == 1:
        raise ValueError("can't draw a scatterplot matrix for a single variable")
    mlev = float(np.max(lev))
    spl = {n: x.forward[n] for n in names}
    bsp = {n: x.backward[n] for n in names}
    frange = np.array([[np.min(spl[n].knots), np.max(spl[n].knots)] for n in names]).T
    pfr = np.column_stack([predy(bsp[n], [-mlev, mlev]) for n in names])
    pfr[0] = np.where(np.isnan(pfr[0]), frange[0], np.fmax(pfr[0], frange[0]))
    pfr[1] = np.where(np.isnan(pfr[1]), frange[1], np.fmin(pfr[1], frange[1]))
    pfr = pd.DataFrame(pfr, columns=names)
    sub = fr0.loc[fr0[".par"].isin(names), names + [".par"]].copy()
    for n in names:
        sub[n] = predy(spl[n], sub[n].values.astype(float))
    traces = {}
    for j in range(1, len(names)):
        frj = sub.loc[sub[".par"] == names[j]]
        for i in range(j):
            fri = sub.loc[sub[".par"] == names[i]]
            sij = interpSpline(fri[names[i]].values, fri[names[j]].values)
            sji = interpSpline(frj[names[j]].values, frj[names[i]].values)
            traces[(j, i)] = {"sij": sij, "sji": sji, "ll": cont(sij, sji, lev)}
    return {"names": names, "levels": lev, "mlev": mlev, "spl": spl, "bsp": bsp,
            "pfr": pfr, "traces": traces}


def splom_thpr(x, levels=None, conf=(0.50, 0.80, 0.90, 0.95, 0.99), which=None,
               draw_lower=True, draw_upper=True, figsize=None, main=None):
    """Profile pairs plot (``splom(<thpr>)``).  Contours are for the marginal
    two-dimensional regions (``sqrt(qchisq(conf, 2))``).  As in lattice's
    splom, variable k sits on the diagonal at column k from the left and row
    k from the bottom.  Panels below the diagonal are on the zeta scale,
    panels above it on the original scale; the thin lines are the profile
    traces (conditional estimates of one parameter given the other).
    Returns the matplotlib Figure."""
    plt = _plt()
    tr = splom_traces(x, levels=levels, conf=conf, which=which)
    names, lev, mlev, bsp, pfr = tr["names"], tr["levels"], tr["mlev"], tr["bsp"], tr["pfr"]
    n = len(names)
    if figsize is None:
        figsize = (2.3 * n, 2.3 * n)
    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    zl = (-1.07 * mlev, 1.07 * mlev)
    lims = {nm: _extend(pfr[nm].values) for nm in names}
    labels = stripExpr(names)
    ckw = dict(color="C0", lw=0.9)

    def ax_at(col, row_from_bottom):
        return axes[n - 1 - row_from_bottom, col]

    for c in range(n):
        for r in range(n):
            ax = ax_at(c, r)
            ax.tick_params(labelsize=7)
            if c == r:   # diagonal: variable name
                ax.text(0.5, 0.5, labels[c], ha="center", va="center", fontsize=12,
                        transform=ax.transAxes)
                ax.set_xlim(lims[names[c]])
                ax.set_ylim(lims[names[c]])
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.grid(True, color="0.9", lw=0.8)
            ax.set_axisbelow(True)
            if c > r:    # lower panel (zeta scale): horizontal = var c, vertical = var r
                j, i = c, r
                if not draw_lower:
                    ax.set_visible(False)
                    continue
                t = tr["traces"][(j, i)]
                ll = t["ll"]
                ax.set_xlim(zl)
                ax.set_ylim(zl)
                dd = np.diff(zl)[0] / 50
                gx, gy = _predict_grid(t["sij"])
                ax.plot(gy, gx, **ckw)
                hx, hy = _predict_grid(t["sji"])
                ax.plot(hx, hy, **ckw)
                tki, tkj = ll["tki"], ll["tkj"]
                ax.hlines(tki[0], tki[1] - dd, tki[1] + dd, **ckw)
                ax.vlines(tkj[0], tkj[1] - dd, tkj[1] + dd, **ckw)
                for k in range(len(lev)):
                    ax.plot(ll["pts"][k, :, 0], ll["pts"][k, :, 1], **ckw)
            else:        # upper panel (original scale): horizontal = var c, vertical = var r
                ii, jj = c, r
                if not draw_upper:
                    ax.set_visible(False)
                    continue
                t = tr["traces"][(jj, ii)]
                pts = t["ll"]["pts"]
                gx, gy = _predict_grid(t["sij"])
                hx, hy = _predict_grid(t["sji"])
                bi, bj = bsp[names[ii]], bsp[names[jj]]
                ax.plot(predy(bi, gx), predy(bj, gy), **ckw)
                ax.plot(predy(bi, hy), predy(bj, hx), **ckw)
                for k in range(len(lev)):
                    ax.plot(predy(bi, pts[k, :, 1]), predy(bj, pts[k, :, 0]), **ckw)
                ax.set_xlim(lims[names[ii]])
                ax.set_ylim(lims[names[jj]])
            if r != 0:
                ax.set_xticklabels([])
            if c != 0:
                ax.set_yticklabels([])
    fig.text(0.5, 0.005, "Scatter Plot Matrix", ha="center", fontsize=10)
    if main:
        fig.suptitle(main)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig
