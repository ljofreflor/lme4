"""Diagnostic plots for fitted mixed models (transcription of lme4's R/plot.R and R/plots.R).

R's lme4 draws these with lattice; here they are drawn with matplotlib, which is an
optional dependency imported only when a plot is requested.  The *data* shown (which points,
in which order, with which intervals and axis limits) follow the R code line by line:

* :func:`plot_merMod`   -- ``plot.merMod``: residuals vs fitted values (default), or any
  ``y ~ x | g`` formula (scatter plot, box plots, histogram)
* :func:`qqmath_merMod` -- ``qqmath.merMod``: normal QQ plot of the standardized residuals
* :func:`dotplot_ranef` -- ``dotplot.ranef.mer``: caterpillar plot of the conditional modes
  with conditional-variance intervals
* :func:`qqmath_ranef`  -- ``qqmath.ranef.mer``: normal QQ plot of the conditional modes

Every function returns matplotlib Figure(s) (one per grouping factor for the ranef plots,
as R returns a list of trellis objects).
"""
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["plot_merMod", "qqmath_merMod", "dotplot_ranef", "qqmath_ranef", "ppoints"]

# lattice's default theme colours (trellis.par.get())
_SYMBOL_COL = "#0080ff"       # plot.symbol / dot.symbol
_DOT_LINE_COL = "#e6e6e6"     # dot.line
_REF_LINE_COL = "#e6e6e6"     # reference.line (panel.grid)
_AXIS_PADDING = 0.07          # lattice.getOption("axis.padding")$numeric


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib:  pip install matplotlib") from e
    return plt


def ppoints(n, a=None):
    """R's stats::ppoints"""
    if a is None:
        a = 3 / 8 if n <= 10 else 1 / 2
    if n <= 0:
        return np.zeros(0)
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def _extend_limits(lim, prop=_AXIS_PADDING):
    """lattice:::extend.limits"""
    lo, hi = float(lim[0]), float(lim[1])
    if lo == hi:
        return np.array([lo - 1, hi + 1]) if lo == 0 else np.array([lo - abs(lo) * 0.5, hi + abs(hi) * 0.5])
    d = hi - lo
    return np.array([lo - prop * d, hi + prop * d])


def _grid_shape(n):
    """a lattice-like default layout (columns, rows) for n panels"""
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)
    return nr, nc


# ---------------------------------------------------------------------------
# helpers for plot.merMod (R/plot.R)
# ---------------------------------------------------------------------------

def getData(model):
    """original data of the fit (getData.merMod), restricted to the rows used in the fit"""
    data = model.call.get("_data") if isinstance(model.call, dict) else None
    if data is None:
        return model.frame.to_data_frame() if hasattr(model.frame, "to_data_frame") else None
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    rn = model.frame.row_names
    if len(data) != len(rn):           # subset / NA rows: keep the rows of the model frame
        data = data.loc[rn]
    return data


def getIDLabels(model):
    """getIDLabels(): interaction of the grouping factors, one label per observation"""
    labs = None
    for f in model.flist.values():
        s = np.asarray(f.as_strings(), dtype=object)
        labs = s if labs is None else np.char.add(np.char.add(labs.astype(str), "."), s.astype(str))
    return np.asarray(labs, dtype=str)


_DOT = re.compile(r"(?<![\w.])\.(?![\w.])")


def _r_to_py(expr):
    e = _DOT.sub("_obj_", expr.strip())
    e = re.sub(r"\bTRUE\b", "True", e)
    e = re.sub(r"\bFALSE\b", "False", e)
    return e


def _split_formula(form):
    """split 'y ~ x | g' into (y or None, x, g or None) strings"""
    if "~" not in form:
        raise ValueError('"form" must be a formula')
    lhs, rhs = form.split("~", 1)
    lhs = lhs.strip() or None
    grp = None
    depth = 0
    for i, ch in enumerate(rhs):           # top-level '|'
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            rhs, grp = rhs[:i], rhs[i + 1:].strip()
            break
    return lhs, rhs.strip(), grp


def _eval_env(model, data):
    def resid(obj, type=None, scaled=False):
        return np.asarray(obj.residuals(type=type, scaled=scaled), float)

    def fitted(obj):
        return np.asarray(obj.fitted(), float)

    env = {"resid": resid, "residuals": resid, "fitted": fitted, "_obj_": model,
           "np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs}
    if data is not None:
        for c in data.columns:
            env[str(c)] = data[c].to_numpy() if not isinstance(data[c].dtype, pd.CategoricalDtype) else data[c]
        env[".obs"] = np.arange(1, len(data) + 1)
    return env


def _eval(expr, env):
    v = eval(_r_to_py(expr), {"__builtins__": {}}, env)  # noqa: S307 -- user-supplied formula
    if isinstance(v, pd.Series):
        v = v if isinstance(v.dtype, pd.CategoricalDtype) else v.to_numpy()
    return v


def _as_factor(v):
    if isinstance(v, pd.Series) and isinstance(v.dtype, pd.CategoricalDtype):
        return v.astype(str).to_numpy(), [str(l) for l in v.cat.categories]
    if isinstance(v, pd.Categorical):
        return np.asarray(v.astype(str)), [str(l) for l in v.categories]
    v = np.asarray(v)
    levs = sorted(set(v.tolist()))
    return v.astype(str), [str(l) for l in levs]


def plot_merMod(x, form=None, abline="missing", id=None, idLabels=None, grid=None,
                xlab=None, ylab=None, ax=None, **kw):
    """Diagnostic plot of a fitted model (R's ``plot.merMod``).

    ``form`` is an R-style formula string evaluated with the original data and the model
    available as ``.``, e.g. ``"resid(., scaled=TRUE) ~ fitted(.) | Sex"``,
    ``"Subject ~ resid(., scaled=TRUE)"`` (box plots), ``"~ resid(.)"`` (histogram).
    The default is ``resid(., type = "pearson") ~ fitted(.)`` with a horizontal line at 0.
    ``id``: a significance level (points with absolute standardized residuals beyond the
    normal quantile are labelled with ``idLabels``, by default the grouping factor levels).
    Extra keyword arguments are passed to the matplotlib drawing call.
    Returns the matplotlib Figure.
    """
    plt = _plt()
    object = x
    form_missing = form is None
    if form is None:
        form = 'resid(., type = "pearson") ~ fitted(.)'
    lhs, rhs, grp = _split_formula(form)
    data = getData(object)
    env = _eval_env(object, data)

    # covariate -- must always be present
    xv = _eval(rhs, env)
    xv = np.asarray(xv, float) if not isinstance(xv, (pd.Series, pd.Categorical)) else xv
    if not np.issubdtype(np.asarray(xv).dtype, np.number):
        raise ValueError("Covariate must be numeric")
    xv = np.asarray(xv, float)
    if xlab is None:
        xlab = rhs
    # response -- need not be present
    yv = None
    if lhs is not None:
        yv = _eval(lhs, env)
        if ylab is None:
            ylab = lhs
    # groups -- need not be present
    gv, glevs = None, None
    if grp is not None:
        parts = [p.strip() for p in re.split(r"[*+]", grp)]
        gvals = [_as_factor(_eval(p, env)) for p in parts]
        if len(gvals) == 1:
            gv, glevs = gvals[0]
        else:
            gv = gvals[0][0]
            for v, _ in gvals[1:]:
                gv = np.char.add(np.char.add(gv, " * "), v)
            glevs = [l for l in pd.unique(gv)]

    if id is not None:                # identify points in plot
        if isinstance(id, (int, float)) and not isinstance(id, bool):
            if id <= 0 or id >= 1:
                raise ValueError("'id' must be between 0 and 1")
            ids = np.abs(np.asarray(object.residuals(type="pearson"))) / object.sigma() > -norm.ppf(id / 2)
        elif isinstance(id, str):
            ids = np.asarray(_eval(id.lstrip("~ "), env), bool)
        else:
            ids = np.asarray(id, bool)
        if idLabels is None:
            idLabels = getIDLabels(object)
        elif isinstance(idLabels, str):
            idLabels = np.asarray(_eval(idLabels.lstrip("~ "), env)).astype(str)
        else:
            idLabels = np.asarray(idLabels).astype(str)
            if len(idLabels) != len(ids):
                raise ValueError('"idLabels" of incorrect length')
    else:
        ids = None

    # defining abline, if needed
    if isinstance(abline, str) and abline == "missing":
        abline = (0, 0) if form_missing else None
    if abline is not None:
        abline = np.atleast_1d(np.asarray(abline, float))

    # defining the type of plot
    if yv is not None:
        is_num = not isinstance(yv, (pd.Series, pd.Categorical)) and np.issubdtype(np.asarray(yv).dtype, np.number)
        plotFun = "xyplot" if is_num else "bwplot"
    else:
        plotFun = "histogram"
    if grid is None:
        grid = plotFun == "xyplot"

    panels = [None] if gv is None else glevs
    if ax is None:
        nr, nc = _grid_shape(len(panels))
        fig, axes = plt.subplots(nr, nc, squeeze=False, sharex=True, sharey=True,
                                 figsize=kw.pop("figsize", (6, 4.5) if len(panels) == 1 else
                                                (min(4 * nc, 12), min(3.2 * nr, 10))))
        axes = axes.ravel()
        for a in axes[len(panels):]:
            a.set_visible(False)
    else:
        fig, axes = ax.figure, [ax]
        if len(panels) > 1:
            raise ValueError("'ax' can only be given for a single-panel plot")

    for k, lev in enumerate(panels):
        a = axes[k]
        sub = np.ones(len(xv), bool) if lev is None else (gv == lev)
        if lev is not None:
            a.set_title(str(lev), fontsize="medium")
        if grid:
            a.grid(True, color=_REF_LINE_COL, zorder=0)
        if plotFun == "xyplot":
            yy = np.asarray(yv, float)
            a.scatter(xv[sub], yy[sub], facecolors="none", edgecolors=kw.pop("color", _SYMBOL_COL),
                      zorder=2, **kw)
            if ids is not None and np.any(ids[sub]):
                for xi, yi, li in zip(xv[sub][ids[sub]], yy[sub][ids[sub]], idLabels[sub][ids[sub]]):
                    a.text(xi, yi, li, ha="center", va="center", fontsize="small")
            if abline is not None:
                if len(abline) == 2:
                    lims = a.get_xlim(), a.get_ylim()      # panel.abline does not rescale
                    a.axline((0, abline[0]), slope=abline[1], color=_SYMBOL_COL, lw=1)
                    a.set_xlim(*lims[0])
                    a.set_ylim(*lims[1])
                else:
                    a.axhline(abline[0], color=_SYMBOL_COL, lw=1)
        elif plotFun == "bwplot":
            yf, ylevs = _as_factor(yv)
            groups = [xv[sub & (yf == l)] for l in ylevs]
            try:                                   # matplotlib >= 3.10
                a.boxplot(groups, orientation="horizontal", tick_labels=ylevs, **kw)
            except TypeError:                      # older matplotlib
                a.boxplot(groups, vert=False, labels=ylevs, **kw)
            if abline is not None:
                a.axvline(abline[0], color=_SYMBOL_COL, lw=1)
        else:
            xs = xv[sub]
            a.hist(xs, weights=np.full(len(xs), 100.0 / max(len(xs), 1)), color="#00ffff",
                   edgecolor="black", zorder=2, **kw)
            ylab = ylab or "Percent of Total"
            if abline is not None:
                a.axvline(abline[0], color=_SYMBOL_COL, lw=1)
    fig.supxlabel(xlab)
    if ylab is not None:
        fig.supylabel(ylab)
    return fig


def qqmath_merMod(x, id=None, idLabels=None, ax=None, **kw):
    """Normal QQ plot of the standardized (scaled Pearson) residuals (``qqmath.merMod``),
    with the line through the quartiles (``panel.qqmathline``)."""
    plt = _plt()
    values = np.asarray(x.residuals(type="pearson", scaled=True), float)
    n = len(values)
    if id is not None:
        if isinstance(id, (int, float)) and not isinstance(id, bool):
            if id <= 0 or id >= 1:
                raise ValueError("'id' must be between 0 and 1")
            ids = np.abs(values) > -norm.ppf(id / 2)
        else:
            ids = np.asarray(id, bool)
        idLabels = getIDLabels(x) if idLabels is None else np.asarray(idLabels).astype(str)
    if ax is None:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (5, 4.5)))
    fig = ax.figure
    o = np.argsort(values, kind="stable")
    qx = norm.ppf(ppoints(n))
    # panel.qqmathline: line through the quartiles
    yq = np.quantile(values, [0.25, 0.75])
    xq = norm.ppf([0.25, 0.75])
    slope = (yq[1] - yq[0]) / (xq[1] - xq[0])
    ax.axline((xq[0], yq[0]), slope=slope, color=_SYMBOL_COL, lw=1)
    ax.scatter(qx, values[o], facecolors="none", edgecolors=kw.pop("color", _SYMBOL_COL), zorder=2, **kw)
    if id is not None and np.any(ids):
        rank = np.empty(n, int)
        rank[o] = np.arange(n)
        for i in np.flatnonzero(ids):
            ax.text(qx[rank[i]], values[i], idLabels[i], ha="center", va="center", fontsize="small")
    ax.set_xlabel("Standard normal quantiles")
    ax.set_ylabel("Standardized residuals")
    return fig


# ---------------------------------------------------------------------------
# plots of the random effects (R/plots.R)
# ---------------------------------------------------------------------------

def _ranef_frame(xt):
    if isinstance(xt, pd.Series):
        df = xt.to_frame()
        pv = xt.attrs.get("postVar")
        if pv is not None and np.ndim(pv) == 1:
            pv = np.asarray(pv).reshape(1, 1, -1)
        return df, pv
    pv = xt.attrs.get("postVar", xt.attrs.get("condVar"))
    return xt, pv


def _r_order_strings(levels):
    """factor(rownames): levels in sorted order (numeric-looking strings sort as R's
    locale collation does for digits)"""
    return sorted(levels)


def asDf0(x, nx):
    """lme4:::asDf0 -- stacked ranef data frame with columns values, ind, .nn (factor
    ordered by the first column) and se"""
    xt, pv = _ranef_frame(x[nx])
    nr, nc = xt.shape
    rn = [str(r) for r in xt.index]
    # reorder(factor(rownames(xt)), xt[[1]], FUN = mean): stable sort of the sorted levels
    base = _r_order_strings(list(dict.fromkeys(rn)))
    first = pd.Series(np.asarray(xt.iloc[:, 0], float), index=rn).groupby(level=0).mean()
    scores = np.array([first[l] for l in base])
    levels = [base[i] for i in np.argsort(scores, kind="stable")]
    ss = pd.DataFrame({
        "values": np.asarray(xt.to_numpy(float), float).ravel(order="F"),
        "ind": pd.Categorical(np.repeat([str(c) for c in xt.columns], nr),
                              categories=[str(c) for c in xt.columns]),
        ".nn": pd.Categorical(np.tile(rn, nc), categories=levels),
    })
    if pv is not None:
        pv = np.asarray(pv, float)
        ss["se"] = np.concatenate([np.sqrt(pv[i, i, :]) for i in range(pv.shape[0])])
    return ss


def dotplot_ranef(x, main=True, transf=None, level=0.95, scales_free=False, **kw):
    """Caterpillar plot of the conditional modes (lattice ``dotplot.ranef.mer``).

    ``x`` is the result of ``model.ranef(condVar=True)``.  One Figure per grouping factor
    (a dict name -> Figure), one panel per random-effect term; the levels are ordered by the
    first term, and the intervals are ``value +/- qnorm((1+level)/2) * condsd``.  Panels
    share the x axis as in lattice's default (``scales_free=True`` gives
    ``scales = list(x = list(relation = "free"))``)."""
    plt = _plt()
    transf = (lambda v: v) if transf is None else transf
    rng = norm.ppf((1 + level) / 2)
    out = {}
    for nx in x:
        ss = asDf0(x, nx)
        has_se = "se" in ss
        terms = list(ss["ind"].cat.categories)
        levels = list(ss[".nn"].cat.categories)
        ypos = {l: i + 1 for i, l in enumerate(levels)}
        # prepanel.ci
        lims = []
        for t in terms:
            s = ss[ss["ind"] == t]
            xv = s["values"].to_numpy()
            if has_se:
                hw = rng * s["se"].to_numpy()
                v = np.concatenate([transf(xv - hw), transf(xv + hw)])
            else:
                v = transf(xv)
            v = v[np.isfinite(v)]
            lims.append(np.array([v.min(), v.max()]))
        if not scales_free:
            lims = [np.array([min(l[0] for l in lims), max(l[1] for l in lims)])] * len(terms)
        lims = [_extend_limits(l) for l in lims]
        fig, axes = plt.subplots(1, len(terms), squeeze=False, sharey=True,
                                 figsize=kw.pop("figsize", (3.5 * len(terms) + 0.8, 0.22 * len(levels) + 1.4)))
        axes = axes.ravel()
        for a, t, lim in zip(axes, terms, lims):
            s = ss[ss["ind"] == t]
            xv = s["values"].to_numpy()
            yv = np.array([ypos[l] for l in s[".nn"].astype(str)])
            for p in range(1, len(levels) + 1):           # panel.abline(h = levels.fos)
                a.axhline(p, color=_DOT_LINE_COL, lw=1, zorder=0)
            a.axvline(0, color=_DOT_LINE_COL, lw=1, zorder=0)
            if has_se:
                se = s["se"].to_numpy()
                a.hlines(yv, transf(xv - rng * se), transf(xv + rng * se), colors="black", zorder=1)
            a.scatter(transf(xv), yv, color=kw.get("color", _SYMBOL_COL), s=kw.get("s", 18), zorder=2)
            a.set_title(t, fontsize="medium")
            a.set_xlim(*lim)
            a.set_ylim(0.4, len(levels) + 0.6)
        axes[0].set_yticks(range(1, len(levels) + 1))
        axes[0].set_yticklabels(levels)
        if main:
            fig.suptitle(nx)
        fig.tight_layout()
        out[nx] = fig
    return out


def qqmath_ranef(x, main=True, level=0.95, **kw):
    """Normal QQ plot of the conditional modes (lattice ``qqmath.ranef.mer``): one Figure
    per grouping factor, one panel per term with free x scales; with conditional variances,
    horizontal intervals ``value +/- qnorm((1+level)/2) * condsd``."""
    plt = _plt()
    rng = norm.ppf((1 + level) / 2)
    out = {}
    for nx in x:
        xt, pv = _ranef_frame(x[nx])
        nr, nc = xt.shape
        terms = [str(c) for c in xt.columns]
        fig, axes = plt.subplots(1, nc, squeeze=False, sharey=pv is not None,
                                 figsize=kw.pop("figsize", (3.5 * nc + 0.8, 4)))
        axes = axes.ravel()
        for j, a in enumerate(axes):
            vals = np.asarray(xt.iloc[:, j], float)
            o = np.argsort(vals, kind="stable")
            if pv is not None:
                se = np.sqrt(np.asarray(pv, float)[j, j, :])[o]
                xv = vals[o]
                yv = norm.ppf((np.arange(1, nr + 1) - 0.5) / nr)
                a.grid(True, color=_REF_LINE_COL, zorder=0)        # panel.grid(h = -1, v = -1)
                a.axvline(0, color="black", lw=1, zorder=1)
                a.hlines(yv, xv - rng * se, xv + rng * se, colors="black", zorder=1)
                a.scatter(xv, yv, color=kw.get("color", _SYMBOL_COL), s=kw.get("s", 18), zorder=2)
                lim = [np.nanmin(xv - rng * se), np.nanmax(xv + rng * se)]
                a.set_xlim(*_extend_limits(lim))
            else:
                a.scatter(norm.ppf(ppoints(nr)), vals[o], facecolors="none",
                          edgecolors=kw.get("color", _SYMBOL_COL))
            a.set_title(terms[j], fontsize="medium")
        if pv is not None:
            fig.supylabel("Standard normal quantiles")
        else:
            fig.supxlabel("Standard normal quantiles")
        if main:
            fig.suptitle(nx)
        fig.tight_layout()
        out[nx] = fig
    return out
