"""Tests for lme4.plots_profile (xyplot.thpr, dens()/densityplot.thpr,
splom.thpr/cont()), splines::backSpline and drop1(test = "user").
Reference values from R 4.6.2 / lme4 2.1.0
(examples/r/07_inference_profiles_and_tests.R)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import lme4
from lme4 import lmer, load_dataset
from lme4.plots_profile import (dens, xyplot_thpr_data, splom_traces, xyplot_thpr,
                                densityplot_thpr, splom_thpr, stripExpr)
from lme4.profile import interpSpline, backSpline, predy


@pytest.fixture(scope="module")
def fm01ML():
    return lmer("Yield ~ 1|Batch", load_dataset("Dyestuff"), REML=False)


@pytest.fixture(scope="module")
def tpr(fm01ML):
    return fm01ML.profile(signames=False)


def test_backSpline_natural_ends(fm01ML):
    # R: confint(logProf(profile(fm01ML))) -- the lower limit lies in the first
    # interval of the backward spline, where backSpline() is quadratic
    ci = fm01ML.profile().logProf().confint()
    assert list(ci.index) == [".lsig01", ".lsigma", "(Intercept)"]
    assert_allclose(ci.iloc[0], [2.50099757307977, 4.43157115624689], rtol=1e-6)
    assert_allclose(ci.iloc[1], [3.64362262568987, 4.21446098339058], rtol=1e-6)


def test_backSpline_inverts_at_knots():
    x = np.linspace(0, 3, 9)
    sp = interpSpline(x, np.sinh(x - 1))
    b = backSpline(sp)
    assert_allclose(predy(b, sp.coefficients[:, 0]), x, atol=1e-12)
    # (as in R, the result is only meaningful for increasing splines: for a
    # decreasing one R reverses the coefficient rows without re-expanding them)


def test_xyplot_data(tpr):
    fr, spl = xyplot_thpr_data(tpr)
    assert list(spl) == ["sd_(Intercept)|Batch", "sigma", "(Intercept)"]
    s = fr.loc[fr["pnm"] == "sigma", "pval"].values
    assert_allclose(s[:3], [35.5631733102932, 38.2299848825959, 39.732007521526], rtol=1e-7)
    # sd: zeta = -2.576 not reached -> R puts it at zeta 0, min(knots) = 0
    sd = fr.loc[fr["pnm"] == "sd_(Intercept)|Batch"]
    assert sd["zeta"].iloc[0] == 0 and sd["pval"].iloc[0] == 0
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        xyplot_thpr_data(tpr, conf=50)


def test_dens(tpr):
    d = dens(tpr)
    assert d.shape == (603, 3)
    for i, (nm, pv, de) in {50: ("sd_(Intercept)|Batch", 37.6084381418342, 0.0269437383602199),
                            250: ("sigma", 45.8174720192894, 0.0546048180650687),
                            450: ("(Intercept)", 1482.93605124174, 0.00152402139571641)}.items():
        assert d["pnm"].iloc[i] == nm
        assert_allclose(d["pval"].iloc[i], pv, rtol=1e-7)
        assert_allclose(d["density"].iloc[i], de, rtol=1e-5)


def test_splom_traces(tpr):
    tr = splom_traces(tpr)
    assert_allclose(tr["pfr"].values[1], [145.778949068132, 82.6463908460995, 1610.2029089012],
                    rtol=1e-7)
    pts = tr["traces"][(1, 0)]["ll"]["pts"]
    assert pts.shape == (5, 102, 2)
    # R: as.vector(cont(...)$pts[, , 1]) (column-major)
    assert_allclose(pts[:, :, 0].ravel(order="F")[:3],
                    [-1.17741002251547, -1.7941225779941, -2.14596602628935], atol=1e-5)
    assert_allclose(pts[:, :, 1].ravel(order="F")[:3],
                    [0.117873618146014, 0.15987654052572, 0.179747024155466], atol=1e-5)
    with pytest.raises(ValueError, match="single variable"):
        splom_traces(tpr, which=[2])


def test_plots_return_figures(tpr):
    import matplotlib.pyplot as plt
    for fig in (xyplot_thpr(tpr), xyplot_thpr(tpr, absVal=True, conf=[0.95, 0.99]),
                densityplot_thpr(tpr), splom_thpr(tpr), tpr.xyplot(), tpr.splom(),
                tpr.densityplot()):
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    assert stripExpr([".sig01", ".lsigma", ".sigsq02", "x"]) == [
        r"$\sigma_{1}$", r"$\log(\sigma)$", r"${\sigma_{2}}^2$", "x"]


def test_drop1_user_and_anova_function():
    fm1 = lmer("Reaction ~ Days + (Days | Subject)", load_dataset("sleepstudy"))

    def LRTSumFun(obj, objectDrop=None, **kw):
        nms = ["Chisq", "Df", "p.value"]
        if objectDrop is None:
            return pd.Series(np.nan, index=nms)
        a = lme4.anova(objectDrop, obj)
        return pd.Series([a["Chisq"].iloc[1], a["Df"].iloc[1], a["Pr(>Chisq)"].iloc[1]], index=nms)

    du = fm1.drop1(test="user", sumFun=LRTSumFun)
    assert callable(lme4.anova)       # the submodule lme4/anova.py must not shadow anova()
    assert list(du.index) == ["<none>", "Days"]
    assert_allclose(du.loc["Days"], [23.5365374978655, 1, 1.22563995719719e-06], rtol=1e-5)
    with pytest.raises(ValueError, match="sumFun"):
        fm1.drop1(test="user")
    with pytest.raises(ValueError, match="sumFun"):
        fm1.drop1(test="Chisq", sumFun=LRTSumFun)
