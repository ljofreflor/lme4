"""Ports of lme4/tests/testthat/test-lmList.R and tests/lmList-tst.R
(numeric references from lme4 2.1-0)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lme4 import load_dataset
from lme4.lmlist import lmList, lm, glm, LmList, spline, approx

from _rrng import RRNG

sleepstudy = load_dataset("sleepstudy")
cbpp = load_dataset("cbpp")


@pytest.fixture(scope="module")
def fm1():
    return lmList("Reaction ~ Days | Subject", sleepstudy)


@pytest.fixture(scope="module")
def simdat():
    # set.seed(12) with RNGkind(sample.kind = "Rounding") as in test-lmList.R
    r = RRNG(12, sample_kind="Rounding")
    g = r.sample(list("ABCDE"), 250)
    y1 = r.runif(250, max=100)
    y2 = r.sample([0.0, 1.0], 250)
    return pd.DataFrame(dict(g=g, y1=y1, y2=y2))


def test_basic(fm1):
    fm1_ = lmList("Reaction ~ Days | Subject", sleepstudy, pool=False)
    cf = pd.DataFrame({
        "(Intercept)": [244.19267, 205.05495, 203.48423, 289.68509, 285.73897, 264.25161,
                        275.01911, 240.16291, 263.03469, 290.10413, 215.11177, 225.8346,
                        261.14701, 276.37207, 254.96815, 210.44909, 253.63604, 267.0448],
        "Days": [21.764702, 2.2617855, 6.1148988, 3.0080727, 5.2660188, 9.5667679,
                 9.1420455, 12.253141, -2.8810339, 19.025974, 13.493933, 19.504017,
                 6.4334976, 13.566549, 11.348109, 18.056151, 9.1884448, 11.298073]})
    c1 = fm1.coef()
    assert isinstance(fm1, LmList)
    assert np.allclose(c1.values, cf.values, rtol=1e-7)
    assert list(c1.columns) == ["(Intercept)", "Days"]
    assert c1.index[0] == "308"
    pd.testing.assert_frame_equal(fm1_.coef(), c1)
    assert str(fm1.formula()) == "Reaction ~ Days | Subject"
    sm1 = fm1.summary()
    assert sm1["RSE"] == pytest.approx(25.5918156267, rel=1e-10)
    assert sm1["df.residual"] == 144
    assert fm1.sigma() == pytest.approx(25.5918156267, rel=1e-10)
    # grouping formula is preserved
    from lme4.rlang import deparse
    assert deparse(fm1.groupsForm) == "Subject"
    assert "Residual standard error: 25.59182" in str(fm1)


def test_confint_pooled_and_not(fm1):
    ci1 = fm1.confint(pool=False)
    ci2 = fm1.confint()
    pd.testing.assert_frame_equal(ci1, ci2)
    ref = np.array([[179.433862895996, 308.951475285822],
                    [193.026448122379, 217.083442786712],
                    [186.785722998616, 220.182727910474]])
    assert np.allclose(ci1["(Intercept)"].iloc[:3].values, ref, rtol=1e-10)
    assert list(ci1["(Intercept)"].columns) == ["2.5 %", "97.5 %"]
    # pooled: coef +/- qt(.975, 144) * pooledSD * sqrt(diag(cov.unscaled))  (nlme intervals)
    cip = fm1.confint(pool=True)
    sd = float(fm1.pooledSD())
    q = stats.t.ppf(0.975, 144)
    X = np.column_stack([np.ones(10), np.arange(10)])
    se = sd * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    cf = fm1.coef()
    assert np.allclose(cip["(Intercept)"]["97.5 %"], cf["(Intercept)"] + q * se[0])
    assert np.allclose(cip["Days"]["2.5 %"], cf["Days"] - q * se[1])


def test_methods(fm1):
    fe = fm1.fixef()
    assert np.allclose(fe.values, fm1.coef().mean().values)
    re = fm1.ranef()
    assert np.allclose(re.mean().values, 0, atol=1e-10)
    # logLik, pooled (nlme's logLik.lmList) vs separate fits
    ll = fm1.logLik()
    rss = sum(np.sum(v.residuals().values ** 2) for v in fm1.values())
    N = 180
    assert ll.value == pytest.approx(-N / 2 * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(rss)))
    assert ll.df == 37
    llu = fm1.logLik(pool=False)
    assert llu.value == pytest.approx(sum(v.logLik().value for v in fm1.values()))
    assert llu.df == 54
    fv = fm1.fitted()
    assert len(fv) == 180 and list(fv.index[:10]) == ["308"] * 10
    assert np.allclose(fv.values + fm1.residuals().values, sleepstudy["Reaction"].values)
    upd = fm1.update("Reaction ~ 1 | Subject")
    assert list(upd.coef().columns) == ["(Intercept)"]
    upd2 = fm1.update(pool=False)
    assert upd2.pool is False
    pr = fm1.predict(sleepstudy.iloc[:3])
    assert np.allclose(pr.values, fv.values[:3])


def test_simulated(simdat):
    fm31 = lmList("y1 ~ 1 | g", simdat)
    assert np.allclose(fm31.coef()["(Intercept)"].values,
                       [45.8945525606396, 50.1127995110841, 49.5320538515225,
                        52.4286874305165, 48.7716343882989], rtol=1e-12)
    assert list(fm31.coef().index) == list("ABCDE")
    fm32 = lmList("y2 ~ 1 | g", simdat, family="binomial")
    assert fm32.pool is False
    cf32 = fm32.confint()        # profile likelihood (confint.glm)
    assert cf32.shape == (5, 2)
    ref = np.array([[-0.400041, -0.311489, -1.07774, -0.841075, -0.273828],
                    [0.743188, 0.768538, 0.0723138, 0.274392, 0.890795]]).T
    assert np.allclose(cf32.values, ref, rtol=5e-6)
    assert fm32.sigma() == 1


def test_cbpp():
    with pytest.warns(UserWarning, match="Fitting failed for "):
        fm4 = lmList("cbind(incidence, size - incidence) ~ period | herd", cbpp,
                     family="binomial")
    cf4 = fm4.coef()
    assert cf4.shape == (15, 4)
    na_pos = np.nonzero(np.isnan(cf4.values.T.ravel()))[0] + 1   # R's which(is.na()), column-major
    assert list(na_pos) == sorted([8 + 15 * k for k in range(4)] + [47])
    with pytest.warns(UserWarning, match="Fitting failed"):
        lmList("incidence ~ period | herd", cbpp)


def test_pooledSD_aliased():
    # GH #377: df must use the rank, not the number of coefficients
    d = sleepstudy.copy()
    d["D2"] = 2 * d["Days"]
    L2 = lmList("Reaction ~ Days + D2 | Subject", d)
    ps = L2.pooledSD()
    assert ps.df == 18 * (10 - 2)
    rss = sum(np.sum(m.residuals().values ** 2) for m in L2.values())
    assert float(ps) == pytest.approx(np.sqrt(rss / 144))
    s2 = L2.summary()
    assert s2["df.residual"] == 144
    assert np.all(np.isnan(L2.coef()["D2"]))
    assert not np.any(np.isnan(s2["coefficients"]["Days"].values))


def test_NA_weights_offsets():
    dat = pd.DataFrame(dict(x=np.arange(1, 9.0), y=[2, 2, 5, 4, 3, 1, 2, 1.0],
                            g=[1, 1, 1, 2, 2, 3, 3, 3]))
    m1 = lmList("y ~ x | g", dat)
    assert not np.any(np.isnan(m1.coef().values))
    r = RRNG(101)
    w = r.runif(len(sleepstudy))
    m2 = lmList("Reaction ~ Days | Subject", sleepstudy, weights=w)
    ss = sleepstudy[sleepstudy["Subject"] == sleepstudy["Subject"].cat.categories[0]]
    m2X = lm("Reaction ~ Days", ss, weights=w[:len(ss)])
    assert np.allclose(m2X.coef().values, m2.coef().iloc[0].values)
    m3 = lmList("Reaction ~ Days | Subject", sleepstudy)
    m4 = lmList("Reaction ~ Days | Subject", sleepstudy, offset=w)
    m4X = lm("Reaction ~ Days", ss, offset=w[:len(ss)])
    assert np.allclose(m4X.coef().values, m4.coef().iloc[0].values)
    assert not np.allclose(m2.coef().values, m3.coef().values)
    assert not np.allclose(m4.coef().values, m3.coef().values)
    sw = sleepstudy.copy()
    sw["w"] = w
    m5 = lmList("Reaction ~ Days + offset(w) | Subject", sw)
    pd.testing.assert_frame_equal(m5.coef(), m4.coef())
    # NAs in a column that is not used
    dat2 = dat.copy()
    dat2["xx"] = [np.nan, np.nan, np.nan, 1, 2, 3, 4, np.nan]
    m5 = lmList("y ~ x | g", dat2)
    ref1 = lm("y ~ x", dat[dat.g == 1]).coef().values
    ref3 = lm("y ~ x", dat[dat.g == 3]).coef().values
    assert np.allclose(m5.coef().iloc[0].values, ref1)
    assert np.allclose(m5.coef().iloc[2].values, ref3)


def test_derived_variables():
    fm = lmList("log(Reaction) ~ Days | Subject", sleepstudy)
    ss = sleepstudy[sleepstudy["Subject"] == "308"]
    X = np.column_stack([np.ones(len(ss)), ss["Days"]])
    b = np.linalg.lstsq(X, np.log(ss["Reaction"].values), rcond=None)[0]
    assert np.allclose(fm.coef().loc["308"].values, b)


def test_subset():
    fm = lmList("Reaction ~ Days | Subject", sleepstudy, subset="Days > 2")
    ss = sleepstudy[(sleepstudy["Subject"] == "309") & (sleepstudy["Days"] > 2)]
    X = np.column_stack([np.ones(len(ss)), ss["Days"]])
    b = np.linalg.lstsq(X, ss["Reaction"].values, rcond=None)[0]
    assert np.allclose(fm.coef().loc["309"].values, b)


def test_grouping_errors():
    d = sleepstudy.copy()
    d["Side"] = np.where(d["Days"] > 4, "L", "R")
    with pytest.raises(ValueError, match="simple grouping"):
        lmList("Reaction ~ Days | Side/Subject", d)
    with pytest.raises(ValueError):
        lmList("Reaction ~ Days + (1|Side) + (1|Subject)", d)


def test_glm_fit_and_spline():
    # glm.fit agrees with closed-form logistic intercept, and summary SE
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1.0])
    d = pd.DataFrame(dict(y=y))
    g = glm("y ~ 1", "binomial", d)
    p = y.mean()
    assert g.coef().iloc[0] == pytest.approx(np.log(p / (1 - p)), rel=1e-8)
    se = g.summary()["coefficients"]["Std. Error"].iloc[0]
    assert se == pytest.approx(1 / np.sqrt(len(y) * p * (1 - p)), rel=1e-6)
    # fmm spline interpolates a cubic exactly
    x = np.linspace(-2, 3, 7)
    sx, sy = spline(x, x ** 3 - x)
    assert np.allclose(sy, sx ** 3 - sx, atol=1e-10)
    assert approx([0, 1, 2], [0, 10, 20], [0.5, 1.5, 3])[:2] == pytest.approx([5, 15])
