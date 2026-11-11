"""Tests for lme4.profile (profile.merMod / confint.merMod), ported from
lme4/tests/profile-tst.R, tests/confint.R and tests/testthat/test-methods.R."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lme4 import lmer, glmer, load_dataset
from lme4.profile import (profile_merMod, confint_merMod, interpSpline, backSpline,
                          isMonotoneSpline, profSplines, predy, varianceProf, logProf)


@pytest.fixture(scope="module")
def sleepstudy():
    return load_dataset("sleepstudy")


@pytest.fixture(scope="module")
def fm1(sleepstudy):     # fit_sleepstudy_1
    return lmer("Reaction ~ Days + (1|Subject)", sleepstudy)


@pytest.fixture(scope="module")
def fm2(sleepstudy):     # fit_sleepstudy_2
    return lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)


@pytest.fixture(scope="module")
def pr_fm2(fm2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_merMod(fm2)


def all_equal(target, current, tolerance=1.5e-8):
    """R's all.equal() for numeric vectors: mean relative difference"""
    target = np.asarray(target, float).ravel()
    current = np.asarray(current, float).ravel()
    return np.mean(np.abs(target - current)) / np.mean(np.abs(target)) < tolerance


# --------------------------------------------------------------------------- R references

def test_dyestuff_ML_profile():
    # profile-tst.R: lme4a reference values (sigma on the log scale there)
    fm01ML = lmer("Yield ~ 1|Batch", load_dataset("Dyestuff"), REML=False)
    tpr = profile_merMod(fm01ML)
    CIpr = tpr.confint()
    assert CIpr.shape == (3, 2)
    assert list(CIpr.columns) == ["2.5 %", "97.5 %"]
    assert all_equal(np.exp([3.64362, 4.21446]), CIpr.loc[".sigma"].values, tolerance=1e-6)
    assert all_equal([1486.451500, 1568.548494], CIpr.loc["(Intercept)"].values)
    # confint(fm01ML) in R: .sig01 12.19785 84.06290
    np.testing.assert_allclose(CIpr.loc[".sig01"].values, [12.19785, 84.06290], rtol=1e-4)
    # all combinations of 'which' work
    for w in ([0, 1, 2], [0, 1], [0], [1, 2], [1], [2], [0, 2]):
        p = profile_merMod(fm01ML, which=w)
        assert sorted(p.frame[".par"].unique()) == sorted(np.array(tpr.frame[".par"].unique())[w])


def test_penicillin_ML_profile():
    # profile-tst.R (testLevel > 2): lme4a reference
    fm2ML = lmer("diameter ~ 1 + (1|plate) + (1|sample)", load_dataset("Penicillin"), REML=False)
    CIpr2 = profile_merMod(fm2ML).confint()
    ref = np.array([[0.633565787613112, 1.1821039843372],
                    [1.09578224011285, 3.55631937954106],
                    [np.exp(-0.721864513060904), np.exp(-0.462903300019305)],
                    [21.2666273835452, 24.6778176174587]])
    assert list(CIpr2.index) == [".sig01", ".sig02", ".sigma", "(Intercept)"]
    assert all_equal(ref, CIpr2.values, tolerance=1e-6)   # R: tolerance=1e-6


def test_sleepstudy_confint_profile(fm2, pr_fm2):
    # confint(fm1) for Reaction ~ Days + (Days|Subject) (?confint.merMod, lme4 2.x
    # ordering: sd's, then correlation)
    ci = pr_fm2.confint()
    ref = pd.DataFrame([[14.3814761, 37.7159958],
                        [3.8011641, 8.7533831],
                        [-0.4815007, 0.6849863],
                        [22.8982669, 28.8579965],
                        [237.6806955, 265.1295147],
                        [7.3586533, 13.5755560]],
                       index=[".sig01", ".sig02", ".sig03", ".sigma", "(Intercept)", "Days"],
                       columns=["2.5 %", "97.5 %"])
    assert list(ci.index) == list(ref.index)
    np.testing.assert_allclose(ci.values, ref.values, rtol=1e-4)
    # confint.merMod gives the same
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci2 = confint_merMod(fm2, quiet=True)
    np.testing.assert_allclose(ci2.values, ci.values, rtol=1e-6)


def test_wald_vs_profile_fixed(fm2):
    # tests/confint.R
    c0 = confint_merMod(fm2, method="Wald")
    c0B = confint_merMod(fm2, method="Wald", parm="Days")
    np.testing.assert_allclose(c0.loc["Days"].values, c0B.loc["Days"].values)
    np.testing.assert_allclose(c0B.values.ravel(), [7.437592, 13.496980], rtol=1e-6)
    assert c0.iloc[:4].isna().all().all()
    pr156 = profile_merMod(fm2, which=[4, 5])
    c1 = pr156.confint()
    # expect Wald and profile _reasonably_ close  (R: all.equal(.., tolerance=2e-3))
    assert np.mean(np.abs(c0.iloc[4:6].values - c1.values)) / np.mean(np.abs(c0.iloc[4:6].values)) < 2e-3
    c6 = pr156.confint("Days")
    pd.testing.assert_frame_equal(c1.iloc[[1]], c6)


def test_wald_names_and_values(fm1):
    # test-methods.R "confint"
    cw = confint_merMod(fm1, method="Wald", parm="beta_")
    np.testing.assert_allclose(cw.values.T.ravel(), [232.301892, 8.891041, 270.508318, 12.043531],
                               rtol=1e-5)
    assert confint_merMod(fm1, method="Wald", parm="theta_").isna().all().all()
    ci_p = confint_merMod(fm1, quiet=True)
    ci_w = confint_merMod(fm1, method="Wald")
    assert list(ci_p.index) == [".sig01", ".sigma", "(Intercept)", "Days"]
    assert list(ci_p.columns) == ["2.5 %", "97.5 %"]
    assert list(ci_p.index) == list(ci_w.index)
    ci_pn = confint_merMod(fm1, quiet=True, signames=False)
    ci_wn = confint_merMod(fm1, method="Wald", signames=False)
    assert list(ci_pn.index) == ["sd_(Intercept)|Subject", "sigma", "(Intercept)", "Days"]
    assert list(ci_wn.index) == list(ci_pn.index)
    with pytest.warns(UserWarning, match="oldNames"):
        confint_merMod(fm1, parm=0, oldNames=True, quiet=True)


def test_profile_varcov_scale(fm1):
    # test-methods.R "profile": no correlations => squaring the sd profile gives
    # the variance-scale profile
    p1 = profile_merMod(fm1, which="theta_")
    p2 = profile_merMod(fm1, which="theta_", prof_scale="varcov")
    assert all_equal(p1.confint().values ** 2, p2.confint().values, tolerance=1e-5)
    assert all_equal(varianceProf(p1).confint().values, p2.confint().values, tolerance=1e-5)
    # sub("^(\\.sig(ma)?)", "\\1sq", .) as in R
    assert list(varianceProf(p1).confint().index) == [".sigsq01", ".sigmasq"]


def test_profile_varcov_names(fm2, pr_fm2):
    p4 = profile_merMod(fm2, which="theta_", prof_scale="varcov", signames=False)
    assert list(p4.frame[".par"].unique()) == ["var_(Intercept)|Subject", "var_Days|Subject",
                                               "cov_Days.(Intercept)|Subject", "sigma"]
    # variance terms: squared sd-scale CIs (R: tolerance 5e-3)
    s = [0, 1, 3]
    a = pr_fm2.confint().values[s] ** 2
    b = p4.confint().values[s]
    assert np.mean(np.abs(a - b)) / np.mean(np.abs(a)) < 5e-3


def test_profile_names_signames_false(fm2):
    pp = profile_merMod(fm2, which="theta_", signames=False)
    ci = pp.confint()
    assert np.all(np.abs(ci.loc["cor_Days.(Intercept)|Subject"].values) < 1)


def test_confint_thpr_bounds_subset(fm2):
    # test-methods.R: confint.thpr uses the right bounds for a subset of parameters
    p = profile_merMod(fm2, which=[0, 2])
    lev = 0.99999
    ci_all = p.confint(level=lev)
    ci_cor = p.confint(parm=".sig03", level=lev)
    assert ci_cor.loc[".sig03"].iloc[1] == 1
    pd.testing.assert_series_equal(ci_cor.loc[".sig03"], ci_all.loc[".sig03"])
    pd.testing.assert_frame_equal(p.confint(parm=1, level=lev), ci_cor)
    assert list(ci_all.columns) == ["0.0005 %", "99.9995 %"]


def test_profile_switched_optimizer_no_warning(fm1):
    # GH #913
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        profile_merMod(fm1, which=[0], optimizer="bobyqa")
    assert not any("unused control arguments" in str(w.message) for w in wl)


def test_profile_frame_and_helpers(pr_fm2):
    fr = pr_fm2.frame
    assert list(fr.columns) == [".zeta", ".sig01", ".sig02", ".sig03", ".sigma", "(Intercept)",
                                "Days", ".par"]
    df = pr_fm2.as_data_frame()
    assert ".focal" in df.columns
    sub = df[df[".par"] == "Days"]
    np.testing.assert_allclose(sub[".focal"].values, sub["Days"].values)
    # zeta is increasing in the focal parameter, and zero at the optimum
    for nm, g in df.groupby(".par", observed=True):
        z = g.sort_values(".focal")[".zeta"].values
        assert np.all(np.diff(z) > 0)
        assert np.min(np.abs(z)) == 0
    # natural forward splines for a well-behaved profile
    assert pr_fm2.forward[".sig01"].cls == "npolySpline"
    lp = logProf(pr_fm2)
    assert ".lsig01" in lp.backward and ".lsigma" in lp.backward
    np.testing.assert_allclose(np.exp(lp.confint().loc[[".lsig01", ".lsigma"]].values),
                               pr_fm2.confint().loc[[".sig01", ".sigma"]].values, rtol=2e-3)


def test_glmm_profile():
    cbpp = load_dataset("cbpp")
    gm1 = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
                family="binomial")
    pr = profile_merMod(gm1, which="theta_")
    ci = pr.confint()
    assert list(ci.index) == [".sig01"]
    assert 0.25 < ci.iloc[0, 0] < 0.45 and 0.9 < ci.iloc[0, 1] < 1.25
    cib = confint_merMod(gm1, parm="beta_", quiet=True)
    cw = confint_merMod(gm1, parm="beta_", method="Wald")
    assert list(cib.index) == ["(Intercept)", "period2", "period3", "period4"]
    # profile and Wald intervals for the fixed effects are close
    assert np.max(np.abs(cib.values - cw.values)) < 0.1


# --------------------------------------------------------------------------- splines

def test_spline_basics():
    x = np.linspace(-2, 3, 9)
    y = x ** 3 / 10 + x
    sp_ = interpSpline(x, y)
    np.testing.assert_allclose(sp_.predict(x), y, atol=1e-12)
    # natural: second derivative 0 at the ends, linear extrapolation
    np.testing.assert_allclose(sp_.predict([x[0], x[-1]], deriv=2), 0, atol=1e-10)
    s1 = sp_.predict([x[-1]], deriv=1)[0]
    np.testing.assert_allclose(sp_.predict([x[-1] + 1])[0], y[-1] + s1)
    bs = backSpline(sp_)
    np.testing.assert_allclose(bs.predict(y), x, atol=1e-12)
    assert np.isnan(bs.predict([y[-1] + 1])[0])       # polySpline: NA outside the knots
    assert np.isnan(predy(None, [0.0])[0])


def test_monotone_profile_splines_GH834():
    # test-methods.R "monotone profile splines (GH #834)"
    pp = pd.DataFrame({
        ".sig03": [-1, -0.9419788, -0.7712586, -0.5406735, -0.283924, -0.03107754,
                   0.1966026, 0.4020445, 0.4467161, 0.4511833, 0.4958549,
                   0.6916766, 0.8531035, 0.9558708, 1],
        ".zeta": [-5.537997, -3.829926, -2.803105, -2.110852, -1.540167, -1.03543,
                  -0.5739137, -0.1114451, 0, 0.01221793, 0.1291381, 0.7508699,
                  1.55595, 2.642026, 4.739299]})
    isp = interpSpline(pp[".sig03"], pp[".zeta"])
    assert not isMonotoneSpline(isp)
    xx = np.linspace(-1, 1, 10001)
    assert np.any(isp.predict(xx, deriv=1) < 0)
    fw, bw = profSplines(pp, ".sig03")
    assert isMonotoneSpline(fw) and isMonotoneSpline(bw)
    assert np.all(fw.predict(xx, deriv=1) >= 0)
    np.testing.assert_allclose(predy(fw, pp[".sig03"]), pp[".zeta"], atol=1e-12)
    np.testing.assert_allclose(predy(bw, pp[".zeta"]), pp[".sig03"], atol=1e-12)
    ci = predy(bw, stats.norm.ppf([0.025, 0.975]))
    np.testing.assert_allclose(ci, [-0.4788, 0.9026], rtol=1e-3)
    assert np.all(np.abs(ci) < 1)
