"""Ports of the influence-measure tests in lme4/tests/testthat/test-methods.R
(references from lme4 2.1-0; fit_sleepstudy_1 / fit_cbpp_1 of lme-tst-fits.R)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from lme4 import glmer, lmer, load_dataset
from lme4.influence import (influence_merMod, cooks_distance, rstudent, rstandard, dfbeta,
                            dfbetas, InfluenceMerMod)

sleepstudy = load_dataset("sleepstudy")
cbpp = load_dataset("cbpp")


@pytest.fixture(scope="module")
def fm1():
    return lmer("Reaction ~ Days + (1|Subject)", sleepstudy)


@pytest.fixture(scope="module")
def gm1():
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
                 family="binomial")


def test_hatvalues(fm1):
    h = influence_merMod(fm1, do_coef=False)["hat"]
    assert np.allclose(h.values[:6], [0.107483311203734, 0.102096105816528,
                                      0.0980557017761242, 0.0953620990825215,
                                      0.0940152977357202, 0.0940152977357202], rtol=1e-6)


def test_small_case_deletion():
    ss = sleepstudy.iloc[:60]
    smallfit = lmer("Reaction ~ 1 + (1 | Subject)", ss)
    i1 = influence_merMod(smallfit)
    assert isinstance(i1, InfluenceMerMod)
    fe = i1["fixed.effects[-case]"]
    assert list(fe.columns) == ["(Intercept)"]
    assert np.allclose(fe.values[:6, 0], [286.35044481665, 286.179896199062, 286.327301507498,
                                          285.014692121823, 284.36060419176, 283.297551183126],
                       rtol=1e-6)
    assert len(i1["converged"]) == 60
    assert list(i1)[:6] == ["fixed.effects", "fixed.effects[-case]", "var.cov.comps",
                            "var.cov.comps[-case]", "vcov", "vcov[-case]"]
    assert list(i1["var.cov.comps"].index) == ["sigma^2", "v[(Intercept)]"]
    db = dfbeta(i1)
    assert np.allclose(db.values[:, 0], fe.values[:, 0] - smallfit.fixef().iloc[0])
    dbs = dfbetas(i1)
    assert dbs.shape == db.shape
    assert dfbeta(i1, "var.cov").shape == (60, 2)


def test_rstudent(fm1):
    r = rstudent(fm1)
    assert np.allclose(r.values[:6], [-1.45598270922089, -1.49664543508657, -2.11747425025103,
                                      -0.0729690066951975, 0.772716397142335, 2.37859408861768],
                       rtol=1e-6)
    with pytest.raises(TypeError):
        rstandard(fm1)


def test_cooks_distance(fm1):
    assert np.allclose(cooks_distance(fm1).values[:6],
                       [0.127645976734753, 0.127346548123793, 0.243724627125036,
                        0.000280638917214881, 0.0309804642689636, 0.293554225380831],
                       rtol=1e-6)


def test_cooks_subject_level(fm1):
    i = influence_merMod(fm1, "Subject")
    assert np.allclose(cooks_distance(i).values[:2], [0.33921460279262, 0.290309061006305],
                       rtol=1e-6)
    assert i["groups"] == "Subject"


def test_full_case_influence(fm1):
    # testLevel > 1 in R
    i1 = influence_merMod(fm1)
    assert np.allclose(i1["fixed.effects[-case]"].values[0], [252.323536264131, 10.3222704729148],
                       rtol=1e-6)
    assert np.allclose(cooks_distance(i1).values[:2], [0.016503344184025, 0.0106634053477361],
                       rtol=1e-5)


def test_cooks_glmer(gm1):
    # R references are reproduced to ~5e-5 (relative): the GLMM refits converge
    # (Nelder-Mead stage) to slightly different points than in R
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = influence_merMod(gm1)
        inf_h = influence_merMod(gm1, "herd")
    assert np.allclose(cooks_distance(inf).values[:3],
                       [0.0524063139071626, 0.0396850631091166, 0.246678190526733], rtol=2e-4)
    assert np.allclose(cooks_distance(inf_h).values[:3],
                       [0.251033719710455, 0.00519895610205458, 0.101113708827459], rtol=2e-4)


def test_glmer_residual_measures(gm1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = gm1.hatvalues()
        rs = rstandard(gm1)
        rt = rstudent(gm1)
    pr = gm1.residuals(type="pearson").values
    # NB: with binomial size weights, hat values can exceed 1 (as in R) -> NaN
    with np.errstate(invalid="ignore"):
        assert np.allclose(rs.values, pr / np.sqrt(1 - h), equal_nan=True)
        assert np.allclose(rt.values, rs.values, equal_nan=True)
        rd = rstandard(gm1, type="deviance")
        assert np.allclose(rd.values, gm1.residuals(type="deviance").values / np.sqrt(1 - h),
                           equal_nan=True)


def test_influence_nAGQ0(gm1):
    gm1Q0 = gm1.update(nAGQ=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i = influence_merMod(gm1Q0, "herd")
    assert isinstance(i, InfluenceMerMod)


def test_cooks_zero_variance_matches_lm():
    # cook's distance comparison (testLevel > 1): with a zero RE variance,
    # case-deletion Cook's distance equals lm's
    rng = np.random.default_rng(101)
    n = 50
    x = rng.normal(size=n)
    f = np.repeat(["1", "2"], n // 2)
    e = rng.normal(scale=5, size=n)
    for lev in ("1", "2"):          # identical group means of the residuals => theta = 0
        e[f == lev] -= e[f == lev].mean()
    dd = pd.DataFrame(dict(x=x, f=pd.Categorical(f), y=2 + 2 * x + e))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm2 = lmer("y ~ x + (1|f)", dd, REML=False)
        assert fm2.theta[0] == pytest.approx(0, abs=1e-6)
        i2 = influence_merMod(fm2)
    X = np.column_stack([np.ones(n), x])
    H = X @ np.linalg.solve(X.T @ X, X.T)
    h = np.diag(H)
    r = dd["y"].values - H @ dd["y"].values
    s2 = np.sum(r ** 2) / (n - 2)
    cd_lm = r ** 2 / (2 * s2) * h / (1 - h) ** 2
    assert np.allclose(cooks_distance(i2).values, cd_lm, rtol=1e-3)
    # hat-value version uses sigma_ML^2 = RSS/n instead of RSS/(n-p)
    assert np.allclose(cooks_distance(fm2).values, cd_lm * n / (n - 2), rtol=1e-4)
