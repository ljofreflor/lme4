"""Ports of lme4/tests/testthat/test-glmer.R (references from lme4 2.1-0)."""
import numpy as np
import pandas as pd
import pytest

from lme4 import glmer, lmer, glmerControl, binomial, poisson, load_dataset, getME, V, cbind
from lme4.optimizers import GHrule

cbpp = load_dataset("cbpp")
REF_FIXEF = np.array([-1.39854982537216, -0.992335519118859, -1.12867532780426, -1.58030423764517])


@pytest.fixture(scope="module")
def gm1():
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=binomial)


def test_cbpp_reference(gm1):
    assert np.allclose(gm1.fixef().values, REF_FIXEF, rtol=5e-4)
    assert gm1.VarCorr()["herd"].cov[0, 0] == pytest.approx(0.41245527438386, rel=6e-4)
    assert gm1.deviance() == pytest.approx(73.47154, rel=1e-5)
    assert gm1.sigma() == 1
    df, aic = gm1.extractAIC()
    assert df == 5 and aic == pytest.approx(194.052674598026, rel=1e-5)
    assert getME(gm1, "theta").values == pytest.approx([0.642226809144453], rel=6e-4)
    Zt = getME(gm1, "Zt")
    assert Zt.shape == (15, 56) and np.all(Zt.data == 1)
    th = gm1.theta[0]
    assert np.allclose(getME(gm1, "Lambdat").toarray(), np.diag(np.repeat(th, 15)))


def test_symbolic_glmer(gm1):
    incidence, size, period, herd = V("incidence size period herd")
    g = glmer(cbind(incidence, size - incidence) == period + (1 | herd), cbpp, family=binomial)
    assert np.allclose(g.fixef().values, gm1.fixef().values)


def test_proportion_with_weights(gm1):
    d = cbpp.copy()
    d["prop"] = d.incidence / d["size"]
    g = glmer("prop ~ period + (1 | herd)", d, family=binomial, weights=d["size"].values)
    assert np.allclose(g.fixef().values, gm1.fixef().values, rtol=1e-6)
    assert g.deviance() == pytest.approx(gm1.deviance(), rel=1e-6)


def test_nlev_error():
    one = cbpp[cbpp.herd == cbpp.herd.cat.categories[0]]
    with pytest.raises(ValueError, match="must have > 1"):
        glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", one, family=binomial)


def test_gaussian_shortcut():
    sleepstudy = load_dataset("sleepstudy")
    with pytest.warns(UserWarning, match="calling glmer"):
        m = glmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
    assert type(m).__name__ == "lmerMod"


def test_agq():
    g9 = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
               family=binomial, nAGQ=9)
    assert g9.devcomp["dims"]["nAGQ"] == 9
    assert "Adaptive Gauss-Hermite Quadrature, nAGQ = 9" in g9.methTitle()
    # AGQ estimates are close to (but not identical to) Laplace
    assert np.allclose(g9.fixef().values, REF_FIXEF, atol=0.02)
    with pytest.raises(ValueError, match="nAGQ > 1 is only available"):
        glmer("Reaction ~ Days + (Days|Subject)", load_dataset("sleepstudy"),
              family=poisson, nAGQ=5)


def test_nAGQ0(gm1):
    g0 = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
               family=binomial, nAGQ=0)
    # nAGQ = 0 optimizes beta inside PIRLS: estimates differ slightly from Laplace
    assert np.allclose(g0.fixef().values, gm1.fixef().values, atol=0.05)
    assert g0.devcomp["dims"]["nAGQ"] == 0


def test_probit_link(gm1):
    g = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
              family=binomial(link="probit"))
    assert g.family().link == "probit"


def test_GHrule():
    r = GHrule(3)
    assert np.allclose(r[:, 0], [-np.sqrt(3), 0, np.sqrt(3)])
    assert np.allclose(r[:, 1], [1 / 6, 2 / 3, 1 / 6])
    assert GHrule(0).shape == (0, 3)


def test_poisson_grouseticks():
    gt = load_dataset("grouseticks")
    m = glmer("TICKS ~ YEAR + (1|BROOD)", gt, family=poisson)
    assert m.isGLMM()
    assert np.isfinite(m.AIC())
    s = str(m.summary())
    assert "Family: poisson  ( log )" in s
