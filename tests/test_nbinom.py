"""Tests for lme4.nbinom (ports of lme4/tests/nbinom.R, testthat/test-nbinom.R,
test-glmernb.R).  The R references there use data simulated with
rnbinom(), which is not reproducible without R's rgamma/rpois, so the
checks here are the structural / self-consistency ones from those files
plus checks of the transcribed numerical building blocks."""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import fminbound
from scipy.special import digamma

from lme4 import load_dataset, getME, glmer, glmerControl, negative_binomial
from lme4.nbinom import (Brent_fmin, optimize, theta_ml, glmer_nb, getNBdisp, setNBdisp,
                         refitNB, est_theta)

grouseticks = load_dataset("grouseticks")
FORM = "TICKS ~ YEAR + cHEIGHT + (1|LOCATION)"


@pytest.fixture(scope="module")
def g1():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmer_nb(FORM, grouseticks)


def test_optimize_R_example():
    # ?optimize:  f <- function (x, a) (x - a)^2 ; optimize(f, c(0, 1), tol = 0.0001, a = 1/3)
    r = optimize(lambda x: (x - 1 / 3) ** 2, (0, 1), tol=1e-4)
    assert r["minimum"] == pytest.approx(1 / 3, abs=1e-8)
    assert r["objective"] == pytest.approx(0, abs=1e-12)
    r = optimize(lambda x: -(x - 1) ** 2, (0, 3), maximum=True)
    assert r["maximum"] == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("f,a,b,tol", [
    (lambda x: np.sin(x) + 0.1 * x * x, -5, 5, 5e-5),
    (lambda x: abs(x - 0.7) + (x - 0.2) ** 4, -3, 3, 1e-6),
])
def test_brent_matches_fmin_port(f, a, b, tol):
    # scipy's fminbound is an independent port of the same Fortran fmin
    ev1, ev2 = [], []
    x1 = Brent_fmin(a, b, lambda x: (ev1.append(x), f(x))[1], tol)
    x2 = fminbound(lambda x: (ev2.append(x), f(x))[1], a, b, xtol=tol)
    assert x1 == x2
    assert np.allclose(ev1, ev2[:len(ev1)]) and len(ev1) == len(ev2)


def test_theta_ml():
    rng = np.random.default_rng(1)
    mu = np.exp(rng.normal(1, 0.5, 500))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    th = theta_ml(y, mu, limit=50)
    score = np.sum(digamma(th + y) - digamma(th) + math.log(th) + 1 - np.log(th + mu)
                   - (y + th) / (mu + th))
    assert abs(score) < 1e-6
    assert 1.2 < th < 3.2 and th.SE > 0
    with pytest.warns(UserWarning, match="iteration limit reached"):
        theta_ml(y, mu, limit=2)


def test_glmer_nb_basic(g1):
    th = getNBdisp(g1)
    assert th == getME(g1, "glmer.nb.theta")
    assert type(g1).__name__ == "glmerMod"
    assert g1.family().family.startswith("Negative Binomial")
    # family in the call is the optimal theta (effects:::mer.to.glm can use it)
    assert g1.call["family"].theta == pytest.approx(th, rel=1e-14)
    assert g1.nevals > 3
    # one extra parameter for theta
    assert g1.logLik().df == len(g1.beta) + 1 + 1
    # refit at the same theta gives the same fit (tests/nbinom.R)
    g1B = refitNB(g1, theta=th)
    assert abs(g1.deviance() - g1B.deviance()) < 1e-6
    assert np.all(np.abs((g1.fixef() - g1B.fixef()) / g1.fixef()) < 1e-6)
    # the profile deviance is minimal at theta
    d0 = -2 * g1.logLik().value
    for f in (0.95, 1.05):
        assert -2 * refitNB(g1, theta=th * f).logLik().value > d0


def test_glmer_nb_vs_fixed_theta(g1):
    th = getNBdisp(g1)
    gf = glmer(FORM, grouseticks, family=negative_binomial(th))
    assert np.allclose(gf.fixef().values, g1.fixef().values, atol=2e-4)
    assert gf.logLik().value == pytest.approx(g1.logLik().value, abs=1e-4)


def test_setNBdisp_does_not_modify(g1):
    th = getNBdisp(g1)
    g2 = setNBdisp(g1, 2 * th)
    assert getNBdisp(g2) == pytest.approx(2 * th)
    assert getNBdisp(g1) == th


def test_est_theta(g1):
    t = est_theta(g1, limit=20, eps=1e-4)
    assert t > 0 and np.isfinite(t.SE)


def test_start_vals(g1):
    # GH 423
    g2 = glmer_nb(FORM, grouseticks, initCtrl=dict(theta=getNBdisp(g1)))
    assert np.allclose(g1.fixef().values, g2.fixef().values, rtol=1e-4, atol=1e-5)


def test_control_arguments():
    sub = grouseticks.iloc[:200]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = glmer_nb(FORM, sub, initCtrl=dict(theta=10))
        assert type(g).__name__ == "glmerMod"
        g = glmer_nb(FORM, sub, nb_control=glmerControl(optimizer="bobyqa"))
        assert g.optinfo["optimizer"] == "bobyqa"
        g = glmer_nb(FORM, sub, nb_control=glmerControl(optCtrl=dict(maxfun=2)))
        assert g.optinfo["feval"] == 3


def test_subset_offset():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = glmer_nb(FORM, grouseticks, subset="YEAR != '97'")
        assert g.nobs() == int((grouseticks["YEAR"] != "97").sum())
        g = glmer_nb("TICKS ~ YEAR + (1|LOCATION)", grouseticks,
                     offset=np.zeros(len(grouseticks)))
        assert np.isfinite(g.logLik().value)


def test_glmer_nb_singular_message_once(capfd):
    # R's glmer.nb() runs the initial fits under suppressMessages() and records the
    # messages of each refit (factory()), re-emitting only those of the last fit:
    # the singular-fit message of the ?glmer.nb example appears exactly once.
    # Data: a small deterministic set with no grouping variance.
    rng = np.random.default_rng(1)
    import pandas as pd
    d = pd.DataFrame({"x": np.tile(np.arange(6.0), 20),
                      "g": pd.Categorical(np.repeat(np.arange(10), 12).astype(str))})
    d["y"] = rng.negative_binomial(0.8, 0.8 / (0.8 + np.exp(1 + 0.2 * d["x"])))
    capfd.readouterr()
    m = glmer_nb("y ~ x + (1|g)", data=d)
    err = capfd.readouterr().err
    assert m.isSingular()
    assert err.count("boundary (singular) fit") == 1
