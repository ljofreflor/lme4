"""Tests for lme4.bootmer (bootMer, refit, boot confidence intervals), ported
from lme4/tests/bootMer.R, tests/confint.R and tests/testthat/test-methods.R.
Bootstrap replicates use numpy's RNG, so R's seeded numbers are not
reproducible; checks are structural or statistical."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lme4 import lmer, glmer, load_dataset
from lme4.bootmer import (bootMer, refit_merMod, boot_ci, norm_inter, _simulate_fallback)
from lme4.profile import confint_merMod


def mySumm(m):
    s = m.sigma()
    return pd.Series(np.concatenate([m.beta, [s], s * m.theta]),
                     index=[f"beta{i + 1}" for i in range(len(m.beta))] + ["sigma"] +
                     [f"sig{i + 1:02d}" for i in range(len(m.theta))])


@pytest.fixture(scope="module")
def dyefit():
    return lmer("Yield ~ 1|Batch", load_dataset("Dyestuff"))


@pytest.fixture(scope="module")
def fm2():
    return lmer("Reaction ~ Days + (Days|Subject)", load_dataset("sleepstudy"))


@pytest.fixture(scope="module")
def gm1():
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)",
                 load_dataset("cbpp"), family="binomial")


def test_refit_same_response(dyefit, fm2, gm1):
    for m in (dyefit, fm2):
        r = refit_merMod(m, m.resp.y)
        np.testing.assert_allclose(r.theta, m.theta, rtol=1e-4)
        np.testing.assert_allclose(r.beta, m.beta, rtol=1e-6)
        assert r.isREML() == m.isREML()
    cb = load_dataset("cbpp")
    r = refit_merMod(gm1, np.column_stack([cb["incidence"], cb["size"] - cb["incidence"]]))
    np.testing.assert_allclose(r.theta, gm1.theta, rtol=1e-3)
    np.testing.assert_allclose(r.beta, gm1.beta, rtol=1e-4)
    # proportions with the original weights give the same fit
    r2 = refit_merMod(gm1, gm1.resp.y)
    np.testing.assert_allclose(r2.beta, r.beta, rtol=1e-6)


def test_bootMer_basic(dyefit):
    boo01 = bootMer(dyefit, mySumm, nsim=10, seed=101)
    assert boo01.t.shape == (10, 3)
    assert list(boo01.t0.index) == ["beta1", "sigma", "sig01"]
    assert boo01.bootFail == 0
    boo01b = bootMer(dyefit, mySumm, nsim=10, seed=101)
    np.testing.assert_array_equal(boo01.t, boo01b.t)
    boo02 = bootMer(dyefit, mySumm, nsim=10, use_u=True, seed=101)
    assert boo02.t.shape == (10, 3)
    assert boo02.as_data_frame().shape == (10, 3)
    ci = boo02.confint(parm=1)
    assert list(ci.index) == ["sigma"] and list(ci.columns) == ["2.5 %", "97.5 %"]
    fm2c = lmer("angle ~ recipe * temperature + (1|recipe:replicate)", load_dataset("cake"))
    k = len(fm2c.beta) + 2
    assert bootMer(fm2c, mySumm, nsim=3, seed=1).t.shape == (3, k)
    assert bootMer(fm2c, mySumm, nsim=3, use_u=True, seed=1).t.shape == (3, k)


def test_bootMer_parametric_distribution(dyefit):
    # the parametric bootstrap of the intercept has sd close to its standard error
    b = bootMer(dyefit, lambda m: m.fixef(), nsim=300, seed=1)
    se = np.sqrt(dyefit.vcov().values[0, 0])
    assert abs(np.std(b.t[:, 0], ddof=1) / se - 1) < 0.2
    # use_u=True: only residual noise => much smaller spread
    bu = bootMer(dyefit, lambda m: m.fixef(), nsim=100, seed=1, use_u=True)
    assert np.std(bu.t[:, 0], ddof=1) < np.std(b.t[:, 0], ddof=1)


def test_bootMer_semiparametric(dyefit):
    # test-methods.R: sd(boo01.sp$t) == 8.215586 (R's RNG); here: same ballpark
    b = bootMer(dyefit, lambda m: m.fixef(), nsim=100, use_u=True, type="semiparametric",
                seed=1)
    assert 5.5 < np.std(b.t, ddof=1) < 11
    with pytest.raises(NotImplementedError):
        bootMer(dyefit, lambda m: m.fixef(), nsim=2, type="semiparametric")


def test_bootMer_glmm(gm1):
    f = lambda m: np.concatenate([m.beta, m.theta])
    b5 = bootMer(gm1, f, nsim=5, seed=2)
    b6 = bootMer(gm1, f, nsim=5, use_u=True, seed=2)
    assert b5.t.shape == (5, 5) and b6.t.shape == (5, 5)
    assert not np.isnan(b5.t).any()


def test_bootMer_failures(dyefit):
    calls = {"n": 0}

    def bad(m):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("boom")
        return m.fixef()
    with pytest.warns(UserWarning, match="some bootstrap runs failed"):
        b = bootMer(dyefit, bad, nsim=3, seed=1)
    assert b.bootFail == 3
    assert np.isnan(b.t).all()
    assert dict(b.boot_fail_msgs) == {"boom": 3}


def test_bootMer_re_form_predict():
    fm3 = lmer("strength ~ (1|batch/cask)", load_dataset("Pastes"))
    assert bootMer(fm3, lambda m: m.predict(), nsim=3, seed=1).t.shape == (3, 60)
    assert bootMer(fm3, lambda m: m.predict(), nsim=3, seed=1, use_u=True).t.shape == (3, 60)
    assert bootMer(fm3, lambda m: m.predict(), nsim=3, seed=1,
                   re_form="~(1|batch)").t.shape == (3, 60)
    c_name = confint_merMod(fm3, method="boot", FUN=lambda m: m.getME("theta"), nsim=3,
                            seed=101)
    c_noname = confint_merMod(fm3, method="boot",
                              FUN=lambda m: np.asarray(m.getME("theta")), nsim=3, seed=101)
    np.testing.assert_allclose(c_name.values, c_noname.values)


def test_confint_boot(fm2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bt in ("norm", "basic", "perc"):
            ci = confint_merMod(fm2, method="boot", boot_type=bt, nsim=10, quiet=True, seed=1)
            assert ci.shape == (6, 2)
            assert list(ci.index) == [".sig01", ".sig02", ".sig03", ".sigma", "(Intercept)",
                                      "Days"]
        for bt in ("stud", "bca", "junk"):
            with pytest.raises(ValueError, match="should be one of"):
                confint_merMod(fm2, method="boot", boot_type=bt, nsim=10)
        ci = confint_merMod(fm2, method="boot", nsim=10, seed=101, quiet=True)
        ci2 = confint_merMod(fm2, method="boot", nsim=10, seed=101, quiet=True)
        pd.testing.assert_frame_equal(ci, ci2)
        ci50 = confint_merMod(fm2, method="boot", nsim=10, seed=101, quiet=True, level=0.5)
        assert list(ci50.columns) == ["25 %", "75 %"]
        assert np.all(ci50["25 %"].values > ci["2.5 %"].values)
        assert np.all(ci50["75 %"].values < ci["97.5 %"].values)
        assert np.all(np.abs(ci.loc[".sig03"].values) <= 1)
        cin = confint_merMod(fm2, method="boot", nsim=10, seed=1, quiet=True, signames=False)
        assert np.all(np.abs(cin.loc["cor_Days.(Intercept)|Subject"].values) <= 1)
        cb = confint_merMod(fm2, method="boot", nsim=10, seed=1, quiet=True, parm="beta_")
        assert list(cb.index) == ["(Intercept)", "Days"]


def test_confint_boot_vs_profile(fm2):
    # tests/confint.R: profile and bootstrap intervals for the fixed effects agree
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = confint_merMod(fm2, parm=[4, 5], quiet=True)
        c2 = confint_merMod(fm2, method="boot", nsim=200, parm=[4, 5], seed=101, quiet=True)
    assert np.mean(np.abs(c1.values - c2.values)) / np.mean(np.abs(c1.values)) < 2e-2


def test_boot_ci_formulas():
    t = np.arange(1.0, 200.0)                       # R = 199: (R+1)*0.025 = 5 exactly
    np.testing.assert_allclose(norm_inter(t, [0.025, 0.975])[:, 1], [5.0, 195.0])
    perc = boot_ci(100.0, t, 0.95, "perc")
    np.testing.assert_allclose(perc[3:], [5.0, 195.0])
    basic = boot_ci(100.0, t, 0.95, "basic")
    np.testing.assert_allclose(basic[3:], [200 - 195.0, 200 - 5.0])
    norm = boot_ci(100.0, t, 0.95, "norm")
    sd = np.std(t, ddof=1)
    q = stats.norm.ppf(0.975)
    np.testing.assert_allclose(norm[1:], [200 - t.mean() - sd * q, 200 - t.mean() + sd * q])
    # interpolation between order statistics (boot:::norm.inter)
    t2 = np.arange(1.0, 100.0)                      # (R+1)*0.025 = 2.5
    v = norm_inter(t2, [0.025])[0, 1]
    z = stats.norm.ppf
    np.testing.assert_allclose(v, 2 + (z(0.025) - z(0.02)) / (z(0.03) - z(0.02)))
    # constant t => no interval
    assert boot_ci(1.0, np.ones(10), 0.95, "perc") is None


def test_simulate_fallback(dyefit, gm1):
    rng = np.random.default_rng(1)
    s = _simulate_fallback(dyefit, 500, rng, use_u=False)
    sims = np.vstack(s)
    assert sims.shape == (500, 30)
    np.testing.assert_allclose(sims.mean(), dyefit.beta[0], rtol=0.01)
    s2 = _simulate_fallback(gm1, 2, rng, use_u=False)
    w = gm1.resp.weights
    assert np.allclose(np.round(s2[0] * w), s2[0] * w)
