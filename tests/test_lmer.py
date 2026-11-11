"""Ports of lme4/tests/testthat/test-lmer.R (numeric references from lme4 2.1-0)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from lme4 import lmer, lmerControl, load_dataset, V, getME, gaussian

Dyestuff = load_dataset("Dyestuff")
Dyestuff2 = load_dataset("Dyestuff2")
sleepstudy = load_dataset("sleepstudy")


def rel(a, b):
    return abs(a - b) / abs(b)


@pytest.fixture(scope="module")
def fm1():
    return lmer("Yield ~ 1|Batch", Dyestuff)


def test_dyestuff_basic(fm1):
    assert fm1.fixef().values == pytest.approx([1527.5])
    assert rel(fm1.VarCorr()["Batch"].cov[0, 0], 1764.0375195) < 1e-5
    assert fm1.isREML()
    REMLfun = fm1.as_function()
    assert REMLfun(np.array([1.0])) == pytest.approx(319.792389042002, rel=1e-10)
    assert REMLfun(np.array([0.0])) == pytest.approx(326.023232155879, rel=1e-10)
    assert fm1.family() == gaussian()
    assert fm1.REMLcrit() == pytest.approx(319.654276842342, rel=1e-8)
    assert rel(fm1.sigma(), 49.5101272946856) < 1e-6
    assert rel(fm1.vcov().iloc[0, 0], 375.714676744) < 1e-5
    th = getME(fm1, "theta")
    assert rel(th.values[0], 0.84832031) < 6e-6
    assert list(th.index) == ["Batch.(Intercept)"]
    Zt = getME(fm1, "Zt")
    assert Zt.shape == (6, 30)
    assert np.all(Zt.data == 1)
    Lt = getME(fm1, "Lambdat").toarray()
    assert np.allclose(Lt, np.diag(np.repeat(th.values[0], 6)))
    assert np.allclose(getME(fm1, "X"), np.ones((30, 1)))


def test_dyestuff_ML(fm1):
    fm1ML = fm1.refitML()
    assert not fm1ML.isREML()
    assert fm1ML.deviance() == pytest.approx(327.327059881135, rel=1e-7)
    assert rel(fm1ML.sigma(), 49.5100999308089) < 1e-6
    df, aic = fm1.extractAIC()
    assert df == 3 and aic == pytest.approx(333.327059881135, rel=1e-7)
    assert rel(fm1ML.vcov().iloc[0, 0], 313.09721874266) < 1e-6


def test_refit_singular(fm1):
    fm2 = fm1.refit(Dyestuff2["Yield"].values)
    assert fm2.fixef().values == pytest.approx([5.6656])
    assert fm2.VarCorr()["Batch"].cov[0, 0] == pytest.approx(0, abs=1e-10)
    assert getME(fm2, "theta").values == pytest.approx([0], abs=1e-10)
    assert fm2.isSingular()


def test_sleepstudy_reference_values():
    fm = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    assert fm.REMLcrit() == pytest.approx(1743.6282720, rel=1e-8)
    assert fm.fixef().values == pytest.approx([251.405104848485, 10.467285959596], rel=1e-8)
    vc = fm.VarCorr()["Subject"]
    assert vc.stddev == pytest.approx([24.7404, 5.9221], rel=1e-3)
    assert vc.correlation[1, 0] == pytest.approx(0.0656, abs=2e-3)
    assert fm.sigma() == pytest.approx(25.5918, rel=1e-4)
    se = np.sqrt(np.diag(fm.vcov().values))
    assert se == pytest.approx([6.8245, 1.5458], rel=1e-3)


def test_two_terms_same_factor():
    fm3 = lmer("Reaction ~ Days + (1|Subject) + (0+Days|Subject)", sleepstudy)
    assert getME(fm3, "n_rtrms") == 2
    assert getME(fm3, "n_rfacs") == 1
    low = getME(fm3, "lower")
    assert list(low.index) == ["Subject.(Intercept)", "Subject.Days"]
    assert list(low.values) == [0, 0]


def test_symbolic_equals_string():
    Reaction, Days, Subject = V("Reaction Days Subject")
    a = lmer(Reaction == Days + (Days | Subject), data=sleepstudy)
    b = lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
    assert np.allclose(a.theta, b.theta)
    assert np.allclose(a.beta, b.beta)
    c = lmer(Reaction == Days + Days.dbar(Subject), data=sleepstudy)
    d = lmer("Reaction ~ Days + (Days || Subject)", data=sleepstudy)
    assert np.allclose(c.theta, d.theta)


def test_nlev_checks():
    one = sleepstudy[sleepstudy.Subject == sleepstudy.Subject.cat.categories[0]]
    with pytest.raises(ValueError, match="must have > 1"):
        lmer("Reaction ~ Days + (1|Subject)", one)
    four = sleepstudy[sleepstudy.Subject.isin(sleepstudy.Subject.cat.categories[:4])]
    with pytest.warns(UserWarning, match="< 5 sampled levels"):
        lmer("Reaction ~ Days + (1|Subject)", four,
             control=lmerControl(check_nlev_gtreq_5="warning"))


def test_nobs_vs_nRE():
    with pytest.raises(ValueError, match=r"number of observations \(=36\) <= number of random effects \(=36\)"):
        lmer("Reaction ~ 1 + Days + (1 + Days | Subject)", sleepstudy,
             subset="Days == 1 | Days == 9")


def test_scale_warning():
    ss = sleepstudy.copy()
    ss["Days"] = ss["Days"] * 1e6
    with pytest.warns(UserWarning, match="predictor variables are on very different scales"):
        lmer("Reaction ~ Days + (1|Subject)", ss)


@pytest.mark.parametrize("kw", [dict(REML=True), dict(start=None), dict(verbose=0),
                                dict(weights=np.ones(30)), dict(na_action="na.exclude"),
                                dict(offset=np.zeros(30)), dict(contrasts=None),
                                dict(control=lmerControl(optimizer="Nelder_Mead")),
                                dict(control=lmerControl())])
def test_argument_variants(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = lmer("Yield ~ 1|Batch", Dyestuff, **kw)
    assert m.fixef().values == pytest.approx([1527.5])


def test_bad_optimizer():
    with pytest.raises(ValueError, match="couldn't find optimizer function"):
        lmer("Yield ~ 1|Batch", Dyestuff, control=lmerControl(optimizer="junk"))


def test_rank_deficient_drop(capsys):
    d = sleepstudy.copy()
    d["Days2"] = 2 * d["Days"]
    m = lmer("Reaction ~ Days + Days2 + (1|Subject)", d)
    assert "dropping 1 column" in capsys.readouterr().err
    assert list(m.fixef().index) == ["(Intercept)", "Days"]
    assert list(m.fixef(add_dropped=True).index) == ["(Intercept)", "Days", "Days2"]


def test_summary_print_runs():
    fm = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    s = str(fm.summary())
    assert "REML criterion at convergence: 1743.6" in s
    assert "251.405" in s and "36.838" in s
    assert "Number of obs: 180, groups:  Subject, 18" in s


def test_ranef_condvar():
    fm = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    re = fm.ranef()
    sub = re["Subject"]
    assert sub.shape == (18, 2)
    pv = sub.attrs["postVar"]
    assert pv.shape == (2, 2, 18)
    assert np.all(pv[0, 0] > 0)
    # coef = fixef + ranef
    cf = fm.coef()["Subject"]
    assert np.allclose(cf["Days"].values, fm.fixef()["Days"] + sub["Days"].values)


def test_crossed_and_nested():
    Penicillin = load_dataset("Penicillin")
    fm = lmer("diameter ~ 1 + (1|plate) + (1|sample)", Penicillin)
    # lme4 reference: REML criterion 330.8606, sd plate 0.8467, sample 1.9316, sigma 0.5499
    assert fm.REMLcrit() == pytest.approx(330.8606, abs=1e-3)
    sds = {k: v.stddev[0] for k, v in fm.VarCorr().items()}
    assert sds["plate"] == pytest.approx(0.8467, abs=1e-3)
    assert sds["sample"] == pytest.approx(1.9316, abs=1e-3)
    assert fm.sigma() == pytest.approx(0.5499, abs=1e-3)
    Pastes = load_dataset("Pastes")
    fp = lmer("strength ~ 1 + (1|batch/cask)", Pastes)
    assert set(fp.flist) == {"cask:batch", "batch"}
    # lme4 reference: REML 246.99, variances cask:batch 8.434, batch 1.657, residual 0.678
    assert fp.REMLcrit() == pytest.approx(246.9907, abs=1e-3)
    vc = {k: v.cov[0, 0] for k, v in fp.VarCorr().items()}
    assert vc["cask:batch"] == pytest.approx(8.434, abs=2e-3)
    assert vc["batch"] == pytest.approx(1.657, abs=2e-3)
    assert fp.sigma() ** 2 == pytest.approx(0.678, abs=1e-3)
