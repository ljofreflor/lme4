"""Ports of lme4/tests/testthat/test-allFit.R."""
import warnings

import numpy as np
import pytest

from lme4 import glmer, lmer, lmerControl, glmerControl, load_dataset
from lme4.allfit import allFit, AllFit

cbpp = load_dataset("cbpp")
sleepstudy = load_dataset("sleepstudy")
NAMES = ["bobyqa", "Nelder_Mead", "nlminbwrap", "nloptwrap.NLOPT_LN_NELDERMEAD",
         "nloptwrap.NLOPT_LN_BOBYQA"]


@pytest.fixture(scope="module")
def gm1():
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp,
                 family="binomial")


@pytest.fixture(scope="module")
def gm_all(gm1):
    return allFit(gm1, verbose=False)


def test_allfit_basic(gm_all, gm1):
    assert isinstance(gm_all, AllFit)
    assert list(gm_all) == NAMES
    s = gm_all.summary()
    assert s["which_OK"].all()
    assert np.allclose(s["llik"], gm1.logLik().value, atol=1e-5)
    assert np.allclose(s["fixef"].values, gm1.fixef().values[None, :], atol=1e-3)
    assert list(s["sdcor"].columns) == ["herd.(Intercept)"]
    assert s["theta"].shape == (5, 1)
    assert "differences in negative log-likelihoods" in str(gm_all)


def test_nloptwrap_switches_optimizer(gm_all):
    assert gm_all["nloptwrap.NLOPT_LN_BOBYQA"].optCtrl == dict(maxeval=1e5,
                                                              algorithm="NLOPT_LN_BOBYQA")
    assert gm_all["nloptwrap.NLOPT_LN_NELDERMEAD"].optCtrl == dict(
        maxeval=1e5, algorithm="NLOPT_LN_NELDERMEAD")
    assert gm_all["bobyqa"].optinfo["optimizer"] == "bobyqa"


def test_nostart(gm1):
    a = allFit(gm1, verbose=False, start_from_mle=False)
    assert a.summary()["which_OK"].all()


def test_lmerControl_arg():
    fm0 = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    fm = fm0.update(control=lmerControl(optCtrl=dict(xtol_rel=1e-8, ftol_rel=1e-8),
                                        calc_derivs=False))
    s0 = allFit(fm0, verbose=False).summary()
    s1 = allFit(fm, verbose=False).summary()
    for k in ("fixef", "sdcor", "theta"):
        assert np.allclose(s0[k].values, s1[k].values, rtol=1e-2)
    assert np.allclose(s0["llik"].values, s1["llik"].values, rtol=1e-2)
    assert list(s0["sdcor"].columns)[-1] == "sigma"


def test_glmerControl_arg(gm1):
    u = gm1.update(control=glmerControl(optimizer="nloptwrap",
                                        optCtrl=dict(xtol_abs=1e-10, ftol_abs=1e-10)))
    af = allFit(u, verbose=False)
    assert isinstance(af, AllFit)


def test_control_variable():
    cc = lmerControl(optimizer="bobyqa", optCtrl=dict(maxfun=1e5))
    fm = lmer("Reaction ~ Days + (1|Subject)", sleepstudy, control=cc)
    aa = allFit(fm, verbose=False)
    assert aa.summary()["which_OK"].all()


def test_maxfun(gm1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = allFit(gm1, verbose=False, maxfun=10)
    v = a.summary()["feval"]
    assert all(np.isnan(x) or x < 12 for x in v)


def test_show_meth_tab(gm1):
    mt = allFit(gm1, show_meth_tab=True)
    assert list(mt["optimizer"]) == ["bobyqa", "Nelder_Mead", "nlminbwrap", "nloptwrap",
                                     "nloptwrap"]
