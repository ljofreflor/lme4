"""Profile/confint, predict, glmer.nb, lmList, nlmer and anova compared with
R's lme4 2.1.0 (tests/reference/extras.json, from make_extras.R)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lme4 import lmer, glmer, nlmer, binomial, load_dataset

REF = json.loads((Path(__file__).parent / "reference" / "extras.json").read_text(encoding="utf-8"))
sleepstudy = load_dataset("sleepstudy")
cbpp = load_dataset("cbpp")


@pytest.fixture(scope="module")
def fm1():
    return lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)


@pytest.fixture(scope="module")
def gm1():
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=binomial)


def _cmp_ci(py, r, rtol):
    assert list(py.index) == r["rows"]
    num = lambda v: np.array([np.nan if x in ("NA", None) else x for x in v], float)
    np.testing.assert_allclose(py.iloc[:, 0].values, num(r["lower"]), rtol=rtol, equal_nan=True)
    np.testing.assert_allclose(py.iloc[:, 1].values, num(r["upper"]), rtol=rtol, equal_nan=True)


def test_confint_profile_sleep(fm1):
    _cmp_ci(fm1.confint(method="profile"), REF["confint_sleep_profile"], 1e-3)


def test_confint_wald_sleep(fm1):
    _cmp_ci(fm1.confint(method="Wald"), REF["confint_sleep_wald"], 1e-6)


def test_confint_profile_dyestuff():
    fd = lmer("Yield ~ 1 | Batch", load_dataset("Dyestuff"))
    _cmp_ci(fd.confint(), REF["confint_dyestuff_prof"], 1e-3)


def test_confint_profile_cbpp(gm1):
    _cmp_ci(gm1.confint(), REF["confint_cbpp_profile"], 2e-3)


def test_predict(fm1, gm1):
    nd = pd.DataFrame({"Days": [0, 2.5, 10],
                       "Subject": pd.Categorical(["308", "335", "999"])})
    np.testing.assert_allclose(fm1.predict(nd, allow_new_levels=True).values,
                               REF["predict_sleep_newlevel"], rtol=1e-6)
    np.testing.assert_allclose(fm1.predict(nd, re_form="NA").values,
                               REF["predict_sleep_reNA"], rtol=1e-6)
    np.testing.assert_allclose(gm1.predict(type="response").values,
                               REF["predict_cbpp_resp"], rtol=1e-5)
    nd2 = pd.DataFrame({"period": pd.Categorical(["1", "4"], categories=["1", "2", "3", "4"]),
                        "herd": pd.Categorical(["1", "15"], categories=[str(i) for i in range(1, 16)])})
    np.testing.assert_allclose(gm1.predict(nd2, type="response").values,
                               REF["predict_cbpp_newdata"], rtol=1e-5)


def test_glmer_nb():
    from lme4.nbinom import glmer_nb
    g = glmer_nb("TICKS ~ YEAR + (1 | BROOD)", load_dataset("grouseticks"))
    r = REF["nb"]
    assert g.getME("glmer.nb.theta") == pytest.approx(r["theta"], rel=1e-3)
    np.testing.assert_allclose(g.fixef().values, r["fixef"], rtol=1e-3)
    assert g.logLik().value == pytest.approx(r["logLik"], rel=1e-6)


def test_lmList():
    from lme4.lmlist import lmList
    ll = lmList("Reaction ~ Days | Subject", sleepstudy)
    r = REF["lmList"]
    cf = ll.coef()
    assert [str(x) for x in cf.index] == r["rows"]
    np.testing.assert_allclose(cf.values, np.array(r["coef"]), rtol=1e-8)


def test_nlmer_orange():
    Orange = pd.DataFrame({
        "Tree": pd.Categorical(np.repeat(["1", "2", "3", "4", "5"], 7),
                               categories=["3", "1", "5", "2", "4"], ordered=True),
        "age": np.tile([118, 484, 664, 1004, 1231, 1372, 1582], 5).astype(float),
        "circumference": [30, 58, 87, 115, 120, 142, 145, 33, 69, 111, 156, 172, 203, 203,
                          30, 51, 75, 108, 115, 139, 140, 32, 62, 112, 167, 179, 209, 214,
                          30, 49, 81, 125, 142, 174, 177]})
    nm1 = nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym | Tree", Orange,
                start=dict(Asym=200, xmid=725, scal=350))
    r = REF["nlmer"]
    assert nm1.logLik().value == pytest.approx(r["logLik"], rel=1e-7)
    np.testing.assert_allclose(nm1.fixef().values, r["fixef"], rtol=1e-4)
    np.testing.assert_allclose(nm1.theta, np.atleast_1d(r["theta"]), rtol=1e-3)
    assert nm1.sigma() == pytest.approx(r["sigma"], rel=1e-4)


def test_anova(fm1):
    f0 = lmer("Reaction ~ 1 + (Days | Subject)", sleepstudy)
    a = f0.anova(fm1)
    r = REF["anova"]
    assert a["Chisq"].iloc[1] == pytest.approx(r["Chisq"], rel=1e-5)
    assert a["Df"].iloc[1] == r["Df"]
    assert a["Pr(>Chisq)"].iloc[1] == pytest.approx(r["p"], rel=1e-3)
    np.testing.assert_allclose(a["AIC"].values, r["AIC"], rtol=1e-8)


def test_drop1():
    cake = load_dataset("cake")
    m = lmer("angle ~ recipe + temperature + (1 | recipe:replicate)", cake)
    d = m.drop1(test="Chisq")
    r = REF["drop1"]
    num = lambda v: np.array([np.nan if x in ("NA", None) else x for x in v], float)
    assert list(d.index) == r["rows"]
    np.testing.assert_allclose(d["npar"].values, num(r["npar"]), equal_nan=True)
    np.testing.assert_allclose(d["AIC"].values, r["AIC"], rtol=1e-8)
    np.testing.assert_allclose(d["LRT"].values, num(r["LRT"]), rtol=1e-5, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(d["Pr(Chi)"].values, num(r["p"]), rtol=1e-4, equal_nan=True)
