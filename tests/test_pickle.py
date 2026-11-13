"""Fitted models survive pickle round trips (incl. the C++ predictor and
response modules) and give identical results afterwards."""
import copy
import io
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from lme4 import (lmer, glmer, glmer_nb, nlmer, load_dataset, binomial, poisson, Gamma, V,
                  profile_merMod)

sleepstudy = load_dataset("sleepstudy")
cbpp = load_dataset("cbpp")
grouseticks = load_dataset("grouseticks")


def _orange():
    circ = {1: [30, 58, 87, 115, 120, 142, 145], 2: [33, 69, 111, 156, 172, 203, 203],
            3: [30, 51, 75, 108, 115, 139, 140], 4: [32, 62, 112, 167, 179, 209, 214],
            5: [30, 49, 81, 125, 142, 174, 177]}
    age = [118, 484, 664, 1004, 1231, 1372, 1582]
    return pd.DataFrame({
        "Tree": pd.Categorical([str(t) for t in range(1, 6) for _ in age],
                               categories=["3", "1", "5", "2", "4"], ordered=True),
        "age": np.array(age * 5, float),
        "circumference": np.array([c for t in range(1, 6) for c in circ[t]], float)})


def roundtrip(m):
    return pickle.loads(pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL))


def _fits():
    Reaction, Days, Subject = V("Reaction Days Subject")
    gamma_d = sleepstudy.assign(Reaction=sleepstudy["Reaction"] / 100)
    return {
        "lmer_REML": lambda: lmer("Reaction ~ Days + (Days | Subject)", sleepstudy),
        "lmer_ML": lambda: lmer("Reaction ~ Days + (Days || Subject)", sleepstudy, REML=False),
        "lmer_symbolic": lambda: lmer(Reaction == Days + (Days | Subject), data=sleepstudy),
        "glmer_binomial": lambda: glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)",
                                        cbpp, family=binomial),
        "glmer_AGQ": lambda: glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)",
                                   cbpp, family=binomial, nAGQ=9),
        "glmer_poisson": lambda: glmer("TICKS ~ YEAR + (1|BROOD)", grouseticks, family=poisson),
        "glmer_Gamma": lambda: glmer("Reaction ~ Days + (1|Subject)", gamma_d,
                                     family=Gamma(link="log")),
        "glmer_nb": lambda: glmer_nb("TICKS ~ YEAR + cHEIGHT + (1|LOCATION)", grouseticks),
        "nlmer": lambda: nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
                               _orange(), start=dict(Asym=200, xmid=725, scal=350)),
    }


FITS = _fits()


@pytest.fixture(scope="module", params=list(FITS))
def pair(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = FITS[request.param]()
    return request.param, m, roundtrip(m)


def _same(a, b):
    if isinstance(a, (pd.DataFrame, pd.Series)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert list(a.index) == list(b.index)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_results_identical(pair):
    name, m, m2 = pair
    assert type(m2) is type(m)
    _same(m.fixef(), m2.fixef())
    _same(m.getME("theta"), m2.getME("theta"))
    _same(m.fitted(), m2.fitted())
    _same(m.residuals(), m2.residuals())
    _same(np.asarray(m.vcov()), np.asarray(m2.vcov()))
    assert float(m.logLik()) == float(m2.logLik())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert m.deviance() == m2.deviance()
    for g, re in m.ranef().items():
        _same(re, m2.ranef()[g])
    assert str(m.summary()) == str(m2.summary())
    assert str(m) == str(m2)


def test_predictor_state_identical(pair):
    name, m, m2 = pair
    p, p2 = m.pp, m2.pp
    for k in ("ldL2", "ldRX2", "CcNumer", "sqrL"):
        a, b = getattr(p, k), getattr(p2, k)
        a, b = (a(1.0), b(1.0)) if k == "sqrL" else (a(), b())
        assert a == b, k
    np.testing.assert_array_equal(p.RX(), p2.RX())
    np.testing.assert_array_equal(np.asarray(p.RZX), np.asarray(p2.RZX))
    np.testing.assert_array_equal(p.L().L.toarray(), p2.L().L.toarray())
    np.testing.assert_array_equal(p.Pvec(), p2.Pvec())
    np.testing.assert_array_equal(np.asarray(p.Vtr), np.asarray(p2.Vtr))
    np.testing.assert_array_equal(np.asarray(m.resp.wtres), np.asarray(m2.resp.wtres))
    assert m.resp.wrss() == m2.resp.wrss()


def test_methods_after_unpickle(pair):
    name, m, m2 = pair
    if name == "nlmer":
        _same(m.predict(), m2.predict())
        _same(m.residuals(type="pearson"), m2.residuals(type="pearson"))
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same(m.predict(), m2.predict())
        _same(m.simulate(nsim=2, seed=101), m2.simulate(nsim=2, seed=101))
        # refitting works from the restored C++ modules
        r1, r2 = m.refit(), m2.refit()
    np.testing.assert_allclose(r1.fixef(), r2.fixef(), rtol=1e-10)


def test_update_after_unpickle():
    m2 = roundtrip(lmer("Reaction ~ Days + (Days | Subject)", sleepstudy))
    u = m2.update(". ~ . - Days")
    ref = lmer("Reaction ~ (Days | Subject)", sleepstudy)
    np.testing.assert_allclose(u.fixef(), ref.fixef(), rtol=1e-10)


def test_formula_env_is_pruned():
    def my_fun(x):        # local function: cannot be pickled, so it is dropped
        return x
    m = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    m.call["formula"].env = dict(io=io, Days=1.0, my_fun=my_fun, unrelated=np.arange(10 ** 6))
    m2 = roundtrip(m)
    assert m2.call["formula"].env == {"Days": 1.0}


def test_deepcopy():
    m = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    m2 = copy.deepcopy(m)
    assert m2.pp.ptr is not m.pp.ptr
    assert str(m2.summary()) == str(m.summary())


def test_load_in_new_process(tmp_path):
    m = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=binomial)
    f = tmp_path / "gm1.pkl"
    f.write_bytes(pickle.dumps(m))
    code = ("import pickle,sys; m=pickle.load(open(sys.argv[1],'rb')); "
            "print(repr(float(m.logLik()))); print(m.predict(type='response')[:3].tolist())")
    import lme4, os
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(lme4.__file__)))
    out = subprocess.run([sys.executable, "-c", code, str(f)], capture_output=True, text=True,
                         check=True, env=env).stdout.splitlines()
    assert float(out[0]) == float(m.logLik())
    assert out[1] == str(m.predict(type="response")[:3].tolist())


def test_profile_pickles():
    m = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
    pr = profile_merMod(m)
    pr2 = roundtrip(pr)
    np.testing.assert_array_equal(np.asarray(pr.confint()), np.asarray(pr2.confint()))
