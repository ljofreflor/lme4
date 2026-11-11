"""Tests for nlmer(), following lme4/tests/nlmer.R (+ nlmer.Rout.save),
lme4/tests/testthat/test-nlmer.R, test-rank.R ("nlmerRank") and the
examples of man/nlmer.Rd."""
import numpy as np
import pandas as pd
import pytest

from lme4 import nlmerControl
from lme4.modular import DevFun
from lme4.nlmer import nlmer, selfStart_models


def _orange():
    """datasets::Orange"""
    circ = {1: [30, 58, 87, 115, 120, 142, 145], 2: [33, 69, 111, 156, 172, 203, 203],
            3: [30, 51, 75, 108, 115, 139, 140], 4: [32, 62, 112, 167, 179, 209, 214],
            5: [30, 49, 81, 125, 142, 174, 177]}
    age = [118, 484, 664, 1004, 1231, 1372, 1582]
    return pd.DataFrame({
        "Tree": pd.Categorical([str(t) for t in range(1, 6) for _ in age],
                               categories=["3", "1", "5", "2", "4"], ordered=True),
        "age": np.array(age * 5, float),
        "circumference": np.array([c for t in range(1, 6) for c in circ[t]], float)})


Orange = _orange()
startvec = dict(Asym=200, xmid=725, scal=350)
FORM = "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree"


def allEQ(x, y, tolerance=4e-4):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.mean(np.abs(x - y)) / np.mean(np.abs(y)) < tolerance


@pytest.fixture(scope="module")
def nm1():
    return nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ (Asym|Tree)", Orange,
                 start=startvec)


def test_nm1_matches_Rout_save(nm1):
    # fixef(nm1) in nlmer.Rout.save
    assert allEQ(nm1.fixef().values, [192.0528, 727.9045, 348.0721])
    assert list(nm1.fixef().index) == ["Asym", "xmid", "scal"]
    np.testing.assert_allclose(nm1.fixef().values, [192.0528, 727.9045, 348.0721], rtol=2e-5)
    # AIC / BIC / logLik / -2*log(L) / df.resid
    assert nm1.AIC() == pytest.approx(273.1438, abs=1e-3)
    assert nm1.BIC() == pytest.approx(280.9205, abs=1e-3)
    assert nm1.logLik().value == pytest.approx(-131.5719, abs=1e-3)
    assert nm1.devcomp["cmp"]["dev"] == pytest.approx(263.1438, abs=1e-3)
    assert nm1.df_residual() == 30
    # random effects: Tree Asym 31.646, Residual 7.843
    vc = nm1.VarCorr()
    assert nm1.sigma() == pytest.approx(7.843, abs=5e-4)
    assert nm1.theta[0] * nm1.sigma() == pytest.approx(31.646, abs=5e-3)
    assert vc is not None
    assert nm1.nobs() == 35 and int(nm1.ngrps()["Tree"]) == 5
    assert nm1.isNLMM() and not nm1.isGLMM() and not nm1.isLMM()


def test_nm1_print(nm1):
    s = str(nm1)
    assert "Nonlinear mixed model fit by maximum likelihood" in s
    assert "['nlmerMod']" in s
    assert "Formula: circumference ~ SSlogis(age, Asym, xmid, scal) ~ (Asym | Tree)" in s
    assert "Data: Orange" in s
    assert "273.1438" in s and "280.9205" in s and "-131.5719" in s and "263.1438" in s
    assert "31.646" in s and "7.843" in s
    assert "Number of obs: 35, groups:  Tree, 5" in s
    assert "192.1" in s and "727.9" in s and "348.1" in s


def test_nm1_dims(nm1):
    dims = nm1.devcomp["dims"]
    assert dims["N"] == 105 and dims["n"] == 35 and dims["p"] == 3 and dims["q"] == 5
    assert dims["NLMM"] == 1 and dims["nAGQ"] == 1
    # ranef sums close to zero; ordering follows the levels of Tree (3<1<5<2<4)
    re = nm1.ranef()["Tree"]
    assert list(re.index) == ["3", "1", "5", "2", "4"]
    assert np.all(np.diff(re["Asym"].values) > 0)


def test_bounds_testthat():
    """testthat/test-nlmer.R: lower bound on the fixed effect Asym"""
    nm = nlmer(FORM, Orange, start=startvec,
               control=nlmerControl(optCtrl=dict(lower=[0, 200, -np.inf, -np.inf])))
    assert nm.fixef().iloc[0] == pytest.approx(200)


def test_nAGQ0():
    nm = nlmer(FORM, Orange, start=startvec, nAGQ=0)
    assert nm.devcomp["dims"]["nAGQ"] == 0
    assert allEQ(nm.fixef().values, [192.0528, 727.9045, 348.0721], tolerance=2e-2)
    assert nm.sigma() == pytest.approx(7.84, abs=0.05)


def test_user_functions(nm1):
    def nfun2(input, Asym, xmid, scal):
        e = np.exp((xmid - input) / scal)
        value = Asym / (1 + e)
        grad = np.column_stack([1 / (1 + e), -Asym / (1 + e) ** 2 / scal * e,
                                -Asym / (1 + e) ** 2 * -(xmid - input) / scal ** 2 * e])
        return value, grad

    nmc = nlmer("circumference ~ nfun2(age, Asym, xmid, scal) ~ Asym | Tree", Orange,
                start=startvec)
    np.testing.assert_allclose(nmc.fixef().values, nm1.fixef().values, rtol=1e-6)

    def nfun_named(input, Asym, xmid, scal):
        v, g = nfun2(input, Asym, xmid, scal)
        return v, pd.DataFrame(g, columns=["Asym", "xmid", "scal"])
    nmd = nlmer("circumference ~ nfun_named(age, Asym, xmid, scal) ~ Asym | Tree", Orange,
                start=startvec)
    np.testing.assert_allclose(nmd.fixef().values, nm1.fixef().values, rtol=1e-6)

    def mymodel(params, data):
        return nfun2(data["age"], params["Asym"], params["xmid"], params["scal"])
    nme = nlmer("circumference ~ mymodel ~ Asym | Tree", Orange, start=startvec)
    np.testing.assert_allclose(nme.fixef().values, nm1.fixef().values, rtol=1e-6)


def test_start_forms_and_order(nm1):
    nm = nlmer(FORM, Orange, start=pd.Series(startvec))
    np.testing.assert_allclose(nm.fixef().values, nm1.fixef().values, rtol=1e-10)
    nm = nlmer(FORM, Orange, start={"nlpars": startvec})
    np.testing.assert_allclose(nm.fixef().values, nm1.fixef().values, rtol=1e-10)
    # a different order of 'start' only permutes the fixed effects
    nm = nlmer(FORM, Orange, start=dict(xmid=725, Asym=200, scal=350))
    assert list(nm.fixef().index) == ["xmid", "Asym", "scal"]
    assert allEQ(nm.fixef()[["Asym", "xmid", "scal"]].values, nm1.fixef().values, 1e-4)


def test_us_equals_default(nm1):
    nm = nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ us(Asym|Tree)", Orange,
               start=startvec)
    np.testing.assert_allclose(nm.fixef().values, nm1.fixef().values, rtol=1e-8)
    assert nm.sigma() == pytest.approx(nm1.sigma(), rel=1e-8)


def test_vector_valued_RE():
    nm = nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ (Asym + xmid | Tree)",
               Orange, start=startvec)
    assert len(nm.theta) == 3
    assert nm.logLik().df == 7


def test_devFunOnly():
    df = nlmer(FORM, Orange, start=startvec, devFunOnly=True)
    assert isinstance(df, DevFun)
    assert len(df.rho.lower) == 4
    df0 = nlmer(FORM, Orange, start=startvec, devFunOnly=True, nAGQ=0)
    assert np.isfinite(df0([1.0]))


def test_formula_errors():
    with pytest.raises(ValueError, match="1 is not meaningful"):
        nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ 1|Tree", Orange, start=startvec)
    with pytest.raises(ValueError, match="pnames"):
        nlmer(FORM, Orange, start=dict(Asym=200, xmid=725, foo=350))
    with pytest.raises(ValueError):
        nlmer(FORM, Orange)


def test_nlmerRank():
    """test-rank.R 'nlmerRank' (data simulated with numpy instead of R's RNG)"""
    rng = np.random.default_rng(101)
    n = 1000
    x = np.abs(rng.normal(size=n))
    y = rng.normal(size=n)
    z = rng.normal(x ** y)
    r = rng.integers(1, 16, n)
    d = pd.DataFrame(dict(x=x, y=y, z=z, r=pd.Categorical(r)))

    def fModf(a, b):
        v = (np.exp(a) * x) ** (b * y)
        return v, {"a": v * b * y, "b": v * y * np.log(np.exp(a) * x)}

    def fModf2(a, b, c):
        v = (np.exp(a + c) * x) ** (b * y)
        return v, {"a": v * b * y, "b": v * y * np.log(np.exp(a + c) * x), "c": v * b * y}

    nlmer("y ~ fModf(a,b) ~ a|r", d, start=dict(a=1, b=1))
    with pytest.raises(Exception, match="Downdated VtV"):
        nlmer("y ~ fModf2(a,b,c) ~ a|r", d, start=dict(a=1, b=1, c=1))


_SS_ARGS = {
    "SSlogis": dict(input=np.linspace(100, 1600, 7), Asym=200., xmid=725., scal=350.),
    "SSasymp": dict(input=np.linspace(0, 10, 7), Asym=10., R0=1., lrc=-1.),
    "SSasympOff": dict(input=np.linspace(0, 10, 7), Asym=10., lrc=-1., c0=0.5),
    "SSasympOrig": dict(input=np.linspace(0, 10, 7), Asym=10., lrc=-1.),
    "SSmicmen": dict(input=np.linspace(0.02, 1.1, 7), Vm=200., K=0.05),
    "SSfol": dict(Dose=np.full(7, 4.0), input=np.linspace(0.25, 24, 7), lKe=-2.5, lKa=0.5,
                  lCl=-3.),
    "SSfpl": dict(input=np.linspace(0, 10, 7), A=1., B=10., xmid=4., scal=1.5),
    "SSgompertz": dict(x=np.linspace(0, 10, 7), Asym=10., b2=2., b3=0.7),
    "SSbiexp": dict(input=np.linspace(0.25, 8, 7), A1=3., lrc1=1., A2=0.6, lrc2=-1.3),
    "SSweibull": dict(x=np.linspace(0.5, 10, 7), Asym=10., Drop=8., lrc=-2., pwr=1.5),
}


@pytest.mark.parametrize("name", sorted(selfStart_models))
def test_selfStart_gradients(name):
    fun, formals, pars = selfStart_models[name]
    args = _SS_ARGS[name]
    val = fun(**args)
    assert val.gradient.shape == (7, len(pars))
    assert val.gradient_names == pars
    for j, p in enumerate(pars):
        h = 1e-6 * max(1.0, abs(args[p]))
        up, dn = dict(args), dict(args)
        up[p] += h
        dn[p] -= h
        num = (np.asarray(fun(**up)) - np.asarray(fun(**dn))) / (2 * h)
        np.testing.assert_allclose(val.gradient[:, j], num, rtol=1e-5, atol=1e-7)


def test_selfStart_values():
    v = selfStart_models["SSlogis"][0](np.array([725.0]), 200, 725, 350)
    assert float(v[0]) == pytest.approx(100.0)
    v = selfStart_models["SSmicmen"][0](np.array([1.0]), 2.0, 1.0)
    assert float(v[0]) == pytest.approx(1.0)
