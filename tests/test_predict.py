"""Tests for lme4.predict.predict_merMod, ported from lme4/tests/testthat/test-predict.R,
lme4/tests/predsim.R and lme4/tests/predict_basis.R (R is not available: numeric
references are those printed in the R tests for the same models)."""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from lme4 import lmer, glmer, load_dataset, binomial, poisson, lmerControl
from lme4.predict import predict_merMod, mkNewReTrms, levelfun, get_orig_levs, setParams, isRE, NA
from lme4.rlang import Formula


@pytest.fixture(scope="module")
def sleepstudy():
    return load_dataset("sleepstudy")


@pytest.fixture(scope="module")
def cbpp():
    return load_dataset("cbpp")


@pytest.fixture(scope="module")
def fm1(sleepstudy):
    return lmer("Reaction ~ Days + (1|Subject)", sleepstudy)


@pytest.fixture(scope="module")
def fm2(sleepstudy):
    return lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)


@pytest.fixture(scope="module")
def gm1(cbpp):
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=binomial)


@pytest.fixture(scope="module")
def fm3():
    return lmer("diameter ~ (1|plate) + (1|sample)", load_dataset("Penicillin"))


@pytest.fixture(scope="module")
def fm4():
    return lmer("angle ~ temp + recipe + (1 | replicate)", load_dataset("cake"))


def _all_equal(current, target, tol):
    """R's all.equal for numeric vectors: mean relative difference <= tol"""
    current, target = np.asarray(current, float), np.asarray(target, float)
    return np.mean(np.abs(target - current)) / np.mean(np.abs(target)) <= tol


def _cbpp_grid(cbpp):
    return pd.DataFrame(list(itertools.product(cbpp["herd"].unique(), cbpp["period"].unique())),
                        columns=["herd", "period"])[["period", "herd"]]


# ------------------------------------------------------------------ basics
def test_isRE():
    assert not isRE(NA)
    assert not isRE(Formula("~0"))
    assert isRE(Formula("~y+x"))
    assert isRE(None)
    assert isRE(Formula("~0+x"))


def test_lmm_fitted(fm2, sleepstudy):
    f = fm2.fitted()
    assert np.allclose(fm2.predict().values, f.values)
    assert list(fm2.predict().index) == list(f.index)
    assert np.allclose(fm2.predict(newdata=sleepstudy).values, f.values)
    assert np.allclose(fm2.predict(re_form="~(Days|Subject)").values, f.values)
    Xb = fm2.getME("X") @ fm2.beta
    for na in ("NA", np.nan, False, NA, "~0"):
        assert np.allclose(fm2.predict(re_form=na).values, Xb)


def test_glmm_fitted(gm1, cbpp):
    p0 = gm1.predict()
    assert np.allclose(p0.values, gm1.resp.eta)
    p0B = gm1.predict(newdata=cbpp)
    assert np.allclose(p0.values, p0B.values, atol=2e-5)
    # level-0
    p1 = gm1.predict(re_form="NA")
    assert len(np.unique(np.round(p1.values, 10))) == cbpp["period"].nunique()
    # random only
    p1R = gm1.predict(random_only=True)
    assert np.allclose(p1.values + p1R.values, p0.values)
    assert np.all(gm1.predict(re_form="NA", random_only=True).values == 0)
    # response scale
    p5 = gm1.predict(type="response")
    assert np.allclose(p5.values, 1 / (1 + np.exp(-p0.values)))
    assert np.allclose(p5.values, gm1.fitted().values)
    assert np.allclose(gm1.predict(newdata=cbpp, type="response").values, gm1.fitted().values)


def test_glmm_newdata(gm1, cbpp):
    nd = _cbpp_grid(cbpp)
    p2 = gm1.predict(nd)
    p3 = gm1.predict(nd, re_form="NA")
    p3R = gm1.predict(nd, random_only=True)
    assert np.allclose(p3.values + p3R.values, p2.values)
    p4 = gm1.predict(nd, re_form="~(1|herd)")
    assert np.allclose(p2.values, p4.values)
    assert list(p2.index) == list(nd.index)


def test_new_levels(gm1, cbpp):
    nd = _cbpp_grid(cbpp)
    nd2 = pd.concat([nd, pd.DataFrame({"period": ["1", "2", "3", "4"], "herd": ["new"] * 4})],
                    ignore_index=True)
    with pytest.raises(ValueError, match="new levels detected in newdata: new"):
        gm1.predict(nd2)
    nd3 = pd.concat([nd, pd.DataFrame({"period": "1", "herd": [f"new{i}" for i in range(1, 51)]})],
                    ignore_index=True)
    with pytest.raises(ValueError, match="new levels detected in newdata: new1, new2,"):
        gm1.predict(nd3)
    p2 = gm1.predict(nd)
    p6 = gm1.predict(nd2, allow_new_levels=True)
    assert np.allclose(p2.values, p6.values[:len(p2)])
    p_pop = gm1.predict(pd.DataFrame({"period": ["1", "2", "3", "4"]}), re_form="NA")
    assert np.allclose(p6.values[-4:], p_pop.values)


def test_na_grouping(fm2):
    nd = pd.DataFrame({"Days": [1.0, 2.0], "Subject": ["308", None]})
    with pytest.raises(ValueError, match="NAs are not allowed in prediction data"):
        fm2.predict(nd)
    p = fm2.predict(nd, allow_new_levels=True)
    assert np.isclose(p.values[1], fm2.predict(nd.iloc[1:], re_form="NA").values[0])


def test_multi_group(fm3):
    p0 = fm3.predict()
    assert np.allclose(np.round(p0.values[:6], 4),
                       [25.9638, 22.7663, 25.7147, 23.6799, 23.7629, 20.773])
    p1 = fm3.predict(re_form="NA")
    assert np.allclose(np.unique(p1.values), 22.9722222222251)


def test_multi_group_newdata(fm3):
    Pen = load_dataset("Penicillin")
    nd = pd.DataFrame(list(itertools.product(Pen["sample"].cat.categories, Pen["plate"].cat.categories)),
                      columns=["sample", "plate"])[["plate", "sample"]]
    p2 = fm3.predict(nd)
    p4 = fm3.predict(nd, re_form="~(1|plate)+(~1|sample)")
    p4B = fm3.predict(nd, re_form="~(1|sample)+(~1|plate)")
    assert np.allclose(p2.values, p4.values)
    assert np.allclose(p4.values, p4B.values)
    p3 = fm3.predict(nd, re_form="NA")
    p5 = fm3.predict(nd, re_form="~(1|sample)")
    p6 = fm3.predict(nd, re_form="~(1|plate)")
    # additive decomposition of the two random effects
    assert np.allclose(p5.values + p6.values - p3.values, p2.values)


def test_random_slopes(fm2, sleepstudy):
    nd = pd.DataFrame(list(itertools.product(sleepstudy["Subject"].cat.categories, [0.0, 9.0])),
                      columns=["Subject", "Days"])[["Days", "Subject"]]
    p2 = fm2.predict(nd).values[:6]
    p3 = fm2.predict(nd, re_form="NA").values[:6]
    p4 = fm2.predict(nd, re_form="~(0+Days|Subject)").values[:6]
    p5 = fm2.predict(nd, re_form="~(1|Subject)").values[:6]
    np.testing.assert_allclose(p2, [253.663652396798, 430.66001930835, 211.006415533628,
                                    227.634788908917, 212.444742696829, 257.61053840953], rtol=5e-7)
    np.testing.assert_allclose(p3, [251.405104848485, 345.610678484848] * 3, rtol=5e-7)
    np.testing.assert_allclose(p4, [251.405104848485, 428.401471760037, 251.405104848485,
                                    268.033478223774, 251.405104848485, 296.570900561186], rtol=5e-7)
    np.testing.assert_allclose(p5, [253.663652396798, 347.869226033161, 211.006415533628,
                                    305.211989169991, 212.444742696829, 306.650316333193], rtol=5e-7)


def test_predict_random_slopes_sleepstudy(fm2, sleepstudy):
    def tmpf(**kw):
        return np.round(fm2.predict(sleepstudy, **kw).values[:6], 3)
    np.testing.assert_allclose(tmpf(), [253.664, 273.33, 292.996, 312.662, 332.329, 351.995], atol=2e-3)
    np.testing.assert_allclose(tmpf(re_form="NA"), [251.405, 261.872, 272.34, 282.807, 293.274, 303.742],
                               atol=2e-3)
    np.testing.assert_allclose(tmpf(re_form="~(0+Days|Subject)"),
                               [251.405, 271.071, 290.738, 310.404, 330.07, 349.736], atol=2e-3)
    np.testing.assert_allclose(tmpf(re_form="~(1|Subject)"),
                               [253.664, 264.131, 274.598, 285.066, 295.533, 306], atol=2e-3)


def test_fewer_levels(fm4, fm1, sleepstudy):
    cake = load_dataset("cake")
    assert np.allclose(fm4.predict(newdata=cake).values, fm4.fitted().values)
    fm4.predict(newdata=cake.iloc[17:], re_form="~ (1 | replicate)")
    fm4.predict(newdata=cake.iloc[18:], re_form="NA")
    a = fm4.predict(newdata=cake.iloc[18:], re_form="~ (1 | replicate)")
    b = fm4.predict(newdata=cake.iloc[18:], re_form="~ (1 | replicate)", allow_new_levels=True)
    assert np.allclose(a.values, b.values)
    p0 = fm1.predict(pd.DataFrame({"Days": 6.0, "Subject": ["308", "309"]}))
    p1 = fm1.predict(pd.DataFrame({"Days": [6.0] * 4, "Subject": ["308", "309"] * 2}))
    assert np.allclose(np.tile(p0.values, 2), p1.values)
    p2 = fm1.predict(pd.DataFrame({"Days": [6.0], "Subject": ["308"]}))
    nd = pd.DataFrame({"Days": [6.0], "Subject": pd.Categorical(
        ["308"], categories=sleepstudy["Subject"].cat.categories)})
    p3 = fm1.predict(nd)
    assert np.allclose(p2.values, p3.values)
    assert np.isclose(p2.values[0], p0.values[0])


def test_factor_fixed_new_level(fm4):
    cake = load_dataset("cake").iloc[:3].copy()
    cake["recipe"] = cake["recipe"].astype(str)
    cake.loc[cake.index[0], "recipe"] = "Z"
    with pytest.raises(ValueError):
        fm4.predict(cake, re_form="NA")


def test_nested_pastes():
    P = load_dataset("Pastes")
    fm = lmer("strength ~ (1|batch/cask)", P)
    assert np.allclose(fm.predict(newdata=P).values, fm.fitted().values)
    rb = fm.ranef(condVar=False)["batch"]
    pb = fm.predict(newdata=P, re_form="~(1|batch)").values
    expect = fm.fixef().values[0] + rb.loc[P["batch"].astype(str).values].values[:, 0]
    assert np.allclose(pb, expect)


def test_interaction_and_nested_groups():
    rng = np.random.default_rng(101)
    d = pd.DataFrame(list(itertools.product("abcdefghij", "abcdefghij", range(10))),
                     columns=["f", "g", "rep"])
    d["f"] = pd.Categorical(d["f"])
    d["g"] = pd.Categorical(d["g"])
    d["y"] = rng.normal(size=len(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = lmer("y ~ (1|f:g)", d)
        m2 = lmer("y ~ (1|f/g)", d)
        dn = d.copy()
        dn["f"] = d["f"].cat.codes.astype(float) + 1
        dn["g"] = d["g"].cat.codes.astype(float) + 1
        m1N = lmer("y ~ (1|f:g)", dn)
    assert np.allclose(m1.predict().values, m1.predict(newdata=d).values)
    assert np.allclose(m2.predict().values, m2.predict(newdata=d).values)
    assert np.allclose(m1N.predict().values, m1N.predict(newdata=dn).values)


def test_poly_basis(sleepstudy):
    f2 = lmer("Reaction ~ poly(Days,2) + (1|Subject)", sleepstudy)
    f3 = lmer("Reaction ~ poly(Days,2,raw=TRUE) + (1|Subject)", sleepstudy)
    nd = pd.DataFrame({"Days": np.arange(5, 13, dtype=float)})
    assert np.allclose(f2.predict(nd, re_form="NA").values, f3.predict(nd, re_form="NA").values)
    assert np.allclose(f2.predict(newdata=sleepstudy).values, f2.fitted().values)


def test_poly_in_random_and_partial_reform():
    rng = np.random.default_rng(101)
    n = 200
    dd = pd.DataFrame({"x": np.arange(1, n + 1, dtype=float),
                       "f": pd.Categorical(np.tile(np.arange(1, 11), 20).astype(str)),
                       "g": pd.Categorical(np.repeat(np.arange(1, 21), 10).astype(str)),
                       "h": pd.Categorical(np.tile(np.arange(1, 6), 40).astype(str)),
                       "y": rng.normal(size=n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = lmer("y~1 + f + (1|h/f) + (poly(x,2)|g)", dd, control=lmerControl(calc_derivs=False))
    assert np.allclose(m1.predict().values, m1.predict(newdata=dd).values)
    pa = m1.predict(newdata=dd, re_form="~1 | h/f").values
    pb = m1.predict(newdata=dd, re_form="~poly(x,2) | g").values
    p0 = m1.predict(newdata=dd, re_form="NA").values
    assert np.allclose(pa + pb - p0, m1.fitted().values)
    # one row only
    assert np.isclose(m1.predict(newdata=dd.iloc[:1], re_form="~1 | h/f").values[0], pa[0])
    assert np.isclose(m1.predict(newdata=dd.iloc[:1], re_form="~poly(x,2) | g").values[0], pb[0])


def test_offset():
    rng = np.random.default_rng(101)
    d = pd.DataFrame({"y": rng.poisson(5, 100), "x": rng.lognormal(1, 1, 100),
                      "f": pd.Categorical(rng.integers(1, 11, 100).astype(str))})
    gm = glmer("y ~ offset(log(x)) + (1|f)", d, family=poisson)
    assert np.allclose(gm.predict(newdata=d).values, gm.resp.eta)
    assert np.allclose(gm.predict(re_form="NA").values,
                       gm.fixef().values[0] + np.log(d["x"].values))
    assert np.allclose(gm.predict(newdata=d, re_form="NA").values,
                       gm.fixef().values[0] + np.log(d["x"].values))


def test_na_in_newdata_and_fit(fm2, sleepstudy):
    nd = pd.DataFrame({"Days": [1.0, np.nan, 3.0], "Subject": ["308", "309", "310"]})
    p = fm2.predict(nd)
    assert np.isnan(p.values[1]) and not np.isnan(p.values[[0, 2]]).any()
    po = predict_merMod(fm2, nd, na_action="na.omit")
    assert list(po.index) == [0, 2]
    assert np.allclose(po.values, p.values[[0, 2]])
    d = sleepstudy.copy()
    d.loc[0, "Reaction"] = np.nan
    m = lmer("Reaction ~ Days + (Days|Subject)", d)
    assert len(m.predict()) == 179
    assert len(m.predict(re_form="NA")) == 179
    assert len(predict_merMod(m, re_form="NA", na_action="na.exclude")) == 180
    m2 = lmer("Reaction ~ Days + (Days|Subject)", d, na_action="na.exclude")
    p = m2.predict(re_form="NA")
    assert len(p) == 180 and np.isnan(p.values[0])
    assert np.allclose(m2.predict(re_form="~(Days|Subject)").values[1:], m2.fitted().values)


def test_space_in_names(cbpp):
    d = cbpp.copy()
    d["silly period"] = d["period"]
    m = glmer("cbind(incidence,size-incidence) ~ `silly period` + (1|herd)", d, family=binomial)
    ref = np.array([-0.809, -1.801, -1.937, -2.388, -1.697, -2.689])
    assert _all_equal(m.predict().values[:6], ref, 5e-4)
    assert np.allclose(m.predict(newdata=d).values, m.predict().values, atol=1e-6)


def test_newparams(fm1):
    th = fm1.theta
    p = predict_merMod(fm1, newparams=dict(beta=[0.0, 0.0], theta=th))
    b = fm1.ranef(condVar=False)["Subject"].iloc[:, 0]
    assert np.allclose(p.values[:10], b.iloc[0])
    with pytest.warns(UserWarning, match="some parameters not specified"):
        predict_merMod(fm1, newparams=dict(beta=[0.0, 0.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        setParams(fm1, dict(beta=[1.0, 2.0, 3.0], theta=th))
    with pytest.raises(ValueError, match="params should be"):
        setParams(fm1, dict(foo=1))
    fm1M = setParams(fm1, dict(theta=[1.0], beta=fm1.beta))
    assert fm1M.getME("theta").values[0] == 1.0
    assert fm1.getME("theta").values[0] == th[0]          # original untouched
    s = setParams(fm1, dict(beta=pd.Series({"Days": 1.0, "(Intercept)": 2.0}), theta=th,
                            sigma=3.0))
    assert list(s.beta) == [2.0, 1.0] and s.sigma() == 3.0


def test_errors(fm1):
    with pytest.raises(ValueError):
        fm1.predict(type="foo")
    with pytest.raises(ValueError, match="deprecated"):
        predict_merMod(fm1, ReForm=None)
    with pytest.raises(ValueError):
        fm1.predict(re_form="~(1|foo)", newdata=pd.DataFrame({"Days": [1.0], "foo": ["a"],
                                                               "Subject": ["308"]}))


def test_levelfun():
    x = pd.DataFrame({"(Intercept)": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    y = levelfun(x, ["a", "c"])
    assert list(y.index) == ["a", "c"]
    with pytest.raises(ValueError, match="new levels detected in newdata: d"):
        levelfun(x, ["a", "d"])
    z = levelfun(x, ["d", "a"], allow_new_levels=True)
    assert list(z.index) == ["d", "a"] and list(z.iloc[:, 0]) == [0.0, 1.0]


def test_get_orig_levs(fm4):
    lv = get_orig_levs(fm4, fixed_only=True)
    assert set(lv) == {"recipe"}
    lv = get_orig_levs(fm4, random_only=True, newdata=pd.DataFrame({"replicate": ["99"],
                                                                     "recipe": ["A"]}))
    assert lv["replicate"][-1] == "99" and lv["recipe"] == ["A", "B", "C"]


def test_mkNewReTrms_shapes(fm2, sleepstudy):
    r = mkNewReTrms(fm2, sleepstudy)
    assert r["Zt"].shape == fm2.getME("Zt").shape
    assert np.allclose(r["b"], fm2.getME("b"))
    assert np.allclose(r["Lambdat"].toarray(), fm2.getME("Lambdat").toarray())


# ------------------------------------------------------------------ se.fit
# reference values: R 4.6.2 / lme4 2.1.0 (examples/r/06_prediction_and_simulation.R)
def test_se_fit_lmm(fm2):
    from lme4.predict import quad_tdiag
    r = fm2.predict(se_fit=True)
    assert set(r) == {"fit", "se.fit"}
    assert np.allclose(r["fit"].values, fm2.fitted().values)
    assert np.allclose(r["se.fit"].values[:6], [12.2548108256166, 10.5433848237068,
                       9.11383151932835, 8.11648154297436, 7.72068692313019,
                       8.01605431277466], rtol=1e-6)
    pop = pd.DataFrame({"Days": np.arange(10.0)})
    p = fm2.predict(newdata=pop, re_form="NA", se_fit=True)
    assert np.allclose(p["se.fit"].values[[0, 5, 9]], [6.8245966949549, 9.58127739200983,
                       14.6288155374647], rtol=1e-6)
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    assert np.allclose(p["se.fit"].values ** 2, quad_tdiag(fm2.vcov().values, X))
    # subset of levels (Cmat subsetting) and a new level (pad_Cmat)
    nd = pd.DataFrame({"Days": [0.0, 5, 9, 0, 5, 9],
                       "Subject": pd.Categorical(["308"] * 3 + ["new"] * 3)})
    p = fm2.predict(newdata=nd, allow_new_levels=True, se_fit=True)
    assert np.allclose(p["fit"].values, [253.663655798386, 351.994964424196, 430.660011324843,
                                         251.405104848485, 303.741534646465, 345.610678484849],
                       rtol=1e-6)
    assert np.allclose(p["se.fit"].values, [12.2548108256166, 8.01605431277466, 13.8568925085997,
                                            6.8245966949549, 9.58127739200983, 14.6288155374647],
                       rtol=1e-6)


def test_se_fit_partial_reform(fm3):
    se = fm3.predict(re_form="~(1|plate)", se_fit=True)["se.fit"].values
    assert len(se) == 144
    assert np.allclose(se, 0.817976123319972, rtol=1e-4)   # theta agrees with R to ~1e-5


def test_se_fit_glmm(gm1):
    pp = pd.DataFrame({"period": pd.Categorical(["1", "2", "3", "4"])})
    with pytest.warns(UserWarning, match="approximation"):
        pl = gm1.predict(newdata=pp, re_form="NA", se_fit=True)
    with pytest.warns(UserWarning):
        pr = gm1.predict(newdata=pp, re_form="NA", type="response", se_fit=True)
    assert np.allclose(pl["se.fit"].values, [0.227892641940324, 0.310303375240942,
                                             0.330776779572552, 0.432643895881694], rtol=1e-5)
    assert np.allclose(pr["se.fit"].values, [0.0361952057965772, 0.0238432560358867,
                                             0.0226586745261182, 0.0199228228834064], rtol=1e-5)


def test_pad_Cmat():
    from lme4.predict import pad_Cmat
    C = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=["g.a.(Intercept)", "g.b.(Intercept)", "x"],
                     columns=["g.a.(Intercept)", "g.b.(Intercept)", "x"])
    r = pad_Cmat(C, {"g": ["a", "b"]}, {"g": ["a", "z"]}, list(C.index), [("g", ["(Intercept)"])])
    assert r["Cmat_names"] == ["g.a.(Intercept)", "g.b.(Intercept)", "g.z.(Intercept)", "x"]
    M = np.asarray(r["Cmat"])
    assert np.all(M[2] == 0) and np.all(M[:, 2] == 0) and M[3, 3] == 8 and M[0, 3] == 2
    assert r["C_factors"]["g"] == ["a", "b", "z"]
