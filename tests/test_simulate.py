"""Tests for lme4.predict.simulate_merMod, ported from lme4/tests/predsim.R,
lme4/tests/simulate.R and testthat/test-predict.R.  Random streams differ from
R's, so these check shapes, determinism and distributional properties."""
import warnings

import numpy as np
import pandas as pd
import pytest

from lme4 import lmer, glmer, load_dataset, binomial, poisson, Gamma, inverse_gaussian
from lme4.predict import simulate_merMod, predict_merMod


def _all_equal(current, target, tol):
    """R's all.equal for numeric vectors: mean relative difference <= tol"""
    current, target = np.asarray(current, float), np.asarray(target, float)
    return np.mean(np.abs(target - current)) / np.mean(np.abs(target)) <= tol


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
def gm1(cbpp):
    return glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=binomial)


@pytest.fixture(scope="module")
def glmm_data():
    rng = np.random.default_rng(42)
    n = 400
    f = rng.integers(0, 20, n)
    b = rng.normal(0, .5, 20)
    x = rng.normal(size=n)
    mu = np.exp(1 + .5 * x + b[f])
    return pd.DataFrame({
        "x": x, "f": pd.Categorical(f.astype(str)),
        "yg": rng.gamma(2, mu / 2), "yp": rng.poisson(mu), "yi": rng.wald(mu, 5.0),
        "ynb": rng.negative_binomial(2, 2 / (2 + mu)),
        "yb": pd.Categorical(np.where(rng.random(n) < 1 / (1 + np.exp(-(x + b[f]))), "yes", "no"))})


def test_shape_names_determinism(fm1, sleepstudy):
    s = fm1.simulate(nsim=3, seed=1)
    assert isinstance(s, pd.DataFrame)
    assert s.shape == (180, 3)
    assert list(s.columns) == ["sim_1", "sim_2", "sim_3"]
    assert list(s.index) == list(fm1.fitted().index)
    s2 = fm1.simulate(nsim=3, seed=1)
    pd.testing.assert_frame_equal(s, s2)
    assert not np.allclose(s.values, fm1.simulate(nsim=3, seed=2).values)
    assert "seed" in s.attrs and "na.action" in s.attrs
    rng = np.random.default_rng(7)
    a = simulate_merMod(fm1, seed=rng)
    assert a.shape == (180, 1)


def test_use_u_vs_reform(fm1):
    a = simulate_merMod(fm1, seed=3, use_u=True)
    b = simulate_merMod(fm1, seed=3, re_form="~(1|Subject)")
    pd.testing.assert_frame_equal(a, b)
    c = simulate_merMod(fm1, seed=3, use_u=False)
    d = simulate_merMod(fm1, seed=3, re_form="NA")
    e = simulate_merMod(fm1, seed=3)
    pd.testing.assert_frame_equal(c, d)
    pd.testing.assert_frame_equal(c, e)
    with pytest.raises(ValueError, match="only one of"):
        simulate_merMod(fm1, use_u=True, re_form="NA")


def test_conditional_distribution(fm1):
    pp = fm1.predict()
    ss = fm1.simulate(nsim=2000, seed=101, use_u=True)
    assert _all_equal(ss.median(axis=1).values, pp.values, 5e-3)
    # residual sd only
    assert abs(ss.values.std(axis=1).mean() / fm1.sigma() - 1) < 0.03


def test_unconditional_distribution(fm1):
    pp2 = fm1.predict(re_form="NA")
    ss2 = fm1.simulate(nsim=2000, seed=101)
    assert _all_equal(ss2.median(axis=1).values, pp2.values, 8e-3)
    tot = np.sqrt(fm1.sigma() ** 2 * (1 + fm1.theta[0] ** 2))
    assert abs(ss2.values.std(axis=1).mean() / tot - 1) < 0.03
    # within-subject correlation induced by the random effect
    c = np.corrcoef(ss2.iloc[0].values, ss2.iloc[1].values)[0, 1]
    rho = fm1.theta[0] ** 2 / (1 + fm1.theta[0] ** 2)
    assert abs(c - rho) < 0.06


def test_newparams_newdata(fm1, sleepstudy):
    s1 = simulate_merMod(fm1, seed=1)
    s2 = simulate_merMod(fm1, seed=1, newdata=sleepstudy,
                         newparams=dict(theta=fm1.theta, beta=fm1.beta, sigma=fm1.sigma()))
    np.testing.assert_allclose(s1.values, s2.values)
    s3 = simulate_merMod(fm1, nsim=500, seed=1, newparams=dict(theta=[0.0], beta=[100.0, 0.0],
                                                               sigma=1.0))
    assert abs(s3.values.mean() - 100) < 0.1 and abs(s3.values.std() - 1) < 0.05


def test_newdata_new_levels(fm1):
    nd = pd.DataFrame({"Days": [0.0, 1.0], "Subject": ["new1", "new2"]})
    with pytest.raises(ValueError, match="new levels"):
        simulate_merMod(fm1, newdata=nd, seed=1)
    s = simulate_merMod(fm1, newdata=nd, seed=1, allow_new_levels=True, nsim=4)
    assert s.shape == (2, 4)


def test_binomial_cbind(gm1, cbpp):
    s = gm1.simulate(nsim=3, seed=2)
    assert s.shape == (56, 6)
    assert isinstance(s.columns, pd.MultiIndex)
    for k in ("sim_1", "sim_2", "sim_3"):
        m = s[k].values
        assert np.allclose(m.sum(axis=1), cbpp["size"].values)
        assert np.all(m >= 0) and np.allclose(m, np.round(m))
    big = gm1.simulate(nsim=400, seed=5, use_u=True)
    inc = np.column_stack([big[f"sim_{i + 1}"].values[:, 0] for i in range(400)])
    np.testing.assert_allclose(inc.mean(axis=1), gm1.fitted().values * cbpp["size"].values,
                               atol=0.6)
    # newdata: sizes default to the response matrix row sums
    s2 = simulate_merMod(gm1, newdata=cbpp, seed=1)
    assert np.allclose(s2["sim_1"].values.sum(axis=1), cbpp["size"].values)


def test_binomial_weights(cbpp):
    d = cbpp.copy()
    d["prop"] = d["incidence"] / d["size"]
    gw = glmer("prop ~ period + (1|herd)", d, family=binomial, weights=d["size"].values)
    s = gw.simulate(nsim=3, seed=2)
    k = s.values * d["size"].values[:, None]
    assert np.allclose(k, np.round(k)) and np.all((s.values >= 0) & (s.values <= 1))


def test_binomial_factor(glmm_data):
    m = glmer("yb ~ x + (1|f)", glmm_data, family=binomial)
    s = m.simulate(nsim=200, seed=1, use_u=True)
    assert all(isinstance(s[c].dtype, pd.CategoricalDtype) for c in s)
    assert list(s["sim_1"].cat.categories) == ["no", "yes"]
    frac = np.mean([(s[c] == "yes").mean() for c in s])
    assert abs(frac - m.fitted().mean()) < 0.01


def test_binomial_factor_one_level():
    y = pd.Categorical(["0"] * 1000 + ["1"] + ["0"] * 1000 + ["1"], categories=["0", "1"])
    x = ["A"] * 1001 + ["B"] * 1001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = glmer("y ~ 1 + (1|x)", pd.DataFrame({"y": y, "x": x}), family=binomial)
        s = simulate_merMod(mod, newdata=pd.DataFrame({"x": ["A"]}), nsim=10, seed=1)
    assert s.shape == (1, 10)
    assert all((s[c] == "0").all() for c in s)


@pytest.mark.parametrize("form,fam", [("yp ~ x + (1|f)", poisson),
                                      ("yg ~ x + (1|f)", Gamma(link="log")),
                                      ("yi ~ x + (1|f)", inverse_gaussian(link="log"))])
def test_glmm_families_mean(glmm_data, form, fam):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmer(form, glmm_data, family=fam)
    s = m.simulate(nsim=300, seed=11, use_u=True)
    assert s.shape == (400, 300)
    assert abs(s.values.mean() / m.fitted().mean() - 1) < 0.02
    if form.startswith("yp"):
        assert np.allclose(s.values, np.round(s.values))
    else:
        assert np.all(s.values > 0)
    if form.startswith("yg"):
        # Gamma: var = sigma^2 mu^2
        mu = m.fitted().values
        v = s.values.var(axis=1)
        assert abs(np.mean(v / mu ** 2) / m.sigma() ** 2 - 1) < 0.1


def test_negative_binomial(glmm_data):
    from lme4.nbinom import glmer_nb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = glmer_nb("ynb ~ x + (1|f)", glmm_data)
    assert np.allclose(nb.predict(newdata=glmm_data).values, nb.resp.eta)
    s = nb.simulate(nsim=500, seed=1, use_u=True)
    mu = nb.fitted().values
    th = nb.resp.family.theta
    assert abs(s.values.mean() / mu.mean() - 1) < 0.02
    assert abs(s.values.var(axis=1).mean() / (mu + mu ** 2 / th).mean() - 1) < 0.1


def test_offset_glmm():
    rng = np.random.default_rng(101)
    d = pd.DataFrame({"y": rng.poisson(5, 100), "x": rng.lognormal(1, 1, 100),
                      "f": pd.Categorical(rng.integers(1, 11, 100).astype(str))})
    gm = glmer("y ~ offset(log(x)) + (1|f)", d, family=poisson)
    s = gm.simulate(nsim=2000, seed=1, use_u=True)
    np.testing.assert_allclose(s.mean(axis=1).values, gm.fitted().values, rtol=0.15, atol=0.3)


def test_na_handling(sleepstudy):
    d = sleepstudy.copy()
    d.loc[0, "Reaction"] = np.nan
    fm = lmer("Reaction ~ Days + (Days | Subject)", d)
    ss = simulate_merMod(fm, seed=101, use_u=True)
    ss0 = simulate_merMod(fm, seed=101, re_form="NA")
    assert len(ss) == len(ss0) == 179
    fm2 = lmer("Reaction ~ Days + (Days | Subject)", d, na_action="na.exclude")
    ss2 = simulate_merMod(fm2, seed=101, use_u=True)
    ss3 = simulate_merMod(fm2, seed=101, re_form="NA")
    assert len(ss2) == len(ss3) == 180
    assert np.isnan(ss2.iloc[0, 0]) and not np.isnan(ss2.iloc[1:, 0]).any()


def test_cond_sim_false(fm1):
    s = simulate_merMod(fm1, seed=1, use_u=True, cond_sim=False, nsim=2)
    np.testing.assert_allclose(s["sim_1"].values, fm1.predict().values)


def test_crossed_partial_reform():
    Pen = load_dataset("Penicillin")
    fm = lmer("diameter ~ (1|plate) + (1|sample)", Pen)
    s = simulate_merMod(fm, nsim=2000, seed=4, re_form="~(1|plate)")
    np.testing.assert_allclose(s.mean(axis=1).values, fm.predict(re_form="~(1|plate)").values,
                               atol=0.15)
