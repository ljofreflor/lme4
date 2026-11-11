"""Formula machinery: parser/deparser, terms(), model.matrix() and the
reformulas helpers, compared with R's behaviour."""
import numpy as np
import pandas as pd
import pytest

from lme4.rlang import parse, deparse, Formula
from lme4.modelframe import terms, model_frame, model_matrix
from lme4.reformulas import findbars, nobars, subbars, expandDoubleVerts, findbars_x
from lme4 import V, I, log


@pytest.mark.parametrize("src,labels,intercept", [
    ("y ~ a*b - 1", ["a", "b", "a:b"], 0),
    ("y ~ b + a + a:b", ["b", "a", "b:a"], 1),
    ("y ~ (a+b+c)^2", ["a", "b", "c", "a:b", "a:c", "b:c"], 1),
    ("y ~ a/b", ["a", "a:b"], 1),
    ("y ~ a %in% b", ["a:b"], 1),
    ("y ~ 0 + f:x", ["f:x"], 0),
    ("y ~ x - 1 + 1", ["x"], 1),
    ("y ~ x + offset(z)", ["x"], 1),
])
def test_terms(src, labels, intercept):
    t = terms(Formula(src))
    assert t.term_labels == labels
    assert t.intercept == intercept


def test_deparse_roundtrip():
    for s in ["y ~ x + (Days | Subject)", "y ~ I(x^2) + log(z, 2)", "~x || g",
              "cbind(a, b - a) ~ f + (1 | g:h)", "y ~ -1 + x", "y ~ x %in% g"]:
        assert deparse(parse(s)) == s


def test_model_matrix_names():
    d = pd.DataFrame({"f": pd.Categorical(list("abcabc")), "g": list("xxyyzz"),
                      "x": np.arange(6.), "y": np.arange(6.) ** 2})
    cases = {
        "y ~ f*x": ["(Intercept)", "fb", "fc", "x", "fb:x", "fc:x"],
        "y ~ 0 + f": ["fa", "fb", "fc"],
        "y ~ x + f:x": ["(Intercept)", "x", "x:fb", "x:fc"],
        "y ~ poly(x, 2)": ["(Intercept)", "poly(x, 2)1", "poly(x, 2)2"],
        "y ~ I(x^2) + log(x + 1)": ["(Intercept)", "I(x^2)", "log(x + 1)"],
    }
    for f, cn in cases.items():
        mf = model_frame(Formula(f), d)
        X, nm, asg, _ = model_matrix(mf.terms, mf)
        assert nm == cn, f


def test_poly_matches_R():
    # R: poly(1:5, 2)
    d = pd.DataFrame({"x": np.arange(1., 6.), "y": np.ones(5)})
    mf = model_frame(Formula("y ~ poly(x, 2)"), d)
    X, nm, _, _ = model_matrix(mf.terms, mf)
    expect = np.array([[-6.324555e-01, 0.5345225], [-3.162278e-01, -0.2672612],
                       [0, -0.5345225], [3.162278e-01, -0.2672612], [6.324555e-01, 0.5345225]])
    assert np.allclose(np.abs(X[:, 1:]), np.abs(expect), atol=1e-6)


def test_findbars_nobars_subbars():
    f = Formula("y ~ x + (1 | g) + (x | h)")
    assert [deparse(b) for b in findbars(f)] == ["1 | g", "x | h"]
    assert deparse(nobars(f).expr) == "y ~ x"
    assert deparse(subbars(f).expr) == "y ~ x + (1 + g) + (x + h)"
    assert deparse(expandDoubleVerts(Formula("y ~ x + (x || g)")).expr) == \
        "y ~ x + ((1 | g) + (0 + x | g))"
    bb = findbars_x(Formula("y ~ (1 | a/b)"), default_special=None)
    assert [deparse(b) for b in bb] == ["1 | b:a", "1 | a"]


def test_symbolic_formula_builds_same_tree():
    y, x, g, h = V("y x g h")
    assert deparse((y == x + (1 | g) + (x | h)).expr) == "y ~ x + (1 | g) + (x | h)"
    assert deparse((y == x + (x & g)).expr) == "y ~ x + x:g"
    assert deparse((y == I(x ** 2) + log(x)).expr) == "y ~ I(x^2) + log(x)"
    assert deparse((y == 0 + x).expr) == "y ~ 0 + x"
    assert deparse((y == np.log(x) + (1 + x | g)).expr) == "y ~ log(x) + (1 + x | g)"
