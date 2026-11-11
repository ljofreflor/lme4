"""Comparison against reference results produced by R's lme4 2.1.0.

The references (tests/reference/reference.json) are produced from
tests/reference/specs.json by tests/reference/make_reference.R, run inside a
docker image with R + lme4 2.1.0:

    docker run --rm -v <repo>/tests/reference:/ref -w /ref lme4ref:2.1 \
        Rscript make_reference.R specs.json reference.json

Tolerances: the objective (logLik) must agree to 1e-8 relative; parameter
estimates may differ where the likelihood surface is flat, because the sparse
Cholesky ordering (Eigen AMD vs CHOLMOD) changes rounding at the 1e-15 level,
which perturbs where the optimizer stops.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "reference"))
from fit_python import SPECS, REF, fit, quantities, maxrel  # noqa: E402

TOL = dict(logLik=1e-8, devcrit=1e-5, fixef=5e-3, theta=5e-3, sigma=1e-4, vcov_rx=1e-3,
           vcov=5e-2, vc_sdcor=5e-3, fitted=1e-3)


@pytest.fixture(scope="module", params=[s["id"] for s in SPECS])
def pair(request):
    spec = next(s for s in SPECS if s["id"] == request.param)
    m = fit(spec)
    return spec, m, REF[spec["id"]]


def test_numbers(pair):
    spec, m, r = pair
    q = quantities(m)
    for k, tol in TOL.items():
        assert maxrel(q[k], r[k]) < tol, (spec["id"], k, maxrel(q[k], r[k]))
    assert q["fixef"].shape == np.atleast_1d(r["fixef"]).shape
    assert list(m.fixef().index) == list(np.atleast_1d(r["fixef_names"]))
    assert list(m.getME("theta").index) == (r["theta_names"] if isinstance(r["theta_names"], list)
                                            else [r["theta_names"]])
    for k in r["ranef"]:
        assert maxrel(q["ranef"][k], r["ranef"][k]) < 0.05
    assert m.isSingular() == r["singular"]
    assert m.nobs() == r["nobs"]
    assert m.logLik().df == r["df"]


# lines depending on variable names or on the optimizer path (convergence warnings)
_SKIP = ("   Data:", "Weights:", "optimizer (", "Model failed to converge", "  See ?lme4",
         "boundary (singular)")


def _norm(lines):
    return [l.rstrip() for l in lines if l.strip() and not l.startswith(_SKIP)]


def test_print_matches_R(pair):
    """the printed summary is identical to R's, except for lines carrying
    estimates (which may differ in the last digits, see TOL) and the data name"""
    spec, m, r = pair
    py = _norm(str(m.summary()).splitlines())
    R = _norm(r["summary"])
    assert len(py) == len(R), "\n".join(py)
    for a, b in zip(py, R):
        if b.startswith(("   Data:", "Weights:")):
            continue
        if a != b:
            # allow numeric differences only: same tokens apart from numbers
            ta, tb = a.split(), b.split()
            assert len(ta) == len(tb), (a, b)
            for x, y in zip(ta, tb):
                if x != y:
                    fx, fy = float(x.replace("e", "E")), float(y.replace("e", "E"))
                    assert abs(fx - fy) <= 2e-2 * max(abs(fy), 1e-3) + 1e-3, (a, b)
