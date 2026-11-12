"""rePCA (rePCA.R), checked against R 4.6.2 / lme4 2.1.0
(values from examples/r/08_covariance_structures_and_diagnostics.R)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lme4 import lmer, load_dataset, rePCA


@pytest.fixture(scope="module")
def sleepstudy():
    return load_dataset("sleepstudy")


def test_rePCA_sleepstudy(sleepstudy):
    # rePCA(lmer(Reaction ~ Days + (Days | Subject), sleepstudy))
    fm1 = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
    pc = rePCA(fm1)
    assert list(pc) == ["Subject"]
    assert_allclose(pc["Subject"].sdev, [0.966867972251729, 0.230879814201474], rtol=1e-6)
    R_rot = np.array([[-0.999861584050719, -0.0166376903320651],
                      [-0.0166376903320651, 0.99986158405072]])
    assert_allclose(np.abs(pc["Subject"].rotation), np.abs(R_rot), rtol=1e-5)
    imp = pc.summary()["Subject"]
    assert list(imp.index) == ["Standard deviation", "Proportion of Variance",
                               "Cumulative Proportion"]
    assert_allclose(imp.values[2], [np.round(1 / (1 + (0.230879814201474 / 0.966867972251729) ** 2), 5), 1])
    assert "Standard deviations (1, .., p=2):" in str(pc)


def test_rePCA_singular_cake():
    # a singular 3 x 3 fit with no zero variance: one null component
    cake = load_dataset("cake")
    m = lmer("angle ~ recipe + temp + (recipe | replicate)", cake)
    pc = rePCA(m)
    assert_allclose(pc["replicate"].sdev[:2], [1.26186598350985, 0.870149973929518], rtol=1e-3)
    assert pc["replicate"].sdev[2] < 1e-4
    assert_allclose(pc.summary()["replicate"].values.ravel(order="F"),
                    [1.26186598350985, 0.67773, 0.67773, 0.870149973929518, 0.32227, 1,
                     1.44943281762783e-05, 0, 1], atol=1e-3)


def test_rePCA_repeated_grouping_factor(sleepstudy):
    # (1|Subject) + (0+Days|Subject): both terms are combined block-diagonally
    m = lmer("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", sleepstudy)
    pc = rePCA(m)
    assert list(pc) == ["Subject"]
    assert_allclose(pc["Subject"].sdev, np.sort(m.theta)[::-1], rtol=1e-12)
    assert_allclose(np.abs(pc["Subject"].rotation), np.eye(2), atol=1e-12)
