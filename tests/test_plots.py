"""Diagnostic plots (plots.py, transcription of lme4 R/plot.R and R/plots.R).

Reference values: R 4.6.2, lme4 2.1.0, lattice (examples/r/01_linear_mixed_models.R)."""
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lme4 import lmer, load_dataset  # noqa: E402
from lme4.plots import ppoints, asDf0, getIDLabels, _extend_limits  # noqa: E402


@pytest.fixture(scope="module")
def fm1():
    return lmer("Reaction ~ Days + (Days | Subject)", load_dataset("sleepstudy"))


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_ppoints():
    # R: ppoints(5); ppoints(12)[1:2]
    np.testing.assert_allclose(ppoints(5), [0.1190476190476190, 0.3095238095238095, 0.5,
                                            0.6904761904761905, 0.8809523809523809])
    np.testing.assert_allclose(ppoints(12)[:2], [0.04166666666666666, 0.125])


def test_plot_default(fm1):
    fig = fm1.plot()
    ax = fig.axes[0]
    off = ax.collections[0].get_offsets()
    np.testing.assert_allclose(off[:, 0], fm1.fitted().to_numpy())
    np.testing.assert_allclose(off[:, 1], fm1.residuals(type="pearson").to_numpy())
    assert ax.get_xlim()[0] > 150     # the reference line does not rescale the panel
    # abline c(0, 0) for the default form
    assert len(ax.lines) == 1
    assert fig._supxlabel.get_text() == "fitted(.)"
    assert fig._supylabel.get_text() == 'resid(., type = "pearson")'


def test_plot_id(fm1):
    fig = fm1.plot(id=0.05)
    texts = fig.axes[0].texts
    # R: which(abs(resid(fm1, type="pearson"))/sigma(fm1) > -qnorm(0.05/2))
    assert [t.get_text() for t in texts] == ["308", "308", "332", "332", "352"]
    np.testing.assert_allclose([t.get_position()[0] for t in texts],
                               fm1.fitted().to_numpy()[np.array([6, 8, 57, 60, 131]) - 1])
    assert list(getIDLabels(fm1)[:2]) == ["308", "308"]


def test_plot_panels_and_bw(fm1):
    fig = fm1.plot("resid(., scaled=TRUE) ~ fitted(.) | Subject", abline=0)
    vis = [a for a in fig.axes if a.get_visible()]
    assert len(vis) == 18
    assert [a.get_title() for a in vis][:3] == ["308", "309", "310"]
    assert all(len(a.collections[0].get_offsets()) == 10 for a in vis)
    fig = fm1.plot("Subject ~ resid(., scaled=TRUE)")
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()][:2] == ["308", "309"]
    fig = fm1.plot("~ resid(., scaled=TRUE)")
    assert len(fig.axes[0].patches) > 0
    with pytest.raises(ValueError):
        fm1.plot("Subject ~ Subject")


def test_dotplot_ranef(fm1):
    rr = fm1.ranef(condVar=True)
    ss = asDf0(rr, "Subject")
    lev = ['309', '310', '370', '349', '350', '334', '335', '371', '308', '369', '351',
           '332', '372', '333', '352', '331', '330', '337']
    assert list(ss[".nn"].cat.categories) == lev
    np.testing.assert_allclose(ss["se"].iloc[0], 12.0708569505143, rtol=1e-5)
    figs = rr.dotplot()
    fig = figs["Subject"]
    assert len(fig.axes) == 2
    assert [a.get_title() for a in fig.axes] == ["(Intercept)", "Days"]
    a = fig.axes[0]
    np.testing.assert_allclose(a.get_xlim(), [-72.6396109918765, 67.1313597421569], rtol=1e-5)
    pts = a.collections[-1].get_offsets()
    # point for level '309' sits at the bottom (y = 1)
    np.testing.assert_allclose(pts[1], [-40.3987380788051, 1.0], rtol=1e-5)
    assert [t.get_text() for t in a.get_yticklabels()] == lev
    # free scales
    fig = rr.dotplot(scales_free=True)["Subject"]
    assert fig.axes[0].get_xlim() != fig.axes[1].get_xlim()


def test_qqmath(fm1):
    rr = fm1.ranef(condVar=True)
    fig = rr.qqmath()["Subject"]
    pts = fig.axes[0].collections[-1].get_offsets()
    # R: qqmath(ranef(fm1))$Subject$panel.args[[1]]$y[1:3]
    np.testing.assert_allclose(pts[:3, 1], [-1.91450582505556, -1.38299412710064, -1.08532490807676])
    assert np.all(np.diff(pts[:, 0]) >= 0)
    fig = fm1.qqmath()
    pts = fig.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(pts[:, 1], np.sort(fm1.residuals(type="pearson", scaled=True)))
    np.testing.assert_allclose(pts[:2, 0], [-2.772921294608662, -2.393979799818510], rtol=1e-9)  # qnorm(ppoints(180))[1:2]


def test_extend_limits():
    np.testing.assert_allclose(_extend_limits([0, 10]), [-0.7, 10.7])
