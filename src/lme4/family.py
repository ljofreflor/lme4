"""GLM family objects (R's ``family``), backed by the C++ glmFamily class.

``binomial()``, ``poisson()``, ``Gamma()``, ``gaussian()``,
``inverse_gaussian()`` and ``negative_binomial(theta)`` mirror the R
constructors, including their ``initialize`` expressions.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._backend import cpp
from .modelframe import Factor

_VALID_LINKS = {
    "binomial": ["logit", "probit", "cloglog", "cauchit", "log"],
    "poisson": ["log", "identity", "sqrt"],
    "Gamma": ["inverse", "identity", "log"],
    "gaussian": ["identity", "log", "inverse"],
    "inverse.gaussian": ["1/mu^2", "inverse", "identity", "log"],
    "negative.binomial": ["log", "sqrt", "identity"],
}
_DEFAULT_LINK = {"binomial": "logit", "poisson": "log", "Gamma": "inverse",
                 "gaussian": "identity", "inverse.gaussian": "1/mu^2",
                 "negative.binomial": "log"}


class Family:
    """An R ``family`` object."""

    def __init__(self, family, link=None, theta=None):
        base = "negative.binomial" if family.startswith("Negative Binomial") else family
        link = link or _DEFAULT_LINK[base]
        if link not in _VALID_LINKS[base]:
            raise ValueError(f"link \"{link}\" not available for {family} family")
        self.base = base
        self.link = link
        self.theta = theta
        if base == "negative.binomial":
            if theta is None:
                raise ValueError("negative binomial family requires theta")
            self.family = f"Negative Binomial({_fmt_theta(theta)})"
        else:
            self.family = family
        self._cpp = cpp.glmFamily(self.family, link, float(theta) if theta is not None else float("nan"))

    # R's family components -------------------------------------------------
    def linkfun(self, mu):
        return np.asarray(self._cpp.linkFun(np.asarray(mu, float)))

    def linkinv(self, eta):
        return np.asarray(self._cpp.linkInv(np.asarray(eta, float)))

    def mu_eta(self, eta):
        return np.asarray(self._cpp.muEta(np.asarray(eta, float)))

    def variance(self, mu):
        return np.asarray(self._cpp.variance(np.asarray(mu, float)))

    def dev_resids(self, y, mu, wt):
        return np.asarray(self._cpp.devResid(np.asarray(y, float), np.asarray(mu, float),
                                             np.asarray(wt, float)))

    def aic(self, y, n, mu, wt, dev):
        return self._cpp.aic(np.asarray(y, float), np.asarray(n, float), np.asarray(mu, float),
                             np.asarray(wt, float), float(dev))

    @property
    def dispersion(self):
        """R >= 4.3 family$dispersion: NaN if free, else fixed value"""
        return 1.0 if self.base in ("binomial", "poisson", "negative.binomial") else float("nan")

    def initialize(self, y, weights, nobs):
        """R's family$initialize; returns dict(y, weights, n, mustart)"""
        y = y
        weights = np.asarray(weights, float).copy()
        if self.base == "binomial":
            if isinstance(y, Factor):
                y = (y.codes != 0).astype(float)
                y = np.asarray(y, float)
            ya = np.asarray(y)
            if ya.dtype == bool:
                ya = ya.astype(float)
            if ya.ndim == 1 or ya.shape[1] == 1:
                ya = ya.ravel().astype(float).copy()
                n = np.ones(nobs)
                ya[weights == 0] = 0
                if np.any((ya < 0) | (ya > 1)):
                    raise ValueError("y values must be 0 <= y <= 1")
                mustart = (weights * ya + 0.5) / (weights + 1)
                m = weights * ya
                if np.any(np.abs(m - np.round(m)) > 1e-3):
                    warnings.warn("non-integer #successes in a binomial glm!")
                return dict(y=ya, weights=weights, n=n, mustart=mustart)
            if ya.shape[1] == 2:
                y1 = ya[:, 0].astype(float)
                n = y1 + ya[:, 1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    yy = np.where(n == 0, 0.0, y1 / n)
                if np.any(np.abs(n - np.round(n)) > 1e-3):
                    warnings.warn("non-integer counts in a binomial glm!")
                weights = weights * n
                mustart = (n * yy + 0.5) / (n + 1)
                return dict(y=yy, weights=weights, n=n, mustart=mustart)
            raise ValueError("for the 'binomial' family, y must be a vector of 0 and 1's\n"
                             "or a 2 column matrix where col 1 is no. successes and col 2 is no. failures")
        ya = np.asarray(y, float).ravel()
        n = np.ones(nobs)
        if self.base == "poisson":
            if np.any(ya < 0):
                raise ValueError("negative values not allowed for the 'Poisson' family")
            mustart = ya + 0.1
        elif self.base == "Gamma":
            if np.any(ya <= 0):
                raise ValueError("non-positive values not allowed for the 'Gamma' family")
            mustart = ya.copy()
        elif self.base == "inverse.gaussian":
            if np.any(ya <= 0):
                raise ValueError("positive values only are allowed for the 'inverse.gaussian' family")
            mustart = ya.copy()
        elif self.base == "negative.binomial":
            if np.any(ya < 0):
                raise ValueError("negative values not allowed for the negative binomial family")
            mustart = ya + (ya == 0) / 6
        else:  # gaussian
            if self.link == "inverse" and np.any(ya == 0):
                raise ValueError("cannot find valid starting values: please specify some")
            if self.link == "log" and np.any(ya <= 0):
                raise ValueError("cannot find valid starting values: please specify some")
            mustart = ya.copy()
        return dict(y=ya, weights=weights, n=n, mustart=mustart)

    def __eq__(self, other):
        return isinstance(other, Family) and other.family == self.family and other.link == self.link

    def __repr__(self):
        return f"\nFamily: {self.family} \nLink function: {self.link} \n"

    def is_gaussian_identity(self):
        return self.base == "gaussian" and self.link == "identity"


def _fmt_theta(theta):
    """MASS: format(round(theta, 4))"""
    return "%.7g" % round(float(theta), 4)


def binomial(link="logit"):
    return Family("binomial", link)


def poisson(link="log"):
    return Family("poisson", link)


def Gamma(link="inverse"):
    return Family("Gamma", link)


def gaussian(link="identity"):
    return Family("gaussian", link)


def inverse_gaussian(link="1/mu^2"):
    return Family("inverse.gaussian", link)


def negative_binomial(theta, link="log"):
    """MASS::negative.binomial(theta)"""
    return Family("negative.binomial", link, theta=theta)


def get_family(family):
    if family is None:
        return None
    if isinstance(family, Family):
        return family
    if isinstance(family, str):
        f = {"binomial": binomial, "poisson": poisson, "Gamma": Gamma, "gamma": Gamma,
             "gaussian": gaussian, "inverse.gaussian": inverse_gaussian,
             "inverse_gaussian": inverse_gaussian}.get(family)
        if f is None:
            raise ValueError(f"unknown family {family!r}")
        return f()
    if callable(family):
        return family()
    raise TypeError("family must be a Family object, a function or a name")


def hasNoScale(family):
    return family is not None and family.family[:12] in ("poisson", "binomial", "negative.bin", "Negative Bin")


def isNBfamily(s):
    import re
    return bool(re.match(r"^Negative ?Binomial", s, re.I))
