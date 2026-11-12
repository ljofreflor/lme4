"""Python counterparts of the R reference classes in lme4/R/AllClass.R
(merPredD, lmResp, lmerResp, glmResp, nlsResp) wrapping the C++ objects."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._backend import cpp


def _csc(m):
    m = sp.csc_matrix(m, dtype=float)
    m.sort_indices()
    return m


class merPredD:
    """predictor module; ``self.ptr`` is the C++ object"""

    def __init__(self, X, Zt, Lambdat, Lind, theta, n=None, Xwts=None, beta0=None, u0=None,
                 Xattrs=None, _ptr=None):
        if _ptr is not None:
            self.ptr = _ptr
        else:
            X = np.asarray(X, float)
            n = X.shape[0] if n is None else int(n)
            self.ptr = cpp.merPredD(np.asfortranarray(X), _csc(Zt), _csc(Lambdat),
                                    np.asarray(Lind, np.int32), np.asarray(theta, float), n,
                                    np.asarray(Xwts if Xwts is not None else [], float),
                                    np.asarray(beta0 if beta0 is not None else [], float),
                                    np.asarray(u0 if u0 is not None else [], float))
        # attributes of X (colnames, msgs, col.dropped, scaling, ...)
        self.Xattrs = dict(Xattrs or {})

    def copy(self):
        return merPredD(None, None, None, None, None, _ptr=self.ptr.copy(), Xattrs=dict(self.Xattrs))

    def __getattr__(self, name):
        return getattr(self.ptr, name)

    @property
    def X(self):
        return np.asarray(self.ptr.X)

    @property
    def colnames(self):
        return self.Xattrs.get("colnames")

    def L(self):
        return CHMfactor(self.ptr)


class CHMfactor:
    """the sparse Cholesky factor: L L' = P (Lambda'Z'WZLambda + I) P'"""

    def __init__(self, ptr):
        self._ptr = ptr
        self.L = sp.csc_matrix(ptr.L())
        self.perm = np.asarray(ptr.Pvec())        # 0-based: L L' = A[perm, perm]

    def solve(self, B, system="A"):
        """Matrix::solve(L, B, system=...) for 'A', 'L', 'Lt', 'P', 'Pt'"""
        B = B.toarray() if sp.issparse(B) else np.asarray(B, float)
        vec = B.ndim == 1
        res = np.asarray(self._ptr.solveSystem(np.atleast_2d(B.T).T if not vec else B.reshape(-1, 1), system))
        return res.ravel() if vec else res

    def __repr__(self):
        return f"CHMfactor (dim {self.L.shape[0]}, nnz {self.L.nnz})"


class _RespBase:
    def __getattr__(self, name):
        return getattr(self.ptr, name)

    @property
    def y(self):
        return np.asarray(self.ptr.y)

    @property
    def mu(self):
        return np.asarray(self.ptr.mu)

    @property
    def offset(self):
        return np.asarray(self.ptr.offset)

    @property
    def weights(self):
        return np.asarray(self.ptr.weights)

    @property
    def wtres(self):
        return np.asarray(self.ptr.wtres)

    @property
    def sqrtXwt(self):
        return np.asarray(self.ptr.sqrtXwt)

    @property
    def sqrtrwt(self):
        return np.asarray(self.ptr.sqrtrwt)


class lmerResp(_RespBase):
    cls = "lmerResp"

    def __init__(self, y=None, weights=None, offset=None, mu=None, REML=0, _ptr=None, **kw):
        if _ptr is not None:
            self.ptr = _ptr
            return
        y = np.asarray(y, float)
        n = len(y)
        weights = np.ones(n) if weights is None else np.asarray(weights, float)
        offset = np.zeros(n) if offset is None else np.asarray(offset, float)
        if n == 0:
            # trivial response (simulate() from a formula): R recycles the length-0 'y'
            # and never creates the external pointer; the C++ module needs equal lengths
            weights, offset = np.ones(0), np.zeros(0)
        mu = np.zeros(n) if mu is None else np.asarray(mu, float)
        sqrtXwt = np.sqrt(weights)
        sqrtrwt = np.sqrt(weights)
        wtres = sqrtrwt * (y - mu)
        self.ptr = cpp.lmerResp(y, weights, offset, mu, sqrtXwt, sqrtrwt, wtres)
        # initializePtr(): lm_updateMu(Ptr, mu - offset); setREML
        self.ptr.updateMu(mu - offset)
        self.ptr.REML = int(REML or 0)

    @property
    def REML(self):
        return self.ptr.REML

    def copy(self):
        return lmerResp(_ptr=self.ptr.copy())

    def objective(self, ldL2, ldRX2, sqrL, sigma_sq=None):
        return self.ptr.Laplace(ldL2, ldRX2, sqrL, sigma_sq)


class glmResp(_RespBase):
    cls = "glmResp"

    def __init__(self, family=None, y=None, weights=None, offset=None, mu=None, n=None,
                 etastart=None, _ptr=None, **kw):
        self.family = family
        if _ptr is not None:
            self.ptr = _ptr
            return
        y = np.asarray(y, float)
        nobs = len(y)
        weights = np.ones(nobs) if weights is None else np.asarray(weights, float)
        offset = np.zeros(nobs) if offset is None else np.asarray(offset, float)
        if nobs == 0:
            # trivial response (simulate() from a formula); see lmerResp
            weights, offset = np.ones(0), np.zeros(0)
        mu = np.zeros(nobs) if mu is None else np.asarray(mu, float)
        sqrtXwt = np.sqrt(weights)
        sqrtrwt = np.sqrt(weights)
        wtres = sqrtrwt * (y - mu)
        nn = np.ones(nobs) if n is None else np.asarray(n, float)
        eta = np.zeros(nobs) if etastart is None else np.asarray(etastart, float)
        theta = family.theta if family.theta is not None else float("nan")
        self.ptr = cpp.glmResp(family.family, family.link, float(theta), y, weights, offset, mu,
                               sqrtXwt, sqrtrwt, wtres, eta, nn)
        self.ptr.updateMu(eta - offset)

    def copy(self):
        return glmResp(family=self.family, _ptr=self.ptr.copy())

    @property
    def eta(self):
        return np.asarray(self.ptr.eta)

    @property
    def n(self):
        return np.asarray(self.ptr.n)

    def devResid(self):
        return np.asarray(self.ptr.devResid())

    def setTheta(self, theta):
        """negative binomial theta"""
        from .family import negative_binomial
        self.ptr.setTheta(float(theta))
        self.family = negative_binomial(theta, link=self.family.link)


class nlsResp(_RespBase):
    cls = "nlsResp"

    def __init__(self, y=None, weights=None, offset=None, mu=None, sqrtXwt=None, gam=None,
                 model=None, pnames=None, _ptr=None, **kw):
        self.model = model
        self.pnames = pnames
        if _ptr is not None:
            self.ptr = _ptr
            return
        y = np.asarray(y, float)
        n = len(y)
        weights = np.ones(n) if weights is None else np.asarray(weights, float)
        mu = np.asarray(mu, float)
        sqrtrwt = np.sqrt(weights)
        wtres = sqrtrwt * (y - mu)
        gam = np.asarray(gam, float)
        offset = np.zeros(len(gam)) if offset is None else np.asarray(offset, float)
        self.ptr = cpp.nlsResp(y, weights, offset, mu, np.asarray(sqrtXwt, float), sqrtrwt, wtres,
                               gam, model, len(pnames))
        self.ptr.updateMu(gam)

    def copy(self):
        return nlsResp(model=self.model, pnames=self.pnames, _ptr=self.ptr.copy())
