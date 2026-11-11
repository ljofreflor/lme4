"""Transcription of lme4/R/covariance.R: structured covariance classes
for random-effects terms (``us``, ``diag``, ``cs``, ``ar1``).

Index arithmetic is kept 1-based (as in R) inside helper computations and
converted to 0-based only when indexing numpy arrays.
"""
from __future__ import annotations

import copy as _copy
import warnings
from typing import List

import numpy as np
import scipy.sparse as sp

SING_TOL = 1e-4


def getSingTol():
    from . import options
    return options.get("lme4.singular.tolerance", SING_TOL)


def sequence(nvec, from_=1, by=1):
    """R's sequence.default(nvec, from, by) (1-based values)"""
    nvec = np.atleast_1d(np.asarray(nvec, int))
    from_ = np.broadcast_to(np.atleast_1d(np.asarray(from_, int)), nvec.shape)
    by = np.broadcast_to(np.atleast_1d(np.asarray(by, int)), nvec.shape)
    out = [f + b * np.arange(n) for n, f, b in zip(nvec, from_, by)]
    return np.concatenate(out).astype(int) if out else np.zeros(0, int)


def _diag_pos(nc):
    """cumsum(c(if (nc > 0L) 1L, if (nc > 1L) nc:2L)) -- 1-based"""
    v = ([1] if nc > 0 else []) + (list(range(nc, 1, -1)) if nc > 1 else [])
    return np.cumsum(v).astype(int)


def semichol(x, tol=256 * np.finfo(float).eps):
    try:
        return np.linalg.cholesky(x).T
    except np.linalg.LinAlgError:
        pass
    vals, vecs = np.linalg.eigh(x.T)
    r = (vals.min(), vals.max())
    if r[1] < 0 or r[0] < -tol * r[1]:
        raise ValueError("'x' is not positive semidefinite")
    vals = np.where(vals < 0, 0, vals)
    R = np.linalg.qr(np.sqrt(vals)[:, None] * vecs.T, mode="r")
    return R


class Covariance:
    """virtual base class"""
    kind = None

    def __init__(self, nc, par=None, hom=None, simulate=False):
        self.nc = int(nc)
        if hom is not None:
            self.hom = bool(hom)
        if par is None:
            self.par = self._default_par(simulate)
        else:
            self.par = np.asarray(par, float).copy()

    def copy(self):
        return _copy.deepcopy(self)

    # par
    def getPar(self):
        return self.par

    def getParLength(self):
        return len(self.par)

    def setPar(self, value, pos=0):
        value = np.asarray(value, float)
        np_ = len(self.par)
        _validValuePos(np_, value, pos)
        o = self.copy()
        o.par = value.copy() if np_ == len(value) else value[pos:pos + np_].copy()
        return o

    def getLambdat_dp(self):
        return np.arange(1, self.nc + 1)

    def getLambdat_i(self):
        return sequence(np.arange(1, self.nc + 1), from_=0)

    def __repr__(self):
        h = f", hom={self.hom}" if hasattr(self, "hom") else ""
        return f"Covariance.{self.kind}(nc={self.nc}{h}, par={self.par})"


def _validValuePos(n, value, pos):
    if n > len(value) - pos:
        raise ValueError("attempt to read past end of 'value'")


class CovarianceUS(Covariance):
    kind = "us"

    def _default_par(self, simulate):
        nc = self.nc
        par = np.random.normal(size=(nc * (nc - 1)) // 2 + nc) if simulate else np.zeros((nc * (nc - 1)) // 2 + nc)
        par[_diag_pos(nc) - 1] = np.random.lognormal(size=nc) if simulate else 1.0
        return par

    def getParNames(self, cnm, gnm, prf=None, old=True):
        nc = self.nc
        inc = np.arange(1, nc + 1)
        dec = np.arange(nc, 0, -1)
        i = sequence(dec, from_=inc)
        j = np.repeat(inc, dec)
        cnm = list(cnm)
        if old:
            ans = [f"{gnm}.{cnm[a-1]}.{cnm[b-1]}" for a, b in zip(i, j)]
        elif prf is None:
            ans = [f"{cnm[a-1]}.{cnm[b-1]}|{gnm}" for a, b in zip(i, j)]
        else:
            ans = [f"{prf[1]}_{cnm[a-1]}.{cnm[b-1]}|{gnm}" for a, b in zip(i, j)]
        for k, p in enumerate(_diag_pos(nc)):
            ans[p - 1] = (f"{gnm}.{cnm[k]}" if old else f"{cnm[k]}|{gnm}" if prf is None
                          else f"{prf[0]}_{cnm[k]}|{gnm}")
        return ans

    getThetaNames = getParNames

    def getTheta(self):
        return self.par

    def getThetaLength(self):
        return len(self.par)

    def getThetaIndex(self):
        nc = self.nc
        snc = np.arange(1, nc + 1)
        base = np.repeat(np.arange(1 - nc, 1 - nc + nc), snc)
        cs = np.cumsum(np.arange(nc, 0, -1))
        return base + cs[sequence(snc, from_=1) - 1]

    def getThetaIndexLength(self):
        nc = self.nc
        return (nc * (nc - 1)) // 2 + nc

    def setTheta(self, value, pos=0):
        value = np.asarray(value, float)
        nt = len(self.par)
        _validValuePos(nt, value, pos)
        o = self.copy()
        o.par = value.copy() if nt == len(value) else value[pos:pos + nt].copy()
        return o

    def getLower(self):
        lo = np.full(len(self.par), -np.inf)
        lo[_diag_pos(self.nc) - 1] = 0.0
        return lo

    def getUpper(self):
        return np.full(len(self.par), np.inf)

    def getLambda(self):
        nc = self.nc
        L = np.zeros((nc, nc))
        if nc > 0:
            ii, jj = _lower_tri_colmajor(nc)
            L[ii, jj] = self.par
        return L

    def getVC(self):
        nc = self.nc
        if nc <= 1:
            return self.par.copy(), np.zeros(0)
        L = self.getLambda()
        S = L @ L.T
        vcomp = np.sqrt(np.diag(S))
        cor = S / vcomp[:, None] / vcomp[None, :]
        ii, jj = _strict_lower_colmajor(nc)
        return vcomp, cor[ii, jj]

    def getVCNames(self, cnm, gnm, prf=None, old=True):
        nms = self.getParNames(cnm, gnm, prf, old)
        ii = set((_diag_pos(self.nc) - 1).tolist())
        return [nms[k] for k in sorted(ii)], [n for k, n in enumerate(nms) if k not in ii]

    def setVC(self, vcomp, ccomp):
        nc = self.nc
        o = self.copy()
        vcomp = np.asarray(vcomp, float)
        ccomp = np.asarray(ccomp, float)
        if nc <= 1:
            o.par = vcomp.copy()
            return o
        S = np.eye(nc)
        ii, jj = _strict_lower_colmajor(nc)
        S[ii, jj] = ccomp         # lower triangle (semichol uses upper -> transpose below)
        S[jj, ii] = ccomp
        R = semichol(S) * vcomp[None, :]       # (semichol(S) * rep(vcomp, each = nc))
        # take i1: upper-triangle (row <= col) in column-major order of R == lower of R'
        Lt = R.T
        li, lj = _lower_tri_colmajor(nc)
        o.par = Lt[li, lj].copy()
        return o

    def getProfPar(self, profscale="sdcor", sc=None):
        vcomp, ccomp = self.getVC()
        if sc is not None:
            vcomp = vcomp * sc
        if profscale == "varcov":
            nc = self.nc
            if nc > 1:
                a = sequence(np.arange(nc - 1, 0, -1), from_=np.arange(2, nc + 1))
                b = np.repeat(np.arange(1, nc), np.arange(nc - 1, 0, -1))
                ccomp = ccomp * vcomp[a - 1] * vcomp[b - 1]
            vcomp = vcomp * vcomp
        return np.concatenate([vcomp, ccomp])

    def setProfPar(self, value, profscale="sdcor", sc=None, pos=0):
        np_ = len(self.par)
        nc = self.nc
        value = np.asarray(value, float)
        vcomp = value[pos:pos + nc]
        ccomp = value[pos + nc:pos + np_]
        if profscale == "varcov":
            vcomp = np.sqrt(vcomp)
            if nc > 1:
                a = sequence(np.arange(nc - 1, 0, -1), from_=np.arange(2, nc + 1))
                b = np.repeat(np.arange(1, nc), np.arange(nc - 1, 0, -1))
                ccomp = ccomp / vcomp[a - 1] / vcomp[b - 1]
        if sc is not None:
            vcomp = vcomp / (np.sqrt(sc) if profscale == "varcov" else sc)
        return self.setVC(vcomp, ccomp)

    def getProfLower(self, profscale="sdcor", sc=None):
        nc = self.nc
        return np.repeat([0.0, -np.inf if profscale == "varcov" else -1.0], [nc, (nc * (nc - 1)) // 2])

    def getProfUpper(self, profscale="sdcor", sc=None):
        nc = self.nc
        return np.repeat([np.inf, np.inf if profscale == "varcov" else 1.0], [nc, (nc * (nc - 1)) // 2])

    def isSingular(self, tol=None):
        tol = getSingTol() if tol is None else tol
        if self.nc > 0:
            return bool(self.par[_diag_pos(self.nc) - 1].min() < tol)
        return False


def _lower_tri_colmajor(nc):
    ii, jj = np.tril_indices(nc)
    o = np.lexsort((ii, jj))
    return ii[o], jj[o]


def _strict_lower_colmajor(nc):
    ii, jj = np.tril_indices(nc, -1)
    o = np.lexsort((ii, jj))
    return ii[o], jj[o]


class CovarianceDiag(Covariance):
    kind = "diag"

    def __init__(self, nc, par=None, hom=False, simulate=False):
        self.hom = bool(hom)
        super().__init__(nc, par, None, simulate)

    def _default_par(self, simulate):
        n = (1 if self.nc > 0 else 0) if self.hom else self.nc
        return np.random.lognormal(size=n) if simulate else np.ones(n)

    def getParNames(self, cnm, gnm, prf=None, old=True):
        cnm = list(cnm)
        if self.hom and self.nc > 0:
            cnm = ["*"]
        if old:
            return [f"{gnm}.{c}" for c in cnm]
        if prf is None:
            return [f"{c}|{gnm}" for c in cnm]
        return [f"{prf[0]}_{c}|{gnm}" for c in cnm]

    getThetaNames = getParNames

    def getTheta(self):
        return self.par

    def getThetaLength(self):
        return len(self.par)

    def getThetaIndex(self):
        return np.ones(self.nc, int) if self.hom else np.arange(1, self.nc + 1)

    def getThetaIndexLength(self):
        return self.nc

    def setTheta(self, value, pos=0):
        return CovarianceUS.setTheta(self, value, pos)

    def getLower(self):
        return np.zeros(len(self.par))

    def getUpper(self):
        return np.full(len(self.par), np.inf)

    def getLambda(self):
        p = self.par if not self.hom else np.repeat(self.par, self.nc)
        return np.diag(p) if self.nc > 0 else np.zeros((0, 0))

    def getLambdat_dp(self):
        return np.ones(self.nc, int)

    def getLambdat_i(self):
        return np.arange(self.nc)

    def getVC(self):
        return self.par.copy(), np.zeros(0)

    def getVCNames(self, cnm, gnm, prf=None, old=True):
        return self.getParNames(cnm, gnm, prf, old), []

    def setVC(self, vcomp, ccomp):
        o = self.copy()
        o.par = np.asarray(vcomp, float).copy()
        return o

    def getProfPar(self, profscale="sdcor", sc=None):
        v = self.par.copy()
        if sc is not None:
            v = v * sc
        if profscale == "varcov":
            v = v * v
        return v

    def setProfPar(self, value, profscale="sdcor", sc=None, pos=0):
        np_ = len(self.par)
        value = np.asarray(value, float)
        v = value.copy() if np_ == len(value) else value[pos:pos + np_]
        if profscale == "varcov":
            v = np.sqrt(v)
        if sc is not None:
            v = v / (np.sqrt(sc) if profscale == "varcov" else sc)
        return self.setPar(v)

    def getProfLower(self, profscale="sdcor", sc=None):
        return self.getLower()

    def getProfUpper(self, profscale="sdcor", sc=None):
        return self.getUpper()

    def isSingular(self, tol=None):
        tol = getSingTol() if tol is None else tol
        return bool(self.nc > 0 and self.par.min() < tol)


def _rhoFallback(L, ar1=False):
    d = np.sqrt((L * L).sum(axis=1))
    h = np.argsort(-d, kind="stable")[:2]
    nz = np.nonzero(d != 0)[0]
    if len(nz) >= 2:
        k = int(np.argmin(np.diff(nz)))
        h = nz[[k, k + 1]]
    d12 = d[h].prod()
    if d12 == 0:
        warnings.warn("'rho' is not identifiable as there is no pair of nonzero standard deviations")
        rho = np.nan
    else:
        ratio = (L[h[0]] * L[h[1]]).sum() / d12
        rho = np.sign(ratio) * abs(ratio) ** (1 / abs(h[0] - h[1])) if ar1 else ratio
    return d, rho


class _CovCSAR1Base(Covariance):
    def __init__(self, nc, par=None, hom=None, simulate=False):
        self.hom = self._default_hom if hom is None else bool(hom)
        super().__init__(nc, par, None, simulate)

    def _nsd(self):
        return (1 if self.nc > 0 else 0) if self.hom else self.nc

    def getParNames(self, cnm, gnm, prf=None, old=True):
        nc = self.nc
        cnm = list(cnm)
        if self.hom and nc > 0:
            cnm = ["*"]
        lab = cnm + (["rho"] if nc > 1 else [])
        if old:
            return [f"{gnm}.{c}" for c in lab]
        if prf is None:
            return [f"{c}|{gnm}" for c in lab]
        return [f"{prf[0]}_{c}|{gnm}" for c in cnm] + ([f"{prf[1]}_rho|{gnm}"] if nc > 1 else [])

    def getThetaLength(self):
        nc = self.nc
        return 2 * nc - (1 if nc > 0 else 0) if self.hom else (nc * (nc - 1)) // 2 + nc

    def getThetaIndexLength(self):
        nc = self.nc
        return (nc * (nc - 1)) // 2 + nc

    def getUpper(self):
        nc = self.nc
        return np.concatenate([np.full(self._nsd(), np.inf), [1.0] if nc > 1 else []])

    def getVC(self):
        if self.nc <= 1:
            return self.par.copy(), np.zeros(0)
        return self.par[:-1].copy(), self.par[-1:].copy()

    def getVCNames(self, cnm, gnm, prf=None, old=True):
        nms = self.getParNames(cnm, gnm, prf, old)
        if self.nc <= 1:
            return nms, []
        return nms[:self.nc], [nms[self.nc]]

    def setVC(self, vcomp, ccomp):
        o = self.copy()
        o.par = np.concatenate([np.asarray(vcomp, float), np.asarray(ccomp, float)])
        return o

    def getProfPar(self, profscale="sdcor", sc=None):
        if profscale == "varcov":
            raise NotImplementedError(f'profscale="varcov" not implemented for class "Covariance.{self.kind}"')
        vcomp, ccomp = self.getVC()
        if sc is not None:
            vcomp = vcomp * sc
        return np.concatenate([vcomp, ccomp])

    def setProfPar(self, value, profscale="sdcor", sc=None, pos=0):
        if profscale == "varcov":
            raise NotImplementedError(f'profscale="varcov" not implemented for class "Covariance.{self.kind}"')
        np_ = len(self.par)
        nc_ = self._nsd()
        value = np.asarray(value, float)
        vcomp = value[pos:pos + nc_]
        ccomp = value[pos + nc_:pos + np_]
        if sc is not None:
            vcomp = vcomp / sc
        return self.setVC(vcomp, ccomp)

    def getProfLower(self, profscale="sdcor", sc=None):
        if profscale == "varcov":
            raise NotImplementedError
        return self.getLower()

    def getProfUpper(self, profscale="sdcor", sc=None):
        if profscale == "varcov":
            raise NotImplementedError
        return self.getUpper()


class CovarianceCS(_CovCSAR1Base):
    kind = "cs"
    _default_hom = False

    def _default_par(self, simulate):
        nc = self.nc
        sd = np.random.lognormal(size=self._nsd()) if simulate else np.ones(self._nsd())
        rho = ([np.random.uniform(-1 / (nc - 1), 1)] if simulate else [0.0]) if nc > 1 else []
        return np.concatenate([sd, rho])

    def getTheta(self):
        nc = self.nc
        par = self.par
        if nc <= 1:
            return par.copy()
        rho = par[-1]
        rho2 = rho * rho
        a = np.zeros(nc)
        a_ = 0.0
        for j in range(nc):
            a[j] = a_
            a_ = a_ + (1 - rho * a_) ** 2 / (1 - rho2 * a_)
        with np.errstate(invalid="ignore"):
            v_ = np.sqrt(1 - rho2 * a)
        v = np.vstack([v_, (rho - rho2 * a) / v_])      # 2 x nc, column-major flatten
        vflat = v.T.ravel()
        if self.hom:
            return par[0] * vflat[:2 * nc - 1]
        idx = sequence(np.arange(nc, 0, -1), from_=np.arange(1, nc + 1))
        reps = np.vstack([np.ones(nc, int), np.arange(nc - 1, -1, -1)]).T.ravel()
        return par[idx - 1] * np.repeat(vflat, reps)

    def getThetaNames(self, cnm, gnm, prf=None, old=True):
        if self.hom:
            nc = self.nc
            cnm = list(cnm)
            a = [f"{gnm}.{c}" if old else f"{c}|{gnm}" if prf is None else f"{prf[0]}_{c}|{gnm}" for c in cnm]
            b = [f"{gnm}.*.{c}" if old else f"*.{c}|{gnm}" if prf is None else f"{prf[1]}_*.{c}|{gnm}" for c in cnm]
            inter = [x for pair in zip(a, b) for x in pair]
            return inter[:2 * nc - 1]
        return CovarianceUS.getParNames(self, cnm, gnm, prf, old)

    def getThetaIndex(self):
        if self.hom:
            nc = self.nc
            snc = np.arange(1, nc + 1)
            out = sequence(snc, from_=2, by=2)
            out[np.cumsum(snc) - 1] = np.arange(1, 2 * nc, 2)
            return out
        return CovarianceUS.getThetaIndex(self)

    def setTheta(self, value, pos=0):
        nc = self.nc
        hom = self.hom
        value = np.asarray(value, float)
        nt = self.getThetaLength()
        _validValuePos(nt, value, pos)
        o = self.copy()
        if nc <= 1:
            o.par = value[pos:pos + 1].copy() if nc > 0 else np.zeros(0)
            return o
        l21 = value[pos + 1]
        l22 = value[pos + (2 if hom else nc)]
        d = None
        if l21 != 0 or l22 != 0:
            rho = np.sign(l21) * 1 / np.sqrt(1 + (l22 / l21) ** 2) if l21 != 0 else 0.0
        else:
            i = sequence(np.arange(nc, 0, -1), from_=np.arange(1, nc * (nc + 1) + 1, nc + 1)[:nc])
            if hom:
                j = np.repeat(np.arange(pos + 1, pos + 2 * nc), np.vstack(
                    [np.ones(nc, int), np.arange(nc - 1, -1, -1)]).T.ravel()[:2 * nc - 1])
            else:
                j = np.arange(pos + 1, pos + (nc * (nc - 1)) // 2 + nc + 1)
            L = np.zeros(nc * nc)
            L[i - 1] = value[j - 1]
            d, rho = _rhoFallback(L.reshape(nc, nc, order="F"))
        if hom:
            sigma = value[pos:pos + 1]
        elif np.isnan(rho):
            sigma = d
        elif rho == 0:
            sigma = value[np.cumsum([pos + 1] + list(range(nc, 1, -1))) - 1]
        else:
            sigma = value[pos:pos + nc] / np.concatenate([[1.0], np.full(nc - 1, rho)])
        o.par = np.concatenate([sigma, [rho]])
        return o

    def getLower(self):
        nc = self.nc
        return np.concatenate([np.zeros(self._nsd()), [-1 / (nc - 1)] if nc > 1 else []])

    def getLambda(self):
        nc = self.nc
        L = np.zeros((nc, nc))
        if nc > 0:
            th = self.getTheta()
            if self.hom:
                reps = np.vstack([np.ones(nc, int), np.arange(nc - 1, -1, -1)]).T.ravel()[:2 * nc - 1]
                th = np.repeat(th, reps)
            ii, jj = _lower_tri_colmajor(nc)
            L[ii, jj] = th
        return L

    def isSingular(self, tol=None):
        tol = getSingTol() if tol is None else tol
        nc = self.nc
        if nc > 0:
            vcomp, ccomp = self.getVC()
            return bool(vcomp.min() < tol or (nc > 1 and min(ccomp[0] + 1 / (nc - 1), 1 - ccomp[0]) < tol))
        return False


class CovarianceAR1(_CovCSAR1Base):
    kind = "ar1"
    _default_hom = True

    def _default_par(self, simulate):
        nc = self.nc
        sd = np.random.lognormal(size=self._nsd()) if simulate else np.ones(self._nsd())
        rho = ([np.random.uniform(-1, 1)] if simulate else [0.0]) if nc > 1 else []
        return np.concatenate([sd, rho])

    def getTheta(self):
        nc = self.nc
        par = self.par
        if nc <= 1:
            return par.copy()
        rho = par[-1]
        rho2 = rho * rho
        v1 = rho ** np.arange(nc)
        v2 = rho ** np.arange(nc - 1) * np.sqrt(1 - rho2)
        if self.hom:
            return par[0] * np.concatenate([v1, v2])
        idx = sequence(np.arange(nc, 0, -1), from_=np.arange(1, nc + 1))
        return par[idx - 1] * np.concatenate([v1, v2[sequence(np.arange(nc - 1, 0, -1), from_=1) - 1]])

    def getThetaNames(self, cnm, gnm, prf=None, old=True):
        if self.hom:
            nc = self.nc
            cnm = list(cnm)
            i = np.concatenate([np.arange(1, nc + 1), np.arange(2, nc + 1)]) if nc > 1 else np.array([nc])
            j = np.repeat([1, 2], [nc, nc - 1]) if nc > 1 else np.array([nc])
            k = np.arange(1, min(2, nc) + 1)
            fmt = (lambda a, b: f"{gnm}.{a}.{b}") if old else (lambda a, b: f"{a}.{b}|{gnm}") if prf is None \
                else (lambda a, b: f"{prf[1]}_{a}.{b}|{gnm}")
            ans = [fmt(cnm[a - 1], cnm[b - 1]) for a, b in zip(i, j)]
            pos = np.array([1, nc + 1])[k - 1]
            for p, kk in zip(pos, k):
                ans[p - 1] = (f"{gnm}.{cnm[kk-1]}" if old else f"{cnm[kk-1]}|{gnm}" if prf is None
                              else f"{prf[1]}_{cnm[kk-1]}|{gnm}")
            return ans
        return CovarianceUS.getParNames(self, cnm, gnm, prf, old)

    def getThetaIndex(self):
        if self.hom:
            nc = self.nc
            snc = np.arange(1, nc + 1)
            out = sequence(snc, from_=np.arange(nc + 1, 2 * nc + 1), by=-1)
            out[1 + np.cumsum(np.arange(nc)) - 1] = snc
            return out
        return CovarianceUS.getThetaIndex(self)

    def setTheta(self, value, pos=0):
        nc = self.nc
        hom = self.hom
        value = np.asarray(value, float)
        nt = self.getThetaLength()
        _validValuePos(nt, value, pos)
        o = self.copy()
        if nc <= 1:
            o.par = value[pos:pos + 1].copy() if nc > 0 else np.zeros(0)
            return o
        l21 = value[pos + 1]
        l22 = value[pos + nc]
        d = None
        if l21 != 0 or l22 != 0:
            rho = np.sign(l21) * 1 / np.sqrt(1 + (l22 / l21) ** 2) if l21 != 0 else 0.0
        else:
            i = sequence(np.arange(nc, 0, -1), from_=np.arange(1, nc * (nc + 1) + 1, nc + 1)[:nc])
            if hom:
                j = sequence(np.arange(nc, 0, -1), from_=np.repeat(pos + 1 + np.array([0, nc]), [1, nc - 1]))
            else:
                j = np.arange(pos + 1, pos + (nc * (nc - 1)) // 2 + nc + 1)
            L = np.zeros(nc * nc)
            L[i - 1] = value[j - 1]
            d, rho = _rhoFallback(L.reshape(nc, nc, order="F"), ar1=True)
        if hom:
            sigma = value[pos:pos + 1]
        elif np.isnan(rho):
            sigma = d
        elif rho == 0:
            sigma = value[np.cumsum([pos + 1] + list(range(nc, 1, -1))) - 1]
        else:
            sigma = np.exp(np.log(np.abs(value[pos:pos + nc])) - np.arange(nc) * np.log(abs(rho)))
        o.par = np.concatenate([sigma, [rho]])
        return o

    def getLower(self):
        nc = self.nc
        return np.concatenate([np.zeros(self._nsd()), [-1.0] if nc > 1 else []])

    def getLambda(self):
        nc = self.nc
        L = np.zeros((nc, nc))
        if nc > 0:
            th = self.getTheta()
            if self.hom:
                th = th[sequence(np.arange(nc, 0, -1), from_=np.repeat([1, nc + 1], [1, nc - 1])) - 1]
            ii, jj = _lower_tri_colmajor(nc)
            L[ii, jj] = th
        return L

    def isSingular(self, tol=None):
        tol = getSingTol() if tol is None else tol
        nc = self.nc
        if nc > 0:
            vcomp, ccomp = self.getVC()
            return bool(vcomp.min() < tol or (nc > 1 and 1 - abs(ccomp[0]) < tol))
        return False


COV_CLASSES = {"us": CovarianceUS, "diag": CovarianceDiag, "cs": CovarianceCS, "ar1": CovarianceAR1}


def new_cov(kind, nc, hom=None):
    cls = COV_CLASSES[kind]
    if kind == "us":
        return cls(nc)
    return cls(nc, hom=hom) if hom is not None else cls(nc)


def mkMkPar(reCovs):
    if all(isinstance(c, CovarianceUS) for c in reCovs):
        return lambda theta: np.asarray(theta, float)
    nt = [c.getThetaLength() for c in reCovs]
    np_ = [c.getParLength() for c in reCovs]
    jt = np.cumsum([0] + nt)
    jp = np.cumsum([0] + np_)

    def mkPar(theta):
        theta = np.asarray(theta, float)
        par = np.zeros(jp[-1])
        for i, c in enumerate(reCovs):
            par[jp[i]:jp[i + 1]] = c.setTheta(theta[jt[i]:jt[i + 1]]).getPar()
        return par
    return mkPar


def mkMkTheta(reCovs):
    if all(isinstance(c, CovarianceUS) for c in reCovs):
        return lambda par: np.asarray(par, float)
    nt = [c.getThetaLength() for c in reCovs]
    np_ = [c.getParLength() for c in reCovs]
    jt = np.cumsum([0] + nt)
    jp = np.cumsum([0] + np_)

    def mkTheta(par):
        par = np.asarray(par, float)
        theta = np.zeros(jt[-1])
        for i, c in enumerate(reCovs):
            theta[jt[i]:jt[i + 1]] = c.setPar(par[jp[i]:jp[i + 1]]).getTheta()
        return theta
    return mkTheta


def anyStructured(reCovs):
    return not all(isinstance(c, CovarianceUS) for c in reCovs)


def upReTrms(reTrms, spCalls):
    """lme4:::upReTrms -- build Lambdat/Lind/theta/par/lower/upper from the
    covariance structures named by the special calls"""
    from .rlang import Const
    reCovs = []
    for (gnm, cn), sc in zip(reTrms["cnms"], spCalls):
        kind = sc.fn.name
        hom = sc.arg("hom")
        if hom is not None:
            if not (isinstance(hom, Const) and isinstance(hom.value, bool)):
                raise ValueError("'hom' is not TRUE or FALSE in special call")
            hom = hom.value
        if kind not in COV_CLASSES:
            raise ValueError(f"covariance structure '{kind}' is not supported")
        reCovs.append(new_cov(kind, len(cn), hom))
    nl = reTrms["nl"]
    reTrms["Lambdat"], reTrms["Lind"], reTrms["theta"] = _build_lambdat(reCovs, nl)
    reTrms["par"] = np.concatenate([c.getPar() for c in reCovs])
    reTrms["lower"] = np.concatenate([c.getLower() for c in reCovs])
    reTrms["upper"] = np.concatenate([c.getUpper() for c in reCovs])
    reTrms["reCovs"] = reCovs
    return reTrms


def _build_lambdat(reCovs, nl, theta=None):
    nc = np.array([c.nc for c in reCovs])
    thetas = [c.getTheta() for c in reCovs]
    tidx = [c.getThetaIndex() for c in reCovs]
    nt = np.array([len(t) for t in thetas])
    nti = np.array([len(t) for t in tidx])
    nl = np.asarray(nl, int)
    nc_nl = np.repeat(nc, nl)
    nti_nl = np.repeat(nti, nl)
    R_dp = np.concatenate([np.tile(c.getLambdat_dp(), l) for c, l in zip(reCovs, nl)])
    R_p = np.concatenate([[0], np.cumsum(R_dp)])
    starts = np.concatenate([[0], np.cumsum(nc_nl)])[:len(nc_nl)]
    R_i = np.repeat(starts, nti_nl) + np.concatenate(
        [np.tile(c.getLambdat_i(), l) for c, l in zip(reCovs, nl)])
    toff = np.concatenate([[0], np.cumsum(nt)])[:len(nt)]
    R_x = np.repeat(toff, nti * nl) + np.concatenate([np.tile(t, l) for t, l in zip(tidx, nl)])
    q = len(R_dp)
    th = np.concatenate(thetas) if theta is None else np.asarray(theta, float)
    Lt = sp.csc_matrix((th[R_x - 1], R_i, R_p), shape=(q, q))
    return Lt, R_x.astype(np.int32), np.concatenate(thetas)


def upReCovs(reCovs, theta):
    pos = 0
    out = []
    for c in reCovs:
        out.append(c.setTheta(theta, pos))
        pos += c.getThetaLength()
    return out
