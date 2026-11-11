"""The fitted-model classes (``merMod``, ``lmerMod``, ``glmerMod``,
``nlmerMod``) and their methods, transcribed from lme4/R/lmer.R,
lme4/R/methods.R and lme4/R/utilities.R."""
from __future__ import annotations

import math
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from . import options
from .covariance import getSingTol, upReCovs, mkMkPar, mkMkTheta
from .family import gaussian, hasNoScale, isNBfamily
from .modelframe import Factor
from .rformat import r_format, printCoefmat, print_named_vector, format_char_matrix, abbreviate, abbreviate_vec
from .rlang import deparse
from .varcorr import VarCorr as _VarCorr, mkVarCorr, formatVC


class merMod:
    """Fitted mixed-effects model (R's S4 class ``merMod``)."""

    def __init__(self, call, frame, flist, flist_assign, cnms, Gp, theta, beta, u, lower,
                 devcomp, pp, resp, optinfo):
        self.call = call
        self.frame = frame
        self.flist = flist
        self.flist_assign = np.asarray(flist_assign, int)
        self.cnms = cnms                 # list of (group name, colnames)
        self.Gp = np.asarray(Gp, int)
        self.theta = np.asarray(theta, float)
        self.beta = np.asarray(beta, float)
        self.u = np.asarray(u, float)
        self.lower = np.asarray(lower, float)
        self.devcomp = devcomp
        self.pp = pp
        self.resp = resp
        self.optinfo = optinfo
        self.upper = None
        self.reCovs = None
        self.wmsgs = {}

    # ------------------------------------------------------------ predicates
    def isREML(self):
        return bool(self.devcomp["dims"]["REML"])

    def isGLMM(self):
        return bool(self.devcomp["dims"]["GLMM"])

    def isNLMM(self):
        return bool(self.devcomp["dims"]["NLMM"])

    def isLMM(self):
        return not self.isGLMM() and not self.isNLMM()

    def isSingular(self, tol=None):
        tol = getSingTol() if tol is None else tol
        return any(c.isSingular(tol) for c in self.getReCovs())

    # ------------------------------------------------------------ basics
    @property
    def colnames_X(self):
        return list(self.pp.Xattrs.get("colnames") or [f"X{i+1}" for i in range(len(self.beta))])

    def getReCovs(self):
        return self.reCovs

    def getUpper(self):
        return self.upper if self.upper is not None else np.full(len(self.lower), np.inf)

    def nobs(self):
        return self.frame.nrow

    def ngrps(self):
        return pd.Series({k: f.nlevels for k, f in self.flist.items()}, dtype=int)

    def sigma(self):
        dc = self.devcomp
        if dc["dims"]["useSc"]:
            return float(dc["cmp"]["sigmaREML" if dc["dims"]["REML"] else "sigmaML"])
        return 1.0

    def family(self):
        if isinstance(self, glmerMod):
            return self.resp.family
        return gaussian()

    def fixef(self, add_dropped=False, noScale=None):
        ff = pd.Series(self.beta, index=self.colnames_X, dtype=float)
        xa = self.pp.Xattrs
        if not noScale and xa.get("scaled:scale") is not None:
            sc = xa["scaled:scale"]
            ce = xa.get("scaled:center", {k: 0.0 for k in sc})
            if "(Intercept)" in ff.index:
                shift = sum(ff[k] * ce[k] / sc[k] for k in sc)
                ff["(Intercept)"] -= shift
            for k in sc:
                ff[k] = ff[k] / sc[k]
        if add_dropped and xa.get("col.dropped"):
            dd = xa["col.dropped"]
            names = list(ff.index)
            vals = list(ff.values)
            for nm, pos in sorted(dd.items(), key=lambda kv: kv[1]):
                names.insert(pos - 1, nm)
                vals.insert(pos - 1, np.nan)
            ff = pd.Series(vals, index=names)
        return ff

    def getParLength(self):
        return sum(c.getParLength() for c in self.getReCovs())

    def getParNames(self, old=True, prf=None):
        out = []
        for c, (g, cn) in zip(self.getReCovs(), self.cnms):
            out += c.getParNames(cn, g, prf, old)
        return out

    def getThetaNames(self, old=True, prf=None):
        out = []
        for c, (g, cn) in zip(self.getReCovs(), self.cnms):
            out += c.getThetaNames(cn, g, prf, old)
        return out

    def npar(self):
        n = len(self.beta) + self.getParLength() + self.devcomp["dims"]["useSc"]
        if "Negative Binomial" in self.family().family:
            n += 1
        return n

    def df_residual(self):
        return self.nobs() - self.npar()

    # ------------------------------------------------------------ deviance etc
    def devCrit(self, REML=None):
        if REML is True and not self.isLMM():
            raise ValueError("can't compute REML deviance for a non-LMM")
        cmp = self.devcomp["cmp"]
        if REML is None:
            REML = self.isREML()
        n = self.devcomp["dims"]["n"]
        ldW = np.sum(np.log(self.weights("prior")))
        if REML:
            if self.isREML():
                return cmp["REML"]
            lnum = math.log(2 * math.pi * cmp["pwrss"])
            nmp = n - len(self.beta)
            return -ldW + cmp["ldL2"] + cmp["ldRX2"] + nmp * (1 + lnum - math.log(nmp))
        if not self.isREML():
            return cmp["dev"]
        lnum = math.log(2 * math.pi * cmp["pwrss"])
        return -ldW + cmp["ldL2"] + n * (1 + lnum - math.log(n))

    def REMLcrit(self):
        return self.devCrit(REML=True)

    def deviance(self, REML=None):
        if self.isGLMM():
            return float(np.sum(self.residuals(type="deviance") ** 2))
        if self.isREML() and REML is None:
            warnings.warn("deviance() is deprecated for REML fits; use REMLcrit for the REML "
                          "criterion or deviance(.,REML=FALSE) for deviance calculated at the REML fit")
            return self.devCrit(REML=True)
        return self.devCrit(REML=False)

    def logLik(self, REML=None):
        if REML is None:
            REML = self.isREML()
        val = -self.devCrit(REML=REML) / 2
        return LogLik(val, df=self.npar(), nobs=self.nobs(), REML=REML)

    def AIC(self, k=2):
        ll = self.logLik()
        return -2 * ll.value + k * ll.df

    def BIC(self):
        ll = self.logLik()
        return -2 * ll.value + math.log(ll.nobs) * ll.df

    def extractAIC(self, scale=0, k=2):
        L = self.refitML().logLik()
        return L.df, -2 * L.value + k * L.df

    # ------------------------------------------------------------ random effects
    def VarCorr(self, sigma=None):
        useSc = not self.isGLMM()
        sc = self.sigma() if sigma is None else sigma
        nms = [list(self.flist)[a - 1] for a in self.flist_assign]
        cnms = [(n, cn) for n, (_, cn) in zip(nms, self.cnms)]
        items = mkVarCorr(sc, cnms, self.theta, reCovs=self.getReCovs(), relReCovs=useSc)
        return _VarCorr(items, sc, useSc)

    def _b(self):
        return np.asarray(self.pp.b(1.0))

    def ranef(self, condVar=True, drop=False, whichel=None):
        """conditional modes of the random effects: dict name -> DataFrame
        (the conditional variances are in ``df.attrs['postVar']``, shape (nc, nc, nlev))"""
        ans = self._b()
        fl = self.flist
        levs = {k: f.levels for k, f in fl.items()}
        nc = [len(cn) for _, cn in self.cnms]
        nb = np.diff(self.Gp)
        ml = []
        pos = 0
        for i, k in enumerate(nb):
            blk = ans[pos:pos + k].reshape(-1, nc[i])
            ml.append(pd.DataFrame(blk, columns=self.cnms[i][1]))
            pos += k
        out = {}
        for i, nm in enumerate(fl):
            m = [ml[j] for j in range(len(ml)) if self.flist_assign[j] == i + 1]
            ub2 = {len(x) for x in m}
            if len(ub2) > 1:
                raise ValueError("differing numbers of b per group")
            nr = ub2.pop()
            rn = levs[nm] if nr == len(levs[nm]) else list(range(1, nr + 1))
            df = pd.concat([x.reset_index(drop=True) for x in m], axis=1)
            df.index = rn
            out[nm] = df
        if whichel is not None:
            whichel = [whichel] if isinstance(whichel, str) else whichel
            out = {k: v for k, v in out.items() if k in whichel}
        if condVar:
            vv = self.condVar(scaled=True)
            for k in out:
                out[k].attrs["postVar"] = vv[k]
        if drop:
            out2 = {}
            for k, v in out.items():
                if v.shape[1] > 1:
                    out2[k] = v
                else:
                    s = v.iloc[:, 0].copy()
                    if "postVar" in v.attrs:
                        s.attrs["postVar"] = v.attrs["postVar"].reshape(-1)
                    out2[k] = s
            out = out2
        return RanefMer(out)

    def condVar(self, scaled=True):
        """conditional variances of the random effects (per grouping factor)"""
        cnms = self.cnms
        asgn = self.flist_assign
        ncols = [len(cn) for _, cn in cnms]
        fnames = list(self.flist)
        nlevs = [self.flist[k].nlevels for k in fnames]
        nctot = [sum(ncols[j] for j in range(len(asgn)) if asgn[j] == i + 1) for i in range(len(fnames))]
        offsets = np.concatenate([[0], np.cumsum([ncols[j] * nlevs[asgn[j] - 1] for j in range(len(asgn))])]).astype(int)
        terms = [[j for j in range(len(asgn)) if asgn[j] == i + 1] for i in range(len(fnames))]
        s2 = self.sigma() ** 2 if scaled else 1.0
        out = {}
        if all(len(t) == 1 for t in terms):
            vv = self.pp.condVar(nlevs, nctot, offsets.tolist(), terms)
            for i, k in enumerate(fnames):
                m = np.asarray(vv[i]) * s2
                nc_ = nctot[i]
                out[k] = m.reshape(nc_, nc_, nlevs[i], order="F")
        else:
            # several terms per factor: use the full conditional covariance (arrange.condVar)
            L = self.pp.L()
            Lt = sp.csc_matrix(self.pp.Lambdat)
            q = Lt.shape[0]
            for i, k in enumerate(fnames):
                nl = nlevs[i]
                idx_by_term = []
                for j in terms[i]:
                    st = self.Gp[j]
                    idx_by_term.append(st + np.arange(ncols[j] * nl).reshape(nl, ncols[j]))
                arr = np.zeros((nctot[i], nctot[i], nl))
                for lev in range(nl):
                    idx = np.concatenate([t[lev] for t in idx_by_term])
                    LvT = Lt[:, idx].toarray()
                    arr[:, :, lev] = LvT.T @ L.solve(LvT, "A") * s2
                out[k] = arr
        return out

    def coef(self):
        fef = self.fixef()
        ref = self.ranef(condVar=False)
        out = {}
        for k, r in ref.items():
            val = pd.DataFrame(np.tile(fef.values, (len(r), 1)), index=r.index, columns=fef.index)
            for nm in r.columns:
                if nm not in val.columns:
                    val.insert(0, nm, 0.0)
                val[nm] = val[nm] + r[nm].values
            out[k] = val
        return out

    # ------------------------------------------------------------ fitted / residuals
    def fitted(self):
        mu = np.asarray(self.resp.mu)
        return pd.Series(mu, index=self.frame.row_names)

    def residuals(self, type=None, scaled=False):
        if type is None:
            type = "deviance" if self.isGLMM() else "response"
        r = self.resp
        y, mu = r.y, r.mu
        if isinstance(self, glmerMod):
            if type == "deviance":
                d = np.sqrt(np.maximum(0, r.devResid()))
                res = np.where(y > mu, d, -d)
            elif type == "pearson":
                res = r.wtres
            elif type == "working":
                res = np.asarray(r.wrkResids())
            elif type == "response":
                res = y - mu
            else:
                raise NotImplementedError("partial residuals are not implemented yet")
        else:
            if type in ("working", "response"):
                res = y - mu
            elif type in ("deviance", "pearson"):
                res = r.wtres
            else:
                raise NotImplementedError("partial residuals are not implemented yet")
        if scaled:
            res = res / self.sigma()
        return pd.Series(np.asarray(res), index=self.frame.row_names)

    resid = residuals

    def weights(self, type="prior"):
        if type == "prior":
            return np.asarray(self.resp.weights)
        if type == "working":
            if self.isGLMM():
                return np.asarray(self.pp.Xwts) ** 2
            return np.asarray(self.resp.weights)
        raise ValueError(type)

    def hatvalues(self, fullHatMatrix=False):
        if self.isGLMM():
            warnings.warn("the hat matrix may not make sense for GLMMs")
        sqrtW = sp.diags(np.sqrt(self.weights("prior")))
        L = self.pp.L()
        Lt = sp.csc_matrix(self.pp.Lambdat)
        Zt = sp.csc_matrix(self.pp.Zt)
        RX = np.asarray(self.pp.RX())
        X = np.asarray(self.pp.X)
        RZX = np.asarray(self.pp.RZX)
        CL = L.solve(L.solve((Lt @ Zt @ sqrtW).toarray(), "P"), "L")
        CR = np.linalg.solve(RX.T, (X.T @ sqrtW.toarray()) - RZX.T @ CL)
        if fullHatMatrix:
            return CL.T @ CL + CR.T @ CR
        return (CR ** 2).sum(axis=0) + (CL ** 2).sum(axis=0)

    # ------------------------------------------------------------ vcov
    def vcov(self, correlation=True, sigm=None, use_hessian=None, full=False, noScale=None):
        sigm = self.sigma() if sigm is None else sigm
        if full:
            return self.vcov_full(sigm)
        h = (self.optinfo.get("derivs") or {}).get("Hessian")
        npar = self.getParLength()
        hess_avail = h is not None and np.asarray(h).shape[0] > npar
        if use_hessian is None:
            use_hessian = hess_avail
        if use_hessian and not hess_avail:
            raise ValueError("'use.hessian'=TRUE specified, but Hessian is unavailable")
        V = sigm ** 2 * np.asarray(self.pp.unsc())
        bad_V_hess = True
        if hess_avail:
            try:
                hi = np.linalg.solve(np.asarray(h), np.eye(len(h)))
            except np.linalg.LinAlgError:
                hi = np.full(np.shape(h), np.nan)
            hi = hi[npar:, npar:]
            V_hess = hi + hi.T
            bad_V_hess = bool(np.any(np.isnan(V_hess)))
            if not bad_V_hess and np.linalg.eigvalsh(V_hess).min() <= 0:
                bad_V_hess = True
            if not use_hessian and not bad_V_hess and np.any(np.abs(V - V_hess) > 1e-4 * V_hess):
                warnings.warn("variance-covariance matrix computed from finite-difference Hessian\n"
                              "and from RX differ by >0.0001: consider 'use.hessian=TRUE'")
        if use_hessian:
            if not bad_V_hess:
                V = V_hess
            else:
                warnings.warn("variance-covariance matrix computed from finite-difference Hessian is\n"
                              "not positive definite or contains NA values: falling back to var-cov estimated from RX")
        nms = self.colnames_X
        xa = self.pp.Xattrs
        if not noScale and xa.get("scaled:scale") is not None:
            V = _scale_vcov(V, nms, xa["scaled:scale"], xa.get("scaled:center"))
        return pd.DataFrame(V, index=nms, columns=nms)

    def vcov_full(self, s=None):
        s = self.sigma() if s is None else s
        L = self.pp.L()
        RX = np.asarray(self.pp.RX())
        RZX = np.asarray(self.pp.RZX)
        Lt = sp.csc_matrix(self.pp.Lambdat).toarray()
        RXtinv = np.linalg.inv(RX.T)
        LinvLt = L.solve(L.solve(Lt, "P"), "L")
        q, p = Lt.shape[0], RX.shape[0]
        Minv = s * np.block([[LinvLt, np.zeros((q, p))], [-RXtinv @ RZX.T @ LinvLt, RXtinv]])
        C = Minv.T @ Minv
        rr = self.ranef(condVar=False)
        re_nms = []
        for k, r in rr.items():
            re_nms += [f"{k}.{lev}.{c}" for lev in r.index for c in r.columns]
        nms = re_nms + self.colnames_X
        return pd.DataFrame(C, index=nms, columns=nms)

    # ------------------------------------------------------------ getME
    def getME(self, name):
        if isinstance(name, (list, tuple)):
            return {n: self.getME(n) for n in name}
        PR, rsp, dims = self.pp, self.resp, self.devcomp["dims"]
        if name == "X":
            return np.asarray(PR.X)
        if name == "Z":
            return sp.csc_matrix(PR.Zt).T.tocsc()
        if name == "Zt":
            return sp.csc_matrix(PR.Zt)
        if name == "y":
            return rsp.y
        if name == "mu":
            return rsp.mu
        if name == "u":
            return self.u
        if name == "b":
            return sp.csc_matrix(PR.Lambdat).T @ self.u
        if name == "L":
            return PR.L()
        if name == "Lambda":
            return sp.csc_matrix(PR.Lambdat).T.tocsc()
        if name == "Lambdat":
            return sp.csc_matrix(PR.Lambdat)
        if name == "A":
            return sp.csc_matrix(PR.Lambdat) @ sp.csc_matrix(PR.Zt)
        if name == "Lind":
            return np.asarray(PR.Lind)
        if name == "RX":
            cn = self.colnames_X
            return pd.DataFrame(np.asarray(PR.RX()), index=cn, columns=cn)
        if name == "RZX":
            return pd.DataFrame(np.asarray(PR.RZX), columns=self.colnames_X)
        if name == "sigma":
            return self.sigma()
        if name == "Gp":
            return self.Gp
        if name == "Tp":
            sizes = [len(cn) * (len(cn) + 1) // 2 for _, cn in self.cnms]
            return pd.Series(np.concatenate([[0], np.cumsum(sizes)]),
                             index=["beg__"] + [g for g, _ in self.cnms])
        if name == "flist":
            return self.flist
        if name == "fixef":
            return self.fixef()
        if name == "beta":
            return self.beta
        if name == "theta":
            return pd.Series(self.theta, index=self.getThetaNames())
        if name == "par":
            return pd.Series(self.optinfo["val"][:self.getParLength()], index=self.getParNames())
        if name == "Tlist":
            return {g: c.getLambda() for c, (g, _) in zip(self.getReCovs(), self.cnms)}
        if name == "REML":
            return dims["REML"]
        if name == "is_REML":
            return self.isREML()
        if name == "n_rtrms":
            return len(self.cnms)
        if name == "n_rfacs":
            return len(self.flist)
        if name in ("N", "n", "p", "q"):
            return dims[name]
        if name == "npar":
            return dims.get("npar", len(self.lower))
        if name == "cnms":
            return self.cnms
        if name == "devcomp":
            return self.devcomp
        if name == "offset":
            return rsp.offset
        if name == "lower":
            return pd.Series(self.lower, index=self.getParNames())
        if name == "devfun":
            return self.as_function()
        if name == "k":
            return len(self.cnms)
        if name == "m":
            return dims["nth"]
        if name == "l_i":
            return pd.Series({k: f.nlevels for k, f in self.flist.items()})
        if name == "glmer.nb.theta":
            if self.isGLMM() and isNBfamily(rsp.family.family):
                return rsp.family.theta
            return np.nan
        if name == "Ztlist":
            out = {}
            Zt = sp.csr_matrix(PR.Zt)
            for i, (g, cn) in enumerate(self.cnms):
                n = self.Gp[i + 1] - self.Gp[i]
                nt = len(cn)
                for j, c in enumerate(cn):
                    rows = self.Gp[i] + np.arange(j, n, nt)
                    out[f"{g}.{c}"] = Zt[rows]
            return out
        raise ValueError(f"Mixed-Effects extraction of '{name}' is not available for class "
                         f"\"{type(self).__name__}\"")

    # ------------------------------------------------------------ deviance function
    def as_function(self):
        """the deviance function at the fitted model (as.function.merMod / getME 'devfun')"""
        from .modular import Rho, mkdevfun
        from .control import glmerControl
        if self.isGLMM():
            from .modular import mkGlmerDevfun
            reTrms = dict(Zt=sp.csc_matrix(self.pp.Zt), theta=self.theta,
                          Lambdat=sp.csc_matrix(self.pp.Lambdat), Lind=np.asarray(self.pp.Lind),
                          flist=self.flist, cnms=self.cnms, lower=self.lower, upper=self.getUpper(),
                          reCovs=self.getReCovs())
            dims = self.devcomp["dims"]
            return mkGlmerDevfun(self.frame, np.asarray(self.pp.X), reTrms, self.family(),
                                 nAGQ=dims["nAGQ"],
                                 control=glmerControl(compDev=bool(dims["compDev"]),
                                                      tolPwrss=self.devcomp["cmp"]["tolPwrss"]),
                                 Xattrs=self.pp.Xattrs)
        rc = self.getReCovs()
        rho = Rho(pp=self.pp.copy(), resp=self.resp.copy(), lower=self.lower, upper=self.getUpper(),
                  mkPar=mkMkPar(rc), mkTheta=mkMkTheta(rc), reCovs=rc)
        return mkdevfun(rho, self.devcomp["dims"]["nAGQ"] or 0)

    # ------------------------------------------------------------ refit
    def refitML(self, optimizer="bobyqa", **kw):
        """refit an REML-fitted LMM by maximum likelihood (refitML.merMod)"""
        if not self.isREML():
            return self
        from .modular import Rho, mkdevfun
        from .modules import lmerResp, merPredD
        from .optimizers import optwrap
        from .lmer import _optinfo
        rr = self.resp
        rc = self.getReCovs()
        rho = Rho()
        rho.resp = lmerResp(y=rr.y, offset=rr.offset, weights=rr.weights, REML=0)
        xpp = self.pp
        rho.pp = merPredD(np.asarray(xpp.X), sp.csc_matrix(xpp.Zt), sp.csc_matrix(xpp.Lambdat),
                          np.asarray(xpp.Lind), np.asarray(xpp.theta), n=xpp.X.shape[0],
                          Xattrs=dict(xpp.Xattrs))
        rho.mkPar = mkMkPar(rc)
        rho.mkTheta = mkMkTheta(rc)
        devfun = mkdevfun(rho, 0)
        if optimizer == "bobyqa" and "calc_derivs" not in kw:
            kw["calc_derivs"] = True
        opt = optwrap(optimizer, devfun, rho.mkPar(np.asarray(rho.pp.theta)), lower=self.lower,
                      upper=self.getUpper(), rho=rho, **kw)
        n = len(rr.y)
        pp = rho.pp
        p = pp.X.shape[1]
        dims = dict(N=n, n=n, p=p, nmp=n - p, q=pp.Zt.shape[0], nth=len(pp.theta), useSc=1,
                    reTrms=len(self.cnms), spFe=0, REML=0, GLMM=0, NLMM=0, nAGQ=None,
                    npar=len(self.lower))
        wrss = rho.resp.wrss()
        ussq = pp.sqrL(1.0)
        pwrss = wrss + ussq
        cmp = dict(ldL2=pp.ldL2(), ldRX2=pp.ldRX2(), wrss=wrss, ussq=ussq, pwrss=pwrss,
                   drsum=np.nan, dev=opt["fval"], REML=np.nan, sigmaML=math.sqrt(pwrss / n),
                   sigmaREML=math.sqrt(pwrss / (n - p)))
        cl = type(self.call)(self.call.fun, **dict(self.call))
        cl["REML"] = False
        ans = lmerMod(call=cl, frame=self.frame, flist=self.flist, flist_assign=self.flist_assign,
                      cnms=self.cnms, Gp=self.Gp, theta=np.asarray(pp.theta).copy(),
                      beta=np.asarray(pp.delb), u=np.asarray(pp.delu), lower=self.lower,
                      devcomp=dict(cmp=cmp, dims=dims), pp=pp, resp=rho.resp,
                      optinfo=_optinfo(opt))
        ans.upper = self.getUpper()
        ans.reCovs = upReCovs(rc, np.asarray(pp.theta))
        return ans

    def update(self, formula=None, **kw):
        """update.merMod: refit with modified arguments.  ``formula`` may use
        R's ``.`` notation, e.g. ``". ~ . - Days"``."""
        from .lmer import lmer, glmer
        from .rlang import Formula
        args = {k: v for k, v in self.call.items() if not k.startswith("_")}
        args["data"] = self.call.get("_data")
        if formula is not None:
            args["formula"] = update_formula(self.call["formula"], formula)
        args.update(kw)
        args.pop("_data", None)
        f = lmer if self.call.fun == "lmer" else glmer
        if f is lmer:
            for k in ("family", "nAGQ"):
                args.pop(k, None)
        else:
            args.pop("REML", None)
        return f(_env=self.call["formula"].env, **args)

    def refit(self, newresp=None, control=None):
        """refit.merMod: re-optimize with a new response, reusing the model structure"""
        from .bootmer import refit_merMod
        return refit_merMod(self, newresp, control=control)

    # ------------------------------------------------------------ model frame / matrices
    def model_frame(self, fixed_only=False):
        fr = self.frame.to_pandas()
        if fixed_only:
            vars_ = self.frame.terms.extra.get("varnames.fixed")
            if vars_:
                fr = fr[[v for v in vars_ if v in fr.columns]]
        return fr

    def model_matrix(self, type="fixed"):
        if type == "fixed":
            return pd.DataFrame(np.asarray(self.pp.X), columns=self.colnames_X,
                                index=self.frame.row_names)
        if type == "random":
            return self.getME("Z")
        raise ValueError(type)

    def formula(self, fixed_only=False, random_only=False):
        from .reformulas import nobars, lme4_reOnly
        f = self.call["formula"]
        if fixed_only:
            return nobars(f)
        if random_only:
            return lme4_reOnly(f, response=True)
        return f

    # ------------------------------------------------------------ prediction
    def predict(self, newdata=None, re_form=None, type="link", allow_new_levels=False,
                random_only=False, na_action=None, newparams=None, **kw):
        """predict.merMod.  re_form: None = all random effects (R's NULL);
        "NA"/np.nan/False = none (R's NA); or a formula such as "~(1|g)"."""
        from .predict import predict_merMod, _MISSING
        return predict_merMod(self, newdata=newdata, re_form=re_form, type=type,
                              allow_new_levels=allow_new_levels, random_only=random_only,
                              na_action=_MISSING if na_action is None else na_action,
                              newparams=newparams, **kw)

    def simulate(self, nsim=1, seed=None, use_u=None, re_form=None, newdata=None, **kw):
        """simulate.merMod (seed: None, int or numpy Generator)"""
        from .predict import simulate_merMod, _MISSING
        return simulate_merMod(self, nsim=nsim, seed=seed,
                               use_u=_MISSING if use_u is None else use_u,
                               re_form=_MISSING if re_form is None else re_form,
                               newdata=newdata, **kw)

    # ------------------------------------------------------------ inference
    def confint(self, parm=None, level=0.95, method="profile", **kwargs):
        """confint.merMod: method "profile" (R's default), "Wald" or "boot"."""
        from .profile import confint_merMod
        return confint_merMod(self, parm=parm, level=level, method=method, **kwargs)

    def influence(self, *a, **k):
        from .influence import influence_merMod
        return influence_merMod(self, *a, **k)

    def cooks_distance(self, *a, **k):
        from .influence import cooks_distance
        return cooks_distance(self, *a, **k)

    def rstudent(self, *a, **k):
        from .influence import rstudent
        return rstudent(self, *a, **k)

    def profile(self, **kw):
        """profile.merMod"""
        from .profile import profile_merMod
        return profile_merMod(self, **kw)

    def anova(self, *others, refit=True, model_names=None):
        if not others:
            return anova_single(self)
        return anova_merMod([self] + list(others), refit=refit, model_names=model_names)

    def drop1(self, scope=None, test="none", k=2):
        """drop1.merMod: single fixed-term deletions (refitted by ML)"""
        from .anova import drop1_merMod
        return drop1_merMod(self, scope=scope, test=test, k=k)

    # ------------------------------------------------------------ printing
    def methTitle(self):
        return methTitle(self.devcomp["dims"])

    def summary(self, correlation=None, use_hessian=None):
        return SummaryMerMod(self, correlation=correlation, use_hessian=use_hessian)

    def _llikAIC(self, chkREML=True):
        llik = self.logLik()
        if chkREML and self.devcomp["dims"]["REML"]:
            return llik, {"REML": self.devcomp["cmp"]["REML"]}
        tab = {"AIC": self.AIC(), "BIC": self.BIC(), "logLik": llik.value,
               "-2*log(L)": self.devCrit(), "df.resid": self.df_residual()}
        return llik, tab

    def _fitMsgs(self):
        xa = self.pp.Xattrs
        return [xa[k] for k in ("msgRankdrop", "msgScaleX") if xa.get(k)]

    def __str__(self):
        digits = max(3, options.get("digits") - 3)
        out = []
        dims = self.devcomp["dims"]
        out.append(_prt_methTit(methTitle(dims), type(self).__name__))
        out += _prt_family(self)
        out += _prt_call(self.call, long=False)
        _, aictab = self._llikAIC()
        out.append(_prt_aictab(aictab, 4).rstrip("\n"))
        out.append("Random effects:")
        out.append(formatVC(self.VarCorr(), digits=digits, comp=("Std.Dev.",)))
        out.append(_prt_grps(self.ngrps(), dims["n"]))
        cf = self.fixef()
        if len(cf) > 0:
            out.append("Fixed Effects:")
            vals = r_format(cf.values, digits=digits)
            out.append(_print_named_values(vals, list(cf.index)))
        else:
            out.append("No fixed effect coefficients")
        fm = self._fitMsgs()
        if fm:
            out.append("fit warnings:")
            out += fm
        w = _prt_warn(self.optinfo, summary=True)
        if w:
            out.append(w)
        return "\n".join(out)

    def __repr__(self):
        return self.__str__()


class lmerMod(merMod):
    pass


class glmerMod(merMod):
    pass


class nlmerMod(merMod):
    pass


def _pct(p):
    s = r_format([100 * p], digits=3)[0].strip()
    return f"{s} %"


def _scale_vcov(V, nms, sc, ce):
    """lme4:::scale_vcov -- transform vcov of scaled coefficients back to the original scale"""
    p = len(nms)
    A = np.eye(p)
    ce = ce or {k: 0.0 for k in sc}
    for k, s in sc.items():
        j = nms.index(k)
        A[j, j] = 1 / s
        if "(Intercept)" in nms:
            A[nms.index("(Intercept)"), j] = -ce[k] / s
    return A @ V @ A.T


class LogLik:
    def __init__(self, value, df, nobs, REML=False):
        self.value = float(value)
        self.df = int(df)
        self.nobs = int(nobs)
        self.REML = REML

    def __float__(self):
        return self.value

    def __repr__(self):
        kind = "'log Lik.' " if not self.REML else "'log Lik.' "
        v = r_format([self.value], digits=7)[0].strip()
        return f"'log Lik.' {v} (df={self.df})"


class RanefMer(dict):
    """result of ranef(): dict of DataFrames"""

    def as_data_frame(self):
        rows = []
        for grp, df in self.items():
            pv = df.attrs.get("postVar") if hasattr(df, "attrs") else None
            frame = df if isinstance(df, pd.DataFrame) else df.to_frame()
            for j, term in enumerate(frame.columns):
                for i, lev in enumerate(frame.index):
                    sd = math.sqrt(pv[j, j, i]) if pv is not None and pv.ndim == 3 else np.nan
                    rows.append((grp, term, str(lev), frame.iloc[i, j], sd))
        return pd.DataFrame(rows, columns=["grpvar", "term", "grp", "condval", "condsd"])

    def __repr__(self):
        out = []
        for k, v in self.items():
            out.append(f"${k}\n{v}\n")
        if any("postVar" in getattr(v, "attrs", {}) for v in self.values()):
            out.append("with conditional variances for " +
                       ", ".join(f'"{k}"' for k, v in self.items() if "postVar" in getattr(v, "attrs", {})) + " ")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# printing helpers (lmer.R)
# ---------------------------------------------------------------------------

def methTitle(dims):
    GLMM = dims["GLMM"]
    kind = ["Linear", "Nonlinear", "Generalized linear", "Generalized nonlinear"][GLMM * 2 + dims["NLMM"]]
    if dims["REML"]:
        how = "REML"
    else:
        how = "maximum likelihood"
        if GLMM:
            nAGQ = dims["nAGQ"]
            how += (" (Laplace Approximation)" if nAGQ == 1 else
                    f" (Adaptive Gauss-Hermite Quadrature, nAGQ = {nAGQ})")
        else:
            how += " "      # paste("maximum likelihood", NULL) in R adds a trailing space
    return f"{kind} mixed model fit by {how}"


def _prt_methTit(mtit, cls):
    w = options.get("width")
    if len(mtit) + 5 + len(cls) > w:
        import textwrap
        return "\n  ".join(textwrap.wrap(mtit, w - 2, break_on_hyphens=False)) + f" [{cls}]"
    return f"{mtit} ['{cls}']"


def _prt_family(obj):
    if isinstance(obj, glmerMod):
        f = obj.resp.family
        return [f" Family: {f.family}  ( {f.link} )"]
    return []


def _fmt_formula(f):
    return deparse(f.expr)


def _prt_call(call, long=True):
    out = []
    if call is None:
        return out
    if call.get("formula") is not None:
        out.append(f"Formula: {_fmt_formula(call['formula'])}")
    if call.get("data") is not None:
        out.append(f"   Data: {call['data']}")
    if call.get("weights") is not None:
        out.append(f"Weights: {call['weights']}")
    if call.get("offset") is not None:
        out.append(f" Offset: {call['offset']}")
    if call.get("subset") is not None:
        out.append(f" Subset: {call['subset']}")
    return out


def _prt_aictab(aictab, digits=1):
    if list(aictab) == ["REML"]:
        v = round(aictab["REML"], digits)
        return f"REML criterion at convergence: {r_format([v], digits=7)[0].strip()}\n"
    names = list(aictab)
    vals = [round(aictab[k], digits) for k in names]
    f = r_format(vals, digits=7)
    i = names.index("df.resid")
    f[i] = r_format([vals[i]], digits=7)[0].strip()
    return print_named_vector([s.strip() for s in f], names, right=True) + "\n"


def _prt_grps(ngrps, nobs):
    return f"Number of obs: {nobs}, groups:  " + "; ".join(f"{k}, {v}" for k, v in ngrps.items())


def _prt_warn(optinfo, summary=False):
    if not optinfo:
        return ""
    cc = optinfo["conv"]["opt"] or 0
    msgs = list((optinfo["conv"].get("lme4") or {}).get("messages", []))
    warns = list(optinfo.get("warnings", []))
    if cc > 0 or msgs or warns:
        m = "(OK)" if cc == 0 else (f"({optinfo['message']})" if optinfo.get("message") else "")
        opt = optinfo["optimizer"]
        opt = opt if isinstance(opt, str) else getattr(opt, "__name__", str(opt))
        convmsg = f"optimizer ({opt}) convergence code: {cc} {m}"
        if summary:
            return f"{convmsg} ; {len(warns)} optimizer warnings; {len(msgs)} lme4 warnings "
        return "\n".join([convmsg] + msgs + warns) + "\n"
    return ""


def _print_named_values(vals, names):
    vals = [v.strip() for v in vals]
    W = max([len(v) for v in vals] + [len(n) for n in names])
    w = [W] * len(vals)          # common width, print.gap = 2
    width = options.get("width")
    lines = []
    i = 0
    while i < len(vals):
        j, tot = i, 0
        while j < len(vals) and (tot + w[j] + 2 <= width or j == i):
            tot += w[j] + 2
            j += 1
        lines.append("  ".join(n.rjust(k) for n, k in zip(names[i:j], w[i:j])) + "  ")
        lines.append("  ".join(v.rjust(k) for v, k in zip(vals[i:j], w[i:j])) + "  ")
        i = j
    return "\n".join(lines)


class SummaryMerMod:
    """summary.merMod"""

    def __init__(self, obj: merMod, correlation=None, use_hessian=None):
        self.object = obj
        p = len(obj.beta)
        self.corrSet = correlation
        if correlation is None:
            correlation = p <= options.get("lme4.summary.cor.max")
        self.correlation = correlation
        dd = obj.devcomp["dims"]
        self.useScale = bool(dd["useSc"])
        self.sigma = obj.sigma()
        self.methTitle = methTitle(dd)
        self.objClass = type(obj).__name__
        self.family = obj.resp.family.family if isinstance(obj, glmerMod) else None
        self.link = obj.resp.family.link if isinstance(obj, glmerMod) else None
        coefs = obj.fixef()
        vc = obj.vcov(use_hessian=use_hessian)
        se = np.sqrt(np.diag(vc.values))
        tab = {"Estimate": coefs.values, "Std. Error": se}
        if p > 0:
            tv = coefs.values / se
            tab["t value" if self.useScale else "z value"] = tv
            if obj.isGLMM():
                tab["Pr(>|z|)"] = 2 * stats.norm.sf(np.abs(tv))
        self.coefficients = pd.DataFrame(tab, index=coefs.index)
        self.vcov = obj.vcov(correlation=correlation, sigm=self.sigma)
        self.logLik, self.AICtab = obj._llikAIC()
        self.varcor = obj.VarCorr()
        self.ngrps = obj.ngrps()
        self.residuals = obj.residuals(type="pearson", scaled=True)
        self.call = obj.call
        self.optinfo = obj.optinfo
        self.fitMsgs = obj._fitMsgs()
        self.devcomp = obj.devcomp

    def __str__(self, digits=None, ranef_comp=("Variance", "Std.Dev."), show_resids=True,
                correlation=None):
        digits = max(3, options.get("digits") - 3) if digits is None else digits
        obj = self.object
        out = [_prt_methTit(self.methTitle, self.objClass)]
        out += _prt_family(obj)
        out += _prt_call(self.call)
        out.append("")
        out.append(_prt_aictab(self.AICtab))
        if show_resids:
            r = self.residuals.dropna().values
            q = np.quantile(r, [0, 0.25, 0.5, 0.75, 1])
            q = _zapsmall(q, digits + 1)
            out.append("Scaled residuals: ")
            out.append(print_named_vector([s.strip() for s in r_format(q, digits=digits)],
                                          ["Min", "1Q", "Median", "3Q", "Max"]) + "\n")
        out.append("Random effects:")
        out.append(formatVC(self.varcor, digits=digits, comp=ranef_comp))
        out.append(_prt_grps(self.ngrps, self.devcomp["dims"]["n"]))
        p = len(self.coefficients)
        if p > 0:
            out.append("\nFixed effects:")
            cm = self.coefficients
            out.append(printCoefmat(cm.values, list(cm.columns), list(cm.index), digits=digits,
                                    signif_stars_=options.get("show.signif.stars"),
                                    has_p=cm.shape[1] == 4))
            corr = correlation
            if corr is None:
                cor_max = options.get("lme4.summary.cor.max")
                corr = self.corrSet if self.corrSet is not None else p <= cor_max
                if not corr and p > cor_max and self.corrSet is None:
                    # R signals this with message(), i.e. on stderr
                    import sys
                    print(f"\nCorrelation matrix not shown by default, as p = {p} > {cor_max}.\n"
                               "Use print(summary(x), correlation=TRUE)  or\n"
                               "    vcov(x)        if you need it\n", file=sys.stderr)
            if corr and p > 1:
                V = self.vcov.values
                sd = np.sqrt(np.diag(V))
                C = V / sd[:, None] / sd[None, :]
                rn = list(cm.index)
                rns = abbreviate_vec(rn, 11)
                cns = abbreviate_vec(rn, 6)
                fm = r_format(np.round(C, 3).T.ravel(), digits=7, nsmall=3)
                fm = np.array(fm, dtype=object).reshape(p, p).T
                rows = []
                for i in range(1, p):
                    rows.append([fm[i, j] if j < i else "" for j in range(p - 1)])
                out.append("\nCorrelation of Fixed Effects:")
                out.append(format_char_matrix(rows, cns[:-1], rns[1:], right=False))
        if self.fitMsgs:
            out.append("fit warnings:")
            out += self.fitMsgs
        w = _prt_warn(self.optinfo, summary=False)
        if w:
            out.append(w)
        return "\n".join(out)

    __repr__ = __str__


def _zapsmall(x, digits):
    """R (>= 4.4) zapsmall: round(x, digits - log10(max|x|)); R's round()
    rounds a non-integer 'digits' to the nearest integer"""
    x = np.asarray(x, float)
    mx = np.max(np.abs(x))
    if mx == 0:
        return x
    d = max(0.0, digits - math.log10(mx))
    return np.round(x, int(math.floor(d + 0.5)))


# ---------------------------------------------------------------------------
# anova
# ---------------------------------------------------------------------------

def anova_single(obj):
    """anova() for a single model: sequential F tests of the fixed effects"""
    X = np.asarray(obj.pp.X)
    asgn = obj.pp.Xattrs.get("assign")
    if asgn is None:
        raise ValueError("no 'assign' attribute for the fixed-effects model matrix")
    asgn = np.asarray(asgn)
    RX = np.asarray(obj.pp.RX())
    ss = (RX @ obj.beta) ** 2
    tt = obj.frame.terms.extra.get("fixed.terms")
    labels = tt.term_labels if tt is not None else []
    terms_ = sorted(set(asgn[asgn > 0]))
    df = [int(np.sum(asgn == t)) for t in terms_]
    SS = [float(ss[asgn == t].sum()) for t in terms_]
    MS = [s / d for s, d in zip(SS, df)]
    s2 = obj.sigma() ** 2 if obj.devcomp["dims"]["useSc"] else 1.0
    F = [m / s2 for m in MS]
    idx = [labels[t - 1] if t - 1 < len(labels) else str(t) for t in terms_]
    return pd.DataFrame({"npar": df, "Sum Sq": SS, "Mean Sq": MS, "F value": F}, index=idx)


def anova_merMod(mods, refit=True, model_names=None):
    nobs = [m.nobs() for m in mods]
    if len(set(nobs)) > 1:
        raise ValueError("models were not all fitted to the same size of dataset")
    names = model_names or [f"MODEL{i+1}" for i in range(len(mods))]
    reml = [isinstance(m, merMod) and m.isREML() for m in mods]
    if refit:
        if any(reml):
            print("refitting model(s) with ML (instead of REML)", file=sys.stderr)
        mods = [m.refitML() if r else m for m, r in zip(mods, reml)]
    elif any(reml) and not all(reml):
        warnings.warn("some models fit with REML = TRUE, some not")
    llks = [m.logLik() for m in mods]
    order = np.argsort([l.df for l in llks], kind="stable")
    mods = [mods[i] for i in order]
    llks = [llks[i] for i in order]
    names = [names[i] for i in order]
    npar = np.array([l.df for l in llks])
    ll = np.array([l.value for l in llks])
    dev = -2 * ll
    chisq = np.concatenate([[np.nan], np.maximum(0, -2 * np.diff(-ll * -1) * -1)])
    chisq = np.concatenate([[np.nan], 2 * np.diff(ll)])
    chisq = np.where(np.arange(len(chisq)) == 0, np.nan, np.maximum(chisq, 0))
    dfChisq = np.concatenate([[np.nan], np.diff(npar)])
    with np.errstate(invalid="ignore"):
        p = np.where(dfChisq > 0, stats.chi2.sf(chisq, np.where(dfChisq > 0, dfChisq, 1)), np.nan)
    tab = pd.DataFrame({"npar": npar, "AIC": [m.AIC() for m in mods], "BIC": [m.BIC() for m in mods],
                        "logLik": ll, "-2*log(L)": dev, "Chisq": chisq, "Df": dfChisq,
                        "Pr(>Chisq)": p}, index=names)
    return tab


def update_formula(old, new):
    """stats::update.formula with '.' substitution"""
    from .rlang import Formula, Call, Sym, parse, as_formula

    def sub(e, rep):
        if isinstance(e, Sym) and e.name == ".":
            return Call(Sym("("), (rep,)) if isinstance(rep, Call) else rep
        if isinstance(e, Call):
            return Call(e.fn, tuple(sub(a, rep) for a in e.args), e.names)
        return e
    new = as_formula(new)
    old = as_formula(old)
    lhs = new.lhs
    rhs = sub(new.rhs, old.rhs)
    if lhs is not None and old.lhs is not None:
        lhs = sub(lhs, old.lhs)
    elif lhs is None:
        lhs = old.lhs
    f = Formula(Call(Sym("~"), (lhs, rhs)) if lhs is not None else Call(Sym("~"), (rhs,)), old.env)
    # simplify via terms(): drop removed terms like R's update.formula
    return _simplify_formula(f)


def _simplify_formula(f):
    from .modelframe import terms
    from .rlang import Formula, Call, Sym, parse, Const
    from .reformulas import findbars_x, sumTerms, makeOp
    tt = terms(f, keep_order=False)
    parts = [parse(l) for l in tt.term_labels]
    parts = [Call(Sym("("), (p,)) if (isinstance(p, Call) and p.fn in (Sym("|"), Sym("||"))) else p
             for p in parts]
    if not tt.intercept:
        parts = [Const(0.0)] + parts
    if not parts:
        parts = [Const(1.0)]
    rhs = sumTerms(parts)
    return Formula(Call(Sym("~"), (f.lhs, rhs)) if f.lhs is not None else Call(Sym("~"), (rhs,)), f.env)
