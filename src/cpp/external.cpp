// external.cpp: externally callable functions in lme4
//
// Transcription of the algorithmic parts of lme4/src/external.cpp for the
// Python port (the R glue -- external pointers, BEGIN_RCPP ... -- is
// replaced by the pybind11 module in bindings.cpp).
//
// Copyright (C)       2011-2018 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).

#include "external.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace lme4ext {
    typedef   Eigen::VectorXi        iVec;
    typedef   Eigen::MatrixXd         Mat;
    typedef   Eigen::VectorXd         Vec;
    typedef   Eigen::ArrayXd          Ar1;

    using      std::runtime_error;
    using optimizer::printer;

    // utilities

    std::vector<iVec> allPerm_int(const iVec& v_) {
        iVec     v(v_);   // forces a copy
        int     sz(v.size());
        std::vector<iVec> vec;
        std::sort(v.data(), v.data() + sz);
        do {
            vec.push_back(iVec(v));
        } while (std::next_permutation(v.data(), v.data() + sz));
        return vec;
    }

    static inline double pwrss(lmResp *rp, merPredD *pp, double fac) {
        return rp->wrss() + (fac ? pp->sqrL(fac) : pp->u0().squaredNorm());
    }

    static double internal_glmerWrkIter(merPredD *pp, glmResp *rp, bool uOnly) {
        pp->updateXwts(rp->sqrtWrkWt());
        pp->updateDecomp();
        pp->updateRes(rp->wtWrkResp());
        if (uOnly) pp->solveU();
        else pp->solve();
        rp->updateMu(pp->linPred(1.));
        // deviance/data-fit term reweighted by 1/phi (phi==1 reproduces the
        // original formula exactly); the ||u||^2 penalty (sqrL) is untouched
        return rp->resDev() / rp->phi() + pp->sqrL(1.);
    }

    static void pwrssUpdate(glmResp *rp, merPredD *pp, bool uOnly,
                            double tol, int maxit, int verbose) {
        double oldpdev=std::numeric_limits<double>::max();
        double pdev;
        int maxstephalfit = 20;
        bool   cvgd = false, verb = verbose > 2, moreverb = verbose > 10;

        pdev = oldpdev; // define so debugging statements work on first step
        for (int i = 0; i < maxit; i++) {
            if (verb) {
                std::ostringstream os;
                os << "*** pwrssUpdate step " << i << std::endl;
                printer(os.str());
            }
            Vec   olddelu(pp->delu()), olddelb(pp->delb());
            pdev = internal_glmerWrkIter(pp, rp, uOnly);
            if (verb) {
                std::ostringstream os;
                os << "pdev=" << pdev <<
                    "; delu_min: " << pp->delu().minCoeff() <<
                    "; delu_max: " << pp->delu().maxCoeff() <<
                    "; delb_min: " << (pp->delb().size() ? pp->delb().minCoeff() : NAN) <<
                    "; delb_max: " << (pp->delb().size() ? pp->delb().maxCoeff() : NAN) <<
                    std::endl;
                printer(os.str());
            }
            if (std::abs((oldpdev - pdev) / pdev) < tol) {cvgd = true; break;}

            if (std::isnan(pdev) || (pdev > oldpdev)) {
                // PWRSS step led to _larger_ deviation, or nan; try step halving
                if (verb) printer("\npwrssUpdate: Entering step halving loop\n");
                for (int k = 0; k < maxstephalfit &&
                         (std::isnan(pdev) || (pdev > oldpdev)); k++) {
                    pp->setDelu((olddelu + pp->delu())/2.);
                    if (!uOnly) pp->setDelb((olddelb + pp->delb())/2.);
                    rp->updateMu(pp->linPred(1.));
                    pdev = rp->resDev() + pp->sqrL(1.);
                    if (moreverb) {
                        std::ostringstream os;
                        os << "step-halving iteration " << k << ":  pdev=" << pdev << std::endl;
                        printer(os.str());
                    }
                }
                if (std::isnan(pdev)) {
                    throw runtime_error("PIRLS loop resulted in NaN value");
                }
                if ((pdev - oldpdev) > tol) {
                    throw runtime_error("(maxstephalfit) PIRLS step-halvings failed to reduce deviance in pwrssUpdate");
                }
            } // step-halving
            oldpdev = pdev;
        } // pwrss loop
        if (!cvgd)
            throw runtime_error("pwrssUpdate did not converge in (maxit) iterations");
    }

    // Runs pwrssUpdate() to find the conditional mode(s), profiling phi via
    // a nested fixed-point loop around it when the family has a free/
    // estimated dispersion parameter and dispProfile is requested; see the
    // long comment in lme4/src/external.cpp.  Returns false if even the safe
    // fallback failed to converge for this candidate theta/beta.
    static bool profilePhi(glmResp *rp, merPredD *pp, bool uOnly, double tol,
                           int maxit, int verb, bool dispProfile, int maxPhiIter,
                           double qEff) {
        if (rp->hasFreeDispersion() && dispProfile) {
            double phi = 1.;
            double lastGoodPhi = 1.;
            double n = rp->weights().sum();
            if (std::isfinite(qEff)) n -= qEff;
            double phiTol = 1e-8;
            for (int outer = 0; outer < maxPhiIter; ++outer) {
                rp->setPhi(phi);
                try {
                    pwrssUpdate(rp, pp, uOnly, tol, maxit, verb);
                } catch (std::runtime_error&) {
                    pp->setDelu(Vec::Zero(pp->u0().size()));
                    pp->setDelb(Vec::Zero(pp->beta0().size()));
                    rp->updateMu(pp->linPred(1.));
                    phi = lastGoodPhi;
                    rp->setPhi(phi);
                    try {
                        pwrssUpdate(rp, pp, uOnly, tol, maxit, verb);
                    } catch (std::runtime_error&) {
                        return false;
                    }
                    break;
                }
                double phiNew = rp->resDev() / n;
                bool cvgd = std::abs(phiNew - phi) / phi < phiTol;
                lastGoodPhi = phi;
                // Damp the update: take only a small step (in log space)
                // toward phiNew each outer iteration
                phi = std::exp(0.9 * std::log(phi) + 0.1 * std::log(phiNew));
                if (cvgd) break;
            }
            rp->setPhi(phi);
        } else {
            pwrssUpdate(rp, pp, uOnly, tol, maxit, verb);
        }
        // PIRLS updates mu after forming the working factorization. Rebuild
        // it at the accepted mean and final phi before Laplace or AGQ uses it.
        pp->updateXwts(rp->sqrtWrkWt());
        pp->updateDecomp();
        return true;
    }

    double glmerLaplace(merPredD& pp, glmResp& rp, int nAGQ, double tol, int maxit,
                        int verbose, bool dispProfile, int maxPhiIter, double qEff) {
        if (verbose > 100) {
            std::ostringstream os;
            os << "\nglmerLaplace resDev:  " << rp.resDev() << std::endl;
            printer(os.str());
        }
        bool uOnly(nAGQ != 0);
        if (!profilePhi(&rp, &pp, uOnly, tol, maxit, verbose, dispProfile, maxPhiIter, qEff))
            return 1e10;
        return rp.Laplace(pp.ldL2(), pp.ldRX2(), pp.sqrL(1.));
    }

    // fac: 1-based integer vector indicating the factor levels
    static Ar1 devcCol(const iVec& fac, const Ar1& u, const Ar1& devRes, double phi) {
        Ar1  ans(u.square());
        for (int i = 0; i < devRes.size(); ++i) ans[fac[i] - 1] += devRes[i] / phi;
        return ans;
    }

    static const double sqrt2pi = std::sqrt(2. * M_PI);

    double glmerAGQ(merPredD& pp_, glmResp& rp_, double tol, int maxit,
                    const Mat& GQmat, const iVec& fac, int verbose,
                    bool dispProfile, int maxPhiIter, double qEff) {
        glmResp*  rp = &rp_;
        merPredD* pp = &pp_;
        if (fac.size() != rp->mu().size())
            throw std::invalid_argument("size of fac must match dimension of response vector");

        if (!profilePhi(rp, pp, true, tol, maxit, verbose, dispProfile, maxPhiIter, qEff))
            return 1e10;
        double phi(rp->phi());

        const Ar1      devc0(devcCol(fac, pp->u(1.), rp->devResid(), phi));
        const unsigned int q(pp->u0().size());
        if (pp->L().nonZerosL() != (long) q)
            throw std::invalid_argument("AGQ only defined for a single scalar random-effects term");
        // L is diagonal; R uses the factor's x slot in its own (permuted)
        // order.  With a diagonal factor the permutation is applied to
        // map back to the original order of the random effects.
        Vec Ldiag(pp->L().diagL());
        Vec dperm(q);
        {
            Eigen::VectorXi P(pp->Pvec()); // L L' = A[P, P]
            for (unsigned int i = 0; i < q; ++i) dperm[P[i]] = Ldiag[i];
        }
        const Ar1         sd(dperm.array().inverse());

        Ar1             mult(q);

        mult.setZero();
        for (int i = 0; i < GQmat.rows(); ++i) {
            double     zknot(GQmat(i, 0));
            if (zknot == 0)
                mult += Ar1::Constant(q, GQmat(i, 1));
            else {
                pp->setU0((zknot * sd).matrix()); // to be added to current delu
                rp->updateMu(pp->linPred(1.));
                mult += (-0.5 * (devcCol(fac, pp->u(1.), rp->devResid(), phi) - devc0) -
                         GQmat(i, 2)).exp() * GQmat(i, 1)/sqrt2pi;
            }
        }
        pp->setU0(Vec::Zero(q)); // restore settings from pwrssUpdate;
        rp->updateMu(pp->linPred(1.));
        return rp->Laplace(pp->ldL2(), 0., pp->sqrL(1.)) - 2 * std::log(mult.prod());
    }

    // called only from prssUpdate() :
    static void nstepFac(nlsResp *rp, merPredD *pp, double prss0, int verb) {
        for (double fac = 1.; fac > 0.001; fac /= 2.) {
            double prss1 = rp->updateMu(pp->linPred(fac)) + pp->sqrL(fac);
            if (verb > 3) {
                std::ostringstream os;
                os << "   nstepFac(), fac=" << fac << ", prss0-prss1=" << prss0 - prss1 << "\n";
                printer(os.str());
            }
            if (prss1 < prss0) {
                pp->installPars(fac);
                return;
            }
        }
        throw runtime_error("step factor reduced below 0.001 without reducing pwrss");
    }

    static void prssUpdate(nlsResp *rp, merPredD *pp, int verb, bool uOnly,
                           double tol, int maxit) {
        bool cvgd(false);
        for (int it=0; it < maxit; ++it) {
            rp->updateMu(pp->linPred(0.));
            pp->updateXwts(rp->sqrtXwt());
            pp->updateDecomp();
            pp->updateRes(rp->wtres());
            double pwrs0(pwrss(rp, pp, 0.)),
                ccrit((uOnly ? pp->solveU() : pp->solve())/pwrs0);
            if (verb > 3) {
                std::ostringstream os;
                os << " it=" << it << ", pwrs0=" << pwrs0 << ", ccrit=" << ccrit
                   << ", tol=" << tol << "\n";
                printer(os.str());
            }
            if (ccrit < tol) {
                cvgd = true;
                break;
            }
            nstepFac(rp, pp, pwrs0, verb);
        }
        if (!cvgd) throw runtime_error("prss{Update} failed to converge in 'maxit' iterations");
    }

    double nlmerLaplace(merPredD& pp, nlsResp& rp, const Vec& theta,
                        const Vec& u0, const Vec& beta0,
                        int verbose, bool uOnly, double tol, int maxit) {
        pp.setTheta(theta);
        pp.setU0   (u0);
        pp.setBeta0(beta0);
        prssUpdate(&rp, &pp, verbose, uOnly, tol, maxit);
        return rp.Laplace(pp.ldL2(), pp.ldRX2(), pp.sqrL(1.));
    }

    // linear mixed-effects models

    static double lmer_dev(merPredD& ppt, lmerResp& rpt, const Vec& theta) {
        ppt.setTheta(theta);
        ppt.updateXwts(rpt.sqrtXwt());
        ppt.updateDecomp();
        rpt.updateMu(ppt.linPred(0.));
        ppt.updateRes(rpt.wtres());
        ppt.solve();
        rpt.updateMu(ppt.linPred(1.));
        return rpt.Laplace(ppt.ldL2(), ppt.ldRX2(), ppt.sqrL(1.));
    }

    double lmer_Deviance(merPredD& pp, lmerResp& rp, const Vec& theta) {
        return lmer_dev(pp, rp, theta);
    }

    std::pair<double, double> lmer_opt1(merPredD& ppt, lmerResp& rpt, double lower, double upper) {
        Vec     th(1);
        optimizer::Golden gold(lower, upper);
        for (int i = 0; i < 30; ++i) {
            th[0] = gold.xeval();
            gold.newf(lmer_dev(ppt, rpt, th));
        }
        return std::make_pair(gold.xpos(), gold.value());
    }
}
