// respModule.cpp: response modules using Eigen
//
// Transcription of lme4/src/respModule.cpp for the Python port.
// Copyright (C) 2011-2012 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).

#include "respModule.h"
#include <cmath>
#include <stdexcept>

namespace lme4 {
    using Eigen::ArrayXd;
    using Eigen::VectorXd;

    using std::invalid_argument;

    static const double M_LN_SQRT_2PI_ = 0.918938533204672741780329736406;

    lmResp::lmResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                   const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                   const VectorXd& wtres)
        : d_y(y), d_weights(weights), d_offset(offset), d_mu(mu),
          d_sqrtXwt(sqrtXwt), d_sqrtrwt(sqrtrwt), d_wtres(wtres) {
        Eigen::Index n = d_y.size();
        if (d_weights.size() != n || d_mu.size() != n || d_sqrtrwt.size() != n ||
            d_wtres.size() != n)
            throw invalid_argument("lmResp: dimension mismatch");
        updateWrss();
        d_ldW = d_weights.array().log().sum();
    }

    /** Update the (conditional) mean response and weighted residuals */
    double lmResp::updateMu(const VectorXd& gamma) {
        if (gamma.size() != d_offset.size())
            throw invalid_argument("updateMu: Size mismatch");
        d_mu = d_offset + gamma;
        return updateWrss();
    }

    /** wtres <- sqrtrwt * (y - mu); return(wrss <- sum(wtres^2)) */
    double lmResp::updateWrss() {
        d_wtres = d_sqrtrwt.cwiseProduct(d_y - d_mu);
        d_wrss  = d_wtres.squaredNorm();
        return d_wrss;
    }

    void lmResp::setOffset(const VectorXd& oo) {
        if (oo.size() != d_offset.size())
            throw invalid_argument("setOffset: Size mismatch");
        d_offset = oo;
    }

    void lmResp::setResp(const VectorXd& yy) {
        if (yy.size() != d_y.size())
            throw invalid_argument("setResp: Size mismatch");
        d_y = yy;
    }

    void lmResp::setWeights(const VectorXd& ww) {
        if (ww.size() != d_weights.size())
            throw invalid_argument("setWeights: Size mismatch");
        d_weights = ww;
        d_sqrtrwt = ww.array().sqrt();
        d_ldW = ww.array().log().sum();
    }

    lmerResp::lmerResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                       const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                       const VectorXd& wtres)
        : lmResp(y, weights, offset, mu, sqrtXwt, sqrtrwt, wtres),
          d_reml(0) {
    }

    double lmerResp::Laplace(double ldL2, double ldRX2, double sqrL) const {
        double lnum = std::log(2.* M_PI * (d_wrss + sqrL));
        if (d_reml == 0) return ldL2 - d_ldW + d_y.size() * (1. + lnum - std::log((double) d_y.size()));
        double nmp = d_y.size() - d_reml;
        return ldL2 - d_ldW + ldRX2 + nmp * (1. + lnum - std::log(nmp));
    }

    double lmerResp::Laplace(double ldL2, double ldRX2, double sqrL, double sigma_sq) const {
        double df = d_y.size() - d_reml;

        double result = df * (2.0 * M_LN_SQRT_2PI_ + std::log(sigma_sq)); // (2pi sigma_sq)^-df/2
        result += (d_wrss + sqrL) / sigma_sq; // exp(-1/2sigma_sq x |pwrss|)
        result += ldL2 + (d_reml > 0 ? ldRX2 : 0.0); // det|LL'|^-1/2 and similar REML penalty
        result += -d_ldW; // subtract prior weights factor
        return result;
    }

    void lmerResp::setReml(int rr) {
        if (rr < 0) throw invalid_argument("setReml: negative value for REML not meaningful");
        d_reml = rr;
    }

    glmResp::glmResp(const glmFamily& fam, const VectorXd& y, const VectorXd& weights,
                     const VectorXd& offset, const VectorXd& mu, const VectorXd& sqrtXwt,
                     const VectorXd& sqrtrwt, const VectorXd& wtres, const VectorXd& eta,
                     const VectorXd& n)
        : lmResp(y, weights, offset, mu, sqrtXwt, sqrtrwt, wtres),
          d_fam(fam),
          d_eta(eta),
          d_n(n) {
    }

    double  glmResp::aic() const {
        return d_fam.aic(d_y, d_n, d_mu, d_weights, resDev());
    }

    ArrayXd glmResp::devResid() const {
        return d_fam.devResid(d_y, d_mu, d_weights);
    }

    ArrayXd glmResp::muEta() const {
        return d_fam.muEta(d_eta);
    }

    ArrayXd glmResp::variance() const {
        return d_fam.variance(d_mu);
    }

    ArrayXd glmResp::wrkResids() const {
        return (d_y - d_mu).array() / muEta();
    }

    ArrayXd glmResp::wrkResp() const {
        return (d_eta - d_offset).array() + wrkResids();
    }

    ArrayXd glmResp::wtWrkResp() const {
        return wrkResp() * sqrtWrkWt();
    }

    ArrayXd glmResp::sqrtWrkWt() const {
        // d_phi reweights the deviance/data-fit curvature only (d_phi==1
        // reproduces the original formula exactly)
        return muEta() * (d_weights.array() / (d_phi * variance())).sqrt();
    }

    double glmResp::Laplace(double ldL2, double ldRX2, double sqrL) const {
        // see the comments in lme4/src/respModule.cpp about the "+2"
        // bookkeeping constant for families with free dispersion
        double ans = ldL2 + sqrL + aic();
        if (hasFreeDispersion()) ans -= 2.;
        return ans;
    }

    double glmResp::resDev() const {
        return devResid().sum();
    }

    double glmResp::updateMu(const VectorXd& gamma) {
        d_eta = d_offset + gamma; // lengths are checked here
        d_mu  = d_fam.linkInv(d_eta);
        return updateWrss();
    }

    double glmResp::updateWts() {
        d_sqrtrwt = (d_weights.array() / variance()).sqrt();
        d_sqrtXwt = muEta() * d_sqrtrwt.array();
        return updateWrss();
    }

    void glmResp::setN(const VectorXd& n) {
        if (n.size() != d_n.size())
            throw invalid_argument("n size mismatch");
        d_n = n;
    }

    nlsResp::nlsResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                     const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                     const VectorXd& wtres, const VectorXd& gamma, nlsModel mod, int npar)
        : lmResp(y, weights, offset, mu, sqrtXwt, sqrtrwt, wtres),
          d_gamma(gamma), d_nlmod(mod), d_npar(npar) {
    }

    double nlsResp::Laplace(double ldL2, double ldRX2, double sqrL) const {
        double lnum = 2.* M_PI * (d_wrss + sqrL), n = d_y.size();
        return ldL2 + n * (1 + std::log(lnum / n));
    }

    double nlsResp::updateMu(const VectorXd& gamma) {
        int             n = d_y.size();
        if (gamma.size() != d_gamma.size())
            throw invalid_argument("size mismatch in updateMu");
        d_gamma = gamma;
        const VectorXd lp(d_gamma + d_offset); // linear predictor
        MatrixXd pars = Eigen::Map<const MatrixXd>(lp.data(), n, d_npar);
        std::pair<VectorXd, MatrixXd> rr = d_nlmod(pars);
        if (rr.first.size() != n)
            throw invalid_argument("dimension mismatch");
        d_mu = rr.first;
        if (rr.second.size() != d_sqrtXwt.size())
            throw invalid_argument("gradient dimension mismatch");
        std::copy(rr.second.data(), rr.second.data() + rr.second.size(), d_sqrtXwt.data());
        return updateWrss();
    }
}
