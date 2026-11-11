// external.h: algorithms from lme4/src/external.cpp (the .Call'able
// functions), transcribed for the Python port.
//
// Copyright (C) 2011-2018 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).

#ifndef LME4_EXTERNAL_H
#define LME4_EXTERNAL_H

#include "predModule.h"
#include "respModule.h"
#include "optimizer.h"
#include <vector>

namespace lme4ext {
    using lme4::merPredD;
    using lme4::lmResp;
    using lme4::lmerResp;
    using lme4::glmResp;
    using lme4::nlsResp;

    double lmer_Deviance(merPredD& pp, lmerResp& rp, const Eigen::VectorXd& theta);
    std::pair<double, double> lmer_opt1(merPredD& pp, lmerResp& rp, double lower, double upper);

    double glmerLaplace(merPredD& pp, glmResp& rp, int nAGQ, double tol, int maxit,
                        int verbose, bool dispProfile, int maxPhiIter, double qEff);
    double glmerAGQ(merPredD& pp, glmResp& rp, double tol, int maxit,
                    const Eigen::MatrixXd& GQmat, const Eigen::VectorXi& fac,
                    int verbose, bool dispProfile, int maxPhiIter, double qEff);

    double nlmerLaplace(merPredD& pp, nlsResp& rp, const Eigen::VectorXd& theta,
                        const Eigen::VectorXd& u0, const Eigen::VectorXd& beta0,
                        int verbose, bool uOnly, double tol, int maxit);

    std::vector<Eigen::VectorXi> allPerm_int(const Eigen::VectorXi& v);
}

#endif
