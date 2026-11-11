// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// respModule.h: response modules using Eigen
//
// Transcription of lme4/src/respModule.h for the Python port.
// Copyright (C) 2011-2012 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).
//
// The members own their storage.  The nonlinear model in nlsResp is
// evaluated through a callback (a Python function) instead of an R
// language object evaluated in an R environment.

#ifndef LME4_RESPMODULE_H
#define LME4_RESPMODULE_H

#include "glmFamily.h"
#include <functional>
#include <utility>

namespace lme4 {
    using Eigen::VectorXd;
    using Eigen::MatrixXd;
    using glm::glmFamily;

    class lmResp {
    protected:
        double d_wrss, /**< current weighted sum of squared residuals */
            d_ldW;     /**< sum of log prior weights */
        VectorXd d_y,  /**< response vector */
            d_weights, /**< prior weights - always present even if unity */
            d_offset,  /**< offset in the model */
            d_mu,      /**< mean response from current linear predictor */
            d_sqrtXwt, /**< Square roots of the "X weights" */
            d_sqrtrwt, /**< Square roots of the residual weights */
            d_wtres;   /**< Current weighted residuals */
    public:
        lmResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
               const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
               const VectorXd& wtres);
        virtual ~lmResp() {}

        const VectorXd&    sqrtXwt() const {return d_sqrtXwt;}
        const VectorXd&         mu() const {return d_mu;}
        const VectorXd&     offset() const {return d_offset;}
        const VectorXd&    sqrtrwt() const {return d_sqrtrwt;}
        const VectorXd&    weights() const {return d_weights;}
        const VectorXd&      wtres() const {return d_wtres;}
        const VectorXd&          y() const {return d_y;}
        double            wrss() const {return d_wrss;}
        virtual double    updateMu(const VectorXd&);
        double       updateWts()       {return updateWrss();}
        double      updateWrss(); /**< update the weighted residuals and d_wrss */
        void         setOffset(const VectorXd&);
        void           setResp(const VectorXd&);
        void        setWeights(const VectorXd&);
    };

    class lmerResp : public lmResp {
    private:
        int d_reml;             /**< 0 for evaluating the deviance, p
                                 * for evaluating the REML criterion. */
    public:
        lmerResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                 const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                 const VectorXd& wtres);

        double         Laplace(double,double,double) const;
        double         Laplace(double,double,double,double) const;
        int               REML() const {return d_reml;}
        void           setReml(int);
    };

    class glmResp : public lmResp {
    protected:
        glmFamily  d_fam;
        VectorXd   d_eta, d_n;
        double     d_phi = 1.0;  /**< dispersion multiplier for the working weights */
    public:
        glmResp(const glmFamily& fam, const VectorXd& y, const VectorXd& weights,
                const VectorXd& offset, const VectorXd& mu, const VectorXd& sqrtXwt,
                const VectorXd& sqrtrwt, const VectorXd& wtres, const VectorXd& eta,
                const VectorXd& n);

        Eigen::ArrayXd   devResid() const;
        Eigen::ArrayXd      muEta() const;
        Eigen::ArrayXd  sqrtWrkWt() const;
        Eigen::ArrayXd   variance() const;
        Eigen::ArrayXd  wrkResids() const;
        Eigen::ArrayXd    wrkResp() const;
        Eigen::ArrayXd  wtWrkResp() const;

        const VectorXd&           eta() const {return d_eta;}
        const VectorXd&             n() const {return d_n;}

        const std::string& family() const {return d_fam.fam();}
        const std::string&   link() const {return d_fam.lnk();}
        bool     hasFreeDispersion() const {return d_fam.hasFreeDispersion();}
        const glmFamily& fam() const {return d_fam;}

        double                aic() const;
        double            Laplace(double,double,double) const;
        double             resDev() const;
        double              theta() const {return d_fam.theta();}
        double                phi() const {return d_phi;}
        double           updateMu(const VectorXd&);
        double          updateWts();

        void                 setN(const VectorXd&);
        void               setPhi(const double& nphi) {d_phi = nphi;}
        void             setTheta(const double& ntheta) {d_fam.setTheta(ntheta);}
    };

    // model callback: given the matrix of parameter values (n x npar,
    // column-major in a vector of length n * npar) return (mu, gradient)
    typedef std::function<std::pair<VectorXd, MatrixXd>(const MatrixXd&)> nlsModel;

    class nlsResp : public lmResp {
    protected:
        VectorXd        d_gamma;
        nlsModel        d_nlmod;
        int             d_npar;
    public:
        nlsResp(const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                const VectorXd& wtres, const VectorXd& gamma, nlsModel mod, int npar);

        double            Laplace(double, double, double) const;
        double           updateMu(const VectorXd&);
        const VectorXd&     gamma() const {return d_gamma;}
    };
}

#endif
