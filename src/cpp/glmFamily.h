// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// glmFamily.h: glm family class using Eigen
//
// Transcription of lme4/src/glmFamily.h for the Python port.
// Copyright (C)       2012 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).
//
// Differences from the R version: the R implementation keeps the R
// family functions (linkfun, linkinv, mu.eta, variance, dev.resids, aic)
// as a fall-back for families/links not implemented in compiled code.
// In the Python port a family is identified by its name and link name
// only; unknown names raise std::invalid_argument.

#ifndef LME4_GLMFAMILY_H
#define LME4_GLMFAMILY_H

#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <cmath>

namespace glm {
    using Eigen::ArrayXd;

    class glmDist {
    protected:
        std::string d_familyName;    /**< $family, for error messages only */
        bool   d_hasDispersionField; /**< does the family have a $dispersion component */
        double d_dispersionField;    /**< its value, NaN meaning free */
    public:
        glmDist(const std::string& fam, bool hasDisp = true, double disp = NAN)
            : d_familyName(fam), d_hasDispersionField(hasDisp), d_dispersionField(disp) {}
        virtual ~glmDist() {}

        virtual const ArrayXd variance(const ArrayXd&) const;
        virtual const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        virtual double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                                       const ArrayXd&, double) const;
        virtual double           theta() const;
        virtual void          setTheta(const double&);
        virtual bool  hasFreeDispersion() const {
            if (!d_hasDispersionField) return true;
            return std::isnan(d_dispersionField);
        }
    };

    class binomialDist : public glmDist {
    public:
        binomialDist(const std::string& f) : glmDist(f) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        bool  hasFreeDispersion() const {return false;}
    };

    class gammaDist : public glmDist {
    public:
        gammaDist(const std::string& f) : glmDist(f) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        bool  hasFreeDispersion() const {return true;}
    };

    class GaussianDist : public glmDist {
    public:
        GaussianDist(const std::string& f) : glmDist(f) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        bool  hasFreeDispersion() const {return true;}
    };

    class inverseGaussianDist : public glmDist {
    public:
        inverseGaussianDist(const std::string& f) : glmDist(f) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        bool  hasFreeDispersion() const {return true;}
    };

    class negativeBinomialDist : public glmDist {
    protected:
        double  d_theta;
    public:
        negativeBinomialDist(const std::string& f, double theta) : glmDist(f), d_theta(theta) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        double           theta() const {return d_theta;}
        void          setTheta(const double& ntheta) {d_theta = ntheta;}
        bool  hasFreeDispersion() const {return false;}
    };

    class PoissonDist : public glmDist {
    public:
        PoissonDist(const std::string& f) : glmDist(f) {}
        const ArrayXd variance(const ArrayXd&) const;
        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        bool  hasFreeDispersion() const {return false;}
    };

    class glmLink {
    public:
        glmLink() {}
        virtual ~glmLink() {}

        virtual const ArrayXd linkFun(const ArrayXd&) const;
        virtual const ArrayXd linkInv(const ArrayXd&) const;
        virtual const ArrayXd   muEta(const ArrayXd&) const;
    };

    class cauchitLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class cloglogLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const; // R uses the R function here
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class identityLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class  inverseLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class logLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class logitLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class probitLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    // R's sqrt link / 1/mu^2 link are handled through the R fall-back in
    // lme4; we implement them here directly.
    class sqrtLink : public glmLink {
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class inverseSquareLink : public glmLink { // "1/mu^2"
    public:
        const ArrayXd linkFun(const ArrayXd&) const;
        const ArrayXd linkInv(const ArrayXd&) const;
        const ArrayXd   muEta(const ArrayXd&) const;
    };

    class glmFamily {
    protected:
        std::string  d_family, d_linknam; /**< as in the R glmFamily object */
        glmDist     *d_dist;
        glmLink     *d_link;
    public:
        // family: R family name ("binomial", "poisson", "Gamma", "gaussian",
        // "inverse.gaussian", "Negative Binomial(<theta>)"); link: link name;
        // theta: negative binomial theta (ignored otherwise)
        glmFamily(const std::string& family, const std::string& link, double theta = NAN);
        glmFamily(const glmFamily& other);
        glmFamily& operator=(const glmFamily&) = delete;
        ~glmFamily();
        const std::string& fam() const {return d_family;}
        const std::string& lnk() const {return d_linknam;}

        const ArrayXd devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const;
        const ArrayXd  linkFun(const ArrayXd&  mu) const {return d_link->linkFun(mu);}
        const ArrayXd  linkInv(const ArrayXd& eta) const {return d_link->linkInv(eta);}
        const ArrayXd    muEta(const ArrayXd& eta) const {return d_link->muEta(eta);}
        const ArrayXd variance(const ArrayXd&  mu) const {return d_dist->variance(mu);}
        double             aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                               const ArrayXd&, double) const;
        double           theta() const {return d_dist->theta();}
        void          setTheta(const double& theta) {d_dist->setTheta(theta);}
        bool  hasFreeDispersion() const {return d_dist->hasFreeDispersion();}
    private:
        double d_theta0;
        void build();
    };
}

#endif /* LME4_GLMFAMILY_H */
