//
// glmFamily.cpp: implementation of glmFamily and related classes using Eigen
//
// Transcription of lme4/src/glmFamily.cpp for the Python port.
// Copyright (C) 2011-2012 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).

#include "glmFamily.h"
#include "rmath.h"
#include <limits>
#include <cmath>
#include <functional>

namespace glm {
    /** Cumulative probability function of the complement of the Gumbel distribution
     * (i.e. pgumbel(q,0.,1.,0) == 1 - pgumbel2(-q,0.,1.,0))
     */
    static inline double
    pgumbel2(const double& q, const double& loc, const double& scale, int lower_tail) {
        double qq = (q - loc) / scale;
        qq = -std::exp(qq);
        return lower_tail ? -std::expm1(qq) : std::exp(qq);
    }

    /** density of the complement of the Gumbel distribution */
    static inline double
    dgumbel2(const double& x, const double& loc, const double& scale, int give_log) {
        double xx = (x - loc) / scale;
        xx = xx - std::exp(xx) - std::log(scale);
        return give_log ? xx : std::exp(xx);
    }

    //@{ Templated scalar functors used in links, inverse links, etc.
    template <typename T>
    struct logN0 {
        const T operator()(const T& x) const {return x ? std::log(x) : T();}
    };

    static inline ArrayXd Y_log_Y(const ArrayXd& y, const ArrayXd& mu) {
        return y * (y/mu).unaryExpr(logN0<double>());
    }

    template<typename T>
    struct Round {
        const T operator()(const T& x) const {return std::nearbyint(x);}
    };

    template<typename T>
    struct x1mx {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(), x * (1 - x)));
        }
    };

    template<typename T>
    struct Lgamma {
        const T operator() (const T& x) const {
            return std::lgamma(x);
        }
    };

    template<typename T>
    struct cauchitinv {
        const T operator() (const T& x) const {
            return T(std::min(1.-std::numeric_limits<T>::epsilon(),
                              rmath::pcauchy(double(x), 0., 1., 1, 0)));
        }
    };

    template<typename T>
    struct cauchit {
        const T operator() (const T& x) const {
            return T(rmath::qcauchy(double(x), 0., 1., 1, 0));
        }
    };

    template<typename T>
    struct cauchitmueta {
        const T operator() (const T& x) const {
            return T(rmath::dcauchy(double(x), 0., 1., 0));
        }
    };

    template<typename T>
    struct logitinv {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(),
                              std::min(1.-std::numeric_limits<T>::epsilon(),
                                       rmath::plogis(double(x), 0., 1., 1, 0))));
        }
    };

    template<typename T>
    struct logit {
        const T operator() (const T& x) const {
            return T(rmath::qlogis(double(x), 0., 1., 1, 0));
        }
    };

    template<typename T>
    struct logitmueta {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(),
                              rmath::dlogis(double(x), 0., 1., 0)));
        }
    };

    template<typename T>
    struct probitinv {
        const T operator() (const T& x) const {
            return T(std::min(1.-std::numeric_limits<T>::epsilon(),
                              rmath::pnorm(double(x), 0., 1., 1, 0)));
        }
    };

    template<typename T>
    struct probit {
        const T operator() (const T& x) const {
            return T(rmath::qnorm(double(x), 0., 1., 1, 0));
        }
    };

    template<typename T>
    struct probitmueta {
        const T operator() (const T& x) const {
            return T(rmath::dnorm(double(x), 0., 1., 0));
        }
    };

    template<typename T>
    struct clogloginv {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(),
                              std::min(1.-std::numeric_limits<T>::epsilon(),
                                       pgumbel2(double(x), 0., 1., 1))));
        }
    };

    template<typename T>
    struct cloglogmueta {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(),
                              dgumbel2(double(x), 0., 1., 0)));
        }
    };

    template<typename T>
    struct boundexp {
        const T operator() (const T& x) const {
            return T(std::max(std::numeric_limits<T>::epsilon(),
                              std::exp(double(x))));
        }
    };
    //@}

    //@{
    double                binomialDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                                                 const ArrayXd& wt, double dev) const {
        ArrayXd    m((n > 1).any() ? n : wt);
        ArrayXd   yy((m * y).unaryExpr(Round<double>()));
        m = m.unaryExpr(Round<double>());
        double ans(0.);
        for (int i=0; i < mu.size(); ++i)
            ans += (m[i] <= 0. ? 0. : wt[i]/m[i]) * rmath::dbinom(yy[i], m[i], mu[i], true);
        return (-2. * ans);
    }
    const ArrayXd         binomialDist::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return 2. * wt * (Y_log_Y(y, mu) + Y_log_Y(1. - y, 1. - mu));
    }
    const ArrayXd         binomialDist::variance(const ArrayXd& mu) const {return mu.unaryExpr(x1mx<double>());}
    //@}

    //@{
    double                   gammaDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                                                 const ArrayXd& wt, double dev) const {
        double   nn(wt.sum());
        double disp(dev/nn);
        double   ans(0), invdisp(1./disp);
        for (int i = 0; i < mu.size(); ++i)
            ans += wt[i] * rmath::dgamma(y[i], invdisp, mu[i] * disp, true);
        return -2. * ans + 2.;
    }
    const ArrayXd            gammaDist::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return -2. * wt * ((y/mu).unaryExpr(logN0<double>()) - (y - mu)/mu);
    }
    const ArrayXd            gammaDist::variance(const ArrayXd& mu) const {return mu.square();}
    //@}

    //@{
    double                GaussianDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                                                 const ArrayXd& wt, double dev) const {
        double   nn(mu.size());
        return nn * (std::log(2. * M_PI * dev/nn) + 1.) + 2. - wt.log().sum();
    }
    const ArrayXd         GaussianDist::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return wt * (y - mu).square();
    }
    const ArrayXd         GaussianDist::variance(const ArrayXd& mu) const {return ArrayXd::Ones(mu.size());}
    //@}

    //@{
    double         inverseGaussianDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
        const ArrayXd& wt, double dev) const {
        double wtsum(wt.sum());
        return wtsum * (std::log(dev/wtsum * 2. * M_PI) + 1.) + 3. * (y.log() * wt).sum() + 2.;
    }
    const ArrayXd  inverseGaussianDist::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return wt * ((y - mu).square())/(y * mu.square());
    }
    const ArrayXd  inverseGaussianDist::variance(const ArrayXd& mu) const {return mu.cube();}
    //@}

    //@{
    double        negativeBinomialDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                                                 const ArrayXd& wt, double dev) const {
        return 2. * (wt * (y + d_theta) * (mu + d_theta).log() -
                     y * mu.log() + (y + 1).unaryExpr(Lgamma<double>()) -
                     d_theta * std::log(d_theta) + std::lgamma(d_theta) -
                     (d_theta + y).unaryExpr(Lgamma<double>())).sum();
    }
    const ArrayXd negativeBinomialDist::devResid(const ArrayXd &y, const ArrayXd &mu, const ArrayXd &wt) const {
        return 2. * wt * (Y_log_Y(y, mu) - (y + d_theta) * ((y + d_theta)/(mu + d_theta)).log());
    }
    const ArrayXd negativeBinomialDist::variance(const ArrayXd &mu) const {
        return mu + mu.square()/d_theta;
    }
    //@}

    //@{
    double                 PoissonDist::aic     (const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                                                 const ArrayXd& wt, double dev) const {
        double ans(0.);
        for (int i = 0; i < mu.size(); ++i) ans += rmath::dpois(y[i], mu[i], true) * wt[i];
        return (-2. * ans);
    }
    const ArrayXd          PoissonDist::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return 2. * wt * (y * (y/mu).unaryExpr(logN0<double>()) - (y - mu));
    }
    const ArrayXd          PoissonDist::variance(const ArrayXd& mu) const {return mu;}
    //@}

    //@{
    const ArrayXd  cauchitLink::linkFun(const ArrayXd&  mu) const {return  mu.unaryExpr(cauchit<double>());}
    const ArrayXd  cauchitLink::linkInv(const ArrayXd& eta) const {return eta.unaryExpr(cauchitinv<double>());}
    const ArrayXd  cauchitLink::muEta(  const ArrayXd& eta) const {return eta.unaryExpr(cauchitmueta<double>());}
    //@}

    //@{
    const ArrayXd      logLink::linkFun(const ArrayXd&  mu) const {return  mu.log();}
    const ArrayXd      logLink::linkInv(const ArrayXd& eta) const {return eta.unaryExpr(boundexp<double>());}
    const ArrayXd      logLink::muEta(  const ArrayXd& eta) const {return eta.unaryExpr(boundexp<double>());}
    //@}

    //@{
    const ArrayXd    logitLink::linkFun(const ArrayXd&  mu) const {return  mu.unaryExpr(logit<double>());}
    const ArrayXd    logitLink::linkInv(const ArrayXd& eta) const {return eta.unaryExpr(logitinv<double>());}
    const ArrayXd    logitLink::muEta(  const ArrayXd& eta) const {return eta.unaryExpr(logitmueta<double>());}
    //@}

    //@{
    const ArrayXd   probitLink::linkFun(const ArrayXd&  mu) const {return  mu.unaryExpr(probit<double>());}
    const ArrayXd   probitLink::linkInv(const ArrayXd& eta) const {return eta.unaryExpr(probitinv<double>());}
    const ArrayXd   probitLink::muEta(  const ArrayXd& eta) const {return eta.unaryExpr(probitmueta<double>());}
    //@}

    //@{
    const ArrayXd identityLink::linkFun(const ArrayXd&  mu) const {return  mu;}
    const ArrayXd identityLink::linkInv(const ArrayXd& eta) const {return eta;}
    const ArrayXd identityLink::muEta(  const ArrayXd& eta) const {return ArrayXd::Ones(eta.size());}
    //@}

    //@{
    const ArrayXd  inverseLink::linkFun(const ArrayXd&  mu) const {return mu.inverse();}
    const ArrayXd  inverseLink::linkInv(const ArrayXd& eta) const {return eta.inverse();}
    const ArrayXd  inverseLink::muEta(  const ArrayXd& eta) const {return -(eta.inverse().square());}
    //@}

    //@{ the R make.link("cloglog") linkfun: log(-log(1 - mu))
    const ArrayXd  cloglogLink::linkFun(const ArrayXd&  mu) const {return (-(1. - mu).log()).log();}
    const ArrayXd  cloglogLink::linkInv(const ArrayXd& eta) const {return eta.unaryExpr(clogloginv<double>());}
    const ArrayXd  cloglogLink::muEta(  const ArrayXd& eta) const {return eta.unaryExpr(cloglogmueta<double>());}
    //@}

    //@{ R make.link("sqrt")
    const ArrayXd  sqrtLink::linkFun(const ArrayXd&  mu) const {return mu.sqrt();}
    const ArrayXd  sqrtLink::linkInv(const ArrayXd& eta) const {return eta.square();}
    const ArrayXd  sqrtLink::muEta(  const ArrayXd& eta) const {return 2. * eta;}
    //@}

    //@{ R make.link("1/mu^2")
    const ArrayXd  inverseSquareLink::linkFun(const ArrayXd&  mu) const {return mu.square().inverse();}
    const ArrayXd  inverseSquareLink::linkInv(const ArrayXd& eta) const {return eta.sqrt().inverse();}
    const ArrayXd  inverseSquareLink::muEta(  const ArrayXd& eta) const {return -1. / (2. * eta.pow(1.5));}
    //@}

    void glmFamily::build() {
        d_link = nullptr;
        d_dist = nullptr;
        if (d_linknam == "cauchit")  d_link = new cauchitLink();
        if (d_linknam == "cloglog")  d_link = new cloglogLink();
        if (d_linknam == "identity") d_link = new identityLink();
        if (d_linknam == "inverse")  d_link = new inverseLink();
        if (d_linknam == "log")      d_link = new logLink();
        if (d_linknam == "logit")    d_link = new logitLink();
        if (d_linknam == "probit")   d_link = new probitLink();
        if (d_linknam == "sqrt")     d_link = new sqrtLink();
        if (d_linknam == "1/mu^2")   d_link = new inverseSquareLink();
        if (!d_link) throw std::invalid_argument("unknown link: " + d_linknam);

        if (d_family  == "binomial")         d_dist = new binomialDist(d_family);
        if (d_family  == "Gamma")            d_dist = new gammaDist(d_family);
        if (d_family  == "gaussian")         d_dist = new GaussianDist(d_family);
        if (d_family  == "inverse.gaussian") d_dist = new inverseGaussianDist(d_family);
        if (d_family.substr(0, 18) ==
            "Negative Binomial(")             d_dist = new negativeBinomialDist(d_family, d_theta0);
        if (d_family  == "poisson")          d_dist = new PoissonDist(d_family);
        if (!d_dist) {
            delete d_link;
            throw std::invalid_argument("unknown family: " + d_family);
        }
    }

    glmFamily::glmFamily(const std::string& family, const std::string& link, double theta)
        : d_family(family), d_linknam(link), d_dist(nullptr), d_link(nullptr), d_theta0(theta) {
        build();
    }

    glmFamily::glmFamily(const glmFamily& other)
        : d_family(other.d_family), d_linknam(other.d_linknam),
          d_dist(nullptr), d_link(nullptr), d_theta0(other.d_theta0) {
        build();
        if (d_family.substr(0, 18) == "Negative Binomial(") d_dist->setTheta(other.theta());
    }

    glmFamily::~glmFamily() {
        delete d_dist;
        delete d_link;
    }

    const ArrayXd glmFamily::devResid(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt) const {
        return d_dist->devResid(y, mu, wt);
    }

    double glmFamily::aic(const ArrayXd& y, const ArrayXd& n, const ArrayXd& mu,
                          const ArrayXd& wt, double dev) const {
        return d_dist->aic(y, n, mu, wt, dev);
    }

    const ArrayXd glmLink::linkFun(const ArrayXd&) const {
        throw std::runtime_error("linkFun not implemented for this link");
    }
    const ArrayXd glmLink::linkInv(const ArrayXd&) const {
        throw std::runtime_error("linkInv not implemented for this link");
    }
    const ArrayXd glmLink::muEta(const ArrayXd&) const {
        throw std::runtime_error("muEta not implemented for this link");
    }
    const ArrayXd glmDist::variance(const ArrayXd&) const {
        throw std::runtime_error("variance not implemented for family " + d_familyName);
    }
    const ArrayXd glmDist::devResid(const ArrayXd&, const ArrayXd&, const ArrayXd&) const {
        throw std::runtime_error("devResid not implemented for family " + d_familyName);
    }
    double glmDist::aic(const ArrayXd&, const ArrayXd&, const ArrayXd&,
                        const ArrayXd&, double) const {
        throw std::runtime_error("aic not implemented for family " + d_familyName);
    }

    double glmDist::theta() const {
        throw std::invalid_argument("theta accessor applies only to negative binomial");
    }

    void   glmDist::setTheta(const double& theta) {
        throw std::invalid_argument("setTheta applies only to negative binomial");
    }
}
