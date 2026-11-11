// rmath.h: the small subset of R's nmath library used by lme4's C++ code
//
// Transcribed from R's src/nmath (stirlerr.c, bd0.c, dbinom.c, dpois.c,
// dgamma.c, qnorm.c, ...) so that the Python port of lme4 reproduces the
// same density values as R's Rmath without linking to R.
//
// Original code Copyright (C) 1998-2021 The R Core Team, Ross Ihaka,
// Catherine Loader.  GPL (>= 2).

#ifndef LME4PY_RMATH_H
#define LME4PY_RMATH_H

#include <cmath>
#include <limits>

namespace rmath {

    static const double M_LN_SQRT_2PI_ = 0.918938533204672741780329736406; // log(sqrt(2*pi))
    static const double M_LN_SQRT_PId2_ = 0.225791352644727432363097614947; // log(sqrt(pi/2))
    static const double M_2PI_ = 6.283185307179586476925286766559;
    static const double M_1_SQRT_2PI_ = 0.398942280401432677939946059934;
    static const double ML_NAN = std::numeric_limits<double>::quiet_NaN();
    static const double ML_POSINF = std::numeric_limits<double>::infinity();
    static const double ML_NEGINF = -std::numeric_limits<double>::infinity();

    inline bool R_nonint(double x) {
        return std::fabs(x - std::nearbyint(x)) > 1e-7 * std::max(1., std::fabs(x));
    }

    // stirlerr(n) = log(n!) - log( sqrt(2*pi*n)*(n/e)^n )
    inline double stirlerr(double n) {
        const double S0 = 0.083333333333333333333;        /* 1/12 */
        const double S1 = 0.00277777777777777777778;      /* 1/360 */
        const double S2 = 0.00079365079365079365079365;   /* 1/1260 */
        const double S3 = 0.000595238095238095238095238;  /* 1/1680 */
        const double S4 = 0.0008417508417508417508417508; /* 1/1188 */
        static const double sferr_halves[31] = {
            0.0, /* n=0 - wrong, place holder only */
            0.1534264097200273452913848,  /* 0.5 */
            0.0810614667953272582196702,  /* 1.0 */
            0.0548141210519176538961390,  /* 1.5 */
            0.0413406959554092940938221,  /* 2.0 */
            0.03316287351993628748511048, /* 2.5 */
            0.02767792568499833914878929, /* 3.0 */
            0.02374616365629749597132920, /* 3.5 */
            0.02079067210376509311152277, /* 4.0 */
            0.01848845053267318523077934, /* 4.5 */
            0.01664469118982119216319487, /* 5.0 */
            0.01513497322191737887351255, /* 5.5 */
            0.01387612882307074799874573, /* 6.0 */
            0.01281046524292022692424986, /* 6.5 */
            0.01189670994589177009505572, /* 7.0 */
            0.01110455975820691732662991, /* 7.5 */
            0.010411265261972096497478567, /* 8.0 */
            0.009799416126158803298389475, /* 8.5 */
            0.009255462182712732917728637, /* 9.0 */
            0.008768700134139385462952823, /* 9.5 */
            0.008330563433362871256469318, /* 10.0 */
            0.007934114564314020547248100, /* 10.5 */
            0.007573675487951840794972024, /* 11.0 */
            0.007244554301320383179543912, /* 11.5 */
            0.006942840107209529865664152, /* 12.0 */
            0.006665247032707682442354394, /* 12.5 */
            0.006408994188004207068439631, /* 13.0 */
            0.006171712263039457647532867, /* 13.5 */
            0.005951370112758847735624416, /* 14.0 */
            0.005746216513010115682023589, /* 14.5 */
            0.005554733551962801371038690  /* 15.0 */
        };
        double nn;
        if (n <= 15.0) {
            nn = n + n;
            if (nn == (int)nn) return sferr_halves[(int)nn];
            return std::lgamma(n + 1.) - (n + 0.5) * std::log(n) + n - M_LN_SQRT_2PI_;
        }
        nn = n * n;
        if (n > 500) return (S0 - S1 / nn) / n;
        if (n > 80) return (S0 - (S1 - S2 / nn) / nn) / n;
        if (n > 35) return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
        /* 15 < n <= 35 : */
        return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
    }

    // bd0(x, np) = x log(x/np) + np - x, computed carefully
    inline double bd0(double x, double np) {
        if (!std::isfinite(x) || !std::isfinite(np) || np == 0.0) return ML_NAN;
        if (std::fabs(x - np) < 0.1 * (x + np)) {
            double v = (x - np) / (x + np);
            double s = (x - np) * v;
            if (std::fabs(s) < std::numeric_limits<double>::min()) return s;
            double ej = 2 * x * v;
            v = v * v;
            for (int j = 1; j < 1000; j++) {
                ej *= v;
                double s1 = s + ej / ((j << 1) + 1);
                if (s1 == s) return s1;
                s = s1;
            }
        }
        return x * std::log(x / np) + np - x;
    }

    inline double dpois_raw(double x, double lambda, int give_log) {
        if (lambda == 0) return (x == 0) ? (give_log ? 0. : 1.) : (give_log ? ML_NEGINF : 0.);
        if (!std::isfinite(lambda)) return give_log ? ML_NEGINF : 0.;
        if (x < 0) return give_log ? ML_NEGINF : 0.;
        if (x <= lambda * std::numeric_limits<double>::min())
            return give_log ? -lambda : std::exp(-lambda);
        if (lambda < x * std::numeric_limits<double>::min()) {
            if (!std::isfinite(x)) return give_log ? ML_NEGINF : 0.;
            double v = -lambda + x * std::log(lambda) - std::lgamma(x + 1);
            return give_log ? v : std::exp(v);
        }
        double f = M_2PI_ * x, xx = x - lambda, yy = -stirlerr(x) - bd0(x, lambda);
        (void) xx;
        return give_log ? -0.5 * std::log(f) + yy : std::exp(yy) / std::sqrt(f);
    }

    inline double dpois(double x, double lambda, int give_log) {
        if (std::isnan(x) || std::isnan(lambda)) return x + lambda;
        if (lambda < 0) return ML_NAN;
        if (R_nonint(x)) return give_log ? ML_NEGINF : 0.;
        if (x < 0 || !std::isfinite(x)) return give_log ? ML_NEGINF : 0.;
        x = std::nearbyint(x);
        return dpois_raw(x, lambda, give_log);
    }

    inline double dbinom_raw(double x, double n, double p, double q, int give_log) {
        double lf, lc;
        if (p == 0) return (x == 0) ? (give_log ? 0. : 1.) : (give_log ? ML_NEGINF : 0.);
        if (q == 0) return (x == n) ? (give_log ? 0. : 1.) : (give_log ? ML_NEGINF : 0.);
        if (x == 0) {
            if (n == 0) return give_log ? 0. : 1.;
            lc = (p < 0.1) ? -bd0(n, n * q) - n * p : n * std::log(q);
            return give_log ? lc : std::exp(lc);
        }
        if (x == n) {
            lc = (q < 0.1) ? -bd0(n, n * p) - n * q : n * std::log(p);
            return give_log ? lc : std::exp(lc);
        }
        if (x < 0 || x > n) return give_log ? ML_NEGINF : 0.;
        lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
        lf = M_2PI_ * x * (n - x) / n; // = 2pi * x * (1 - x/n)
        return give_log ? lc - 0.5 * std::log(lf) : std::exp(lc) / std::sqrt(lf);
    }

    inline double dbinom(double x, double n, double p, int give_log) {
        if (std::isnan(x) || std::isnan(n) || std::isnan(p)) return x + n + p;
        if (p < 0 || p > 1 || n < 0 || R_nonint(n)) return ML_NAN;
        if (R_nonint(x) || x < 0 || !std::isfinite(x)) return give_log ? ML_NEGINF : 0.;
        n = std::nearbyint(n);
        x = std::nearbyint(x);
        return dbinom_raw(x, n, p, 1 - p, give_log);
    }

    // dgamma(x, shape, scale)
    inline double dgamma(double x, double shape, double scale, int give_log) {
        if (std::isnan(x) || std::isnan(shape) || std::isnan(scale)) return x + shape + scale;
        if (shape < 0 || scale <= 0) return ML_NAN;
        if (x < 0) return give_log ? ML_NEGINF : 0.;
        if (shape == 0) return (x == 0) ? ML_POSINF : (give_log ? ML_NEGINF : 0.);
        if (x == 0) {
            if (shape < 1) return ML_POSINF;
            if (shape > 1) return give_log ? ML_NEGINF : 0.;
            return give_log ? -std::log(scale) : 1 / scale;
        }
        double pr;
        if (shape < 1) {
            pr = dpois_raw(shape, x / scale, give_log);
            return give_log ? pr + (std::isfinite(shape / x) ? std::log(shape / x) : std::log(shape) - std::log(x))
                            : pr * shape / x;
        }
        pr = dpois_raw(shape - 1, x / scale, give_log);
        return give_log ? pr - std::log(scale) : pr / scale;
    }

    // Normal distribution
    inline double dnorm(double x, double mu, double sigma, int give_log) {
        if (std::isnan(x) || std::isnan(mu) || std::isnan(sigma)) return x + mu + sigma;
        if (sigma < 0) return ML_NAN;
        x = (x - mu) / sigma;
        if (!std::isfinite(x) && mu == x) return ML_NAN;
        x = std::fabs(x);
        if (x >= 2 * std::sqrt(std::numeric_limits<double>::max())) return give_log ? ML_NEGINF : 0.;
        if (give_log) return -(M_LN_SQRT_2PI_ + 0.5 * x * x + std::log(sigma));
        if (x < 5) return M_1_SQRT_2PI_ * std::exp(-0.5 * x * x) / sigma;
        // underflow-protected version as in R
        if (x > std::sqrt(-2 * M_LN2 * (std::numeric_limits<double>::min_exponent + 1 - std::numeric_limits<double>::digits)))
            return 0.;
        double x1 = std::ldexp(std::nearbyint(std::ldexp(x, 16)), -16);
        double x2 = x - x1;
        return M_1_SQRT_2PI_ / sigma * (std::exp(-0.5 * x1 * x1) * std::exp((-0.5 * x2 - x1) * x2));
    }

    inline double pnorm(double x, double mu, double sigma, int lower_tail, int log_p) {
        double z = (x - mu) / sigma;
        double p = lower_tail ? 0.5 * std::erfc(-z / M_SQRT2) : 0.5 * std::erfc(z / M_SQRT2);
        return log_p ? std::log(p) : p;
    }

    // qnorm: Wichura's AS241 algorithm, as in R's qnorm.c (non-log scale)
    inline double qnorm(double p, double mu, double sigma, int lower_tail, int log_p) {
        double p_, q, r, val;
        if (std::isnan(p) || std::isnan(mu) || std::isnan(sigma)) return p + mu + sigma;
        if (log_p) p = std::exp(p);
        if (p < 0 || p > 1) return ML_NAN;
        if (p == 0) return lower_tail ? ML_NEGINF : ML_POSINF;
        if (p == 1) return lower_tail ? ML_POSINF : ML_NEGINF;
        if (sigma < 0) return ML_NAN;
        if (sigma == 0) return mu;
        p_ = lower_tail ? p : 1 - p;
        q = p_ - 0.5;
        if (std::fabs(q) <= .425) {
            r = .180625 - q * q;
            val = q * (((((((r * 2509.0809287301226727 +
                             33430.575583588128105) * r + 67265.770927008700853) * r +
                           45921.953931549871457) * r + 13731.693765509461125) * r +
                         1971.5909503065514427) * r + 133.14166789178437745) * r +
                       3.387132872796366608)
                / (((((((r * 5226.495278852545925 +
                         28729.085735721942674) * r + 39307.89580009271061) * r +
                       21213.794301586595867) * r + 5394.1960214247511077) * r +
                     687.1870074920579083) * r + 42.313330701600911252) * r + 1.);
        } else {
            if (q < 0) r = p_; else r = 1 - p_;
            r = std::sqrt(-std::log(r));
            if (r <= 5.) {
                r += -1.6;
                val = (((((((r * 7.7454501427834140764e-4 +
                             .0227238449892691845833) * r + .24178072517745061177) *
                           r + 1.27045825245236838258) * r +
                          3.64784832476320460504) * r + 5.7694972214606914055) *
                        r + 4.6303378461565452959) * r +
                       1.42343711074968357734)
                    / (((((((r *
                             1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                            r + .0151986665636164571966) * r +
                           .14810397642748007459) * r + .68976733498510000455) *
                         r + 1.6763848301838038494) * r +
                        2.05319162663775882187) * r + 1.);
            } else {
                r += -5.;
                val = (((((((r * 2.01033439929228813265e-7 +
                             2.71155556874348757815e-5) * r +
                            .0012426609473880784386) * r + .026532189526576123093) *
                          r + .29656057182850489123) * r +
                         1.7848265399172913358) * r + 5.4637849111641143699) *
                       r + 6.6579046435011037772)
                    / (((((((r *
                             2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                            r + 1.8463183175100546818e-5) * r +
                           7.868691311456132591e-4) * r + .0148753612908506148525)
                         * r + .13692988092273580531) * r +
                        .59983220655588793769) * r + 1.);
            }
            if (q < 0.0) val = -val;
        }
        return mu + sigma * val;
    }

    // Logistic distribution
    inline double plogis(double x, double location, double scale, int lower_tail, int log_p) {
        x = (x - location) / scale;
        if (!lower_tail) x = -x;
        if (log_p) return -std::log1p(std::exp(-x));
        return 1 / (1 + std::exp(-x));
    }

    inline double qlogis(double p, double location, double scale, int lower_tail, int log_p) {
        if (log_p) p = std::exp(p);
        if (p < 0 || p > 1) return ML_NAN;
        if (p == 0) return lower_tail ? ML_NEGINF : ML_POSINF;
        if (p == 1) return lower_tail ? ML_POSINF : ML_NEGINF;
        double v = lower_tail ? std::log(p / (1. - p)) : std::log((1. - p) / p);
        return location + scale * v;
    }

    inline double dlogis(double x, double location, double scale, int give_log) {
        x = std::fabs((x - location) / scale);
        double e = std::exp(-x);
        double f = 1.0 + e;
        return give_log ? -(x + std::log(scale * f * f)) : e / (scale * f * f);
    }

    // Cauchy distribution
    inline double pcauchy(double x, double location, double scale, int lower_tail, int log_p) {
        x = (x - location) / scale;
        if (!std::isfinite(x)) {
            double v = (x < 0) ? (lower_tail ? 0. : 1.) : (lower_tail ? 1. : 0.);
            return log_p ? std::log(v) : v;
        }
        if (!lower_tail) x = -x;
        double v;
        if (std::fabs(x) > 1) {
            double y = std::atan(1 / x) / M_PI;
            v = (x > 0) ? (0.5 - y + 0.5) : -y;
        } else
            v = 0.5 + std::atan(x) / M_PI;
        return log_p ? std::log(v) : v;
    }

    inline double qcauchy(double p, double location, double scale, int lower_tail, int log_p) {
        if (log_p) p = std::exp(p);
        if (p < 0 || p > 1) return ML_NAN;
        if (p == 0.5) return location;
        if (p == 0.) return location + (lower_tail ? scale : -scale) * ML_NEGINF;
        if (p == 1.) return location + (lower_tail ? scale : -scale) * ML_POSINF;
        if (!lower_tail) p = 1 - p;
        // tanpi(p - 0.5) with argument reduction, as in R's qcauchy
        return location + scale * (-1. / std::tan(M_PI * p));
    }

    inline double dcauchy(double x, double location, double scale, int give_log) {
        double y = (x - location) / scale;
        return give_log ? -std::log(M_PI * scale * (1. + y * y))
                        : 1. / (M_PI * scale * (1. + y * y));
    }
}

#endif
