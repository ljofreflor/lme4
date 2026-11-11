// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// bobyqa.cpp: C++ transcription of M.J.D. Powell's BOBYQA, as shipped in the
// R package minqa 1.2.8 (Fortran sources bobyqa.f, bobyqb.f, altmov.f,
// prelim.f, rescue.f, trsbox.f, updatebobyqa.f and the C++ glue minqa.cpp).
//
// Original Fortran: M.J.D. Powell (2009), "The BOBYQA algorithm for bound
//   constrained optimization without derivatives", Report DAMTP 2009/NA06,
//   University of Cambridge.
// minqa modifications (error codes IERR, R output routines minqit/minqir,
//   calfun interface): Douglas Bates, Katharine M. Mullen, John C. Nash,
//   Ravi Varadhan and contributors to the minqa R package.
// C++ transcription for the Python port of lme4.
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.  This program is distributed in the hope that it will
// be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
// Public License for more details.
//
// Transcription notes
// -------------------
// * The control flow follows the Fortran statement by statement, including
//   its GOTOs (as C++ goto/labels named after the Fortran statement labels),
//   and every floating-point expression keeps the Fortran operation order.
//   Floating-point contraction (FMA) is disabled for this file so that the
//   results agree with gfortran's (non-FMA) code on x86_64.
// * Arrays are accessed through 1-based, column-major macros (XPT(k,j),
//   BMAT(i,j), ...) so that the indices read exactly as in the Fortran.  The
//   working space W is partitioned exactly as in BOBYQA (bobyqa.f), and the
//   sub-arrays W(NP), W(NDIM+1), ... are passed as pointer offsets, so the
//   aliasing between work arrays (e.g. TRSBOX's GNEW being read back as W(J)
//   in BOBYQB) is preserved.
// * Deviations from the minqa Fortran (see also the comments in the code):
//   - RESCUE: minqa's rescue.f computes the new point in W(1..N) but then calls
//     CALFUN(N,X,IPRINT) with X an undeclared (implicitly typed, uninitialised)
//     local scalar -- a bug.  Powell's original code evaluates F at W, and so
//     does this transcription.
//   - ALTMOV: KSAV and STPSAV are uninitialised in the Fortran if no line
//     search gives a positive PREDSQ; here they default to KNEW and 0.
//   - Locals that the Fortran may read before assignment on paths that can not
//     actually occur (DIFFC, RATIO, F, ...) are initialised to zero.

#include "bobyqa.h"
#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace minqa {

namespace {

    using std::sqrt;
    using std::abs;

    inline double dmin1(double a, double b) { return std::min(a, b); }
    inline double dmax1(double a, double b) { return std::max(a, b); }
    inline double sq(double a) { return a * a; } // Fortran A**2

    inline std::string fmt(const char* f, double v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), f, v);
        return buf;
    }
    inline std::string fmt(const char* f, int v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), f, v);
        return buf;
    }

    /// The Fortran-callable CALFUN of minqa.cpp
    struct Calfun {
        const std::function<double(const double*, int)>& fn;
        int& feval;             // minqa's .feval. counter

        double operator()(int n, const double* x, int ip) {
            feval++;            // increment func eval count
            for (int i = 0; i < n; i++)
                if (!std::isfinite(x[i]))
                    throw std::range_error("non-finite x values not allowed in calfun");
            double f = fn(x, n);
            if (!std::isfinite(f)) f = std::numeric_limits<double>::max();

            if (ip == 3 || (ip > 3 && feval % ip == 0)) {
                // print eval info when very verbose / every ip evaluations
                std::string s = fmt("%3d:", feval) + fmt("%#14.8g:", f);
                for (int i = 0; i < n; i++) s += fmt(" %#8g", x[i]);
                s += "\n";
                optimizer::printer(s);
            }
            return f;
        }
    };

    /// minqit: iteration output when rho changes and iprint >= 2
    void minqit(int iprint, double rho, int nf, double fopt, int n,
                const double* xbase, const double* xopt) {
        if (iprint >= 2) {
            std::string s = fmt("rho: %#8.2g", rho) + fmt(" eval: %3d", nf) +
                fmt(" fn: %#12g par:", fopt);
            for (int i = 0; i < n; i++) s += fmt("%#8g ", xbase[i] + xopt[i]);
            s += "\n";
            optimizer::printer(s);
        }
    }

    /// minqir: output at return
    void minqir(int iprint, double f, int nf, int n, const double* x) {
        if (iprint > 0) {
            std::string s = "At return\n";
            s += fmt("eval: %3d", nf) + fmt(" fn: %#14.8g par:", f);
            for (int i = 0; i < n; i++) s += fmt(" %#8g", x[i]);
            s += "\n";
            optimizer::printer(s);
        }
    }

// 1-based, column-major accessors mirroring the Fortran declarations
#define X(i)        x[(i) - 1]
#define XL(i)       xl[(i) - 1]
#define XU(i)       xu[(i) - 1]
#define XBASE(i)    xbase[(i) - 1]
#define XPT(k, j)   xpt[((k) - 1) + (std::size_t)((j) - 1) * npt]
#define FVAL(i)     fval[(i) - 1]
#define XOPT(i)     xopt[(i) - 1]
#define GOPT(i)     gopt[(i) - 1]
#define HQ(i)       hq[(i) - 1]
#define PQ(i)       pq[(i) - 1]
#define BMAT(i, j)  bmat[((i) - 1) + (std::size_t)((j) - 1) * ndim]
#define ZMAT(k, j)  zmat[((k) - 1) + (std::size_t)((j) - 1) * npt]
#define SL(i)       sl[(i) - 1]
#define SU(i)       su[(i) - 1]
#define XNEW(i)     xnew[(i) - 1]
#define XALT(i)     xalt[(i) - 1]
#define D(i)        d[(i) - 1]
#define VLAG(i)     vlag[(i) - 1]
#define W(i)        w[(i) - 1]
#define GNEW(i)     gnew[(i) - 1]
#define XBDI(i)     xbdi[(i) - 1]
#define S(i)        s[(i) - 1]
#define HS(i)       hs[(i) - 1]
#define HRED(i)     hred[(i) - 1]
#define GLAG(i)     glag[(i) - 1]
#define HCOL(i)     hcol[(i) - 1]
#define PTSAUX(i, j) ptsaux[((i) - 1) + (std::size_t)((j) - 1) * 2]
#define PTSID(i)    ptsid[(i) - 1]

    // ======================================================================
    // SUBROUTINE PRELIM
    // ======================================================================
    void prelim(int n, int npt, double* x, const double* xl, const double* xu,
                double rhobeg, int iprint, int maxfun, double* xbase,
                double* xpt, double* fval, double* gopt, double* hq, double* pq,
                double* bmat, double* zmat, int ndim, double* sl, double* su,
                int& nf, int& kopt, Calfun& calfun) {
        const double half = 0.5, one = 1.0, two = 2.0, zero = 0.0;
        double rhosq, recip, stepa = zero, stepb = zero, f, fbeg = zero, temp, diff;
        int np, nfm, nfx, itemp, jpt = 0, ipt = 0, ih;

        rhosq = rhobeg * rhobeg;
        recip = one / rhosq;
        np = n + 1;
        // Set XBASE to the initial vector of variables, and set the initial
        // elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
        for (int j = 1; j <= n; j++) {
            XBASE(j) = X(j);
            for (int k = 1; k <= npt; k++) XPT(k, j) = zero;
            for (int i = 1; i <= ndim; i++) BMAT(i, j) = zero;
        }
        for (ih = 1; ih <= (n * np) / 2; ih++) HQ(ih) = zero;
        for (int k = 1; k <= npt; k++) {
            PQ(k) = zero;
            for (int j = 1; j <= npt - np; j++) ZMAT(k, j) = zero;
        }
        // Begin the initialization procedure.
        nf = 0;
    L50:
        nfm = nf;
        nfx = nf - n;
        nf = nf + 1;
        if (nfm <= 2 * n) {
            if (nfm >= 1 && nfm <= n) {
                stepa = rhobeg;
                if (SU(nfm) == zero) stepa = -stepa;
                XPT(nf, nfm) = stepa;
            } else if (nfm > n) {
                stepa = XPT(nf - n, nfx);
                stepb = -rhobeg;
                if (SL(nfx) == zero) stepb = dmin1(two * rhobeg, SU(nfx));
                if (SU(nfx) == zero) stepb = dmax1(-two * rhobeg, SL(nfx));
                XPT(nf, nfx) = stepb;
            }
        } else {
            itemp = (nfm - np) / n;
            jpt = nfm - itemp * n - n;
            ipt = jpt + itemp;
            if (ipt > n) {
                itemp = jpt;
                jpt = ipt - n;
                ipt = itemp;
            }
            XPT(nf, ipt) = XPT(ipt + 1, ipt);
            XPT(nf, jpt) = XPT(jpt + 1, jpt);
        }
        // Calculate the next value of F.
        for (int j = 1; j <= n; j++) {
            X(j) = dmin1(dmax1(XL(j), XBASE(j) + XPT(nf, j)), XU(j));
            if (XPT(nf, j) == SL(j)) X(j) = XL(j);
            if (XPT(nf, j) == SU(j)) X(j) = XU(j);
        }
        f = calfun(n, x, iprint);
        FVAL(nf) = f;
        if (nf == 1) {
            fbeg = f;
            kopt = 1;
        } else if (f < FVAL(kopt)) {
            kopt = nf;
        }
        // Set the nonzero initial elements of BMAT and the quadratic model.
        if (nf <= 2 * n + 1) {
            if (nf >= 2 && nf <= n + 1) {
                GOPT(nfm) = (f - fbeg) / stepa;
                if (npt < nf + n) {
                    BMAT(1, nfm) = -one / stepa;
                    BMAT(nf, nfm) = one / stepa;
                    BMAT(npt + nfm, nfm) = -half * rhosq;
                }
            } else if (nf >= n + 2) {
                ih = (nfx * (nfx + 1)) / 2;
                temp = (f - fbeg) / stepb;
                diff = stepb - stepa;
                HQ(ih) = two * (temp - GOPT(nfx)) / diff;
                GOPT(nfx) = (GOPT(nfx) * stepb - temp * stepa) / diff;
                if (stepa * stepb < zero) {
                    if (f < FVAL(nf - n)) {
                        FVAL(nf) = FVAL(nf - n);
                        FVAL(nf - n) = f;
                        if (kopt == nf) kopt = nf - n;
                        XPT(nf - n, nfx) = stepb;
                        XPT(nf, nfx) = stepa;
                    }
                }
                BMAT(1, nfx) = -(stepa + stepb) / (stepa * stepb);
                BMAT(nf, nfx) = -half / XPT(nf - n, nfx);
                BMAT(nf - n, nfx) = -BMAT(1, nfx) - BMAT(nf, nfx);
                ZMAT(1, nfx) = sqrt(two) / (stepa * stepb);
                ZMAT(nf, nfx) = sqrt(half) / rhosq;
                ZMAT(nf - n, nfx) = -ZMAT(1, nfx) - ZMAT(nf, nfx);
            }
        } else {
            // Off-diagonal second derivatives of the Lagrange functions and
            // the initial quadratic model.
            ih = (ipt * (ipt - 1)) / 2 + jpt;
            ZMAT(1, nfx) = recip;
            ZMAT(nf, nfx) = recip;
            ZMAT(ipt + 1, nfx) = -recip;
            ZMAT(jpt + 1, nfx) = -recip;
            temp = XPT(nf, ipt) * XPT(nf, jpt);
            HQ(ih) = (fbeg - FVAL(ipt + 1) - FVAL(jpt + 1) + f) / temp;
        }
        if (nf < npt && nf < maxfun) goto L50;
    }

    // ======================================================================
    // SUBROUTINE TRSBOX
    // ======================================================================
    void trsbox(int n, int npt, const double* xpt, const double* xopt,
                const double* gopt, const double* hq, const double* pq,
                const double* sl, const double* su, double delta, double* xnew,
                double* d, double* gnew, double* xbdi, double* s, double* hs,
                double* hred, double& dsq, double& crvmin) {
        const double half = 0.5, one = 1.0, onemin = -1.0, zero = 0.0;
        int iterc, nact, itermax = 0, iact = 0, itcsav = 0, isav, iu, ih;
        double sqstp, delsq, qred, beta, stepsq, gredsq = zero, resid, ds, shs,
            temp, blen = zero, stplen = zero, xsum, sdec, ggsav = zero,
            dredsq = zero, dredg = zero, sredg = zero, angbd, ratio, ssq,
            xsav = zero, dhs, dhd, redmax, redsav, angt = zero, sth, rednew,
            rdprev = zero, rdnext = zero, cth, tempa, tempb;
        (void) sqstp; (void) ratio;

        iterc = 0;
        nact = 0;
        sqstp = zero;
        for (int i = 1; i <= n; i++) {
            XBDI(i) = zero;
            if (XOPT(i) <= SL(i)) {
                if (GOPT(i) >= zero) XBDI(i) = onemin;
            } else if (XOPT(i) >= SU(i)) {
                if (GOPT(i) <= zero) XBDI(i) = one;
            }
            if (XBDI(i) != zero) nact = nact + 1;
            D(i) = zero;
            GNEW(i) = GOPT(i);
        }
        delsq = delta * delta;
        qred = zero;
        crvmin = onemin;
        // Set the next search direction of the conjugate gradient method.
    L20:
        beta = zero;
    L30:
        stepsq = zero;
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) != zero) {
                S(i) = zero;
            } else if (beta == zero) {
                S(i) = -GNEW(i);
            } else {
                S(i) = beta * S(i) - GNEW(i);
            }
            stepsq = stepsq + sq(S(i));
        }
        if (stepsq == zero) goto L190;
        if (beta == zero) {
            gredsq = stepsq;
            itermax = iterc + n - nact;
        }
        if (gredsq * delsq <= 1.0e-4 * qred * qred) goto L190;
        // Multiply the search direction by the second derivative matrix of Q.
        goto L210;
    L50:
        resid = delsq;
        ds = zero;
        shs = zero;
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) == zero) {
                resid = resid - sq(D(i));
                ds = ds + S(i) * D(i);
                shs = shs + S(i) * HS(i);
            }
        }
        if (resid <= zero) goto L90;
        temp = sqrt(stepsq * resid + ds * ds);
        if (ds < zero) {
            blen = (temp - ds) / stepsq;
        } else {
            blen = resid / (temp + ds);
        }
        stplen = blen;
        if (shs > zero) {
            stplen = dmin1(blen, gredsq / shs);
        }
        // Reduce STPLEN if necessary in order to preserve the simple bounds.
        iact = 0;
        for (int i = 1; i <= n; i++) {
            if (S(i) != zero) {
                xsum = XOPT(i) + D(i);
                if (S(i) > zero) {
                    temp = (SU(i) - xsum) / S(i);
                } else {
                    temp = (SL(i) - xsum) / S(i);
                }
                if (temp < stplen) {
                    stplen = temp;
                    iact = i;
                }
            }
        }
        // Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.
        sdec = zero;
        if (stplen > zero) {
            iterc = iterc + 1;
            temp = shs / stepsq;
            if (iact == 0 && temp > zero) {
                crvmin = dmin1(crvmin, temp);
                if (crvmin == onemin) crvmin = temp;
            }
            ggsav = gredsq;
            gredsq = zero;
            for (int i = 1; i <= n; i++) {
                GNEW(i) = GNEW(i) + stplen * HS(i);
                if (XBDI(i) == zero) gredsq = gredsq + sq(GNEW(i));
                D(i) = D(i) + stplen * S(i);
            }
            sdec = dmax1(stplen * (ggsav - half * stplen * shs), zero);
            qred = qred + sdec;
        }
        // Restart the conjugate gradient method if it has hit a new bound.
        if (iact > 0) {
            nact = nact + 1;
            XBDI(iact) = one;
            if (S(iact) < zero) XBDI(iact) = onemin;
            delsq = delsq - sq(D(iact));
            if (delsq <= zero) goto L90;
            goto L20;
        }
        // If STPLEN is less than BLEN, then either apply another conjugate
        // gradient iteration or RETURN.
        if (stplen < blen) {
            if (iterc == itermax) goto L190;
            if (sdec <= 0.01 * qred) goto L190;
            beta = gredsq / ggsav;
            goto L30;
        }
    L90:
        crvmin = zero;
        // Prepare for the alternative iteration.
    L100:
        if (nact >= n - 1) goto L190;
        dredsq = zero;
        dredg = zero;
        gredsq = zero;
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) == zero) {
                dredsq = dredsq + sq(D(i));
                dredg = dredg + D(i) * GNEW(i);
                gredsq = gredsq + sq(GNEW(i));
                S(i) = D(i);
            } else {
                S(i) = zero;
            }
        }
        itcsav = iterc;
        goto L210;
        // Let the search direction S be a linear combination of the reduced D
        // and the reduced G that is orthogonal to the reduced D.
    L120:
        iterc = iterc + 1;
        temp = gredsq * dredsq - dredg * dredg;
        if (temp <= 1.0e-4 * qred * qred) goto L190;
        temp = sqrt(temp);
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) == zero) {
                S(i) = (dredg * D(i) - dredsq * GNEW(i)) / temp;
            } else {
                S(i) = zero;
            }
        }
        sredg = -temp;
        // Calculate an upper bound on the tangent of half the angle of the
        // alternative iteration, namely ANGBD.
        angbd = one;
        iact = 0;
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) == zero) {
                tempa = XOPT(i) + D(i) - SL(i);
                tempb = SU(i) - XOPT(i) - D(i);
                if (tempa <= zero) {
                    nact = nact + 1;
                    XBDI(i) = onemin;
                    goto L100;
                } else if (tempb <= zero) {
                    nact = nact + 1;
                    XBDI(i) = one;
                    goto L100;
                }
                ratio = one;
                ssq = sq(D(i)) + sq(S(i));
                temp = ssq - sq(XOPT(i) - SL(i));
                if (temp > zero) {
                    temp = sqrt(temp) - S(i);
                    if (angbd * temp > tempa) {
                        angbd = tempa / temp;
                        iact = i;
                        xsav = onemin;
                    }
                }
                temp = ssq - sq(SU(i) - XOPT(i));
                if (temp > zero) {
                    temp = sqrt(temp) + S(i);
                    if (angbd * temp > tempb) {
                        angbd = tempb / temp;
                        iact = i;
                        xsav = one;
                    }
                }
            }
        }
        // Calculate HHD and some curvatures for the alternative iteration.
        goto L210;
    L150:
        shs = zero;
        dhs = zero;
        dhd = zero;
        for (int i = 1; i <= n; i++) {
            if (XBDI(i) == zero) {
                shs = shs + S(i) * HS(i);
                dhs = dhs + D(i) * HS(i);
                dhd = dhd + D(i) * HRED(i);
            }
        }
        // Seek the greatest reduction in Q for a range of equally spaced values
        // of ANGT in [0,ANGBD].
        redmax = zero;
        isav = 0;
        redsav = zero;
        iu = (int) (17.0 * angbd + 3.1);
        for (int i = 1; i <= iu; i++) {
            angt = angbd * (double) i / (double) iu;
            sth = (angt + angt) / (one + angt * angt);
            temp = shs + angt * (angt * dhd - dhs - dhs);
            rednew = sth * (angt * dredg - sredg - half * sth * temp);
            if (rednew > redmax) {
                redmax = rednew;
                isav = i;
                rdprev = redsav;
            } else if (i == isav + 1) {
                rdnext = rednew;
            }
            redsav = rednew;
        }
        // Return if the reduction is zero. Otherwise, set the sine and cosine
        // of the angle of the alternative iteration, and calculate SDEC.
        if (isav == 0) goto L190;
        if (isav < iu) {
            temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
            angt = angbd * ((double) isav + half * temp) / (double) iu;
        }
        cth = (one - angt * angt) / (one + angt * angt);
        sth = (angt + angt) / (one + angt * angt);
        temp = shs + angt * (angt * dhd - dhs - dhs);
        sdec = sth * (angt * dredg - sredg - half * sth * temp);
        if (sdec <= zero) goto L190;
        // Update GNEW, D and HRED.
        dredg = zero;
        gredsq = zero;
        for (int i = 1; i <= n; i++) {
            GNEW(i) = GNEW(i) + (cth - one) * HRED(i) + sth * HS(i);
            if (XBDI(i) == zero) {
                D(i) = cth * D(i) + sth * S(i);
                dredg = dredg + D(i) * GNEW(i);
                gredsq = gredsq + sq(GNEW(i));
            }
            HRED(i) = cth * HRED(i) + sth * HS(i);
        }
        qred = qred + sdec;
        if (iact > 0 && isav == iu) {
            nact = nact + 1;
            XBDI(iact) = xsav;
            goto L100;
        }
        // If SDEC is sufficiently small, then RETURN after setting XNEW to
        // XOPT+D, giving careful attention to the bounds.
        if (sdec > 0.01 * qred) goto L120;
    L190:
        dsq = zero;
        for (int i = 1; i <= n; i++) {
            XNEW(i) = dmax1(dmin1(XOPT(i) + D(i), SU(i)), SL(i));
            if (XBDI(i) == onemin) XNEW(i) = SL(i);
            if (XBDI(i) == one) XNEW(i) = SU(i);
            D(i) = XNEW(i) - XOPT(i);
            dsq = dsq + sq(D(i));
        }
        return;

        // The following instructions multiply the current S-vector by the
        // second derivative matrix of the quadratic model, putting the product
        // in HS.  They are reached from three different parts of the software
        // above and they can be regarded as an external subroutine.
    L210:
        ih = 0;
        for (int j = 1; j <= n; j++) {
            HS(j) = zero;
            for (int i = 1; i <= j; i++) {
                ih = ih + 1;
                if (i < j) HS(j) = HS(j) + HQ(ih) * S(i);
                HS(i) = HS(i) + HQ(ih) * S(j);
            }
        }
        for (int k = 1; k <= npt; k++) {
            if (PQ(k) != zero) {
                temp = zero;
                for (int j = 1; j <= n; j++) temp = temp + XPT(k, j) * S(j);
                temp = temp * PQ(k);
                for (int i = 1; i <= n; i++) HS(i) = HS(i) + temp * XPT(k, i);
            }
        }
        if (crvmin != zero) goto L50;
        if (iterc > itcsav) goto L150;
        for (int i = 1; i <= n; i++) HRED(i) = HS(i);
        goto L120;
    }

    // ======================================================================
    // SUBROUTINE ALTMOV
    // ======================================================================
    void altmov(int n, int npt, const double* xpt, const double* xopt,
                const double* bmat, const double* zmat, int ndim,
                const double* sl, const double* su, int kopt, int knew,
                double adelt, double* xnew, double* xalt, double& alpha,
                double& cauchy, double* glag, double* hcol, double* w) {
        const double half = 0.5, one = 1.0, zero = 0.0;
        double cnst, temp, ha, presav, dderiv, distsq, subd, slbd, sumin, diff,
            step, vlag, tempd, tempa, tempb, predsq, stpsav, bigstp, wfixsq,
            ggfree, wsqsav, gw, curv, scale, csave = zero;
        int ilbd, iubd, isbd, ksav, ibdsav, iflag;

        cnst = one + sqrt(2.0);
        for (int k = 1; k <= npt; k++) HCOL(k) = zero;
        for (int j = 1; j <= npt - n - 1; j++) {
            temp = ZMAT(knew, j);
            for (int k = 1; k <= npt; k++) HCOL(k) = HCOL(k) + temp * ZMAT(k, j);
        }
        alpha = HCOL(knew);
        ha = half * alpha;
        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.
        for (int i = 1; i <= n; i++) GLAG(i) = BMAT(knew, i);
        for (int k = 1; k <= npt; k++) {
            temp = zero;
            for (int j = 1; j <= n; j++) temp = temp + XPT(k, j) * XOPT(j);
            temp = HCOL(k) * temp;
            for (int i = 1; i <= n; i++) GLAG(i) = GLAG(i) + temp * XPT(k, i);
        }
        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point.
        presav = zero;
        // Deviation: KSAV/STPSAV are uninitialised in the Fortran if PREDSQ is
        // never positive; use a safe default (XNEW = XOPT).
        ksav = knew;
        stpsav = zero;
        ibdsav = 0;
        for (int k = 1; k <= npt; k++) {
            if (k == kopt) continue; // GOTO 80
            dderiv = zero;
            distsq = zero;
            for (int i = 1; i <= n; i++) {
                temp = XPT(k, i) - XOPT(i);
                dderiv = dderiv + GLAG(i) * temp;
                distsq = distsq + temp * temp;
            }
            subd = adelt / sqrt(distsq);
            slbd = -subd;
            ilbd = 0;
            iubd = 0;
            sumin = dmin1(one, subd);
            // Revise SLBD and SUBD if necessary because of the bounds in SL and SU.
            for (int i = 1; i <= n; i++) {
                temp = XPT(k, i) - XOPT(i);
                if (temp > zero) {
                    if (slbd * temp < SL(i) - XOPT(i)) {
                        slbd = (SL(i) - XOPT(i)) / temp;
                        ilbd = -i;
                    }
                    if (subd * temp > SU(i) - XOPT(i)) {
                        subd = dmax1(sumin, (SU(i) - XOPT(i)) / temp);
                        iubd = i;
                    }
                } else if (temp < zero) {
                    if (slbd * temp > SU(i) - XOPT(i)) {
                        slbd = (SU(i) - XOPT(i)) / temp;
                        ilbd = i;
                    }
                    if (subd * temp < SL(i) - XOPT(i)) {
                        subd = dmax1(sumin, (SL(i) - XOPT(i)) / temp);
                        iubd = -i;
                    }
                }
            }
            // Seek a large modulus of the KNEW-th Lagrange function when the
            // index of the other interpolation point on the line through XOPT
            // is KNEW.
            if (k == knew) {
                diff = dderiv - one;
                step = slbd;
                vlag = slbd * (dderiv - slbd * diff);
                isbd = ilbd;
                temp = subd * (dderiv - subd * diff);
                if (abs(temp) > abs(vlag)) {
                    step = subd;
                    vlag = temp;
                    isbd = iubd;
                }
                tempd = half * dderiv;
                tempa = tempd - diff * slbd;
                tempb = tempd - diff * subd;
                if (tempa * tempb < zero) {
                    temp = tempd * tempd / diff;
                    if (abs(temp) > abs(vlag)) {
                        step = tempd / diff;
                        vlag = temp;
                        isbd = 0;
                    }
                }
            } else {
                // Search along each of the other lines through XOPT and
                // another point.
                step = slbd;
                vlag = slbd * (one - slbd);
                isbd = ilbd;
                temp = subd * (one - subd);
                if (abs(temp) > abs(vlag)) {
                    step = subd;
                    vlag = temp;
                    isbd = iubd;
                }
                if (subd > half) {
                    if (abs(vlag) < 0.25) {
                        step = half;
                        vlag = 0.25;
                        isbd = 0;
                    }
                }
                vlag = vlag * dderiv;
            }
            // Calculate PREDSQ for the current line search and maintain PRESAV.
            temp = step * (one - step) * distsq;
            predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
            if (predsq > presav) {
                presav = predsq;
                ksav = k;
                stpsav = step;
                ibdsav = isbd;
            }
        } // 80 CONTINUE
        // Construct XNEW in a way that satisfies the bound constraints exactly.
        // (minqa 1.2.8 sets IBDSAV=0 here, which disables the two exact
        // assignments below; kept for fidelity.)
        ibdsav = 0;
        for (int i = 1; i <= n; i++) {
            temp = XOPT(i) + stpsav * (XPT(ksav, i) - XOPT(i));
            XNEW(i) = dmax1(SL(i), dmin1(SU(i), temp));
        }
        if (ibdsav < 0) XNEW(-ibdsav) = SL(-ibdsav);
        if (ibdsav > 0) XNEW(ibdsav) = SU(ibdsav);
        // Prepare for the iterative method that assembles the constrained
        // Cauchy step in W.
        bigstp = adelt + adelt;
        iflag = 0;
    L100:
        wfixsq = zero;
        ggfree = zero;
        for (int i = 1; i <= n; i++) {
            W(i) = zero;
            tempa = dmin1(XOPT(i) - SL(i), GLAG(i));
            tempb = dmax1(XOPT(i) - SU(i), GLAG(i));
            if (tempa > zero || tempb < zero) {
                W(i) = bigstp;
                ggfree = ggfree + sq(GLAG(i));
            }
        }
        if (ggfree == zero) {
            cauchy = zero;
            goto L200;
        }
        // Investigate whether more components of W can be fixed.
        step = zero;
    L120:
        temp = adelt * adelt - wfixsq;
        if (temp > zero) {
            wsqsav = wfixsq;
            step = sqrt(temp / ggfree);
            ggfree = zero;
            for (int i = 1; i <= n; i++) {
                if (W(i) == bigstp) {
                    temp = XOPT(i) - step * GLAG(i);
                    if (temp <= SL(i)) {
                        W(i) = SL(i) - XOPT(i);
                        wfixsq = wfixsq + sq(W(i));
                    } else if (temp >= SU(i)) {
                        W(i) = SU(i) - XOPT(i);
                        wfixsq = wfixsq + sq(W(i));
                    } else {
                        ggfree = ggfree + sq(GLAG(i));
                    }
                }
            }
            if (wfixsq > wsqsav && ggfree > zero) goto L120;
        }
        // Set the remaining free components of W and all components of XALT,
        // except that W may be scaled later.
        gw = zero;
        for (int i = 1; i <= n; i++) {
            if (W(i) == bigstp) {
                W(i) = -step * GLAG(i);
                XALT(i) = dmax1(SL(i), dmin1(SU(i), XOPT(i) + W(i)));
            } else if (W(i) == zero) {
                XALT(i) = XOPT(i);
            } else if (GLAG(i) > zero) {
                XALT(i) = SL(i);
            } else {
                XALT(i) = SU(i);
            }
            gw = gw + GLAG(i) * W(i);
        }
        // Set CURV to the curvature of the KNEW-th Lagrange function along W.
        // Scale W by a factor less than one if that can reduce the modulus of
        // the Lagrange function at XOPT+W. Set CAUCHY to the final value of
        // the square of this function.
        curv = zero;
        for (int k = 1; k <= npt; k++) {
            temp = zero;
            for (int j = 1; j <= n; j++) temp = temp + XPT(k, j) * W(j);
            curv = curv + HCOL(k) * temp * temp;
        }
        if (iflag == 1) curv = -curv;
        if (curv > -gw && curv < -cnst * gw) {
            scale = -gw / curv;
            for (int i = 1; i <= n; i++) {
                temp = XOPT(i) + scale * W(i);
                XALT(i) = dmax1(SL(i), dmin1(SU(i), temp));
            }
            cauchy = sq(half * gw * scale);
        } else {
            cauchy = sq(gw + half * curv);
        }
        // If IFLAG is zero, then XALT is calculated as before after reversing
        // the sign of GLAG. Thus two XALT vectors become available. The one
        // that is chosen is the one that gives the larger value of CAUCHY.
        if (iflag == 0) {
            for (int i = 1; i <= n; i++) {
                GLAG(i) = -GLAG(i);
                W(n + i) = XALT(i);
            }
            csave = cauchy;
            iflag = 1;
            goto L100;
        }
        if (csave > cauchy) {
            for (int i = 1; i <= n; i++) XALT(i) = W(n + i);
            cauchy = csave;
        }
    L200:
        return;
    }

    // ======================================================================
    // SUBROUTINE UPDATEBOBYQA
    // ======================================================================
    void updatebobyqa(int n, int npt, double* bmat, double* zmat, int ndim,
                      double* vlag, double beta, double denom, int knew,
                      double* w) {
        const double one = 1.0, zero = 0.0;
        int nptm, jp;
        double ztest, temp, tempa, tempb, alpha, tau;

        nptm = npt - n - 1;
        ztest = zero;
        for (int k = 1; k <= npt; k++)
            for (int j = 1; j <= nptm; j++)
                ztest = dmax1(ztest, abs(ZMAT(k, j)));
        ztest = 1.0e-20 * ztest;
        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.
        for (int j = 2; j <= nptm; j++) {
            if (abs(ZMAT(knew, j)) > ztest) {
                temp = sqrt(sq(ZMAT(knew, 1)) + sq(ZMAT(knew, j)));
                tempa = ZMAT(knew, 1) / temp;
                tempb = ZMAT(knew, j) / temp;
                for (int i = 1; i <= npt; i++) {
                    temp = tempa * ZMAT(i, 1) + tempb * ZMAT(i, j);
                    ZMAT(i, j) = tempa * ZMAT(i, j) - tempb * ZMAT(i, 1);
                    ZMAT(i, 1) = temp;
                }
            }
            ZMAT(knew, j) = zero;
        }
        // Put the first NPT components of the KNEW-th column of HLAG into W,
        // and calculate the parameters of the updating formula.
        for (int i = 1; i <= npt; i++) W(i) = ZMAT(knew, 1) * ZMAT(i, 1);
        alpha = W(knew);
        tau = VLAG(knew);
        VLAG(knew) = VLAG(knew) - one;
        // Complete the updating of ZMAT.
        temp = sqrt(denom);
        tempb = ZMAT(knew, 1) / temp;
        tempa = tau / temp;
        for (int i = 1; i <= npt; i++)
            ZMAT(i, 1) = tempa * ZMAT(i, 1) - tempb * VLAG(i);
        // Finally, update the matrix BMAT.
        for (int j = 1; j <= n; j++) {
            jp = npt + j;
            W(jp) = BMAT(knew, j);
            tempa = (alpha * VLAG(jp) - tau * W(jp)) / denom;
            tempb = (-beta * W(jp) - tau * VLAG(jp)) / denom;
            for (int i = 1; i <= jp; i++) {
                BMAT(i, j) = BMAT(i, j) + tempa * VLAG(i) + tempb * W(i);
                if (i > npt) BMAT(jp, i - npt) = BMAT(i, j);
            }
        }
    }

    // ======================================================================
    // SUBROUTINE RESCUE
    // ======================================================================
    void rescue(int n, int npt, const double* xl, const double* xu, int iprint,
                int maxfun, double* xbase, double* xpt, double* fval,
                double* xopt, double* gopt, double* hq, double* pq,
                double* bmat, double* zmat, int ndim, double* sl, double* su,
                int& nf, double delta, int& kopt, double* vlag, double* ptsaux,
                double* ptsid, double* w, Calfun& calfun) {
        const double half = 0.5, one = 1.0, zero = 0.0;
        int np, nptm, ih, jp, jpn, iw, ip, iq, nrem, kold, knew, ihp = 0, ihq;
        double sfrac, sumpq, winc, distsq, temp, fbase, sum, beta = zero,
            bsum, denom = zero, vlmxsq, hdiag, den, dsqmin, xp = zero, xq = zero,
            vquad, f, diff;

        np = n + 1;
        sfrac = half / (double) np;
        nptm = npt - np;
        // Shift the interpolation points so that XOPT becomes the origin, and
        // set the elements of ZMAT to zero.
        sumpq = zero;
        winc = zero;
        for (int k = 1; k <= npt; k++) {
            distsq = zero;
            for (int j = 1; j <= n; j++) {
                XPT(k, j) = XPT(k, j) - XOPT(j);
                distsq = distsq + sq(XPT(k, j));
            }
            sumpq = sumpq + PQ(k);
            W(ndim + k) = distsq;
            winc = dmax1(winc, distsq);
            for (int j = 1; j <= nptm; j++) ZMAT(k, j) = zero;
        }
        // Update HQ so that HQ and PQ define the second derivatives of the
        // model after XBASE has been shifted to the trust region centre.
        ih = 0;
        for (int j = 1; j <= n; j++) {
            W(j) = half * sumpq * XOPT(j);
            for (int k = 1; k <= npt; k++) W(j) = W(j) + PQ(k) * XPT(k, j);
            for (int i = 1; i <= j; i++) {
                ih = ih + 1;
                HQ(ih) = HQ(ih) + W(i) * XOPT(j) + W(j) * XOPT(i);
            }
        }
        // Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to zero, and
        // also set the elements of PTSAUX.
        for (int j = 1; j <= n; j++) {
            XBASE(j) = XBASE(j) + XOPT(j);
            SL(j) = SL(j) - XOPT(j);
            SU(j) = SU(j) - XOPT(j);
            XOPT(j) = zero;
            PTSAUX(1, j) = dmin1(delta, SU(j));
            PTSAUX(2, j) = dmax1(-delta, SL(j));
            if (PTSAUX(1, j) + PTSAUX(2, j) < zero) {
                temp = PTSAUX(1, j);
                PTSAUX(1, j) = PTSAUX(2, j);
                PTSAUX(2, j) = temp;
            }
            if (abs(PTSAUX(2, j)) < half * abs(PTSAUX(1, j))) {
                PTSAUX(2, j) = half * PTSAUX(1, j);
            }
            for (int i = 1; i <= ndim; i++) BMAT(i, j) = zero;
        }
        fbase = FVAL(kopt);
        // Set the identifiers of the artificial interpolation points that are
        // along a coordinate direction from XOPT, and set the corresponding
        // nonzero elements of BMAT and ZMAT.
        PTSID(1) = sfrac;
        for (int j = 1; j <= n; j++) {
            jp = j + 1;
            jpn = jp + n;
            PTSID(jp) = (double) j + sfrac;
            if (jpn <= npt) {
                PTSID(jpn) = (double) j / (double) np + sfrac;
                temp = one / (PTSAUX(1, j) - PTSAUX(2, j));
                BMAT(jp, j) = -temp + one / PTSAUX(1, j);
                BMAT(jpn, j) = temp + one / PTSAUX(2, j);
                BMAT(1, j) = -BMAT(jp, j) - BMAT(jpn, j);
                ZMAT(1, j) = sqrt(2.0) / abs(PTSAUX(1, j) * PTSAUX(2, j));
                ZMAT(jp, j) = ZMAT(1, j) * PTSAUX(2, j) * temp;
                ZMAT(jpn, j) = -ZMAT(1, j) * PTSAUX(1, j) * temp;
            } else {
                BMAT(1, j) = -one / PTSAUX(1, j);
                BMAT(jp, j) = one / PTSAUX(1, j);
                BMAT(j + npt, j) = -half * sq(PTSAUX(1, j));
            }
        }
        // Set any remaining identifiers with their nonzero elements of ZMAT.
        if (npt >= n + np) {
            for (int k = 2 * np; k <= npt; k++) {
                iw = (int) (((double) (k - np) - half) / (double) n);
                ip = k - np - iw * n;
                iq = ip + iw;
                if (iq > n) iq = iq - n;
                PTSID(k) = (double) ip + (double) iq / (double) np + sfrac;
                temp = one / (PTSAUX(1, ip) * PTSAUX(1, iq));
                ZMAT(1, k - np) = temp;
                ZMAT(ip + 1, k - np) = -temp;
                ZMAT(iq + 1, k - np) = -temp;
                ZMAT(k, k - np) = temp;
            }
        }
        nrem = npt;
        kold = 1;
        knew = kopt;
        // Reorder the provisional points in the way that exchanges PTSID(KOLD)
        // with PTSID(KNEW).
    L80:
        for (int j = 1; j <= n; j++) {
            temp = BMAT(kold, j);
            BMAT(kold, j) = BMAT(knew, j);
            BMAT(knew, j) = temp;
        }
        for (int j = 1; j <= nptm; j++) {
            temp = ZMAT(kold, j);
            ZMAT(kold, j) = ZMAT(knew, j);
            ZMAT(knew, j) = temp;
        }
        PTSID(kold) = PTSID(knew);
        PTSID(knew) = zero;
        W(ndim + knew) = zero;
        nrem = nrem - 1;
        if (knew != kopt) {
            temp = VLAG(kold);
            VLAG(kold) = VLAG(knew);
            VLAG(knew) = temp;
            // Update the BMAT and ZMAT matrices so that the status of the
            // KNEW-th interpolation point can be changed from provisional to
            // original.
            updatebobyqa(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w);
            if (nrem == 0) goto L350;
            for (int k = 1; k <= npt; k++) W(ndim + k) = abs(W(ndim + k));
        }
        // Pick the index KNEW of an original interpolation point that has not
        // yet replaced one of the provisional interpolation points.
    L120:
        dsqmin = zero;
        for (int k = 1; k <= npt; k++) {
            if (W(ndim + k) > zero) {
                if (dsqmin == zero || W(ndim + k) < dsqmin) {
                    knew = k;
                    dsqmin = W(ndim + k);
                }
            }
        }
        if (dsqmin == zero) goto L260;
        // Form the W-vector of the chosen original interpolation point.
        for (int j = 1; j <= n; j++) W(npt + j) = XPT(knew, j);
        for (int k = 1; k <= npt; k++) {
            sum = zero;
            if (k == kopt) {
                // CONTINUE
            } else if (PTSID(k) == zero) {
                for (int j = 1; j <= n; j++) sum = sum + W(npt + j) * XPT(k, j);
            } else {
                ip = (int) PTSID(k);
                if (ip > 0) sum = W(npt + ip) * PTSAUX(1, ip);
                iq = (int) ((double) np * PTSID(k) - (double) (ip * np));
                if (iq > 0) {
                    iw = 1;
                    if (ip == 0) iw = 2;
                    sum = sum + W(npt + iq) * PTSAUX(iw, iq);
                }
            }
            W(k) = half * sum * sum;
        }
        // Calculate VLAG and BETA for the required updating of the H matrix if
        // XPT(KNEW,.) is reinstated in the set of interpolation points.
        for (int k = 1; k <= npt; k++) {
            sum = zero;
            for (int j = 1; j <= n; j++) sum = sum + BMAT(k, j) * W(npt + j);
            VLAG(k) = sum;
        }
        beta = zero;
        for (int j = 1; j <= nptm; j++) {
            sum = zero;
            for (int k = 1; k <= npt; k++) sum = sum + ZMAT(k, j) * W(k);
            beta = beta - sum * sum;
            for (int k = 1; k <= npt; k++) VLAG(k) = VLAG(k) + sum * ZMAT(k, j);
        }
        bsum = zero;
        distsq = zero;
        for (int j = 1; j <= n; j++) {
            sum = zero;
            for (int k = 1; k <= npt; k++) sum = sum + BMAT(k, j) * W(k);
            jp = j + npt;
            bsum = bsum + sum * W(jp);
            for (int ipp = npt + 1; ipp <= ndim; ipp++) sum = sum + BMAT(ipp, j) * W(ipp);
            bsum = bsum + sum * W(jp);
            VLAG(jp) = sum;
            distsq = distsq + sq(XPT(knew, j));
        }
        beta = half * distsq * distsq + beta - bsum;
        VLAG(kopt) = VLAG(kopt) + one;
        // KOLD is set to the index of the provisional interpolation point that
        // is going to be deleted to make way for the KNEW-th original
        // interpolation point.
        denom = zero;
        vlmxsq = zero;
        for (int k = 1; k <= npt; k++) {
            if (PTSID(k) != zero) {
                hdiag = zero;
                for (int j = 1; j <= nptm; j++) hdiag = hdiag + sq(ZMAT(k, j));
                den = beta * hdiag + sq(VLAG(k));
                if (den > denom) {
                    kold = k;
                    denom = den;
                }
            }
            vlmxsq = dmax1(vlmxsq, sq(VLAG(k)));
        }
        if (denom <= 1.0e-2 * vlmxsq) {
            W(ndim + knew) = -W(ndim + knew) - winc;
            goto L120;
        }
        goto L80;
        // When label 260 is reached, all the final positions of the
        // interpolation points have been chosen although any changes have not
        // been included yet in XPT.
    L260:
        for (int kpt = 1; kpt <= npt; kpt++) {
            if (PTSID(kpt) == zero) continue; // GOTO 340
            if (nf >= maxfun) {
                nf = -1;
                goto L350;
            }
            ih = 0;
            for (int j = 1; j <= n; j++) {
                W(j) = XPT(kpt, j);
                XPT(kpt, j) = zero;
                temp = PQ(kpt) * W(j);
                for (int i = 1; i <= j; i++) {
                    ih = ih + 1;
                    HQ(ih) = HQ(ih) + temp * W(i);
                }
            }
            PQ(kpt) = zero;
            ip = (int) PTSID(kpt);
            iq = (int) ((double) np * PTSID(kpt) - (double) (ip * np));
            if (ip > 0) {
                xp = PTSAUX(1, ip);
                XPT(kpt, ip) = xp;
            }
            if (iq > 0) {
                xq = PTSAUX(1, iq);
                if (ip == 0) xq = PTSAUX(2, iq);
                XPT(kpt, iq) = xq;
            }
            // Set VQUAD to the value of the current model at the new point.
            vquad = fbase;
            if (ip > 0) {
                ihp = (ip + ip * ip) / 2;
                vquad = vquad + xp * (GOPT(ip) + half * xp * HQ(ihp));
            }
            if (iq > 0) {
                ihq = (iq + iq * iq) / 2;
                vquad = vquad + xq * (GOPT(iq) + half * xq * HQ(ihq));
                if (ip > 0) {
                    iw = std::max(ihp, ihq) - std::abs(ip - iq);
                    vquad = vquad + xp * xq * HQ(iw);
                }
            }
            for (int k = 1; k <= npt; k++) {
                temp = zero;
                if (ip > 0) temp = temp + xp * XPT(k, ip);
                if (iq > 0) temp = temp + xq * XPT(k, iq);
                vquad = vquad + half * PQ(k) * temp * temp;
            }
            // Calculate F at the new interpolation point, and set DIFF to the
            // factor that is going to multiply the KPT-th Lagrange function
            // when the model is updated to provide interpolation to the new
            // function value.
            for (int i = 1; i <= n; i++) {
                W(i) = dmin1(dmax1(XL(i), XBASE(i) + XPT(kpt, i)), XU(i));
                if (XPT(kpt, i) == SL(i)) W(i) = XL(i);
                if (XPT(kpt, i) == SU(i)) W(i) = XU(i);
            }
            nf = nf + 1;
            // Deviation: minqa 1.2.8 calls CALFUN(N,X,IPRINT) here with X an
            // undeclared local scalar (a bug); Powell's original evaluates F
            // at W, as is done here.
            f = calfun(n, w, iprint);
            FVAL(kpt) = f;
            if (f < FVAL(kopt)) kopt = kpt;
            diff = f - vquad;
            // Update the quadratic model.
            for (int i = 1; i <= n; i++) GOPT(i) = GOPT(i) + diff * BMAT(kpt, i);
            for (int k = 1; k <= npt; k++) {
                sum = zero;
                for (int j = 1; j <= nptm; j++) sum = sum + ZMAT(k, j) * ZMAT(kpt, j);
                temp = diff * sum;
                if (PTSID(k) == zero) {
                    PQ(k) = PQ(k) + temp;
                } else {
                    ip = (int) PTSID(k);
                    iq = (int) ((double) np * PTSID(k) - (double) (ip * np));
                    ihq = (iq * iq + iq) / 2;
                    if (ip == 0) {
                        HQ(ihq) = HQ(ihq) + temp * sq(PTSAUX(2, iq));
                    } else {
                        ihp = (ip * ip + ip) / 2;
                        HQ(ihp) = HQ(ihp) + temp * sq(PTSAUX(1, ip));
                        if (iq > 0) {
                            HQ(ihq) = HQ(ihq) + temp * sq(PTSAUX(1, iq));
                            iw = std::max(ihp, ihq) - std::abs(iq - ip);
                            HQ(iw) = HQ(iw) + temp * PTSAUX(1, ip) * PTSAUX(1, iq);
                        }
                    }
                }
            }
            PTSID(kpt) = zero;
        } // 340 CONTINUE
    L350:
        return;
    }

    // ======================================================================
    // SUBROUTINE BOBYQB
    // ======================================================================
    void bobyqb(int n, int npt, double* x, const double* xl, const double* xu,
                double rhobeg, double rhoend, int iprint, int maxfun,
                double* xbase, double* xpt, double* fval, double* xopt,
                double* gopt, double* hq, double* pq, double* bmat,
                double* zmat, int ndim, double* sl, double* su, double* xnew,
                double* xalt, double* d, double* vlag, double* w, int& ierr,
                Calfun& calfun) {
        const double half = 0.5, one = 1.0, ten = 10.0, tenth = 0.1, two = 2.0,
            zero = 0.0;
        int np, nptm, nh, nf = 0, kopt = 0, kbase, nresc, ntrits, itest, nfsav,
            ih, ip, jp, knew = 0, ksav;
        double xoptsq, fsave, rho, delta, diffa, diffb, diffc = zero, dsq = zero,
            crvmin = zero, dnorm = zero, distsq = zero, errbig, frhosq, bdtol,
            bdtest, curv, fracsq, sumpq, sum, temp, sumz, sumw, adelt = zero,
            alpha = zero, cauchy = zero, suma, sumb, beta = zero, bsum, dx,
            denom = zero, delsq, scaden, biglsq, hdiag, den, f = zero,
            fopt = zero, vquad = zero, diff, ratio = zero, densav, pqold, gqsq,
            gisq, dist;

        np = n + 1;
        nptm = npt - np;
        nh = (n * np) / 2;
        // The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ,
        // PQ, BMAT and ZMAT for the first iteration, with the corresponding
        // values of NF and KOPT.
        prelim(n, npt, x, xl, xu, rhobeg, iprint, maxfun, xbase, xpt, fval,
               gopt, hq, pq, bmat, zmat, ndim, sl, su, nf, kopt, calfun);
        xoptsq = zero;
        for (int i = 1; i <= n; i++) {
            XOPT(i) = XPT(kopt, i);
            xoptsq = xoptsq + sq(XOPT(i));
        }
        fsave = FVAL(1);
        if (nf < npt) {
            ierr = 390;
            goto L720;
        }
        kbase = 1;
        // Complete the settings that are required for the iterative procedure.
        rho = rhobeg;
        delta = rho;
        nresc = nf;
        ntrits = 0;
        diffa = zero;
        diffb = zero;
        itest = 0;
        nfsav = nf;
        // Update GOPT if necessary before the first iteration and after each
        // call of RESCUE that makes a call of CALFUN.
    L20:
        if (kopt != kbase) {
            ih = 0;
            for (int j = 1; j <= n; j++) {
                for (int i = 1; i <= j; i++) {
                    ih = ih + 1;
                    if (i < j) GOPT(j) = GOPT(j) + HQ(ih) * XOPT(i);
                    GOPT(i) = GOPT(i) + HQ(ih) * XOPT(j);
                }
            }
            if (nf > npt) {
                for (int k = 1; k <= npt; k++) {
                    temp = zero;
                    for (int j = 1; j <= n; j++) temp = temp + XPT(k, j) * XOPT(j);
                    temp = PQ(k) * temp;
                    for (int i = 1; i <= n; i++) GOPT(i) = GOPT(i) + temp * XPT(k, i);
                }
            }
        }
        // Generate the next point in the trust region that provides a small
        // value of the quadratic model subject to the constraints on the
        // variables.
    L60:
        trsbox(n, npt, xpt, xopt, gopt, hq, pq, sl, su, delta, xnew, d,
               w, w + (np - 1), w + (np + n - 1), w + (np + 2 * n - 1),
               w + (np + 3 * n - 1), dsq, crvmin);
        dnorm = dmin1(delta, sqrt(dsq));
        if (dnorm < half * rho) {
            ntrits = -1;
            distsq = sq(ten * rho);
            if (nf <= nfsav + 2) goto L650;
            // The following choice between labels 650 and 680 depends on
            // whether or not our work with the current RHO seems to be complete.
            errbig = dmax1(dmax1(diffa, diffb), diffc);
            frhosq = 0.125 * rho * rho;
            if (crvmin > zero && errbig > frhosq * crvmin) goto L650;
            bdtol = errbig / rho;
            for (int j = 1; j <= n; j++) {
                bdtest = bdtol;
                if (XNEW(j) == SL(j)) bdtest = W(j);
                if (XNEW(j) == SU(j)) bdtest = -W(j);
                if (bdtest < bdtol) {
                    curv = HQ((j + j * j) / 2);
                    for (int k = 1; k <= npt; k++) curv = curv + PQ(k) * sq(XPT(k, j));
                    bdtest = bdtest + half * curv * rho;
                    if (bdtest < bdtol) goto L650;
                }
            }
            goto L680;
        }
        ntrits = ntrits + 1;
        // Severe cancellation is likely to occur if XOPT is too far from XBASE.
        // If the following test holds, then XBASE is shifted so that XOPT
        // becomes zero.
    L90:
        if (dsq <= 1.0e-3 * xoptsq) {
            fracsq = 0.25 * xoptsq;
            sumpq = zero;
            for (int k = 1; k <= npt; k++) {
                sumpq = sumpq + PQ(k);
                sum = -half * xoptsq;
                for (int i = 1; i <= n; i++) sum = sum + XPT(k, i) * XOPT(i);
                W(npt + k) = sum;
                temp = fracsq - half * sum;
                for (int i = 1; i <= n; i++) {
                    W(i) = BMAT(k, i);
                    VLAG(i) = sum * XPT(k, i) + temp * XOPT(i);
                    ip = npt + i;
                    for (int j = 1; j <= i; j++)
                        BMAT(ip, j) = BMAT(ip, j) + W(i) * VLAG(j) + VLAG(i) * W(j);
                }
            }
            // Then the revisions of BMAT that depend on ZMAT are calculated.
            for (int jj = 1; jj <= nptm; jj++) {
                sumz = zero;
                sumw = zero;
                for (int k = 1; k <= npt; k++) {
                    sumz = sumz + ZMAT(k, jj);
                    VLAG(k) = W(npt + k) * ZMAT(k, jj);
                    sumw = sumw + VLAG(k);
                }
                for (int j = 1; j <= n; j++) {
                    sum = (fracsq * sumz - half * sumw) * XOPT(j);
                    for (int k = 1; k <= npt; k++) sum = sum + VLAG(k) * XPT(k, j);
                    W(j) = sum;
                    for (int k = 1; k <= npt; k++) BMAT(k, j) = BMAT(k, j) + sum * ZMAT(k, jj);
                }
                for (int i = 1; i <= n; i++) {
                    ip = i + npt;
                    temp = W(i);
                    for (int j = 1; j <= i; j++) BMAT(ip, j) = BMAT(ip, j) + temp * W(j);
                }
            }
            // The following instructions complete the shift, including the
            // changes to the second derivative parameters of the quadratic model.
            ih = 0;
            for (int j = 1; j <= n; j++) {
                W(j) = -half * sumpq * XOPT(j);
                for (int k = 1; k <= npt; k++) {
                    W(j) = W(j) + PQ(k) * XPT(k, j);
                    XPT(k, j) = XPT(k, j) - XOPT(j);
                }
                for (int i = 1; i <= j; i++) {
                    ih = ih + 1;
                    HQ(ih) = HQ(ih) + W(i) * XOPT(j) + XOPT(i) * W(j);
                    BMAT(npt + i, j) = BMAT(npt + j, i);
                }
            }
            for (int i = 1; i <= n; i++) {
                XBASE(i) = XBASE(i) + XOPT(i);
                XNEW(i) = XNEW(i) - XOPT(i);
                SL(i) = SL(i) - XOPT(i);
                SU(i) = SU(i) - XOPT(i);
                XOPT(i) = zero;
            }
            xoptsq = zero;
        }
        if (ntrits == 0) goto L210;
        goto L230;
        // XBASE is also moved to XOPT by a call of RESCUE.
    L190:
        nfsav = nf;
        kbase = kopt;
        rescue(n, npt, xl, xu, iprint, maxfun, xbase, xpt, fval, xopt, gopt,
               hq, pq, bmat, zmat, ndim, sl, su, nf, delta, kopt, vlag,
               w, w + (n + np - 1), w + (ndim + np - 1), calfun);
        // XOPT is updated now in case the branch below to label 720 is taken.
        xoptsq = zero;
        if (kopt != kbase) {
            for (int i = 1; i <= n; i++) {
                XOPT(i) = XPT(kopt, i);
                xoptsq = xoptsq + sq(XOPT(i));
            }
        }
        if (nf < 0) {
            nf = maxfun;
            ierr = 390;
            goto L720;
        }
        nresc = nf;
        if (nfsav < nf) {
            nfsav = nf;
            goto L20;
        }
        if (ntrits > 0) goto L60;
        // Pick two alternative vectors of variables, relative to XBASE, that
        // are suitable as new positions of the KNEW-th interpolation point.
    L210:
        altmov(n, npt, xpt, xopt, bmat, zmat, ndim, sl, su, kopt, knew, adelt,
               xnew, xalt, alpha, cauchy, w, w + (np - 1), w + ndim);
        for (int i = 1; i <= n; i++) D(i) = XNEW(i) - XOPT(i);
        // Calculate VLAG and BETA for the current choice of D.
    L230:
        for (int k = 1; k <= npt; k++) {
            suma = zero;
            sumb = zero;
            sum = zero;
            for (int j = 1; j <= n; j++) {
                suma = suma + XPT(k, j) * D(j);
                sumb = sumb + XPT(k, j) * XOPT(j);
                sum = sum + BMAT(k, j) * D(j);
            }
            W(k) = suma * (half * suma + sumb);
            VLAG(k) = sum;
            W(npt + k) = suma;
        }
        beta = zero;
        for (int jj = 1; jj <= nptm; jj++) {
            sum = zero;
            for (int k = 1; k <= npt; k++) sum = sum + ZMAT(k, jj) * W(k);
            beta = beta - sum * sum;
            for (int k = 1; k <= npt; k++) VLAG(k) = VLAG(k) + sum * ZMAT(k, jj);
        }
        dsq = zero;
        bsum = zero;
        dx = zero;
        for (int j = 1; j <= n; j++) {
            dsq = dsq + sq(D(j));
            sum = zero;
            for (int k = 1; k <= npt; k++) sum = sum + W(k) * BMAT(k, j);
            bsum = bsum + sum * D(j);
            jp = npt + j;
            for (int i = 1; i <= n; i++) sum = sum + BMAT(jp, i) * D(i);
            VLAG(jp) = sum;
            bsum = bsum + sum * D(j);
            dx = dx + D(j) * XOPT(j);
        }
        beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta - bsum;
        VLAG(kopt) = VLAG(kopt) + one;
        // If NTRITS is zero, the denominator may be increased by replacing the
        // step D of ALTMOV by a Cauchy step. Then RESCUE may be called if
        // rounding errors have damaged the chosen denominator.
        if (ntrits == 0) {
            denom = sq(VLAG(knew)) + alpha * beta;
            if (denom < cauchy && cauchy > zero) {
                for (int i = 1; i <= n; i++) {
                    XNEW(i) = XALT(i);
                    D(i) = XNEW(i) - XOPT(i);
                }
                cauchy = zero;
                goto L230;
            }
            if (denom <= half * sq(VLAG(knew))) {
                if (nf > nresc) goto L190;
                ierr = 320;
                goto L720;
            }
        } else {
            // Alternatively, if NTRITS is positive, then set KNEW to the index
            // of the next interpolation point to be deleted to make room for a
            // trust region step.
            delsq = delta * delta;
            scaden = zero;
            biglsq = zero;
            knew = 0;
            for (int k = 1; k <= npt; k++) {
                if (k == kopt) continue; // GOTO 350
                hdiag = zero;
                for (int jj = 1; jj <= nptm; jj++) hdiag = hdiag + sq(ZMAT(k, jj));
                den = beta * hdiag + sq(VLAG(k));
                distsq = zero;
                for (int j = 1; j <= n; j++) distsq = distsq + sq(XPT(k, j) - XOPT(j));
                temp = dmax1(one, sq(distsq / delsq));
                if (temp * den > scaden) {
                    scaden = temp * den;
                    knew = k;
                    denom = den;
                }
                biglsq = dmax1(biglsq, temp * sq(VLAG(k)));
            } // 350 CONTINUE
            if (scaden <= half * biglsq) {
                if (nf > nresc) goto L190;
                ierr = 320;
                goto L720;
            }
        }
        // Put the variables for the next calculation of the objective function
        // in XNEW, with any adjustments for the bounds.  Calculate the value of
        // the objective function at XBASE+XNEW, unless the limit on the number
        // of calculations of F has been reached.
    L360:
        for (int i = 1; i <= n; i++) {
            X(i) = dmin1(dmax1(XL(i), XBASE(i) + XNEW(i)), XU(i));
            if (XNEW(i) == SL(i)) X(i) = XL(i);
            if (XNEW(i) == SU(i)) X(i) = XU(i);
        }
        if (nf >= maxfun) {
            ierr = 390;
            goto L720;
        }
        nf = nf + 1;
        f = calfun(n, x, iprint);
        if (ntrits == -1) {
            fsave = f;
            goto L720;
        }
        // Use the quadratic model to predict the change in F due to the step D,
        // and set DIFF to the error of this prediction.
        fopt = FVAL(kopt);
        vquad = zero;
        ih = 0;
        for (int j = 1; j <= n; j++) {
            vquad = vquad + D(j) * GOPT(j);
            for (int i = 1; i <= j; i++) {
                ih = ih + 1;
                temp = D(i) * D(j);
                if (i == j) temp = half * temp;
                vquad = vquad + HQ(ih) * temp;
            }
        }
        for (int k = 1; k <= npt; k++) vquad = vquad + half * PQ(k) * sq(W(npt + k));
        diff = f - fopt - vquad;
        diffc = diffb;
        diffb = diffa;
        diffa = abs(diff);
        if (dnorm > rho) nfsav = nf;
        // Pick the next value of DELTA after a trust region step.
        if (ntrits > 0) {
            if (vquad >= zero) {
                ierr = 430;
                goto L720;
            }
            ratio = (f - fopt) / vquad;
            if (ratio <= tenth) {
                delta = dmin1(half * delta, dnorm);
            } else if (ratio <= 0.7) {
                delta = dmax1(half * delta, dnorm);
            } else {
                delta = dmax1(half * delta, dnorm + dnorm);
            }
            if (delta <= 1.5 * rho) delta = rho;
            // Recalculate KNEW and DENOM if the new F is less than FOPT.
            if (f < fopt) {
                ksav = knew;
                densav = denom;
                delsq = delta * delta;
                scaden = zero;
                biglsq = zero;
                knew = 0;
                for (int k = 1; k <= npt; k++) {
                    hdiag = zero;
                    for (int jj = 1; jj <= nptm; jj++) hdiag = hdiag + sq(ZMAT(k, jj));
                    den = beta * hdiag + sq(VLAG(k));
                    distsq = zero;
                    for (int j = 1; j <= n; j++) distsq = distsq + sq(XPT(k, j) - XNEW(j));
                    temp = dmax1(one, sq(distsq / delsq));
                    if (temp * den > scaden) {
                        scaden = temp * den;
                        knew = k;
                        denom = den;
                    }
                    biglsq = dmax1(biglsq, temp * sq(VLAG(k)));
                }
                if (scaden <= half * biglsq) {
                    knew = ksav;
                    denom = densav;
                }
            }
        }
        // Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
        // moved. Also update the second derivative terms of the model.
        updatebobyqa(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w);
        ih = 0;
        pqold = PQ(knew);
        PQ(knew) = zero;
        for (int i = 1; i <= n; i++) {
            temp = pqold * XPT(knew, i);
            for (int j = 1; j <= i; j++) {
                ih = ih + 1;
                HQ(ih) = HQ(ih) + temp * XPT(knew, j);
            }
        }
        for (int jj = 1; jj <= nptm; jj++) {
            temp = diff * ZMAT(knew, jj);
            for (int k = 1; k <= npt; k++) PQ(k) = PQ(k) + temp * ZMAT(k, jj);
        }
        // Include the new interpolation point, and make the changes to GOPT at
        // the old XOPT that are caused by the updating of the quadratic model.
        FVAL(knew) = f;
        for (int i = 1; i <= n; i++) {
            XPT(knew, i) = XNEW(i);
            W(i) = BMAT(knew, i);
        }
        for (int k = 1; k <= npt; k++) {
            suma = zero;
            for (int jj = 1; jj <= nptm; jj++) suma = suma + ZMAT(knew, jj) * ZMAT(k, jj);
            sumb = zero;
            for (int j = 1; j <= n; j++) sumb = sumb + XPT(k, j) * XOPT(j);
            temp = suma * sumb;
            for (int i = 1; i <= n; i++) W(i) = W(i) + temp * XPT(k, i);
        }
        for (int i = 1; i <= n; i++) GOPT(i) = GOPT(i) + diff * W(i);
        // Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
        if (f < fopt) {
            kopt = knew;
            xoptsq = zero;
            ih = 0;
            for (int j = 1; j <= n; j++) {
                XOPT(j) = XNEW(j);
                xoptsq = xoptsq + sq(XOPT(j));
                for (int i = 1; i <= j; i++) {
                    ih = ih + 1;
                    if (i < j) GOPT(j) = GOPT(j) + HQ(ih) * D(i);
                    GOPT(i) = GOPT(i) + HQ(ih) * D(j);
                }
            }
            for (int k = 1; k <= npt; k++) {
                temp = zero;
                for (int j = 1; j <= n; j++) temp = temp + XPT(k, j) * D(j);
                temp = PQ(k) * temp;
                for (int i = 1; i <= n; i++) GOPT(i) = GOPT(i) + temp * XPT(k, i);
            }
        }
        // Calculate the parameters of the least Frobenius norm interpolant to
        // the current data, the gradient of this interpolant at XOPT being put
        // into VLAG(NPT+I), I=1,2,...,N.
        if (ntrits > 0) {
            for (int k = 1; k <= npt; k++) {
                VLAG(k) = FVAL(k) - FVAL(kopt);
                W(k) = zero;
            }
            for (int j = 1; j <= nptm; j++) {
                sum = zero;
                for (int k = 1; k <= npt; k++) sum = sum + ZMAT(k, j) * VLAG(k);
                for (int k = 1; k <= npt; k++) W(k) = W(k) + sum * ZMAT(k, j);
            }
            for (int k = 1; k <= npt; k++) {
                sum = zero;
                for (int j = 1; j <= n; j++) sum = sum + XPT(k, j) * XOPT(j);
                W(k + npt) = W(k);
                W(k) = sum * W(k);
            }
            gqsq = zero;
            gisq = zero;
            for (int i = 1; i <= n; i++) {
                sum = zero;
                for (int k = 1; k <= npt; k++) sum = sum + BMAT(k, i) * VLAG(k) + XPT(k, i) * W(k);
                if (XOPT(i) == SL(i)) {
                    gqsq = gqsq + sq(dmin1(zero, GOPT(i)));
                    gisq = gisq + sq(dmin1(zero, sum));
                } else if (XOPT(i) == SU(i)) {
                    gqsq = gqsq + sq(dmax1(zero, GOPT(i)));
                    gisq = gisq + sq(dmax1(zero, sum));
                } else {
                    gqsq = gqsq + sq(GOPT(i));
                    gisq = gisq + sum * sum;
                }
                VLAG(npt + i) = sum;
            }
            // Test whether to replace the new quadratic model by the least
            // Frobenius norm interpolant, making the replacement if the test
            // is satisfied.
            itest = itest + 1;
            if (gqsq < ten * gisq) itest = 0;
            if (itest >= 3) {
                for (int i = 1; i <= std::max(npt, nh); i++) {
                    if (i <= n) GOPT(i) = VLAG(npt + i);
                    if (i <= npt) PQ(i) = W(npt + i);
                    if (i <= nh) HQ(i) = zero;
                    itest = 0;
                }
            }
        }
        // If a trust region step has provided a sufficient decrease in F, then
        // branch for another trust region calculation. The case NTRITS=0
        // occurs when the new interpolation point was reached by an
        // alternative step.
        if (ntrits == 0) goto L60;
        if (f <= fopt + tenth * vquad) goto L60;
        // Alternatively, find out if the interpolation points are close enough
        // to the best point so far.
        distsq = dmax1(sq(two * delta), sq(ten * rho));
    L650:
        knew = 0;
        for (int k = 1; k <= npt; k++) {
            sum = zero;
            for (int j = 1; j <= n; j++) sum = sum + sq(XPT(k, j) - XOPT(j));
            if (sum > distsq) {
                knew = k;
                distsq = sum;
            }
        }
        // If KNEW is positive, then ALTMOV finds alternative new positions for
        // the KNEW-th interpolation point within distance ADELT of XOPT. It is
        // reached via label 90. Otherwise, there is a branch to label 60 for
        // another trust region iteration, unless the calculations with the
        // current RHO are complete.
        if (knew > 0) {
            dist = sqrt(distsq);
            if (ntrits == -1) {
                delta = dmin1(tenth * delta, half * dist);
                if (delta <= 1.5 * rho) delta = rho;
            }
            ntrits = 0;
            adelt = dmax1(dmin1(tenth * dist, delta), rho);
            dsq = adelt * adelt;
            goto L90;
        }
        if (ntrits == -1) goto L680;
        if (ratio > zero) goto L60;
        if (dmax1(delta, dnorm) > rho) goto L60;
        // The calculations with the current value of RHO are complete. Pick
        // the next values of RHO and DELTA.
    L680:
        if (rho > rhoend) {
            delta = half * rho;
            ratio = rho / rhoend;
            if (ratio <= 16.0) {
                rho = rhoend;
            } else if (ratio <= 250.0) {
                rho = sqrt(ratio) * rhoend;
            } else {
                rho = tenth * rho;
            }
            delta = dmax1(delta, rho);
            minqit(iprint, rho, nf, FVAL(kopt), n, xbase, xopt);
            ntrits = 0;
            nfsav = nf;
            goto L60;
        }
        // Return from the calculation, after another Newton-Raphson step, if
        // it is too short to have been tried before.
        if (ntrits == -1) goto L360;
    L720:
        if (FVAL(kopt) <= fsave) {
            for (int i = 1; i <= n; i++) {
                X(i) = dmin1(dmax1(XL(i), XBASE(i) + XOPT(i)), XU(i));
                if (XOPT(i) == SL(i)) X(i) = XL(i);
                if (XOPT(i) == SU(i)) X(i) = XU(i);
            }
            f = FVAL(kopt);
        }
        minqir(iprint, f, nf, n, x);
    }

#undef X
#undef XL
#undef XU
#undef XBASE
#undef XPT
#undef FVAL
#undef XOPT
#undef GOPT
#undef HQ
#undef PQ
#undef BMAT
#undef ZMAT
#undef SL
#undef SU
#undef XNEW
#undef XALT
#undef D
#undef VLAG
#undef W
#undef GNEW
#undef XBDI
#undef S
#undef HS
#undef HRED
#undef GLAG
#undef HCOL
#undef PTSAUX
#undef PTSID

} // anonymous namespace

    // ======================================================================
    // SUBROUTINE BOBYQA (driver), plus the workspace allocation of
    // minqa.cpp's bobyqa_cpp().
    // ======================================================================
    int bobyqa(int n, int npt, double* x, const double* xl, const double* xu,
               double rhobeg, double rhoend, int iprint, int maxfun,
               const std::function<double(const double*, int)>& fn,
               int& nfeval) {
        nfeval = 0;
        Calfun calfun{fn, nfeval};
        int ierr = 0;
        const double zero = 0.0;
        int np = n + 1;
        // Return if the value of NPT is unacceptable.
        if (npt < n + 2 || npt > ((n + 2) * np) / 2) {
            ierr = 10;
            return ierr;
        }
        // Working space, of the length allocated in minqa.cpp
        std::vector<double> wv((std::size_t) (npt + 5) * (npt + n) + (3 * n * (n + 5)) / 2);
        double* w = wv.data();
        // Partition the working space array (1-based offsets as in bobyqa.f).
        int ndim = npt + n;
        int ixb = 1;
        int ixp = ixb + n;
        int ifv = ixp + n * npt;
        int ixo = ifv + npt;
        int igo = ixo + n;
        int ihq = igo + n;
        int ipq = ihq + (n * np) / 2;
        int ibmat = ipq + npt;
        int izmat = ibmat + ndim * n;
        int isl = izmat + npt * (npt - np);
        int isu = isl + n;
        int ixn = isu + n;
        int ixa = ixn + n;
        int id = ixa + n;
        int ivl = id + n;
        int iw = ivl + ndim;
        ierr = 0;
        // Return if there is insufficient space between the bounds. Modify the
        // initial X if necessary in order to avoid conflicts between the bounds
        // and the construction of the first quadratic model.
        for (int j = 1; j <= n; j++) {
            double temp = xu[j - 1] - xl[j - 1];
            if (temp < rhobeg + rhobeg) {
                ierr = 20;
                return ierr;
            }
            int jsl = isl + j - 1;
            int jsu = jsl + n;
            double& wsl = w[jsl - 1];
            double& wsu = w[jsu - 1];
            wsl = xl[j - 1] - x[j - 1];
            wsu = xu[j - 1] - x[j - 1];
            if (wsl >= -rhobeg) {
                if (wsl >= zero) {
                    x[j - 1] = xl[j - 1];
                    wsl = zero;
                    wsu = temp;
                } else {
                    x[j - 1] = xl[j - 1] + rhobeg;
                    wsl = -rhobeg;
                    wsu = dmax1(xu[j - 1] - x[j - 1], rhobeg);
                }
            } else if (wsu <= rhobeg) {
                if (wsu <= zero) {
                    x[j - 1] = xu[j - 1];
                    wsl = -temp;
                    wsu = zero;
                } else {
                    x[j - 1] = xu[j - 1] - rhobeg;
                    wsl = dmin1(xl[j - 1] - x[j - 1], -rhobeg);
                    wsu = rhobeg;
                }
            }
        }
        // Make the call of BOBYQB.
        bobyqb(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun,
               w + (ixb - 1), w + (ixp - 1), w + (ifv - 1), w + (ixo - 1),
               w + (igo - 1), w + (ihq - 1), w + (ipq - 1), w + (ibmat - 1),
               w + (izmat - 1), ndim, w + (isl - 1), w + (isu - 1),
               w + (ixn - 1), w + (ixa - 1), w + (id - 1), w + (ivl - 1),
               w + (iw - 1), ierr, calfun);
        return ierr;
    }

} // namespace minqa
