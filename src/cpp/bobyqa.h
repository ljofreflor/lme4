// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// bobyqa.h: C++ transcription of M.J.D. Powell's BOBYQA, as shipped in the
// R package minqa 1.2.8 (Fortran sources bobyqa.f, bobyqb.f, altmov.f,
// prelim.f, rescue.f, trsbox.f, updatebobyqa.f and the C++ glue minqa.cpp).
//
// Original Fortran: M.J.D. Powell (2009), "The BOBYQA algorithm for bound
//   constrained optimization without derivatives", DAMTP 2009/NA06.
// minqa modifications: Douglas Bates, Katharine M. Mullen, John C. Nash,
//   Ravi Varadhan (and contributors), minqa R package.
// C++ transcription for the Python port of lme4.
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.  This program is distributed in the hope that it will
// be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
// Public License for more details.

#ifndef LME4_MINQA_BOBYQA_H
#define LME4_MINQA_BOBYQA_H

#include <functional>

namespace minqa {

    /**
     * Minimize calfun(x, n) subject to xl <= x <= xu (Powell's BOBYQA).
     *
     * @param n       number of variables
     * @param npt     number of interpolation conditions, in [n+2, (n+1)(n+2)/2]
     * @param x       on entry the starting values, on exit the best point found
     * @param xl,xu   lower and upper bounds (length n)
     * @param rhobeg  initial trust region radius
     * @param rhoend  final trust region radius
     * @param iprint  print level (0: none, 1: at return, 2: each new rho,
     *                3: every evaluation, >3: every iprint-th evaluation)
     * @param maxfun  maximum number of calls of calfun
     * @param calfun  objective; called with a pointer to n doubles and n
     * @param nfeval  on exit, the number of calls of calfun (minqa's ".feval.")
     *
     * @return ierr as set by the minqa Fortran: 0 (normal), 10 (NPT out of
     *         range), 20 (some XU-XL < 2*RHOBEG), 320 (too much cancellation
     *         in a denominator), 390 (MAXFUN reached), 430 (trust region step
     *         failed to reduce Q).
     *
     * As in minqa's calfun(), a std::range_error is thrown if a non-finite x
     * would be passed to the objective, and non-finite objective values are
     * replaced by DBL_MAX.
     */
    int bobyqa(int n, int npt, double* x, const double* xl, const double* xu,
               double rhobeg, double rhoend, int iprint, int maxfun,
               const std::function<double(const double*, int)>& calfun,
               int& nfeval);
}

#endif // LME4_MINQA_BOBYQA_H
