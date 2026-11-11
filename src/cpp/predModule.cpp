//
// predModule.cpp: implementation of predictor module using Eigen
//
// Transcription of lme4/src/predModule.cpp for the Python port.
// Copyright (C) 2011-2013 Douglas Bates, Martin Maechler, Ben Bolker and Steve Walker
// GPL (>= 2).

#include "predModule.h"
#include <stdexcept>
#include <algorithm>

namespace lme4 {
    using     std::invalid_argument;
    using     std::runtime_error;

    using   Eigen::ArrayXd;

    typedef merPredD::SpMatrixd SpMatrixd;

    // symbolic product: pattern of A %*% B with all structural entries kept
    static SpMatrixd patternProduct(const SpMatrixd& A, const SpMatrixd& B) {
        SpMatrixd Ao(A), Bo(B);
        for (int k = 0; k < Ao.nonZeros(); ++k) Ao.valuePtr()[k] = 1.;
        for (int k = 0; k < Bo.nonZeros(); ++k) Bo.valuePtr()[k] = 1.;
        SpMatrixd ans = Ao * Bo;    // all values positive => no cancellation
        ans.makeCompressed();
        for (int k = 0; k < ans.nonZeros(); ++k) ans.valuePtr()[k] = 0.;
        return ans;
    }

    merPredD::merPredD(const MatrixXd& X, const SpMatrixd& Zt, const SpMatrixd& Lambdat,
                       const VectorXi& Lind, const VectorXd& theta, Index n,
                       const VectorXd& Xwts, const VectorXd& beta0, const VectorXd& u0)
        : d_X(X), d_Zt(Zt), d_Lambdat(Lambdat), d_theta(theta), d_Lind(Lind),
          d_N(X.rows()), d_p(X.cols()), d_q(Zt.rows()), d_CcNumer(0.), d_ldL2(0.), d_ldRX2(0.),
          d_RX(X.cols())
    {
        // ---- R-level initialize() of the merPredD reference class
        d_Zt.makeCompressed();
        d_Lambdat.makeCompressed();
        if (d_theta.size() == 0 || d_Lind.size() == 0)
            throw invalid_argument("theta and Lind must have positive length");
        {   // all(sort(unique(Lind)) == seq_along(theta))
            std::vector<int> ul(d_Lind.data(), d_Lind.data() + d_Lind.size());
            std::sort(ul.begin(), ul.end());
            ul.erase(std::unique(ul.begin(), ul.end()), ul.end());
            bool ok = (Index) ul.size() == d_theta.size();
            for (size_t i = 0; ok && i < ul.size(); ++i) ok = (ul[i] == (int) i + 1);
            if (!ok) throw invalid_argument("all(sort(unique(Lind)) == seq_along(theta)) is not TRUE");
        }
        d_RZX   = MatrixXd::Zero(d_q, d_p);
        d_Utr   = VectorXd::Zero(d_q);
        d_V     = MatrixXd::Zero(n, d_p);
        d_VtV   = MatrixXd::Zero(d_p, d_p);
        d_Vtr   = VectorXd::Zero(d_p);
        d_beta0 = beta0.size() == d_p ? beta0 : VectorXd::Zero(d_p);
        d_delb  = VectorXd::Zero(d_p);
        d_delu  = VectorXd::Zero(d_q);
        d_u0    = u0.size() == d_q ? u0 : VectorXd::Zero(d_q);
        if (n == d_N) d_Ut = d_Zt;
        else {      // Zt %*% sparseMatrix(i=seq_len(N), j=as.integer(gl(n, 1, N)), x=1)
            SpMatrixd J(d_N, n);
            std::vector<Eigen::Triplet<double> > tr;
            for (Index i = 0; i < d_N; ++i) tr.push_back(Eigen::Triplet<double>(i, i % n, 1.));
            J.setFromTriplets(tr.begin(), tr.end());
            d_Ut = d_Zt * J;
            d_Ut.makeCompressed();
        }
        d_LamtUt = patternProduct(d_Lambdat, d_Ut);
        d_Xwts = Xwts.size() == d_N ? Xwts : VectorXd::Ones(d_N);

        // ---- C++ constructor (merPredD::merPredD in lme4)
        if (d_N != d_Zt.cols())
            throw invalid_argument("Z dimension mismatch");
        if (d_Lind.size() != d_Lambdat.nonZeros())
            throw invalid_argument("size of Lind does not match nonzeros in Lambda");
        for (Index i = 0; i < d_Lind.size(); ++i)
            if (d_Lind[i] < 1 || d_Lind[i] > d_theta.size())
                throw invalid_argument("Lind out of range");
        if (d_V.cols() > 0) {
            d_VtV.setZero();
            d_VtV.selfadjointView<Eigen::Upper>().rankUpdate(d_V.adjoint());
        }
        d_RX.compute(d_VtV);    // ensure d_RX is initialized even in the 0-column X case

        setTheta(d_theta);          // starting values into Lambda
        updateLamtUt();
        d_I.resize(d_q, d_q);
        d_I.setIdentity();
        {   // pattern of LamtUt %*% t(LamtUt) + I
            SpMatrixd Lo = patternProduct(d_LamtUt, SpMatrixd(d_LamtUt.transpose()));
            SpMatrixd Io(d_I);
            d_A = Lo + Io;
            d_A.makeCompressed();
        }
        d_L.analyzePattern(d_A); // perform symbolic analysis
        if (d_L.info() != Eigen::Success)
            throw runtime_error("CholeskyDecomposition.analyzePattern failed");

        // ---- initializePtr() of the merPredD reference class
        setTheta(d_theta);
        updateXwts(d_Xwts);
        updateDecomp();
    }

    merPredD::merPredD(const merPredD& o)
        : d_X(o.d_X), d_RZX(o.d_RZX), d_V(o.d_V), d_VtV(o.d_VtV),
          d_Zt(o.d_Zt), d_Ut(o.d_Ut), d_LamtUt(o.d_LamtUt), d_Lambdat(o.d_Lambdat),
          d_theta(o.d_theta), d_Vtr(o.d_Vtr), d_Utr(o.d_Utr), d_Xwts(o.d_Xwts),
          d_beta0(o.d_beta0), d_delb(o.d_delb), d_delu(o.d_delu), d_u0(o.d_u0),
          d_Lind(o.d_Lind), d_N(o.d_N), d_p(o.d_p), d_q(o.d_q),
          d_CcNumer(o.d_CcNumer), d_ldL2(o.d_ldL2), d_ldRX2(o.d_ldRX2),
          d_RX(o.d_p), d_A(o.d_A), d_I(o.d_I)
    {
        d_RX.compute(d_VtV);
        d_L.analyzePattern(d_A);
        // re-establish the decomposition at the current state
        updateL();
        if (d_p > 0) updateDecomp();
    }

    void merPredD::updateLamtUt() {
        // This complicated code bypasses problems caused by Eigen's
        // sparse/sparse matrix multiplication pruning zeros.  The
        // Cholesky decomposition croaks if the structure of d_LamtUt changes.
        std::fill(d_LamtUt.valuePtr(), d_LamtUt.valuePtr() + d_LamtUt.nonZeros(), 0.);
        for (Index j = 0; j < d_Ut.outerSize(); ++j) {
            for(MSpMatrixd::InnerIterator rhsIt(d_Ut, j); rhsIt; ++rhsIt) {
                Scalar                        y(rhsIt.value());
                Index                         k(rhsIt.index());
                MSpMatrixd::InnerIterator prdIt(d_LamtUt, j);
                for (MSpMatrixd::InnerIterator lhsIt(d_Lambdat, k); lhsIt; ++lhsIt) {
                    Index                     i = lhsIt.index();
                    while (prdIt && prdIt.index() != i) ++prdIt;
                    if (!prdIt) throw runtime_error("logic error in updateLamtUt");
                    prdIt.valueRef()           += lhsIt.value() * y;
                }
            }
        }
    }

    VectorXd merPredD::b(const double& f) const {return d_Lambdat.adjoint() * u(f);}

    VectorXd merPredD::beta(const double& f) const {return d_beta0 + f * d_delb;}

    VectorXd merPredD::linPred(const double& f) const {
        return d_X * beta(f) + d_Zt.adjoint() * b(f);
    }

    std::vector<MatrixXd> merPredD::condVar(const std::vector<int>& nl,
                                            const std::vector<int>& nct,
                                            const std::vector<int>& off,
                                            const std::vector<std::vector<int> >& trmlst) const {
        // nl : nlevs : number of levels for each unique factor
        // nct : nctot : total number of components per factor
        // off : offsets : points to where each term starts (0-based)
        // trmlst : terms : 0-based term indices for each factor
        const int nf(nl.size());
        std::vector<MatrixXd> ans(nf);
        for (int i = 0; i < nf; i++) {
            int         ncti(nct[i]), nli(nl[i]);
            MatrixXd ansi(ncti * ncti, nli);
            const std::vector<int>& trms(trmlst[i]);
            if (trms.size() == 1) { // simple case
                int offset = off[trms[0]];
                for (int j = 0; j < nli; ++j) {
                    MatrixXd LvT(SpMatrixd(d_Lambdat.middleCols(offset + j * ncti, ncti)).toDense());
                    MatrixXd Lv(LvT.adjoint());
                    MatrixXd tmp(LvT);
                    d_L.solveInPlace(tmp, ChmDecomp::A);
                    MatrixXd rr(Lv * tmp);
                    std::copy(rr.data(), rr.data() + rr.size(), ansi.col(j).data());
                }
            } else {
                throw std::runtime_error("multiple terms per factor not yet written");
            }
            ans[i] = ansi;
        }
        return ans;
    }

    VectorXd merPredD::u(const double& f) const {return d_u0 + f * d_delu;}

    merPredD::Scalar merPredD::sqrL(const double& f) const {return u(f).squaredNorm();}

    void merPredD::updateL() {
        updateLamtUt();
        SpMatrixd A = d_LamtUt * SpMatrixd(d_LamtUt.transpose());
        A += d_I;
        A.makeCompressed();
        d_L.factorize(A);
        if (d_L.info() != Eigen::Success)
            throw runtime_error("Cholesky factorization failed");
        d_ldL2 = d_L.logDetL2();
    }

    void merPredD::setTheta(const VectorXd& theta) {
        if (theta.size() != d_theta.size())
            throw invalid_argument("theta size mismatch");
        // update theta
        std::copy(theta.data(), theta.data() + theta.size(), d_theta.data());
        // update Lambdat
        const int *lipt = d_Lind.data();
        double *LamX = d_Lambdat.valuePtr(), *thpt = d_theta.data();
        for (int i = 0; i < d_Lind.size(); ++i) {
            LamX[i] = thpt[lipt[i] - 1];
        }
    }

    void merPredD::setZt(const VectorXd& ZtNonZero) {
        if (ZtNonZero.size() != d_Zt.nonZeros())
            throw invalid_argument("setZt: dimension mismatch");
        double *ZtX = d_Zt.valuePtr(); // where the nonzero values of Zt live
        std::copy(ZtNonZero.data(), ZtNonZero.data() + ZtNonZero.size(), ZtX);
    }

    merPredD::Scalar merPredD::solve() {
        d_delu          = d_Utr - d_u0;
        d_L.solveInPlace(d_delu, ChmDecomp::P);
        d_L.solveInPlace(d_delu, ChmDecomp::L);    // d_delu now contains cu
        d_CcNumer       = d_delu.squaredNorm(); // numerator of convergence criterion

        d_delb          = d_RX.matrixL().solve(d_Vtr - d_RZX.adjoint() * d_delu);
        d_CcNumer      += d_delb.squaredNorm(); // increment CcNumer
        d_RX.matrixU().solveInPlace(d_delb);

        d_delu         -= d_RZX * d_delb;
        d_L.solveInPlace(d_delu, ChmDecomp::Lt);
        d_L.solveInPlace(d_delu, ChmDecomp::Pt);
        return d_CcNumer;
    }

    merPredD::Scalar merPredD::solveU() {
        d_delb.setZero(); // in calculation of linPred delb should be zero after solveU
        d_delu          = d_Utr - d_u0;
        d_L.solveInPlace(d_delu, ChmDecomp::P);
        d_L.solveInPlace(d_delu, ChmDecomp::L);    // d_delu now contains cu
        d_CcNumer       = d_delu.squaredNorm(); // numerator of convergence criterion
        d_L.solveInPlace(d_delu, ChmDecomp::Lt);
        d_L.solveInPlace(d_delu, ChmDecomp::Pt);
        return d_CcNumer;
    }

    void merPredD::updateXwts(const ArrayXd& sqrtXwt) {
        if (d_Xwts.size() != sqrtXwt.size())
            throw invalid_argument("updateXwts: dimension mismatch");
        std::copy(sqrtXwt.data(), sqrtXwt.data() + sqrtXwt.size(), d_Xwts.data());
        if (sqrtXwt.size() == d_V.rows()) { // W is diagonal
            d_V              = d_Xwts.asDiagonal() * d_X;
            for (int j = 0; j < d_N; ++j)
                for (MSpMatrixd::InnerIterator Utj(d_Ut, j), Ztj(d_Zt, j);
                     Utj && Ztj; ++Utj, ++Ztj)
                    Utj.valueRef() = Ztj.value() * d_Xwts.data()[j];
        } else {
            SpMatrixd      W(d_V.rows(), sqrtXwt.size());
            const double *pt = sqrtXwt.data();
            W.reserve(sqrtXwt.size());
            for (Index j = 0; j < W.cols(); ++j, ++pt) {
                W.startVec(j);
                W.insertBack(j % d_V.rows(), j) = *pt;
            }
            W.finalize();
            d_V              = W * d_X;
            SpMatrixd      Ut(d_Zt * W.adjoint());
            if (Ut.cols() != d_Ut.cols())
                throw std::runtime_error("Size mismatch in updateXwts");

            // More complex code to handle the pruning of zeros
            std::fill(d_Ut.valuePtr(), d_Ut.valuePtr() + d_Ut.nonZeros(), 0.);
            for (int j = 0; j < d_Ut.outerSize(); ++j) {
                MSpMatrixd::InnerIterator lhsIt(d_Ut, j);
                for (SpMatrixd::InnerIterator  rhsIt(Ut, j); rhsIt; ++rhsIt, ++lhsIt) {
                    Index                         k(rhsIt.index());
                    while (lhsIt && lhsIt.index() != k) ++lhsIt;
                    if (!lhsIt || lhsIt.index() != k)
                        throw std::runtime_error("Pattern mismatch in updateXwts");
                    lhsIt.valueRef() = rhsIt.value();
                }
            }
        }
        // avoid crossprod of n-by-0 matrix: GH #794
        if (d_V.cols() > 0) d_VtV.setZero().selfadjointView<Eigen::Upper>().rankUpdate(d_V.adjoint());
        updateL();
    }

    void merPredD::updateDecomp() {
        updateDecomp(NULL);
    }

    void merPredD::updateDecomp(const MatrixXd* xPenalty) {  // update L, RZX and RX
        updateL();
        if (d_LamtUt.cols() != d_V.rows())
            throw std::runtime_error("dimension mismatch in updateDecomp()");
        d_RZX         = d_LamtUt * d_V;
        if (d_p > 0) {
            d_L.solveInPlace(d_RZX, ChmDecomp::P);
            d_L.solveInPlace(d_RZX, ChmDecomp::L);
            MatrixXd      VtVdown(d_VtV);
            VtVdown.selfadjointView<Eigen::Upper>().rankUpdate(d_RZX.adjoint(), -1);
            if (xPenalty != NULL)
                VtVdown.selfadjointView<Eigen::Upper>().rankUpdate(*xPenalty, 1);
            MatrixXd full(VtVdown.selfadjointView<Eigen::Upper>());
            d_RX.compute(full);
            if (d_RX.info() != Eigen::Success)
                throw std::runtime_error("Downdated VtV is not positive definite");
            d_ldRX2         = 2. * d_RX.matrixLLT().diagonal().array().abs().log().sum();
        }
    }

    void merPredD::updateRes(const VectorXd& wtres) {
        if (d_V.rows() != wtres.size())
            throw invalid_argument("updateRes: dimension mismatch");
        d_Vtr           = d_V.adjoint() * wtres;
        d_Utr           = d_LamtUt * wtres;
    }

    void merPredD::installPars(const Scalar& f) {
        d_u0 = u(f);
        d_beta0 = beta(f);
        d_delb.setZero();
        d_delu.setZero();
    }

    void merPredD::setBeta0(const VectorXd& nBeta) {
        if (nBeta.size() != d_p) throw invalid_argument("setBeta0: dimension mismatch");
        std::copy(nBeta.data(), nBeta.data() + d_p, d_beta0.data());
    }

    void merPredD::setDelb(const VectorXd& newDelb) {
        if (newDelb.size() != d_p)
            throw invalid_argument("setDelb: dimension mismatch");
        std::copy(newDelb.data(), newDelb.data() + d_p, d_delb.data());
    }

    void merPredD::setDelu(const VectorXd& newDelu) {
        if (newDelu.size() != d_q)
            throw invalid_argument("setDelu: dimension mismatch");
        std::copy(newDelu.data(), newDelu.data() + d_q, d_delu.data());
    }

    void merPredD::setU0(const VectorXd& newU0) {
        if (newU0.size() != d_q) throw invalid_argument("setU0: dimension mismatch");
        std::copy(newU0.data(), newU0.data() + d_q, d_u0.data());
    }

    // z1 (length p) and z2 (length q) are standard normal deviates supplied by
    // the caller (R uses norm_rand())
    void merPredD::MCMC_beta_u(const Scalar& sigma, const VectorXd& z1, const VectorXd& z2) {
        VectorXd del2(d_RX.matrixU().solve(z1 * sigma));
        d_delb += del2;
        VectorXd del1(z2 * sigma - d_RZX * del2);
        d_L.solveInPlace(del1, ChmDecomp::Lt);
        d_delu += del1;
    }

    VectorXi merPredD::Pvec() const {
        // CHOLMOD convention: L L' = A[Perm, Perm] (0-based)
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> Pinv(d_L.d_chol.permutationPinv());
        return Pinv.indices();
    }

    MatrixXd merPredD::RX() const {
        return d_RX.matrixU();
    }

    MatrixXd merPredD::RXi() const { // inverse RX
        return d_RX.matrixU().solve(MatrixXd::Identity(d_p,d_p));
    }

    MatrixXd merPredD::unsc() const { // unscaled var-cov mat of FE
        // R translation: tcrossprod(RXi)
        return MatrixXd(MatrixXd(d_p, d_p).setZero().
                        selfadjointView<Eigen::Lower>().
                        rankUpdate(RXi()));
    }

    VectorXd merPredD::RXdiag() const {
        return d_RX.matrixLLT().diagonal();
    }

    MatrixXd merPredD::solveSystem(const MatrixXd& B, int sys) const {
        MatrixXd ans(B);
        d_L.solveInPlace(ans, sys);
        return ans;
    }
}
