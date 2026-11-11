// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// predModule.h: predictor module using Eigen
//
// Transcription of lme4/src/predModule.h for the Python port.
// Copyright (C)       2011 Douglas Bates, Martin Maechler and Ben Bolker
// GPL (>= 2).
//
// Differences from the R version:
//  * the members own their storage (Eigen objects) instead of mapping
//    memory owned by R objects; accessors return copies to Python;
//  * the sparse Cholesky factor uses Eigen::SimplicialLLT with an AMD
//    fill-reducing ordering instead of CHOLMOD.  The factor of
//    P (Lambda' Z' W Z Lambda + I) P' is mathematically identical; the
//    fill-reducing permutation P may differ from CHOLMOD's.

#ifndef LME4_PREDMODULE_H
#define LME4_PREDMODULE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <vector>
#include <map>
#include <string>

namespace lme4 {

    using Eigen::ArrayXd;
    using Eigen::LLT;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Eigen::VectorXi;

    // Sparse Cholesky decomposition wrapper providing the subset of
    // CHOLMOD functionality used by lme4 (solveInPlace with the CHOLMOD_*
    // system codes, log-determinant, factor/permutation extraction).
    class lme4CholDecomp {
    public:
        typedef Eigen::SparseMatrix<double>                         SpMat;
        typedef Eigen::SimplicialLLT<SpMat, Eigen::Lower,
                                     Eigen::AMDOrdering<int> >      CholBase;
        // expose the stored factor (column-major, diagonal first in each column)
        class Chol : public CholBase {
        public:
            const SpMat& rawL() const {return this->m_matrix;}
        };
        enum System { A = 0, L = 4, Lt = 5, P = 7, Pt = 8 };  // CHOLMOD_* codes
        Chol         d_chol;

        void analyzePattern(const SpMat& A) {d_chol.analyzePattern(A);}
        void factorize(const SpMat& A) {d_chol.factorize(A);}
        Eigen::ComputationInfo info() const {return d_chol.info();}
        double logDetL2() const {             // log(det(L L')) = 2 sum(log(diag(L)))
            const SpMat& Lm = d_chol.rawL();
            double ans = 0.;
            for (int j = 0; j < Lm.outerSize(); ++j) {
                SpMat::InnerIterator it(Lm, j); // first entry in column is the diagonal
                ans += std::log(it.value());
            }
            return 2. * ans;
        }
        template<typename Derived>
        void solveInPlace(Eigen::MatrixBase<Derived>& b, int sys) const {
            switch (sys) {
            case A:
                b = d_chol.solve(MatrixXd(b));
                break;
            case P:
                b = d_chol.permutationP() * MatrixXd(b);
                break;
            case Pt:
                b = d_chol.permutationPinv() * MatrixXd(b);
                break;
            case L:
                d_chol.matrixL().solveInPlace(b);
                break;
            case Lt:
                d_chol.matrixU().solveInPlace(b);
                break;
            default:
                throw std::invalid_argument("unknown system in solveInPlace");
            }
        }
        SpMat matrixL() const {return d_chol.rawL();}
        VectorXi perm() const {return d_chol.permutationP().indices();}
        long nonZerosL() const {return d_chol.rawL().nonZeros();}
        VectorXd diagL() const {
            const SpMat& Lm = d_chol.rawL();
            VectorXd ans(Lm.cols());
            for (int j = 0; j < Lm.outerSize(); ++j) {
                SpMat::InnerIterator it(Lm, j);
                ans[j] = it.value();
            }
            return ans;
        }
    };

    class merPredD {
    public:
        typedef MatrixXd::Scalar                          Scalar;
        typedef Eigen::Index                              Index;
        typedef Eigen::SparseMatrix<double>               SpMatrixd;
        typedef lme4CholDecomp                            ChmDecomp;
        typedef SpMatrixd                                 MSpMatrixd;
    protected:
        MatrixXd      d_X, d_RZX, d_V, d_VtV;
        SpMatrixd     d_Zt, d_Ut, d_LamtUt, d_Lambdat;
        VectorXd      d_theta, d_Vtr, d_Utr, d_Xwts, d_beta0, d_delb, d_delu, d_u0;
        VectorXi      d_Lind;
        Index         d_N, d_p, d_q;
        Scalar        d_CcNumer, d_ldL2, d_ldRX2;
        ChmDecomp     d_L;
        LLT<MatrixXd> d_RX;
        SpMatrixd     d_A;      // LamtUt * LamtUt' + I (pattern fixed at construction)
        SpMatrixd     d_I;      // q x q identity
        void          initialize(const VectorXd& Xwts);
    public:
        // Corresponds to merPredD$new(X, Zt, Lambdat, Lind, theta, n, ...)
        // followed by initializePtr() in lme4/R/AllClass.R
        merPredD(const MatrixXd& X, const SpMatrixd& Zt, const SpMatrixd& Lambdat,
                 const VectorXi& Lind, const VectorXd& theta, Index n,
                 const VectorXd& Xwts, const VectorXd& beta0, const VectorXd& u0);
        merPredD(const merPredD& other);   // deep copy, like merPredD$copy()

        VectorXi             Pvec() const;

        MatrixXd               RX() const;
        MatrixXd              RXi() const;
        MatrixXd             unsc() const;

        VectorXd           RXdiag() const;
        VectorXd                b(const Scalar& f) const;
        VectorXd             beta(const Scalar& f) const;
        VectorXd          linPred(const Scalar& f) const;
        VectorXd                u(const Scalar& f) const;

        // condVar: list over grouping factors of arrays (ncti x ncti x nli)
        std::vector<MatrixXd> condVar(const std::vector<int>& nlevs,
                                      const std::vector<int>& nctot,
                                      const std::vector<int>& offsets,
                                      const std::vector<std::vector<int> >& terms) const;

        Scalar            CcNumer() const {return d_CcNumer;}
        Scalar               ldL2() const {return d_ldL2;}
        Scalar              ldRX2() const {return d_ldRX2;}
        Scalar              solve();
        Scalar             solveU();
        Scalar               sqrL(const Scalar& f) const;

        const ChmDecomp&        L() const {return d_L;}

        const MatrixXd&             X() const {return d_X;}
        const MatrixXd&             V() const {return d_V;}
        const MatrixXd&           VtV() const {return d_VtV;}
        const MatrixXd&           RZX() const {return d_RZX;}

        const SpMatrixd& Lambdat() const {return d_Lambdat;}
        const SpMatrixd&  LamtUt() const {return d_LamtUt;}
        const SpMatrixd&      Ut() const {return d_Ut;}
        const SpMatrixd&      Zt() const {return d_Zt;}

        const VectorXd&           Utr() const {return d_Utr;}
        const VectorXd&           Vtr() const {return d_Vtr;}
        const VectorXd&          delb() const {return d_delb;}
        const VectorXd&          delu() const {return d_delu;}
        const VectorXd&         beta0() const {return d_beta0;}
        const VectorXd&         theta() const {return d_theta;}
        const VectorXd&            u0() const {return d_u0;}
        const VectorXd&          Xwts() const {return d_Xwts;}
        const VectorXi&          Lind() const {return d_Lind;}
        Index                   N() const {return d_N;}
        Index                   p() const {return d_p;}
        Index                   q() const {return d_q;}

        int                  info() const {return (int) d_L.info();}

        void          installPars(const Scalar& f);
        void          MCMC_beta_u(const Scalar& sigma, const VectorXd& z1, const VectorXd& z2);
        void             setBeta0(const VectorXd&);
        void              setDelb(const VectorXd&);
        void              setDelu(const VectorXd&);
        void             setTheta(const VectorXd&);
        void                setZt(const VectorXd&);
        void                setU0(const VectorXd&);
        void         updateDecomp();
        void         updateDecomp(const MatrixXd*);
        void              updateL();
        void         updateLamtUt();
        void            updateRes(const VectorXd&);
        void           updateXwts(const  ArrayXd&);

        // solve(L, B, system=...) as used by the R code (Matrix::solve on a CHMfactor)
        MatrixXd      solveSystem(const MatrixXd& B, int sys) const;
    };
}

#endif // LME4_PREDMODULE_H
