// bindings.cpp: pybind11 glue exposing the lme4 C++ modules to Python.
//
// This plays the role of the R_CallMethodDef table / Rcpp external
// pointers in lme4/src/external.cpp.  GPL (>= 2).

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "external.h"
#include "bobyqa.h"

#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace lme4;
typedef Eigen::SparseMatrix<double> SpMat;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;

static int chmSystem(const std::string& s) {
    if (s == "A")  return lme4CholDecomp::A;
    if (s == "L")  return lme4CholDecomp::L;
    if (s == "Lt") return lme4CholDecomp::Lt;
    if (s == "P")  return lme4CholDecomp::P;
    if (s == "Pt") return lme4CholDecomp::Pt;
    throw std::invalid_argument("unknown system '" + s + "'");
}

static int nmStatus(optimizer::nm_status s) {
    switch (s) {
    case optimizer::nm_evals:         return -4;
    case optimizer::nm_forced:        return -3;
    case optimizer::nm_nofeasible:    return -2;
    case optimizer::nm_x0notfeasible: return -1;
    case optimizer::nm_active:        return 0;
    case optimizer::nm_minf_max:      return 1;
    case optimizer::nm_fcvg:          return 2;
    case optimizer::nm_xcvg:          return 3;
    }
    return 0;
}

PYBIND11_MODULE(_lme4cpp, m) {
    m.doc() = "C++ (Eigen) backend of the Python port of lme4";

    optimizer::printer = [](const std::string& s) {
        py::gil_scoped_acquire gil;
        py::print(s, py::arg("end") = "", py::arg("flush") = true);
    };

    m.def("eigen_version", []() {
        return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) +
            "." + std::to_string(EIGEN_MINOR_VERSION);
    });
    m.def("Eigen_SSE", []() { return std::string(Eigen::SimdInstructionSetsInUse()); });

    // ------------------------------------------------------------------ merPredD
    py::class_<merPredD>(m, "merPredD")
        .def(py::init<const MatrixXd&, const SpMat&, const SpMat&, const VectorXi&,
                      const VectorXd&, Eigen::Index, const VectorXd&, const VectorXd&,
                      const VectorXd&>(),
             py::arg("X"), py::arg("Zt"), py::arg("Lambdat"), py::arg("Lind"),
             py::arg("theta"), py::arg("n"), py::arg("Xwts") = VectorXd(),
             py::arg("beta0") = VectorXd(), py::arg("u0") = VectorXd())
        .def("copy", [](const merPredD& self) { return merPredD(self); })
        .def("Pvec", &merPredD::Pvec)
        .def("RX", &merPredD::RX)
        .def("RXi", &merPredD::RXi)
        .def("unsc", &merPredD::unsc)
        .def("RXdiag", &merPredD::RXdiag)
        .def("b", &merPredD::b)
        .def("beta", &merPredD::beta)
        .def("linPred", &merPredD::linPred)
        .def("u", &merPredD::u)
        .def("condVar", &merPredD::condVar)
        .def("CcNumer", &merPredD::CcNumer)
        .def("ldL2", &merPredD::ldL2)
        .def("ldRX2", &merPredD::ldRX2)
        .def("solve", &merPredD::solve)
        .def("solveU", &merPredD::solveU)
        .def("sqrL", &merPredD::sqrL)
        .def("installPars", &merPredD::installPars)
        .def("MCMC_beta_u", &merPredD::MCMC_beta_u)
        .def("setBeta0", &merPredD::setBeta0)
        .def("setDelb", &merPredD::setDelb)
        .def("setDelu", &merPredD::setDelu)
        .def("setTheta", &merPredD::setTheta)
        .def("setZt", &merPredD::setZt)
        .def("setU0", &merPredD::setU0)
        .def("updateDecomp", [](merPredD& self, py::object xPenalty) {
                 if (xPenalty.is_none()) self.updateDecomp(NULL);
                 else {
                     MatrixXd xp = xPenalty.cast<MatrixXd>();
                     self.updateDecomp(&xp);
                 }
             }, py::arg("XPenalty") = py::none())
        .def("updateL", &merPredD::updateL)
        .def("updateLamtUt", &merPredD::updateLamtUt)
        .def("updateRes", &merPredD::updateRes)
        .def("updateXwts", [](merPredD& self, const VectorXd& w) { self.updateXwts(w.array()); })
        .def("solveSystem", [](const merPredD& self, const MatrixXd& B, const std::string& sys) {
                 return self.solveSystem(B, chmSystem(sys));
             })
        .def("L", [](const merPredD& self) { return self.L().matrixL(); },
             "the lower-triangular sparse Cholesky factor (of the permuted system)")
        .def("Ldiag", [](const merPredD& self) { return self.L().diagL(); })
        .def("Lnnz", [](const merPredD& self) { return self.L().nonZerosL(); })
        .def_property_readonly("X", &merPredD::X)
        .def_property_readonly("V", &merPredD::V)
        .def_property_readonly("VtV", &merPredD::VtV)
        .def_property_readonly("RZX", &merPredD::RZX)
        .def_property_readonly("Lambdat", &merPredD::Lambdat)
        .def_property_readonly("LamtUt", &merPredD::LamtUt)
        .def_property_readonly("Ut", &merPredD::Ut)
        .def_property_readonly("Zt", &merPredD::Zt)
        .def_property_readonly("Utr", &merPredD::Utr)
        .def_property_readonly("Vtr", &merPredD::Vtr)
        .def_property_readonly("delb", &merPredD::delb)
        .def_property_readonly("delu", &merPredD::delu)
        .def_property_readonly("beta0", &merPredD::beta0)
        .def_property_readonly("theta", &merPredD::theta)
        .def_property_readonly("u0", &merPredD::u0)
        .def_property_readonly("Xwts", &merPredD::Xwts)
        .def_property_readonly("Lind", &merPredD::Lind)
        .def_property_readonly("N", &merPredD::N)
        .def_property_readonly("p", &merPredD::p)
        .def_property_readonly("q", &merPredD::q)
        .def(py::pickle(
            [](const merPredD& s) {
                return py::make_tuple(s.X(), s.Zt(), s.Lambdat(), s.Lind(), s.theta(),
                                      s.V().rows(), s.Xwts(), s.beta0(), s.u0(),
                                      s.V(), s.VtV(), s.RZX(), s.Ut(), s.Utr(), s.Vtr(),
                                      s.delb(), s.delu(), s.CcNumer(), s.ldL2(), s.ldRX2());
            },
            [](py::tuple t) {
                if (t.size() != 20) throw std::runtime_error("merPredD: invalid pickle state");
                merPredD* p = new merPredD(t[0].cast<MatrixXd>(), t[1].cast<SpMat>(),
                                           t[2].cast<SpMat>(), t[3].cast<VectorXi>(),
                                           t[4].cast<VectorXd>(), t[5].cast<Eigen::Index>(),
                                           t[6].cast<VectorXd>(), t[7].cast<VectorXd>(),
                                           t[8].cast<VectorXd>());
                p->restoreState(t[9].cast<MatrixXd>(), t[10].cast<MatrixXd>(),
                                t[11].cast<MatrixXd>(), t[12].cast<SpMat>(),
                                t[13].cast<VectorXd>(), t[14].cast<VectorXd>(),
                                t[15].cast<VectorXd>(), t[16].cast<VectorXd>(),
                                t[17].cast<double>(), t[18].cast<double>(), t[19].cast<double>());
                return p;
            }));

    // ------------------------------------------------------------------ responses
    py::class_<lmResp>(m, "lmResp")
        .def(py::init<const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
                      const VectorXd&, const VectorXd&, const VectorXd&>(),
             py::arg("y"), py::arg("weights"), py::arg("offset"), py::arg("mu"),
             py::arg("sqrtXwt"), py::arg("sqrtrwt"), py::arg("wtres"))
        .def_property_readonly("sqrtXwt", &lmResp::sqrtXwt)
        .def_property_readonly("mu", &lmResp::mu)
        .def_property_readonly("offset", &lmResp::offset)
        .def_property_readonly("sqrtrwt", &lmResp::sqrtrwt)
        .def_property_readonly("weights", &lmResp::weights)
        .def_property_readonly("wtres", &lmResp::wtres)
        .def_property_readonly("y", &lmResp::y)
        .def("wrss", &lmResp::wrss)
        .def("updateMu", &lmResp::updateMu)
        .def("updateWts", &lmResp::updateWts)
        .def("updateWrss", &lmResp::updateWrss)
        .def("setOffset", &lmResp::setOffset)
        .def("setResp", &lmResp::setResp)
        .def("setWeights", &lmResp::setWeights);

    py::class_<lmerResp, lmResp>(m, "lmerResp")
        .def(py::init<const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
                      const VectorXd&, const VectorXd&, const VectorXd&>(),
             py::arg("y"), py::arg("weights"), py::arg("offset"), py::arg("mu"),
             py::arg("sqrtXwt"), py::arg("sqrtrwt"), py::arg("wtres"))
        .def("copy", [](const lmerResp& self) { return lmerResp(self); })
        .def_property("REML", &lmerResp::REML, &lmerResp::setReml)
        .def("setReml", &lmerResp::setReml)
        .def("Laplace", [](const lmerResp& self, double ldL2, double ldRX2, double sqrL,
                           py::object sigma_sq) {
                 if (sigma_sq.is_none()) return self.Laplace(ldL2, ldRX2, sqrL);
                 return self.Laplace(ldL2, ldRX2, sqrL, sigma_sq.cast<double>());
             }, py::arg("ldL2"), py::arg("ldRX2"), py::arg("sqrL"),
             py::arg("sigma_sq") = py::none());

    py::class_<glm::glmFamily>(m, "glmFamily")
        .def(py::init<const std::string&, const std::string&, double>(),
             py::arg("family"), py::arg("link"), py::arg("theta") = NAN)
        .def_property_readonly("family", &glm::glmFamily::fam)
        .def_property_readonly("link", &glm::glmFamily::lnk)
        .def("linkFun", [](const glm::glmFamily& f, const VectorXd& x) { return VectorXd(f.linkFun(x.array())); })
        .def("linkInv", [](const glm::glmFamily& f, const VectorXd& x) { return VectorXd(f.linkInv(x.array())); })
        .def("muEta", [](const glm::glmFamily& f, const VectorXd& x) { return VectorXd(f.muEta(x.array())); })
        .def("variance", [](const glm::glmFamily& f, const VectorXd& x) { return VectorXd(f.variance(x.array())); })
        .def("devResid", [](const glm::glmFamily& f, const VectorXd& y, const VectorXd& mu,
                            const VectorXd& wt) {
                 return VectorXd(f.devResid(y.array(), mu.array(), wt.array()));
             })
        .def("aic", [](const glm::glmFamily& f, const VectorXd& y, const VectorXd& n,
                       const VectorXd& mu, const VectorXd& wt, double dev) {
                 return f.aic(y.array(), n.array(), mu.array(), wt.array(), dev);
             })
        .def("theta", &glm::glmFamily::theta)
        .def("setTheta", &glm::glmFamily::setTheta)
        .def("hasFreeDispersion", &glm::glmFamily::hasFreeDispersion)
        .def(py::pickle(
            [](const glm::glmFamily& f) {
                bool nb = f.fam().substr(0, 18) == "Negative Binomial(";
                return py::make_tuple(f.fam(), f.lnk(), nb ? f.theta() : NAN);
            },
            [](py::tuple t) {
                if (t.size() != 3) throw std::runtime_error("glmFamily: invalid pickle state");
                return new glm::glmFamily(t[0].cast<std::string>(), t[1].cast<std::string>(),
                                          t[2].cast<double>());
            }));

    py::class_<glmResp, lmResp>(m, "glmResp")
        .def(py::init([](const std::string& family, const std::string& link, double theta,
                         const VectorXd& y, const VectorXd& weights, const VectorXd& offset,
                         const VectorXd& mu, const VectorXd& sqrtXwt, const VectorXd& sqrtrwt,
                         const VectorXd& wtres, const VectorXd& eta, const VectorXd& n) {
                 glm::glmFamily fam(family, link, theta);
                 return new glmResp(fam, y, weights, offset, mu, sqrtXwt, sqrtrwt, wtres, eta, n);
             }),
             py::arg("family"), py::arg("link"), py::arg("theta"),
             py::arg("y"), py::arg("weights"), py::arg("offset"), py::arg("mu"),
             py::arg("sqrtXwt"), py::arg("sqrtrwt"), py::arg("wtres"), py::arg("eta"),
             py::arg("n"))
        .def("copy", [](const glmResp& self) { return glmResp(self); })
        .def("devResid", [](const glmResp& r) { return VectorXd(r.devResid()); })
        .def("muEta", [](const glmResp& r) { return VectorXd(r.muEta()); })
        .def("sqrtWrkWt", [](const glmResp& r) { return VectorXd(r.sqrtWrkWt()); })
        .def("variance", [](const glmResp& r) { return VectorXd(r.variance()); })
        .def("wrkResids", [](const glmResp& r) { return VectorXd(r.wrkResids()); })
        .def("wrkResp", [](const glmResp& r) { return VectorXd(r.wrkResp()); })
        .def("wtWrkResp", [](const glmResp& r) { return VectorXd(r.wtWrkResp()); })
        .def_property_readonly("eta", &glmResp::eta)
        .def_property_readonly("n", &glmResp::n)
        .def("fam", &glmResp::family)
        .def("link", &glmResp::link)
        .def("hasFreeDispersion", &glmResp::hasFreeDispersion)
        .def("aic", &glmResp::aic)
        .def("Laplace", &glmResp::Laplace)
        .def("resDev", &glmResp::resDev)
        .def("theta", &glmResp::theta)
        .def("phi", &glmResp::phi)
        .def("updateMu", &glmResp::updateMu)
        .def("updateWts", &glmResp::updateWts)
        .def("setN", &glmResp::setN)
        .def("setPhi", &glmResp::setPhi)
        .def("setTheta", &glmResp::setTheta);

    py::class_<nlsResp, lmResp>(m, "nlsResp")
        .def(py::init<const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
                      const VectorXd&, const VectorXd&, const VectorXd&, const VectorXd&,
                      nlsModel, int>(),
             py::arg("y"), py::arg("weights"), py::arg("offset"), py::arg("mu"),
             py::arg("sqrtXwt"), py::arg("sqrtrwt"), py::arg("wtres"), py::arg("gamma"),
             py::arg("model"), py::arg("npar"))
        .def("copy", [](const nlsResp& self) { return nlsResp(self); })
        .def_property_readonly("gamma", &nlsResp::gamma)
        .def("Laplace", &nlsResp::Laplace)
        .def("updateMu", &nlsResp::updateMu);

    // ------------------------------------------------------------------ optimizers
    py::class_<optimizer::Nelder_Mead>(m, "NelderMead")
        .def(py::init([](const VectorXd& lb, const VectorXd& ub, const VectorXd& xstep0,
                         const VectorXd& x, const VectorXd& xtol) {
                 return new optimizer::Nelder_Mead(lb, ub, xstep0, x, optimizer::nl_stop(xtol));
             }), py::arg("lb"), py::arg("ub"), py::arg("xstep0"), py::arg("x"), py::arg("xtol"))
        .def("newf", [](optimizer::Nelder_Mead& nm, double f) { return nmStatus(nm.newf(f)); })
        .def("setForce_stop", &optimizer::Nelder_Mead::setForce_stop)
        // NB: lme4's NelderMead_setFtol_abs() calls setFtol_rel(); kept for fidelity
        .def("setFtolAbs", [](optimizer::Nelder_Mead& nm, double v) { nm.setFtol_rel(v); })
        .def("setFtolRel", &optimizer::Nelder_Mead::setFtol_rel)
        .def("setIprint", &optimizer::Nelder_Mead::set_Iprint)
        .def("setMaxeval", &optimizer::Nelder_Mead::set_Maxeval)
        .def("setMinfMax", &optimizer::Nelder_Mead::setMinf_max)
        .def("xeval", &optimizer::Nelder_Mead::xeval)
        .def("value", &optimizer::Nelder_Mead::value)
        .def("xpos", &optimizer::Nelder_Mead::xpos)
        .def("evals", [](const optimizer::Nelder_Mead& nm) { return (int) nm.evals(); });

    py::class_<optimizer::Golden>(m, "Golden")
        .def(py::init<double, double>())
        .def("newf", &optimizer::Golden::newf)
        .def("xeval", &optimizer::Golden::xeval)
        .def("value", &optimizer::Golden::value)
        .def("xpos", &optimizer::Golden::xpos);

    // ------------------------------------------------------------------ algorithms
    m.def("lmer_Deviance", &lme4ext::lmer_Deviance);
    m.def("lmer_opt1", &lme4ext::lmer_opt1);
    m.def("glmerLaplace", &lme4ext::glmerLaplace,
          py::arg("pp"), py::arg("rp"), py::arg("nAGQ"), py::arg("tol"), py::arg("maxit"),
          py::arg("verbose"), py::arg("dispProfile"), py::arg("maxPhiIter"), py::arg("qEff"));
    m.def("glmerAGQ", &lme4ext::glmerAGQ,
          py::arg("pp"), py::arg("rp"), py::arg("tol"), py::arg("maxit"), py::arg("GQmat"),
          py::arg("fac"), py::arg("verbose"), py::arg("dispProfile"), py::arg("maxPhiIter"),
          py::arg("qEff"));
    m.def("nlmerLaplace", &lme4ext::nlmerLaplace);
    m.def("allPerm_int", &lme4ext::allPerm_int);

    // ------------------------------------------------------------------ minqa
    // Equivalent of minqa.cpp's bobyqa_cpp() + rval(): runs BOBYQA, then (like
    // rval) re-evaluates fn at the returned par for fval; that extra call is
    // not counted in feval.  The R-level preprocessing of minqa::bobyqa()
    // (rhobeg adjustment, start-value checks, ierr -> msg) is left to Python.
    m.def("bobyqa", [](py::function fn, const VectorXd& x0, const VectorXd& lower,
                       const VectorXd& upper, double rhobeg, double rhoend, int iprint,
                       int maxfun, int npt) {
              const int n = (int) x0.size();
              if (lower.size() != n || upper.size() != n)
                  throw std::invalid_argument("bobyqa: x0, lower and upper must have the same length");
              if (npt <= 0) {   // minqa's commonArgs() default
                  npt = std::min(n + 2, 2 * n);
                  npt = std::max(n + 2, std::min(npt, ((n + 1) * (n + 2)) / 2));
              }
              auto call = [&fn](const double* x, int nn) {
                  py::array_t<double> xa(nn);
                  std::copy(x, x + nn, xa.mutable_data());
                  return fn(xa).cast<double>();
              };
              VectorXd par = x0;
              int feval = 0;
              int ierr = minqa::bobyqa(n, npt, par.data(), lower.data(), upper.data(),
                                       rhobeg, rhoend, iprint, maxfun, call, feval);
              double fval = call(par.data(), n);
              py::dict res;
              res["par"] = par;
              res["fval"] = fval;
              res["feval"] = feval;
              res["ierr"] = ierr;
              return res;
          },
          py::arg("fn"), py::arg("x0"), py::arg("lower"), py::arg("upper"),
          py::arg("rhobeg"), py::arg("rhoend"), py::arg("iprint") = 0,
          py::arg("maxfun") = 10000, py::arg("npt") = 0,
          "Powell's BOBYQA as in R's minqa 1.2.8; returns dict(par, fval, feval, ierr) "
          "with the raw Fortran ierr (0, 10, 20, 320, 390, 430). npt <= 0 selects "
          "minqa's default.");
}
