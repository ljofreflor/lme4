# lme4 for Python

**Linear, generalized linear and nonlinear mixed-effects models for Python: a faithful,
independent transcription of the R package [lme4](https://github.com/lme4/lme4) (version 2.1-0).**

[![PyPI](https://img.shields.io/pypi/v/lme4.svg)](https://pypi.org/project/lme4/)
[![wheels](https://github.com/ljofreflor/lme4/actions/workflows/wheels.yml/badge.svg)](https://github.com/ljofreflor/lme4/actions/workflows/wheels.yml)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](LICENSE)

```python
from lme4 import lmer, V, load_dataset

sleepstudy = load_dataset("sleepstudy")
Reaction, Days, Subject = V("Reaction Days Subject")

fm1 = lmer(Reaction == Days + (Days | Subject), data=sleepstudy)
print(fm1.summary())
```

```
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)
   Data: sleepstudy

REML criterion at convergence: 1743.6

Scaled residuals:
    Min      1Q  Median      3Q     Max
-3.9536 -0.4634  0.0231  0.4634  5.1793

Random effects:
 Groups   Name        Variance Std.Dev. Corr
 Subject  (Intercept) 612.10   24.741
          Days         35.07    5.922   0.07
 Residual             654.94   25.592
Number of obs: 180, groups:  Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept)  251.405      6.825  36.838
Days          10.467      1.546   6.771

Correlation of Fixed Effects:
     (Intr)
Days -0.138
```

This is, line for line, the output of R's `summary(lmer(Reaction ~ Days + (Days | Subject), sleepstudy))`.

---

## Motivation

`lme4` (Bates, Mächler, Bolker & Walker, 2015) is the reference implementation of mixed-effects
models in statistics. Its penalized least squares formulation, sparse Cholesky machinery and
careful numerics are the result of more than fifteen years of work, and its results are
the benchmark against which other software is checked.

Python users have had three options, none fully satisfying:

- **call R from Python** (e.g. `pymer4` through `rpy2`), which requires an R installation and
  ties every deployment to it;
- **use a different algorithm** (e.g. `statsmodels.MixedLM`), whose estimates, defaults and
  supported models differ from lme4's;
- **use partial re-implementations**, which cover a subset of lme4 (typically linear models
  with simple random intercepts).

This project takes a fourth route: **transcribe lme4 itself**. Every R function on the fitting
path and every line of its C++ code were translated, preserving algorithms, defaults, operation
order, error messages and output formats, so that a model fitted in Python gives the same answer
as the same model fitted in R — without R.

## Independence

The package does **not** need R, rpy2 or any R package at runtime or at build time.

- **Native C++ backend.** lme4's own C++ modules (`merPredD`, `lmerResp`, `glmResp`,
  `nlsResp`, `glmFamily`, the PIRLS loop, Laplace and adaptive Gauss–Hermite quadrature,
  the Nelder–Mead optimizer) are transcribed to standard C++17 on top of
  [Eigen](https://eigen.tuxfamily.org) and exposed through pybind11. Rcpp and CHOLMOD are
  replaced by plain Eigen (sparse Cholesky via `SimplicialLLT` with AMD ordering).
- **Self-contained optimizers.** Powell's BOBYQA, as distributed in the R package `minqa`, is
  transcribed to C++; `nloptwrap` uses NLopt, the same C library that R's `nloptr` wraps.
- **R's numerical building blocks** that lme4 relies upon — the `nmath` density functions,
  `qr()` with LINPACK's `dqrdc2` pivoting, `poly()`, contrasts, `model.frame()` /
  `model.matrix()` / `terms()`, the `reformulas` package — are re-implemented.
- **pandas in, pandas out.** Data are pandas DataFrames (categoricals are R factors);
  coefficients, variance components and tables are returned as pandas objects.
- **Prebuilt wheels** for Windows, Linux (x86_64, aarch64) and macOS (Intel, Apple Silicon),
  Python 3.9–3.13.

## Fidelity

Fidelity is not claimed, it is measured. The test suite compares the package against reference
results produced by **R 4.6.2 with lme4 2.1.0**, generated in a pinned Docker image
([`tests/reference/`](tests/reference)):

| check | result |
|---|---|
| objective function (`logLik`, REML/ML criterion) on 24 reference models (LMMs, binomial/Poisson/Gamma GLMMs, AGQ, structured covariances, weights, offsets, nested and crossed effects) | relative difference ≤ 1e-10 |
| fixed effects, variance components, `vcov`, BLUPs, conditional variances, fitted values | typically 1e-8 to 1e-15 |
| printed `summary()` | identical to R's output, line by line |
| `confint(method="profile")`, `predict()`, `glmer.nb()`, `lmList()`, `nlmer()`, `anova()` | agree with R |
| lme4's own test suite (`tests/testthat`, `*.Rout.save`) | numeric expectations ported and passing |

Where the likelihood surface is very flat (for instance a GLMM in which R itself reports a
convergence warning) the estimates can differ in the third or fourth significant digit while
the attained likelihood agrees to 1e-10: both are equally optimal points. The cause is
rounding at the 1e-15 level (Eigen's AMD ordering versus CHOLMOD's), which changes where an
optimizer stops on a plateau.

The transcription is also faithful *in structure*: Python modules mirror lme4's source files
(`lmer.py` ↔ `lmer.R`, `modular.py` ↔ `modular.R`, `covariance.R`, `predict.R`, `profile.R`,
`bootMer.R`, `nbinom.R`, `lmList.R`, `allFit.R`, …; `src/cpp/` ↔ `lme4/src/`), so that the R
source remains the documentation of the algorithms.

## Installation

```bash
pip install --pre lme4
# or
poetry add lme4 --allow-prereleases
```

Releases are currently alpha versions (`2.1.0a*`), hence `--pre`. The version number follows
the lme4 release that is transcribed.

From source, a C++17 compiler is used if available; otherwise the build falls back
automatically to the [Zig](https://ziglang.org) toolchain (`ziglang` package), so no Visual
Studio is needed on Windows:

```bash
pip install git+https://github.com/ljofreflor/lme4.git@develop
```

## Usage

### Formulas

Formulas can be written as R strings or as Python expressions. Python has no binary `~`, so the
response is separated from the predictors with `==`:

```python
from lme4 import lmer, glmer, binomial, V, cbind, load_dataset

# R-style strings
fm = lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)

# symbolic
incidence, size, period, herd = V("incidence size period herd")
cbpp = load_dataset("cbpp")
gm = glmer(cbind(incidence, size - incidence) == period + (1 | herd),
           data=cbpp, family=binomial)
```

| R | Python (symbolic) |
|---|---|
| `y ~ x + z` | `y == x + z` |
| `(x \| g)`, `(1 + x \| g)` | `(x \| g)`, `(1 + x \| g)` |
| `(x \|\| g)` | `x.dbar(g)` |
| `a:b` | `a & b` |
| `a * b`, `a / b`, `(a + b + c)^2` | `a * b`, `a / b`, `(a + b + c) ** 2` |
| `0 + x`, `x - 1` | `0 + x`, `x - 1` |
| `I(x^2)`, `log(x)`, `poly(x, 2)` | `I(x ** 2)`, `log(x)` or `np.log(x)`, `poly(x, 2)` |
| `cs(1 + x \| g)`, `ar1(...)`, `diag(...)` | `cs(...)`, `ar1(...)`, `diag(...)` |

In Python `|` and `&` bind less tightly than `+`: write `x + (a & b)` for R's `x + a:b`
(just as `(Days | Subject)` already needs parentheses in R).

### Correspondence with the R API

| R | Python |
|---|---|
| `lmer()`, `glmer()`, `nlmer()`, `glmer.nb()` | `lmer()`, `glmer()`, `nlmer()`, `glmer_nb()` |
| `lmerControl(check.conv.grad = "ignore")` | `lmerControl(check_conv_grad="ignore")` |
| `fixef(m)`, `ranef(m)`, `VarCorr(m)`, `getME(m, "theta")` | `m.fixef()`, `m.ranef()`, `m.VarCorr()`, `m.getME("theta")` (also as functions) |
| `summary(m)`, `print(m)` | `m.summary()`, `print(m)` |
| `predict(m, newdata, re.form = NA)` | `m.predict(newdata, re_form="NA")` |
| `confint(m)`, `profile(m)`, `bootMer(m, FUN, nsim)` | `m.confint()`, `m.profile()`, `bootMer(m, FUN, nsim)` |
| `anova(m0, m1)`, `refitML(m)`, `update(m, . ~ . - x)` | `m0.anova(m1)`, `m.refitML()`, `m.update(". ~ . - x")` |
| `lmList()`, `allFit()`, `influence()` | `lmList()`, `allFit()`, `influence()` |
| `data(sleepstudy)` | `load_dataset("sleepstudy")` |

Argument names use `_` where R uses `.` (`na_action`, `re_form`, `allow_new_levels`).

### Coverage

Implemented: `lmer`, `glmer` (Laplace, `nAGQ = 0`, adaptive Gauss–Hermite quadrature),
`nlmer` (with the self-starting models `SSlogis`, `SSasymp`, `SSmicmen`, `SSfol`, … and
user-supplied models), `glmer.nb`; structured covariances `us`, `diag`, `cs`, `ar1`;
the modular interface (`lFormula`, `mkLmerDevfun`, `optimizeLmer`, `glFormula`,
`mkGlmerDevfun`, …); `summary`, `VarCorr`, `ranef` with conditional variances, `coef`,
`vcov`, `predict`, `simulate`, `profile`, `confint` (profile, Wald, bootstrap), `bootMer`,
`refit`, `refitML`, `update`, `anova`, `drop1`, `getME`, `hatvalues`, `lmList`, `allFit`,
`influence`, Cook's distance; the datasets shipped with lme4.

Not (yet) implemented: plotting methods, `predict(se.fit = TRUE)`, `simulate()` from a formula
without a fitted model, user-defined GLM families. `nlminbwrap` uses SciPy's L-BFGS-B, since
R's PORT library is not available.

Known deliberate differences: random numbers come from NumPy, so simulations and bootstrap
samples do not reproduce R's streams; factor levels created from strings are sorted by code
point (R sorts by locale; pandas categoricals keep their own order); R's `profile()` modifies
the fitted model in place, this implementation does not.

## Development

```bash
pip install numpy scipy pandas nlopt pybind11 ziglang pytest
python tools/build_ext.py --zig        # or: python setup.py build_ext --inplace
pytest
```

Reference results are regenerated with R in Docker:

```bash
docker build -t lme4ref tests/reference
docker run --rm -v "$PWD/tests/reference:/ref" -w /ref lme4ref Rscript make_reference.R specs.json reference.json
docker run --rm -v "$PWD/tests/reference:/ref" -w /ref lme4ref Rscript make_extras.R
```

Layout:

```
src/cpp/       C++ backend (transcription of lme4/src and minqa/src)
src/lme4/      Python package (transcription of lme4/R and reformulas/R)
tests/         tests, including ports of lme4's tests and the R reference comparison
upstream/      the original R sources (lme4 2.1-0, reformulas 0.4.4, minqa 1.2.8)
third_party/   Eigen 3.4.0
```

## Credits and citation

All statistical and algorithmic credit belongs to the authors of lme4: Douglas Bates, Martin
Mächler, Ben Bolker, Steven Walker, and its contributors; to Ben Bolker and the authors of
`reformulas`; to M. J. D. Powell (BOBYQA) and the `minqa` authors; and to the R Core Team.
This project is an independent transcription and is not affiliated with or endorsed by the
lme4 developers. Please cite the original work:

> Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting Linear Mixed-Effects Models
> Using lme4. *Journal of Statistical Software*, 67(1), 1–48.
> https://doi.org/10.18637/jss.v067.i01

## License

GPL (>= 2), the license of lme4. Eigen is distributed under the MPL 2.0.
