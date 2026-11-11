# lme4 para Python

Transcripción fiel del paquete de R **lme4 2.1-0** (Bates, Maechler, Bolker, Walker y otros)
a Python, con el mismo backend numérico en **C++/Eigen** (expuesto con pybind11) y entrada
de datos con **pandas**.

No es un *wrapper* de R ni requiere R instalado: el código C++ de lme4 (`predModule`,
`respModule`, `glmFamily`, Nelder-Mead, PIRLS, Laplace/AGQ) y el BOBYQA de `minqa` fueron
transcritos, y la lógica R (`lFormula`, `mkReTrms`, `mkLmerDevfun`, `optimizeLmer`,
`mkMerMod`, `summary`, `VarCorr`, ...) se transcribió a Python función por función.

## Uso

Las fórmulas se pueden escribir de forma **simbólica**, con `==` en lugar de `~`:

```python
from lme4 import lmer, glmer, binomial, V, cbind, load_dataset

sleepstudy = load_dataset("sleepstudy")
Reaction, Days, Subject = V("Reaction Days Subject")

fm1 = lmer(Reaction == Days + (Days | Subject), data=sleepstudy)
print(fm1.summary())
```

o como en R, con un string:

```python
fm1 = lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)

cbpp = load_dataset("cbpp")
gm1 = glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)",
            data=cbpp, family=binomial)
```

Salida (idéntica a la de R):

```
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)
   Data: sleepstudy

REML criterion at convergence: 1743.6
...
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
```

### Fórmulas simbólicas

| Python                    | R                        |
|---------------------------|--------------------------|
| `y == x + z`              | `y ~ x + z`              |
| `(x | g)`, `(1 + x | g)`  | `(x | g)`, `(1 + x | g)` |
| `x.dbar(g)`               | `(x || g)`               |
| `a & b`                   | `a:b`                    |
| `a * b`, `a / b`          | `a * b`, `a / b`         |
| `(a + b + c) ** 2`        | `(a + b + c)^2`          |
| `0 + x`, `x - 1`          | `0 + x`, `x - 1`         |
| `I(x ** 2)`, `log(x)`, `np.log(x)` | `I(x^2)`, `log(x)` |
| `cbind(s, n - s)`         | `cbind(s, n - s)`        |

En Python `|` y `&` tienen menor precedencia que `+`: escribe `x + (a & b)` para `x + a:b`
(igual que `(Days | Subject)` ya necesita paréntesis en R).

### Métodos del modelo ajustado

`fixef()`, `ranef(condVar=True)`, `coef()`, `VarCorr()`, `sigma()`, `logLik()`, `AIC()`,
`BIC()`, `deviance()`, `REMLcrit()`, `vcov()`, `summary()`, `fitted()`, `residuals()`,
`predict()`, `simulate()`, `confint()`, `anova()`, `refitML()`, `update()`, `refit()`,
`getME(name)`, `isSingular()`, `hatvalues()`, `as_function()` (función de devianza), ...

Los resultados se devuelven como objetos pandas (`Series`/`DataFrame`) con los mismos
nombres que en R.

Los argumentos de control usan `_` en vez de `.`:
`lmerControl(optimizer="bobyqa", check_conv_grad="ignore")`.

## Instalación

Requisitos: Python ≥ 3.9, numpy, scipy, pandas, nlopt, pybind11 y un compilador C++17.

```bash
pip install .
```

Sin compilador (por ejemplo, Windows sin Visual Studio):

```bash
pip install numpy scipy pandas nlopt pybind11 ziglang
python tools/build_ext.py --zig      # compila src/lme4/_lme4cpp con zig c++
```

## Fidelidad respecto de R

- Los tests (`pytest`) reproducen valores numéricos de la suite de tests de lme4 2.1-0
  (por ejemplo: `Dyestuff` REML 319.654276842342, `sleepstudy` REML 1743.628,
  `cbpp` deviance 73.47154).
- Diferencias de implementación:
  - La factorización de Cholesky dispersa usa `Eigen::SimplicialLLT` con ordenamiento AMD
    en lugar de CHOLMOD. Es matemáticamente idéntica, pero la permutación de `getME(m, "L")`
    puede diferir.
  - `nloptwrap` usa el paquete `nlopt` de Python (la misma biblioteca C que `nloptr`).
  - `nlminbwrap` usa L-BFGS-B de scipy (el código PORT de R no está disponible).
  - Las familias GLM se identifican por nombre: no se aceptan familias definidas por el
    usuario.
  - Los niveles de factores creados a partir de strings se ordenan por punto de código
    (R usa el orden del *locale*). Los `Categorical` de pandas conservan su orden.

## Estructura

```
src/cpp/        backend C++ (transcripción de lme4/src y minqa/src)
src/lme4/       paquete Python (transcripción de lme4/R y reformulas/R)
tests/          tests portados de lme4/tests
tools/          compilación y conversión de datos
upstream/       código fuente original de lme4, reformulas y minqa (referencia)
third_party/    Eigen 3.4.0 (MPL2)
```

Licencia: GPL (>= 2), como lme4.
