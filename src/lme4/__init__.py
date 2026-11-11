"""lme4 for Python: a faithful transcription of the R package lme4
(Bates, Maechler, Bolker, Walker et al.) with a C++/Eigen backend and
pandas support.

    >>> from lme4 import lmer, V, load_dataset
    >>> sleepstudy = load_dataset("sleepstudy")
    >>> Reaction, Days, Subject = V("Reaction Days Subject")
    >>> fm1 = lmer(Reaction == Days + (Days | Subject), data=sleepstudy)
    >>> # or, R-style:  lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
    >>> print(fm1.summary())
"""
from .rlang import Formula
from .symbolic import V, Term, I, log, exp, sqrt, factor, poly, offset, scale, cbind, us, diag, cs, ar1
from .control import lmerControl, glmerControl, nlmerControl
from .family import (Family, binomial, poisson, Gamma, gaussian, inverse_gaussian,
                     negative_binomial)
from .lmer import lmer, glmer
from .nlmer import nlmer
from .profile import profile_merMod, confint_merMod
from .bootmer import bootMer
from .nbinom import glmer_nb
from .lmlist import lmList
from .allfit import allFit
from .influence import influence, cooks_distance, rstudent, rstandard, dfbeta, dfbetas
from .modular import (lFormula, glFormula, mkLmerDevfun, optimizeLmer, mkGlmerDevfun,
                      optimizeGlmer, updateGlmerDevfun)
from .lmer import mkMerMod
from .mermod import merMod, lmerMod, glmerMod, nlmerMod
from .reformulas import findbars, nobars, subbars, expandDoubleVerts, mkReTrms
from .optimizers import Nelder_Mead, bobyqa, nloptwrap, GHrule
from .datasets import load_dataset, list_datasets
from . import options

__version__ = "2.1.0a1"   # tracks lme4 2.1-0


def fixef(m, **kw): return m.fixef(**kw)
def ranef(m, **kw): return m.ranef(**kw)
def VarCorr(m, **kw): return m.VarCorr(**kw)
def getME(m, name): return m.getME(name)
def isSingular(m, tol=None): return m.isSingular(tol)
def REMLcrit(m): return m.REMLcrit()
def refitML(m, **kw): return m.refitML(**kw)


def anova(m, *others, **kw):
    return m.anova(*others, **kw)


__all__ = [n for n in dir() if not n.startswith("_")]
