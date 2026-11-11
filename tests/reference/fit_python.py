"""Fit the reference specs with the Python port (shared by tests and diagnostics)."""
import json
import warnings
from pathlib import Path

import numpy as np

from lme4 import lmer, glmer, load_dataset
from lme4.family import get_family, Family
from lme4.modelframe import eval_expr, EvalContext, data_to_dict
from lme4.rlang import parse

HERE = Path(__file__).parent
SPECS = json.loads((HERE / "specs.json").read_text(encoding="utf-8"))
REF = json.loads((HERE / "reference.json").read_text(encoding="utf-8"))


def fit(spec):
    d = load_dataset(spec["data"])
    if spec.get("subset"):
        keep = np.asarray(eval_expr(parse(spec["subset"]), EvalContext(data_to_dict(d))), bool)
        d = d[keep].reset_index(drop=True)
    kw = {}
    if spec.get("weights"):
        kw["weights"] = np.asarray(eval_expr(parse(spec["weights"]), EvalContext(data_to_dict(d))), float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec["fun"] == "lmer":
            return lmer(spec["formula"], d, REML=spec.get("REML", True), **kw)
        fam = get_family(spec["family"])
        if spec.get("link"):
            fam = Family(fam.family, spec["link"])
        return glmer(spec["formula"], d, family=fam, nAGQ=spec.get("nAGQ", 1), **kw)


def quantities(m):
    re = m.ranef(condVar=True)
    vc = m.VarCorr().as_data_frame()
    return dict(fixef=m.fixef().values, theta=m.theta, sigma=m.sigma(), logLik=m.logLik().value,
                AIC=m.AIC(), devcrit=m.REMLcrit() if m.isREML() else (m.deviance() if m.isGLMM() else m.devCrit()),
                vcov=m.vcov().values, vcov_rx=m.vcov(use_hessian=False).values,
                vc_vcov=vc.vcov.values, vc_sdcor=vc.sdcor.values,
                fitted=m.fitted().values, resid=m.residuals().values,
                ranef={k: np.asarray(v) for k, v in re.items()},
                condsd={k: np.sqrt(np.einsum("iik->ki", v.attrs["postVar"])).ravel()
                        for k, v in re.items()})


def maxrel(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    scale = np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-3 + 1e-12)
    return float(np.max(np.abs(a - b) / scale))


if __name__ == "__main__":
    import sys
    ids = sys.argv[1:] or [s["id"] for s in SPECS]
    for s in SPECS:
        if s["id"] not in ids:
            continue
        r = REF[s["id"]]
        if "error" in r:
            print(f"{s['id']:15s} R ERROR: {r['error']}")
            continue
        try:
            q = quantities(fit(s))
        except Exception as e:
            print(f"{s['id']:15s} PY ERROR {type(e).__name__}: {e}")
            continue
        diffs = {k: maxrel(q[k], r[k]) for k in ("devcrit", "logLik", "fixef", "theta", "sigma", "vcov",
                                                 "vcov_rx", "vc_sdcor", "fitted")}
        diffs["ranef"] = max(maxrel(q["ranef"][k], r["ranef"][k]) for k in r["ranef"])
        diffs["condsd"] = max([maxrel(q["condsd"][k], r["condsd"][k]) for k in r["condsd"] if r["condsd"][k] is not None] or [0])
        worst = max(diffs.values())
        print(f"{s['id']:15s} worst={worst:.1e}  " + " ".join(f"{k}={v:.0e}" for k, v in diffs.items()))
