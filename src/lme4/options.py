"""Global options (R's ``options()`` entries used by lme4)."""
_OPTIONS = {
    "lme4.singular.tolerance": 1e-4,
    "lme4.summary.cor.max": 12,
    "lme4.doublevert.default": "split",
    "digits": 7,
    "show.signif.stars": True,
    "width": 80,
}


def get(name, default=None):
    return _OPTIONS.get(name, default)


def set(**kw):
    """lme4.options.set(**{"lme4.summary.cor.max": 20})"""
    old = {k: _OPTIONS.get(k) for k in kw}
    _OPTIONS.update(kw)
    return old
