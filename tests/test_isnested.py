"""reformulas::isNested -- values checked against R (lme4 2.1.0 / reformulas, Docker):
with(Pastes, isNested(cask, batch)) FALSE; isNested(sample, batch) TRUE;
isNested(batch, sample) FALSE; with(Penicillin, isNested(plate, sample)) FALSE;
with(InstEval, isNested(d, dept)) TRUE; isNested(s, d) FALSE."""
import numpy as np
import pandas as pd
import pytest

from lme4 import isNested, load_dataset


def test_isNested_pastes_penicillin():
    P = load_dataset("Pastes")
    assert isNested(P["cask"], P["batch"]) is False
    assert isNested(P["sample"], P["batch"]) is True
    assert isNested(P["batch"], P["sample"]) is False
    Pen = load_dataset("Penicillin")
    assert isNested(Pen["plate"], Pen["sample"]) is False


def test_isNested_insteval():
    I = load_dataset("InstEval")
    assert isNested(I["d"], I["dept"]) is True
    assert isNested(I["s"], I["d"]) is False


def test_isNested_plain_inputs():
    assert isNested(np.array([1, 1, 2, 2]), ["a", "a", "b", "b"]) is True
    assert isNested(["x", "y", "x"], ["a", "a", "b"]) is False
    # unused levels are fine
    f1 = pd.Categorical(["u", "u"], categories=["u", "v"])
    assert isNested(f1, ["a", "a"]) is True
    with pytest.raises(ValueError):
        isNested([1, 2], [1])
