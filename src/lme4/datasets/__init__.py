"""The example data sets shipped with lme4, as pandas DataFrames
(factors become ordered/unordered ``Categorical`` columns with R's levels)."""
import json
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent
_META = None


def list_datasets():
    return sorted(p.name[:-7] for p in _DIR.glob("*.csv.gz"))


def load_dataset(name):
    global _META
    if _META is None:
        _META = json.loads((_DIR / "_factors.json").read_text())
    path = _DIR / f"{name}.csv.gz"
    if not path.exists():
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(list_datasets())}")
    meta = _META.get(name, {})
    df = pd.read_csv(path, dtype={c: str for c in meta}, keep_default_na=True)
    for col, m in meta.items():
        df[col] = pd.Categorical(df[col], categories=m["levels"], ordered=m["ordered"])
    return df
