"""Convert lme4's R data files (data/*.rda, R/sysdata.rda) into
CSV + JSON metadata shipped with the Python package.

Run with:  python -I tools/convert_rdata.py <lme4-source-dir> <out-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdata


def convert_df(obj):
    meta = {}
    df = pd.DataFrame(obj)
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            meta[col] = {"levels": [str(x) for x in s.cat.categories],
                         "ordered": bool(s.cat.ordered)}
            df[col] = s.astype(object).astype(str).where(s.notna(), None)
    return df, meta


def main(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    allmeta = {}
    for f in sorted((src / "data").glob("*.rda")):
        parsed = rdata.parser.parse_file(f)
        conv = rdata.conversion.convert(parsed)
        for name, obj in conv.items():
            try:
                df, meta = convert_df(obj)
            except Exception as e:  # pragma: no cover
                print("skip", name, e)
                continue
            df.to_csv(out / f"{name}.csv.gz", index=False, compression="gzip")
            allmeta[name] = meta
            print(name, df.shape)
    (out / "_factors.json").write_text(json.dumps(allmeta, indent=1))
    # Gauss-Hermite rules
    sysd = rdata.conversion.convert(rdata.parser.parse_file(src / "R" / "sysdata.rda"))
    rules = [{"z": list(map(float, r["z"])), "w": list(map(float, r["w"]))}
             for r in sysd["fgq_rules"]]
    (out.parent / "_fgq_rules.json").write_text(json.dumps(rules))
    print("fgq_rules", len(rules))


if __name__ == "__main__":
    main(*sys.argv[1:3])
