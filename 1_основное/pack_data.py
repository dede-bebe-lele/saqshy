"""Pack the raw downloads into small yearly files (< 25 MB each) so the repo can rebuild everything quickly."""
import glob
import os
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
(DATA / "packed").mkdir(exist_ok=True)
parts = []
for f in sorted(glob.glob(str(DATA / "weather" / "*.csv"))):
    d = pd.read_csv(f)
    d.insert(0, "cell", os.path.basename(f)[:-4])
    parts.append(d)
w = pd.concat(parts)
for c in w.columns[3:]:
    w[c] = w[c].astype("float32")
for y, g in w.groupby("YEAR"):
    g.to_parquet(DATA / "packed" / f"weather_{y}.parquet", compression="zstd", index=False)
for f in sorted(glob.glob(str(DATA / "firms" / "viirs_*.csv"))):
    d = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "type", "frp"])
    d = d[(d.type == 0) & (d.confidence != "l")]
    d.to_parquet(DATA / "packed" / ("fires_" + os.path.basename(f)[6:10] + ".parquet"), compression="zstd", index=False)
print("packed")
