"""Download daily weather for every 0.5° grid cell in Kazakhstan from NASA POWER (no key needed).

Output: data/grid_cells.csv and data/weather/{cell}.csv (2012-01-01 .. 2025-12-31).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from matplotlib.path import Path as Poly

DATA = Path(__file__).parent / "data"
OUT = DATA / "weather"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = "T2M,T2M_MAX,T2M_MIN,QV2M,RH2M,WS2M,WS2M_MAX,PRECTOTCORR,GWETTOP,GWETROOT,PS"
STEP = 0.5


def make_grid():
    geo = json.load(open(DATA / "kaz.geo.json"))
    ring = np.array(geo["features"][0]["geometry"]["coordinates"][0])
    poly = Poly(ring)
    cells = []
    for lat in np.arange(40.5 + STEP / 2, 56, STEP):
        for lon in np.arange(46.0 + STEP / 2, 88, STEP):
            if poly.contains_point((lon, lat)):
                cells.append((f"{lat:.2f}_{lon:.2f}", round(lat, 2), round(lon, 2)))
    df = pd.DataFrame(cells, columns=["cell", "lat", "lon"])
    df.to_csv(DATA / "grid_cells.csv", index=False)
    return df


def fetch(row):
    f = OUT / f"{row.cell}.csv"
    if f.exists() and f.stat().st_size > 1000:
        return "cached"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={PARAMS}&community=AG&longitude={row.lon}&latitude={row.lat}"
        "&start=20120101&end=20251231&format=CSV&header=false"
    )
    for attempt in range(6):
        try:
            txt = urlopen(url, timeout=180).read().decode()
            df = pd.read_csv(StringIO(txt))
            df.to_csv(f, index=False)
            return "ok"
        except Exception as e:  # network hiccups, rate limits
            time.sleep(5 * (attempt + 1))
            err = e
    return f"fail {err}"


if __name__ == "__main__":
    grid = make_grid()
    print(len(grid), "cells")
    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(fetch, grid.itertuples()))
    print(pd.Series([r.split()[0] for r in res]).value_counts())
