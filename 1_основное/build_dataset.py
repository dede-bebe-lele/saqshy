"""Build one row per grid cell per evening (fire season, March-November).

Everything in a row is known on the evening of day t. The target is whether NASA's
satellites detect a vegetation fire in the cell on day t+1.

Output: data/dataset.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from fire_indices import dc_step, dmc_step, ffmc_step, isi_bui_fwi

DATA = Path(__file__).parent / "data"
DAYS = pd.date_range("2012-01-01", "2025-12-31")
D = len(DAYS)


def load_weather(cells):
    cols = ["T2M", "T2M_MAX", "T2M_MIN", "QV2M", "RH2M", "WS2M", "WS2M_MAX", "PRECTOTCORR", "GWETTOP", "GWETROOT", "PS"]
    W = {c: np.full((D, len(cells)), np.nan, dtype=np.float32) for c in cols}
    packed = sorted((DATA / "packed").glob("weather_*.parquet"))
    allw = pd.concat([pd.read_parquet(f) for f in packed]).groupby("cell") if packed else None
    for k, cell in enumerate(cells.cell):
        w = allw.get_group(cell) if packed else pd.read_csv(DATA / "weather" / f"{cell}.csv")
        w = w.replace(-999, np.nan)
        dates = pd.to_datetime(w.YEAR.astype(str), format="%Y") + pd.to_timedelta(w.DOY - 1, unit="D")
        idx = DAYS.get_indexer(dates)
        ok = idx >= 0
        for c in cols:
            W[c][idx[ok], k] = w[c].values[ok]
    for c in cols:  # rare gaps: carry the last value forward
        W[c] = pd.DataFrame(W[c]).ffill().bfill().values.astype(np.float32)
    return W


def load_fires(cells):
    import glob
    packed = sorted((DATA / "packed").glob("fires_*.parquet"))
    if packed:
        f = pd.concat([pd.read_parquet(p) for p in packed])
    else:
        f = pd.concat([pd.read_csv(p, usecols=["latitude", "longitude", "acq_date", "confidence", "type", "frp"])
                       for p in sorted(glob.glob(str(DATA / "firms" / "viirs_*.csv")))])
    f = f[(f.type == 0) & (f.confidence != "l")]  # vegetation fires, nominal/high confidence
    f["cell"] = [f"{40.5 + 0.5 * np.floor((a - 40.5) / 0.5) + 0.25:.2f}_{46.0 + 0.5 * np.floor((o - 46.0) / 0.5) + 0.25:.2f}"
                 for a, o in zip(f.latitude, f.longitude)]
    cpos = pd.Series(np.arange(len(cells)), index=cells.cell)
    f = f[f.cell.isin(cpos.index)]
    di = DAYS.get_indexer(pd.to_datetime(f.acq_date))
    ci = cpos[f.cell].values
    F = np.zeros((D, len(cells)), dtype=np.float32)
    P = np.zeros((D, len(cells)), dtype=np.float32)
    np.add.at(F, (di, ci), 1)
    np.add.at(P, (di, ci), f.frp.values)
    return F, P


def rolling_sum(A, n):
    """Sum of the last n days including today, along axis 0."""
    c = np.cumsum(np.vstack([np.zeros((1, A.shape[1]), A.dtype), A]), axis=0)
    out = c[n:] - c[:-n]
    return np.vstack([c[1:n], out]).astype(np.float32)


def neighbours(A, cells, size):
    """Sum over the size x size block of cells around each cell, excluding the cell itself."""
    ii = ((cells.lat - 40.75) / 0.5).round().astype(int).values
    jj = ((cells.lon - 46.25) / 0.5).round().astype(int).values
    G = np.zeros((A.shape[0], ii.max() + 1, jj.max() + 1), dtype=np.float32)
    G[:, ii, jj] = A
    k = np.ones((1, size, size), dtype=np.float32)
    S = convolve(G, k, mode="constant")
    return S[:, ii, jj] - A


def indices(W):
    T = W["T2M_MAX"]
    q = W["QV2M"] / 1000.0
    e = q * W["PS"] / (0.622 + 0.378 * q)  # vapour pressure, kPa
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    H = np.clip(100 * e / es, 1, 100)  # afternoon relative humidity
    lr = np.log(np.maximum(e, 1e-4) / 0.6108)
    Td = 237.3 * lr / (17.27 - lr)
    Wk = W["WS2M"] * 3.6 * 1.3  # km/h, 2 m -> ~10 m wind
    r = W["PRECTOTCORR"]
    out = {k: np.full_like(T, np.nan) for k in ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "nesterov"]}
    ffmc = dmc = dc = None
    G = np.zeros(T.shape[1], dtype=np.float32)
    for t, day in enumerate(DAYS):
        if day.month in (12, 1, 2):
            ffmc = None
            G[:] = 0
            continue
        if ffmc is None or (day.month == 3 and day.day == 1):
            ffmc, dmc, dc = np.full(T.shape[1], 85.0), np.full(T.shape[1], 6.0), np.full(T.shape[1], 15.0)
        ffmc = ffmc_step(ffmc, T[t], H[t], Wk[t], r[t])
        dmc = dmc_step(dmc, T[t], H[t], r[t], day.month)
        dc = dc_step(dc, T[t], r[t], day.month)
        isi, bui, fwi = isi_bui_fwi(ffmc, dmc, dc, Wk[t])
        G = np.where(r[t] > 3, 0, G + np.where(T[t] > 0, T[t] * (T[t] - Td[t]), 0))
        for k, v in zip(["ffmc", "dmc", "dc", "isi", "bui", "fwi", "nesterov"], [ffmc, dmc, dc, isi, bui, fwi, G]):
            out[k][t] = v
    return out, H


def days_since(cond):
    out = np.zeros(cond.shape, dtype=np.float32)
    run = np.zeros(cond.shape[1], dtype=np.float32)
    for t in range(cond.shape[0]):
        run = np.where(cond[t], 0, np.minimum(run + 1, 60))
        out[t] = run
    return out


def climatology(F):
    """Share of days with fire in this cell and calendar month, over earlier years only."""
    yrs, months = DAYS.year.values, DAYS.month.values
    fire = (F > 0).astype(np.float32)
    out = np.full(F.shape, np.nan, dtype=np.float32)
    for y in range(2013, 2026):
        past = yrs < y
        for m in range(1, 13):
            sel = past & (months == m)
            out[(yrs == y) & (months == m)] = fire[sel].mean(axis=0)
    return out


if __name__ == "__main__":
    cells = pd.read_csv(DATA / "grid_cells.csv")
    W = load_weather(cells)
    F, P = load_fires(cells)
    print("fire detections in cells:", int(F.sum()))
    IDX, H = indices(W)
    fire = (F > 0).astype(np.float32)
    feats = {
        "tmax": W["T2M_MAX"], "tmin": W["T2M_MIN"], "tmean": W["T2M"], "rh_afternoon": H, "rh_mean": W["RH2M"],
        "wind": W["WS2M"], "wind_max": W["WS2M_MAX"], "rain": W["PRECTOTCORR"],
        "soil_top": W["GWETTOP"], "soil_root": W["GWETROOT"],
        "rain_7d": rolling_sum(W["PRECTOTCORR"], 7), "rain_30d": rolling_sum(W["PRECTOTCORR"], 30),
        "days_since_rain": days_since(W["PRECTOTCORR"] > 2),
        **IDX,
        "fires_today": F, "frp_today": P,
        "fire_days_3d": rolling_sum(fire, 3), "fire_days_7d": rolling_sum(fire, 7),
        "fire_days_30d": rolling_sum(fire, 30), "fire_days_365d": rolling_sum(fire, 365),
        "near_fires_today": neighbours(F, cells, 3), "near_fire_days_7d": neighbours(rolling_sum(fire, 7), cells, 3),
        "region_fire_days_7d": neighbours(rolling_sum(fire, 7), cells, 7),
        "clim_month": climatology(F),
    }
    target = np.vstack([fire[1:], np.zeros((1, fire.shape[1]), np.float32)])
    recent = rolling_sum(fire, 3)
    new_target = target * (recent == 0)

    season = np.isin(DAYS.month, range(3, 12)) & (DAYS < "2025-12-31")
    t_idx = np.where(season)[0]
    nC = len(cells)
    df = pd.DataFrame({
        "date": np.repeat(DAYS[t_idx].values, nC),
        "cell": np.tile(cells.cell.values, len(t_idx)),
        "lat": np.tile(cells.lat.values, len(t_idx)).astype(np.float32),
        "lon": np.tile(cells.lon.values, len(t_idx)).astype(np.float32),
    })
    for k, A in feats.items():
        df[k] = A[t_idx].reshape(-1).astype(np.float32)
    doy = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    df["dow_tomorrow"] = ((df.date.dt.dayofweek + 1) % 7).astype(np.int8)
    df["fire_tomorrow"] = target[t_idx].reshape(-1).astype(np.int8)
    df["new_fire_tomorrow"] = new_target[t_idx].reshape(-1).astype(np.int8)
    df["fires_tomorrow_count"] = np.vstack([F[1:], np.zeros((1, nC), np.float32)])[t_idx].reshape(-1)
    df.to_parquet(DATA / "dataset.parquet", index=False)
    print(df.shape, "fire-tomorrow rate", df.fire_tomorrow.mean().round(4), "new", df.new_fire_tomorrow.mean().round(4))
