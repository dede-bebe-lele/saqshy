"""Train Saqshy on 2012-2022 and test it on three fire seasons it never saw (2023, 2024, 2025).

Baselines (no machine learning):
  * Recent fires   - cells that burned in the last 7 days (what a dispatcher sees today)
  * Usual for month - the cell's share of fire days in this month in earlier years
  * Nesterov index - the classic fire-danger index, official in Russia
  * Canadian FWI   - the Fire Weather Index used by the European/Global fire services

Main metric: if rangers can watch only the riskiest 10% of the country tomorrow, what share
of tomorrow's fires happen inside that area? ("fires caught in top 10%").
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

WEATHER = ["tmax", "tmin", "tmean", "rh_afternoon", "rh_mean", "wind", "wind_max", "rain", "soil_top", "soil_root",
           "rain_7d", "rain_30d", "days_since_rain"]
INDICES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "nesterov"]
FIRE = ["fires_today", "frp_today", "fire_days_3d", "fire_days_7d", "fire_days_30d", "fire_days_365d",
        "near_fires_today", "near_fire_days_7d", "region_fire_days_7d", "clim_month"]
PLACE_TIME = ["lat", "lon", "doy_sin", "doy_cos", "dow_tomorrow"]
FEATURES = WEATHER + INDICES + FIRE + PLACE_TIME
WEATHER_ONLY = WEATHER + INDICES + PLACE_TIME  # ablation: no fire history
NEG_KEEP = 0.15  # keep 15% of no-fire rows when training (weighted back), to save memory
TOP = 0.10

REGIONS = {  # rough natural regions, used to check the model works everywhere
    "North (grain belt)": lambda d: (d.lat >= 50.5) & (d.lon < 76),
    "East (Abai, Altai, forests)": lambda d: (d.lon >= 76) & (d.lat >= 47.5),
    "West & Caspian": lambda d: (d.lon < 58) & (d.lat < 50.5),
    "South & centre (deserts, mountains)": lambda d: ~((d.lat >= 50.5) & (d.lon < 76)) & ~((d.lon >= 76) & (d.lat >= 47.5)) & ~((d.lon < 58) & (d.lat < 50.5)),
}


def caught_in_top(df, score, target, top=TOP):
    """Share of target events that fall in the top `top` share of cells, ranking cells within each day."""
    r = df.assign(s=score).groupby("date")["s"].rank(ascending=False, method="first", pct=True)
    hit = df[target].values == 1
    return float((r.values[hit] <= top).mean())


def evaluate(df, scores, target="fire_tomorrow"):
    y = df[target].values
    rows = []
    for name, s in scores.items():
        s = np.nan_to_num(np.asarray(s, dtype=float), nan=-1)
        rows.append({
            "method": name,
            "fires caught in top 10%": caught_in_top(df, s, target),
            "fires caught in top 5%": caught_in_top(df, s, target, 0.05),
            "ROC-AUC": roc_auc_score(y, s),
            "average precision": average_precision_score(y, s),
        })
    return pd.DataFrame(rows)


def sample_train(df, seed=0):
    rng = np.random.default_rng(seed)
    keep = (df.fire_tomorrow == 1) | (rng.random(len(df)) < NEG_KEEP)
    d = df[keep]
    w = np.where(d.fire_tomorrow == 1, 1.0, 1 / NEG_KEEP)
    return d, w


def fit(d, w, cols):
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(d[cols], d.fire_tomorrow, sample_weight=w)
    return m


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    df["year"] = df.date.dt.year
    train = df[df.year <= 2022]
    test = df[df.year >= 2023].reset_index(drop=True)
    print("train rows", len(train), "fire rate", round(train.fire_tomorrow.mean(), 4))
    print("test rows", len(test), "fire rate", round(test.fire_tomorrow.mean(), 4))

    d, w = sample_train(train)
    model = fit(d, w, FEATURES)
    model_w = fit(d, w, WEATHER_ONLY)
    test["risk"] = model.predict_proba(test[FEATURES])[:, 1]
    test["risk_weather_only"] = model_w.predict_proba(test[WEATHER_ONLY])[:, 1]

    scores = {
        "Recent fires (last 7 days)": test.fire_days_7d + 0.01 * test.near_fire_days_7d,
        "Usual for this month": test.clim_month,
        "Nesterov index (used in Russia)": test.nesterov,
        "Canadian FWI (used in Europe)": test.fwi,
        "Saqshy, weather only": test.risk_weather_only,
        "Saqshy": test.risk,
    }
    res = evaluate(test, scores)
    res_new = evaluate(test, scores, "new_fire_tomorrow")
    test["big_fire_tomorrow"] = (test.fires_tomorrow_count >= 20).astype(np.int8)
    res_big = evaluate(test, scores, "big_fire_tomorrow")
    res.to_csv(RES / "scores_all_fires.csv", index=False)
    res_new.to_csv(RES / "scores_new_fires.csv", index=False)
    res_big.to_csv(RES / "scores_big_fires.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nALL FIRES\n", res.round(3).to_string(index=False))
    print("\nNEW FIRES (no fire in the cell in the last 3 days)\n", res_new.round(3).to_string(index=False))
    print("\nBIG FIRES (20+ detections tomorrow)\n", res_big.round(3).to_string(index=False))

    # trade-off curve: share of tomorrow's fires inside the riskiest k% of the map
    curve = []
    for name, s in scores.items():
        s = np.nan_to_num(np.asarray(s, dtype=float), nan=-1)
        r = test.assign(s=s).groupby("date")["s"].rank(ascending=False, method="first", pct=True).values
        hit, hit_new = test.fire_tomorrow.values == 1, test.new_fire_tomorrow.values == 1
        for k in [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50]:
            curve.append({"method": name, "top_pct": k, "caught": float((r[hit] <= k / 100).mean()),
                          "caught_new": float((r[hit_new] <= k / 100).mean())})
    pd.DataFrame(curve).to_csv(RES / "tradeoff_curve.csv", index=False)

    # by year
    by_year = []
    for y, g in test.groupby("year"):
        for name in ["Recent fires (last 7 days)", "Canadian FWI (used in Europe)", "Saqshy"]:
            s = np.nan_to_num(np.asarray(scores[name])[g.index], nan=-1)
            by_year.append({"year": y, "method": name, "fires": int(g.fire_tomorrow.sum()),
                            "caught in top 10%": caught_in_top(g, s, "fire_tomorrow"),
                            "new fires caught in top 10%": caught_in_top(g, s, "new_fire_tomorrow")})
    by_year = pd.DataFrame(by_year)
    by_year.to_csv(RES / "scores_by_year.csv", index=False)
    print("\nBY YEAR\n", by_year.round(3).to_string(index=False))

    # fairness across regions: rank within the whole country, then look at each region's fires
    rank = test.groupby("date")["risk"].rank(ascending=False, pct=True)
    rank_fwi = test.groupby("date")["fwi"].rank(ascending=False, pct=True)
    rank_rec = test.assign(s=scores["Recent fires (last 7 days)"]).groupby("date")["s"].rank(ascending=False, method="first", pct=True)
    reg = []
    for name, f in REGIONS.items():
        m = f(test).values & (test.fire_tomorrow.values == 1)
        mn = f(test).values & (test.new_fire_tomorrow.values == 1)
        reg.append({"region": name, "cells": int(test[f(test)].cell.nunique()), "fires": int(m.sum()),
                    "Saqshy": float((rank[m] <= TOP).mean()), "Canadian FWI": float((rank_fwi[m] <= TOP).mean()),
                    "Recent fires": float((rank_rec[m] <= TOP).mean()),
                    "Saqshy, new fires": float((rank[mn] <= TOP).mean())})
    reg = pd.DataFrame(reg)
    reg.to_csv(RES / "fairness_by_region.csv", index=False)
    print("\nBY REGION (share of the region's fires inside the national top 10%)\n", reg.round(3).to_string(index=False))

    # importance (on a sample of test rows)
    samp = test.sample(150_000, random_state=0)
    pi = permutation_importance(model, samp[FEATURES], samp.fire_tomorrow, scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(RES / "feature_importance.csv", index=False)
    print("\nIMPORTANCE\n", imp.head(12).round(4).to_string(index=False))

    test["risk_rank"] = rank.values
    test["fwi_rank"] = rank_fwi.values
    test[["date", "cell", "lat", "lon", "risk", "risk_rank", "fwi_rank", "fwi", "nesterov", "fire_days_7d", "fires_today",
          "fire_tomorrow", "new_fire_tomorrow", "fires_tomorrow_count"]].to_parquet(RES / "test_predictions.parquet", index=False)
    joblib.dump({"model": model, "features": FEATURES}, RES / "saqshy_model.joblib", compress=3)
    json.dump({"features": FEATURES, "train_years": "2012-2022", "test_years": "2023-2025", "top": TOP,
               "neg_keep": NEG_KEEP}, open(RES / "config.json", "w"), indent=1)
