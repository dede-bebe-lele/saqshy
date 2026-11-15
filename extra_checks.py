"""Extra checks a sceptical judge would ask for (run after train_eval.py).

1. A fairer simple rule: "recent fires + usual for this month", a 4-input logistic regression fitted on
   2021-2022 only (never on the test years). Is Saqshy still better than that?
2. Uncertainty: bootstrap over test days -> 95% interval for Saqshy's lead.
3. How often does Saqshy beat "recent fires" on a single day?
4. Wildfire season (June-September) vs spring/autumn, when many detections are farmers burning stubble.
5. A fairer FWI: rank only squares that had any fire in the last year (somewhere with fuel).

Writes results/extra_checks.json and results/scores_by_season.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from train_eval import caught_in_top

ROOT = Path(__file__).parent
RES = ROOT / "results"
RULE = ["fire_days_7d", "clim_month", "near_fire_days_7d", "fires_today_any"]

cols = ["date", "cell", "fire_days_7d", "clim_month", "near_fire_days_7d", "fires_today", "fire_days_365d",
        "fwi", "fire_tomorrow", "new_fire_tomorrow"]
df = pd.read_parquet(ROOT / "data" / "dataset.parquet", columns=cols)
df["fires_today_any"] = (df.fires_today > 0).astype(float)
df["year"] = df.date.dt.year
val = df[df.year.isin([2021, 2022])]
test = df[df.year >= 2023].reset_index(drop=True)
pred = pd.read_parquet(RES / "test_predictions.parquet", columns=["date", "cell", "risk"])
test = test.merge(pred, on=["date", "cell"], how="left")
assert test.risk.notna().all()

# 1. the fair simple rule, fitted on 2021-2022
lr = LogisticRegression(max_iter=1000).fit(val[RULE].fillna(0), val.fire_tomorrow)
test["rule"] = lr.predict_proba(test[RULE].fillna(0))[:, 1]
test["recent"] = test.fire_days_7d + 0.01 * test.near_fire_days_7d
# 5. FWI only where something burned in the last year (others pushed to the bottom)
test["fwi_fuel"] = np.where(test.fire_days_365d > 0, test.fwi.fillna(-1), -2)

methods = {"Saqshy": "risk", "Recent fires + usual for month (fitted on 2021-22)": "rule",
           "Recent fires (last 7 days)": "recent", "Canadian FWI, only squares that burned in the last year": "fwi_fuel",
           "Canadian FWI": "fwi"}
out = {"rule_inputs": RULE, "rule_coefficients": dict(zip(RULE, lr.coef_[0].round(3).tolist()))}
out["all_fires_top10"] = {m: caught_in_top(test, test[c].fillna(-1).values, "fire_tomorrow") for m, c in methods.items()}
out["all_fires_top5"] = {m: caught_in_top(test, test[c].fillna(-1).values, "fire_tomorrow", .05) for m, c in methods.items()}
out["rule_auc"] = float(roc_auc_score(test.fire_tomorrow, test.rule))
out["new_fires_top10"] = {m: caught_in_top(test, test[c].fillna(-1).values, "new_fire_tomorrow") for m, c in methods.items()}
out["new_fires_top5"] = {m: caught_in_top(test, test[c].fillna(-1).values, "new_fire_tomorrow", .05) for m, c in methods.items()}

# per-day hit counts for the bootstrap
for c in ["risk", "rule", "recent"]:
    test["top_" + c] = test.groupby("date")[c].rank(ascending=False, method="first", pct=True) <= .10
days = test[test.fire_tomorrow == 1].groupby("date").agg(n=("fire_tomorrow", "size"), risk=("top_risk", "sum"),
                                                         rule=("top_rule", "sum"), recent=("top_recent", "sum"))
rng = np.random.default_rng(0)
boot = {"Saqshy minus recent fires": [], "Saqshy minus fair rule": []}
idx = np.arange(len(days))
for _ in range(2000):
    s = days.iloc[rng.choice(idx, len(idx))].sum()
    boot["Saqshy minus recent fires"].append((s.risk - s.recent) / s.n)
    boot["Saqshy minus fair rule"].append((s.risk - s.rule) / s.n)
out["bootstrap_95ci_points"] = {k: [round(100 * np.percentile(v, 2.5), 1), round(100 * np.percentile(v, 97.5), 1)]
                                for k, v in boot.items()}
busy = days[days.n >= 5]
out["days_with_5plus_fire_squares"] = int(len(busy))
out["share_of_those_days_saqshy_beats_recent"] = float((busy.risk > busy.recent).mean())
out["share_of_those_days_saqshy_ties_recent"] = float((busy.risk == busy.recent).mean())
out["share_of_those_days_saqshy_beats_rule"] = float((busy.risk > busy.rule).mean())

# 4. season split
season = np.where(test.date.dt.month.between(6, 9), "June-September (wildfire season)", "March-May + October-November")
rows = []
for sname in np.unique(season):
    g = test[season == sname]
    for m, c in methods.items():
        rows.append({"season": sname, "method": m, "fire square-days": int(g.fire_tomorrow.sum()),
                     "caught in top 10%": caught_in_top(g, g[c].fillna(-1).values, "fire_tomorrow"),
                     "new fires caught in top 10%": caught_in_top(g, g[c].fillna(-1).values, "new_fire_tomorrow")})
pd.DataFrame(rows).to_csv(RES / "scores_by_season.csv", index=False)
out["by_season"] = rows
json.dump(out, open(RES / "extra_checks.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "by_season"}, indent=1, default=float))
print(pd.DataFrame(rows).round(3).to_string(index=False))
