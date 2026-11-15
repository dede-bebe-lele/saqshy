"""Build saqshy.ipynb, the notebook judges open. It rebuilds the dataset from the packed NASA files
in data/packed, retrains Saqshy and re-runs every test (about 5-10 minutes in Colab).

Run:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace saqshy.ipynb
"""
import nbformat as nbf

REPO = "https://github.com/dede-bebe-lele/saqshy"
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t.strip()))


def code(t):
    cells.append(nbf.v4.new_code_cell(t.strip()))


md(r"""
# Saqshy · Сақшы: tomorrow's wildfire risk map for Kazakhstan

**Track:** AI + Climate · **Team:** Dariya

**One-sentence pitch:** Saqshy helps rangers and district emergency services in Kazakhstan decide where to patrol tomorrow, by taking tonight's NASA satellite fire detections and weather for 1,326 squares of the country and giving back tomorrow's fire risk for every square, with the riskiest 10% as a red zone and a Kazakh/Russian alert.

*This is a prototype, not an official warning service.*

**How to run:** Runtime → Run all. It takes about 5–10 minutes in Colab. It rebuilds the dataset from the packed NASA files in `data/packed/`, retrains the model and re-runs every test. No API keys are needed. To download the raw data again, run `get_fires.sh` and `get_weather.py`.

**Website with the replay of every test day:** https://dede-bebe-lele.github.io/saqshy/
""")

code(r"""
# Setup (Colab): get the code and data
import os
if not os.path.exists("train_eval.py"):
    !git clone -q REPO_URL saqshy
    %cd saqshy
    !pip -q install scikit-learn pandas pyarrow matplotlib joblib scipy
import glob
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 200)
""".replace("REPO_URL", REPO))

md(r"""
## 1. The data

* **NASA FIRMS, VIIRS S-NPP (375 m):** every active fire the satellite saw over Kazakhstan from 2012 to 2025. We keep vegetation fires (`type` 0) and drop low-confidence points.
* **NASA POWER:** daily weather for each of the 1,326 squares: temperature, humidity, wind, rain, and soil wetness.

Let's look at the fires first.
""")

code(r"""
fires = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/packed/fires_*.parquet"))])
fires["date"] = pd.to_datetime(fires.acq_date)
print(f"{len(fires):,} vegetation-fire detections, 2012-2025")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
fires.groupby(fires.date.dt.year).size().plot.bar(ax=ax[0], color="#e8590c", title="Detections per year")
fires.groupby(fires.date.dt.month).size().plot.bar(ax=ax[1], color="#1c7ed6", title="Detections per month (all years)")
plt.tight_layout(); plt.show()
""")

code(r"""
# where fires happen: one dot per detection in 2023 (a sample)
s = fires[fires.date.dt.year == 2023].sample(40000, random_state=0)
plt.figure(figsize=(11, 4.5))
plt.scatter(s.longitude, s.latitude, s=1, c="#e8590c", alpha=.3)
plt.title("Fires seen by the satellite in 2023: the northern grain belt, the eastern forests, the south")
plt.gca().set_aspect(1 / 0.67); plt.axis("off"); plt.show()
""")

md(r"""
## 2. Build the dataset: one row per square per evening

`build_dataset.py` makes a row for every square and every evening of the fire season (March–November). Each row has only what is known **on the evening of day t**:

* today's weather, rain over 7 and 30 days, days since rain;
* the two standard fire-danger indices computed from that weather: the **Canadian Fire Weather Index** (used by Europe's EFFIS) and the **Nesterov index** (official in Russia);
* fires in the square today and over the last 3, 7, 30 and 365 days; fires in neighbouring squares;
* the square's usual fire rate for this month, using **earlier years only**;
* position and time of year.

The target is whether the satellite sees a fire in the square **tomorrow**.
""")

code(r"""
!python -W ignore build_dataset.py
df = pd.read_parquet("data/dataset.parquet")
print(df.shape)
df.sample(5, random_state=1)[["date", "cell", "tmax", "rh_afternoon", "wind", "rain_7d", "fwi", "fires_today", "fire_days_365d", "clim_month", "fire_tomorrow"]]
""")

md(r"""
## 3. Train on 2012–2022, test on 2023–2025

`train_eval.py` trains gradient-boosted trees (`HistGradientBoostingClassifier`) on 2012–2022 and tests on **every day of three fire seasons it never saw**. Baselines:

1. **Recent fires:** rank squares by how many days they burned in the last week. This is what a dispatcher can do without AI.
2. **Usual for this month:** the square's historical fire rate.
3. **Nesterov index** and 4. **Canadian FWI:** the classic weather-only fire-danger indices.

**Main metric:** if rangers can watch only the riskiest 10% of the country tomorrow, what share of tomorrow's fires happen inside it?
""")

code(r"""
!python -W ignore train_eval.py
""")

code(r"""
res = pd.read_csv("results/scores_all_fires.csv")
res[["method", "fires caught in top 10%", "fires caught in top 5%", "ROC-AUC"]].round(3)
""")

code(r"""
!python make_figures.py
display(Image("figures/tradeoff.png"))
""")

md(r"""
**Takeaways**

* Saqshy puts **about 69% of tomorrow's fires inside 10% of the country**, against 62% for the best simple rule (recent fires).
* The weather-only indices (FWI, Nesterov) do badly here, at 11%. Their riskiest squares are the hot southern deserts, which have little to burn. Saqshy learns where the fuel is and where fires really start.
""")

md(r"""
### Stress tests: is the lead real?

Most fires tomorrow are where fires burn today, so persistence is strong. `extra_checks.py` builds the fairest simple rule I could (recent fires + usual rate for the month, a logistic regression fitted on **2021-2022 only**), puts a bootstrap interval on Saqshy's lead, splits out the June-September wildfire season (fewer stubble burns), and gives the FWI a fairer chance (only squares that burned in the last year).
""")

code(r"""
!python -W ignore extra_checks.py > /dev/null
import json
X = json.load(open("results/extra_checks.json"))
print("Top-10% catch:", {k: round(v, 3) for k, v in X["all_fires_top10"].items()})
print("New fires, top-10% catch:", {k: round(v, 3) for k, v in X["new_fires_top10"].items()})
print("95% bootstrap interval for Saqshy's lead (points):", X["bootstrap_95ci_points"])
print("Days (5+ fire squares) where Saqshy beats recent fires:", round(X["share_of_those_days_saqshy_beats_recent"], 3))
pd.read_csv("results/scores_by_season.csv").round(3)
""")

md(r"""
**Honest reading:** against the fair rule the lead is small (about +3 points, interval +2.3 to +3.4) but holds in the wildfire season and is bigger on new fires (42% vs 36%).
""")

md(r"""
## 4. A real evening: 29 April 2025 (test season)
""")

code(r"""
p = pd.read_parquet("results/test_predictions.parquet")
day = p[p.date == "2025-04-29"]
f = day[day.fire_tomorrow == 1]
recent_rank = day.fire_days_7d.rank(ascending=False, method="first", pct=True)
print(f"Squares with fire on 30 April: {len(f)}")
print(f"  inside Saqshy's red 10%:       {(f.risk_rank <= .1).sum()}")
print(f"  inside 'recent fires' red 10%: {(recent_rank[f.index] <= .1).sum()}")
print(f"  inside the FWI's red 10%:      {(f.fwi_rank <= .1).sum()}")
display(Image("figures/showcase_2025-04-29.png"))
""")

md(r"""
## 5. Does it work everywhere, and every year?
""")

code(r"""
print(pd.read_csv("results/scores_by_year.csv").round(3).to_string(index=False))
print()
print(pd.read_csv("results/fairness_by_region.csv").round(3).to_string(index=False))
""")

md(r"""
Saqshy beats the simple rule in every year and every region. It is weaker in the north (grain belt) and the east (forests), where fires are less regular than in the south.
""")

md(r"""
## 6. Where it fails
""")

code(r"""
# The Abai fire: lightning, 8 June 2023
ab = p[(p.date == "2023-06-07") & (p.cell == "50.75_80.75")].iloc[0]
nx = p[(p.date == "2023-06-08") & (p.cell == "50.75_80.75")].iloc[0]
print(f"Evening of 7 June: the square was in Saqshy's riskiest {ab.risk_rank:.0%} (outside the red zone). FWI: {ab.fwi_rank:.0%}")
print(f"Evening of 8 June, once burning: top {nx.risk_rank:.1%}")
new = pd.read_csv("results/scores_new_fires.csv")
print()
print("New fires only (no fire in the square in the 3 days before):")
print(new[["method", "fires caught in top 10%"]].round(3).to_string(index=False))
""")

md(r"""
* **It missed the start of the Abai fire.** A lightning strike leaves no trace in yesterday's data. Once the fire was burning, the square went straight into the top 1%.
* **New fires are much harder.** About 43% fall in the red zone, barely better than the usual-for-this-month rule (42%).
* **The satellite is the ground truth.** It misses small fires and fires under cloud, and some detections are farmers burning stubble.
* **Squares are coarse** (about 2,000 km²) because that is the resolution of the weather data.

**Next steps:** add lightning forecasts and land cover; run it live next spring with the previous day's FIRMS fires and send the alert to rangers through a Telegram bot.
""")

code(r"""
imp = pd.read_csv("results/feature_importance.csv").head(12)
imp.plot.barh(x="feature", y="importance", legend=False, figsize=(7, 4), color="#e8590c", title="Permutation importance (test data)")
plt.gca().invert_yaxis(); plt.show()
""")

md(r"""
## Sources
NASA FIRMS (VIIRS S-NPP active fires) · NASA POWER daily weather · Natural Earth border (world.geo.json) · Van Wagner (1987) Canadian FWI · Nesterov index · scikit-learn, pandas, NumPy, SciPy, matplotlib · AI tools: Claude (Anthropic) helped with brainstorming, data search and code. Full list in the README.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "saqshy.ipynb")
print("saqshy.ipynb written,", len(cells), "cells")
