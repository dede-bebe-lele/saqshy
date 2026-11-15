"""Turn the test results into docs/data.json (website) and video/video_data.js (animation)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RES, DATA = ROOT / "results", ROOT / "data"

# approximate centres of Kazakhstan's regions, used only to name an area in the alert text
REGIONS = [
    ("abai", 48.6, 79.6, "Abai region", "Абай облысы", "Абайская область"),
    ("east", 49.4, 84.4, "East Kazakhstan", "Шығыс Қазақстан облысы", "Восточно-Казахстанская область"),
    ("pavlodar", 52.0, 76.5, "Pavlodar region", "Павлодар облысы", "Павлодарская область"),
    ("karaganda", 48.6, 74.5, "Karaganda region", "Қарағанды облысы", "Карагандинская область"),
    ("ulytau", 47.6, 67.3, "Ulytau region", "Ұлытау облысы", "область Ұлытау"),
    ("akmola", 51.6, 70.2, "Akmola region", "Ақмола облысы", "Акмолинская область"),
    ("north", 54.0, 69.5, "North Kazakhstan", "Солтүстік Қазақстан облысы", "Северо-Казахстанская область"),
    ("kostanay", 51.6, 63.5, "Kostanay region", "Қостанай облысы", "Костанайская область"),
    ("aktobe", 48.6, 58.5, "Aktobe region", "Ақтөбе облысы", "Актюбинская область"),
    ("west", 50.4, 51.2, "West Kazakhstan", "Батыс Қазақстан облысы", "Западно-Казахстанская область"),
    ("atyrau", 47.1, 52.6, "Atyrau region", "Атырау облысы", "Атырауская область"),
    ("mangystau", 44.2, 53.6, "Mangystau region", "Маңғыстау облысы", "Мангистауская область"),
    ("kyzylorda", 45.0, 64.0, "Kyzylorda region", "Қызылорда облысы", "Кызылординская область"),
    ("turkistan", 43.4, 68.2, "Turkistan region", "Түркістан облысы", "Туркестанская область"),
    ("jambyl", 44.4, 72.0, "Jambyl region", "Жамбыл облысы", "Жамбылская область"),
    ("almaty", 44.0, 77.2, "Almaty region", "Алматы облысы", "Алматинская область"),
    ("jetisu", 45.4, 79.6, "Jetisu region", "Жетісу облысы", "область Жетісу"),
]
LABELS = {
    "fire_days_365d": "fire days here in the last year", "clim_month": "usual fire rate this month",
    "fire_days_30d": "fire days here, last 30 days", "fire_days_7d": "fire days here, last 7 days",
    "near_fire_days_7d": "fires in neighbouring squares, 7 days", "region_fire_days_7d": "fires in the wider area, 7 days",
    "fires_today": "fires here today", "near_fires_today": "fires next door today", "frp_today": "fire intensity today",
    "fire_days_3d": "fire days here, last 3 days", "doy_sin": "time of year", "doy_cos": "time of year (2)",
    "lat": "latitude", "lon": "longitude", "soil_top": "topsoil wetness", "soil_root": "root-zone soil wetness",
    "rh_afternoon": "afternoon humidity", "tmax": "max temperature", "wind_max": "max wind", "rain_30d": "rain, last 30 days",
    "dc": "FWI drought code", "dmc": "FWI duff moisture", "ffmc": "FWI fine-fuel moisture", "fwi": "FWI", "nesterov": "Nesterov index",
    "days_since_rain": "days since rain", "rain_7d": "rain, last 7 days", "dow_tomorrow": "day of week tomorrow",
    "tmin": "min temperature", "tmean": "mean temperature", "rh_mean": "mean humidity", "wind": "wind", "rain": "rain today",
    "isi": "FWI spread index", "bui": "FWI build-up index",
}


def classes(rank):
    return np.select([rank <= .10, rank <= .20, rank <= .35, rank <= .60], [4, 3, 2, 1], 0)


if __name__ == "__main__":
    cells = pd.read_csv(DATA / "grid_cells.csv")
    cidx = pd.Series(np.arange(len(cells)), index=cells.cell)
    p = pd.read_parquet(RES / "test_predictions.parquet")
    p["i"] = cidx[p.cell].values
    p = p.sort_values(["date", "i"])
    days = {}
    for d, g in p.groupby("date"):
        g = g.sort_values("i")
        r = "".join(map(str, classes(g.risk_rank.values)))
        wc = classes(g.fwi_rank.values)
        f = g[g.fire_tomorrow == 1]
        days[d.strftime("%Y-%m-%d")] = {"r": r, "wc": "".join(map(str, np.minimum(wc, 3))),
                                         "w": [int(x) for x in g.i.values[wc == 4]],
                                         "f": [[int(i), int(n)] for i, n in zip(f.i, f.fires_tomorrow_count)]}

    # Abai fire, evening of 7 June 2023
    ab = p[(p.date == "2023-06-07")]
    ab_cell = ab.loc[ab.fires_tomorrow_count.idxmax()]
    abai = {"date": "2023-06-07", "cell": [float(ab_cell.lat), float(ab_cell.lon)],
            "rank": float(ab_cell.risk_rank), "fwi_rank": float(ab_cell.fwi_rank), "risk": float(ab_cell.risk),
            "fires": int(ab_cell.fires_tomorrow_count)}
    nx = p[(p.date == "2023-06-08")]
    nf = nx[nx.fire_tomorrow == 1]
    nxc = nx[nx.cell == ab_cell.cell].iloc[0]
    abai.update({"date2": "2023-06-08", "rank2": float(nxc.risk_rank), "n2": int(len(nf)), "in2": int((nf.risk_rank <= .1).sum())})
    abai["text"] = (f"<b>Saqshy missed the start.</b> The evening before, the square where lightning started the fire (white ring) "
                    f"was only in Saqshy's riskiest <b>{round(abai['rank'] * 100)}%</b> of the country, outside the red zone "
                    f"(the Canadian FWI: {round(abai['fwi_rank'] * 100)}%). A lightning strike leaves no trace in yesterday's data. "
                    f"<b>But it knew where the fire would keep burning.</b> On the evening of 8 June the burning square was Saqshy's "
                    f"top {max(1, round(abai['rank2'] * 100))}%, and {abai['in2']} of the {abai['n2']} squares that burned on 9 June were in the red zone, "
                    "so crews and aircraft could be sent there first. Lightning forecasts are our next step.")
    # showcase day: 29 April 2025, a big spring outbreak in the northern steppe
    SHOW = "2025-04-29"
    sh = p[p.date == SHOW]
    shf = sh[sh.fire_tomorrow == 1]
    show = {"date": SHOW, "n": int(len(shf)), "saq": int((shf.risk_rank <= .1).sum()), "fwi": int((shf.fwi_rank <= .1).sum()),
            "recent": int((sh.fire_days_7d.rank(ascending=False, method="first", pct=True)[shf.index] <= .1).sum())}

    S = pd.read_csv(RES / "scores_all_fires.csv")
    SN = pd.read_csv(RES / "scores_new_fires.csv").set_index("method")
    scores = [{"method": r.method, "top10": r["fires caught in top 10%"], "top5": r["fires caught in top 5%"],
               "auc": r["ROC-AUC"], "ap": r["average precision"], "new_top10": SN.loc[r.method, "fires caught in top 10%"]}
              for _, r in S.iterrows()]
    X = json.load(open(RES / "extra_checks.json"))
    RN = "Recent fires + usual for month (fitted on 2021-22)"
    scores.insert(1, {"method": RN, "top10": X["all_fires_top10"][RN], "top5": X["all_fires_top5"][RN], "auc": X["rule_auc"],
                      "ap": None, "new_top10": X["new_fires_top10"][RN]})
    for sc in scores:
        if sc["method"] == "Saqshy, weather only":
            sc["method"] = "Saqshy, weather + location only"
    checks = {"rule": X["all_fires_top10"][RN], "ci_rule": X["bootstrap_95ci_points"]["Saqshy minus fair rule"],
              "ci_recent": X["bootstrap_95ci_points"]["Saqshy minus recent fires"],
              "days": X["days_with_5plus_fire_squares"], "beats_recent": X["share_of_those_days_saqshy_beats_recent"],
              "fwi_fuel": X["all_fires_top10"]["Canadian FWI, only squares that burned in the last year"],
              "season": [r for r in X["by_season"] if r["season"].startswith("June")]}
    curve = pd.read_csv(RES / "tradeoff_curve.csv").to_dict("records")
    by = pd.read_csv(RES / "scores_by_year.csv")
    by_year = []
    for y, g in by.groupby("year"):
        g = g.set_index("method")["caught in top 10%"]
        by_year.append({"year": int(y), "fires": int(by[by.year == y].fires.iloc[0]), "saqshy": g["Saqshy"],
                        "fwi": g["Canadian FWI (used in Europe)"], "recent": g["Recent fires (last 7 days)"]})
    reg = pd.read_csv(RES / "fairness_by_region.csv")
    regions_score = [{"region": r.region, "fires": int(r.fires), "saqshy": r.Saqshy, "fwi": r["Canadian FWI"],
                      "recent": r["Recent fires"]} for _, r in reg.iterrows()]
    imp = pd.read_csv(RES / "feature_importance.csv").head(10)
    importance = [{"label": LABELS.get(f, f), "v": float(v)} for f, v in zip(imp.feature, imp.importance)]

    # where it fails
    me = next(s for s in scores if s["method"] == "Saqshy")
    big = p[(p.fires_tomorrow_count >= 50) & (p.risk_rank > 0.10)].sort_values("fires_tomorrow_count", ascending=False)
    big_all = p[p.fires_tomorrow_count >= 50]
    fails = [
        f"<b>New fires are much harder.</b> For squares with no fire in the three days before, only {me['new_top10']:.0%} fall in the red 10% "
        f"(all fires: {me['top10']:.0%}), about the same as the plain \"usual for this month\" map. A lightning strike or a match leaves no trace in yesterday's data.",
        f"<b>Some big fires come from nowhere.</b> Of {len(big_all)} square-days with 50+ detections, {len(big)} were outside the red zone the evening before"
        + (f", e.g. {big.iloc[0].date + pd.Timedelta(days=1):%d %b %Y} near {big.iloc[0].lat:.1f}°N {big.iloc[0].lon:.1f}°E ({int(big.iloc[0].fires_tomorrow_count)} detections; it was ranked in the top {big.iloc[0].risk_rank:.0%})." if len(big) else "."),
        "<b>Satellites miss small and cloudy fires</b>, and some detections are farmers burning stubble, not wildfires. Saqshy predicts what the satellite will see.",
        f"<b>Uneven by region:</b> it works best where fires repeat every year; see the region table.",
        "<b>Timing.</b> Satellite dates are in UTC, so a live version would run before dawn (~05:00 local), after the night satellite pass, and would need a weather forecast because NASA POWER arrives a few days late.",
        "<b>Not tested with rangers yet.</b> The users and the alert are my design; no forestry office has used it.",
    ]

    region_of = []
    for la, lo in zip(cells.lat, cells.lon):
        d = [((la - r[1]) ** 2 + ((lo - r[2]) * 0.66) ** 2) for r in REGIONS]
        region_of.append(int(np.argmin(d)))
    geo = json.load(open(DATA / "kaz.geo.json"))["features"][0]["geometry"]["coordinates"][0]

    # a showcase day for the hero: many fires and a typical catch rate
    st = pd.DataFrame([{"d": d, "n": len(v["f"]), "c": sum(1 for i, _ in v["f"] if v["r"][i] == "4")} for d, v in days.items()])
    st["share"] = st.c / st.n.clip(lower=1)
    cand = st[(st.n >= 40)]
    hero = cand.iloc[(cand.share - me["top10"]).abs().argsort().iloc[0]].d if len(cand) else "2023-06-07"

    out = {"cells": [[float(a), float(b)] for a, b in zip(cells.lat, cells.lon)], "outline": geo, "days": days,
           "abai": abai, "scores": scores, "curve": curve, "by_year": by_year, "regions_score": regions_score,
           "importance": importance, "fails": fails, "checks": checks, "n_test": int(len(p)), "hero_day": SHOW, "start_day": SHOW, "show": show,
           "cell_region": region_of, "regions": [{"en": r[3], "kk": r[4], "ru": r[5]} for r in REGIONS]}
    (ROOT / "docs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "docs" / "data.json", "w"), separators=(",", ":"))
    print("data.json", round((ROOT / "docs" / "data.json").stat().st_size / 1e6, 2), "MB; hero day", hero)
    print("abai", abai)

    # ---- data for the video
    import glob
    fz = pd.concat([pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "type"]) for f in glob.glob(str(DATA / "firms" / "viirs_2025.csv"))])
    fz = fz[(fz.acq_date == "2025-04-30") & (fz.type == 0) & (fz.confidence != "l")]
    fz = fz.sample(min(400, len(fz)), random_state=0)
    vd = {"cells": [[float(a), float(b), int(c)] for (a, b), c in zip(zip(cells.lat, cells.lon), days[SHOW]["r"])],
          "fires": [[float(a), float(b)] for a, b in zip(fz.latitude, fz.longitude)],
          "show": show, "fwi_red": days[SHOW]["w"],
          "results": [["Recent\nfires", next(s for s in scores if s['method'].startswith('Recent'))["top10"], "#9aa3b5"],
                      ["Nesterov\nindex", next(s for s in scores if s['method'].startswith('Nesterov'))["top10"], "#c7b28a"],
                      ["Canadian\nFWI", next(s for s in scores if s['method'].startswith('Canadian'))["top10"], "#c7b28a"],
                      ["Saqshy", me["top10"], "#ff7a1a"]]}
    (ROOT / "video" / "video_data.js").write_text("const DATA=" + json.dumps(vd) + ";const OUTLINE=" + json.dumps(geo) + ";")
    print("video data written")
