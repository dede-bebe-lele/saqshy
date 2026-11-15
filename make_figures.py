"""Figures for the README: the trade-off curve and the 29 April 2025 showcase map."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BG, INK, MUTED, LINE = "#07080b", "#eef0f6", "#8a90a6", "#262a38"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "font.family": "monospace",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": MUTED, "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": MUTED})

COL = {"Saqshy": "#b6ff3b", "Recent fires (last 7 days)": "#e6e8ef", "Usual for this month": "#9b7bff",
       "Canadian FWI (used in Europe)": "#ff4f9a", "Nesterov index (used in Russia)": "#ff9ec7",
       "Saqshy, weather only": "#7fd420"}

c = pd.read_csv(ROOT / "results" / "tradeoff_curve.csv")
fig, ax = plt.subplots(figsize=(8, 5.2))
for m, g in c.groupby("method"):
    ax.plot(g.top_pct, g.caught * 100, color=COL[m], lw=3.2 if m == "Saqshy" else 1.8, label=m)
    if m == "Saqshy": ax.annotate(f"{g.caught.iloc[-1] * 100:.0f}%", (g.top_pct.iloc[-1], g.caught.iloc[-1] * 100), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=10, color="#b6ff3b")
ax.axvline(10, color="#ff3b5c", ls="--", lw=1)
ax.text(10.5, 4, "the red zone (10%)", color="#ff3b5c", fontsize=10)
ax.set_xlabel("% of Kazakhstan watched (riskiest squares first)")
ax.set_ylabel("% of tomorrow's fires inside that area")
ax.set_title("Test seasons 2023-2025, never seen in training", loc="left", fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(0, 53)
ax.grid(color="#1a1d27")
ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
fig.tight_layout()
fig.savefig(FIG / "tradeoff.png", dpi=160)

# showcase map
D = json.load(open(ROOT / "docs" / "data.json"))
day = D["days"][D["show"]["date"]]
cells = D["cells"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
pal = ["#2b3b66", "#4e5d8c", "#f2c14e", "#ff8c2b", "#ff3d2e"]
for ax, title, red in [(axs[0], "Saqshy's red zone", [i for i, ch in enumerate(day["r"]) if ch == "4"]),
                       (axs[1], "Canadian FWI's red zone", day["w"])]:
    red = set(red)
    xs = [lo for la, lo in cells]
    ys = [la for la, lo in cells]
    ax.scatter(xs, ys, c=["#ff3b5c" if i in red else "#2c2950" for i in range(len(cells))], marker="s", s=22, lw=0)
    fx = [cells[i][1] for i, _ in day["f"]]
    fy = [cells[i][0] for i, _ in day["f"]]
    caught = sum(1 for i, _ in day["f"] if i in red)
    ax.scatter(fx, fy, s=30, facecolor="#b6ff3b", edgecolor=BG, lw=.8, zorder=3)
    ax.set_title(f"{title}: {caught} of {len(day['f'])} fires inside", loc="left", fontsize=12)
    ax.set_aspect(1 / 0.67)
    ax.axis("off")
fig.suptitle("Evening of 29 April 2025 → fires seen by NASA satellites on 30 April (green)", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "showcase_2025-04-29.png", dpi=150)
print("figures saved")
