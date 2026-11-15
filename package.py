"""Copy everything that belongs in the public GitHub repo into ../saqshy_github (every file < 25 MB,
so it can be uploaded by drag-and-drop on github.com)."""
import shutil
from pathlib import Path

SRC = Path(__file__).parent
DST = SRC.parent / "saqshy_github"
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir()

FILES = ["README.md", "saqshy.ipynb", "requirements.txt", "get_fires.sh", "get_weather.py", "pack_data.py",
         "fire_indices.py", "build_dataset.py", "train_eval.py", "make_figures.py", "make_site.py", "build_notebook.py", "extra_checks.py",
         "package.py", "data/grid_cells.csv", "data/kaz.geo.json",
         "results/scores_all_fires.csv", "results/scores_new_fires.csv", "results/scores_big_fires.csv",
         "results/scores_by_year.csv", "results/fairness_by_region.csv", "results/feature_importance.csv",
         "results/tradeoff_curve.csv", "results/extra_checks.json", "results/scores_by_season.csv", "results/config.json", "results/saqshy_model.joblib",
                  "figures/tradeoff.png", "figures/showcase_2025-04-29.png",
         "docs/index.html", "docs/data.json",
         "video/anim.html", "video/script.json", "video/timing.json", "video/timing.js", "video/video_data.js",
         "video/make_voice.py", "video/render.py", "video/saqshy_demo.mp4"]
for f in FILES:
    (DST / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / f, DST / f)
for f in sorted((SRC / "data" / "packed").glob("*.parquet")):
    (DST / "data" / "packed").mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, DST / "data" / "packed" / f.name)
for f in sorted((SRC / "video" / "audio").glob("*.mp3")):
    (DST / "video" / "audio").mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, DST / "video" / "audio" / f.name)
(DST / "docs" / ".nojekyll").write_text("")
(DST / ".gitignore").write_text("data/dataset.parquet\ndata/weather/\ndata/firms/\n__pycache__/\n.ipynb_checkpoints/\nvideo/voice.wav\nvideo/still_*.png\nresults/test_predictions.parquet\n")
files = [p for p in DST.rglob("*") if p.is_file()]
big = max(files, key=lambda p: p.stat().st_size)
print(len(files), "files,", round(sum(p.stat().st_size for p in files) / 1e6, 1), "MB; largest:", big.name, round(big.stat().st_size / 1e6, 1), "MB")
