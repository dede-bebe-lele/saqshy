"""Render anim.html frame by frame with headless Chromium and join it with the voiceover.

Output: video/saqshy_demo.mp4 (1920x1080, 30 fps)
"""
import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
FPS = 30
PAD, START = 0.5, 0.4

timing = json.load(open(HERE / "timing.json"))
(HERE / "timing.js").write_text("const TIMING=" + json.dumps(timing) + ";")

# voice track: each scene's audio placed at its start time
starts, acc = [], START
for s in timing:
    starts.append(acc)
    acc += s["dur"] + PAD
total = acc + 1.2
inputs, filt = [], []
for i, (s, st) in enumerate(zip(timing, starts)):
    inputs += ["-i", str(HERE / "audio" / f"{s['id']}.mp3")]
    filt.append(f"[{i}:a]adelay={int(st * 1000)}|{int(st * 1000)}[a{i}]")
mix = "".join(f"[a{i}]" for i in range(len(timing)))
filt.append(f"{mix}amix=inputs={len(timing)}:normalize=0,apad=whole_dur={total}[out]")
if len(sys.argv) == 1:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-t", f"{total}", str(HERE / "voice.wav")], check=True)

only = float(sys.argv[1]) if len(sys.argv) > 1 else None  # render one still at time t, for checking
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1920, "height": 1080})
    page.goto((HERE / "anim.html").as_uri())
    page.evaluate("window.READY")
    page.wait_for_timeout(800)
    if only is not None:
        page.evaluate(f"render({only})")
        page.screenshot(path=str(HERE / f"still_{only:.1f}.png"))
        sys.exit()
    n = int(total * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                           "-i", str(HERE / "voice.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-preset", "medium", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           str(HERE / "saqshy_demo.mp4")], stdin=subprocess.PIPE)
    for f in range(n):
        page.evaluate(f"render({f / FPS})")
        ff.stdin.write(page.screenshot(type="jpeg", quality=92))
        if f % 300 == 0:
            print(f"frame {f}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    b.close()
print("done", round(total, 1), "s")
