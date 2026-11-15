"""Make the English voiceover with a free Microsoft Edge neural voice (edge-tts).

Reads video/script.json ([{"id":..., "text":...}, ...]), writes video/audio/{id}.mp3 and
video/timing.json with each scene's duration and word timings (for captions).
"""
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

HERE = Path(__file__).parent
VOICE = "en-US-AvaNeural"
RATE = "+4%"


async def speak(scene):
    out = HERE / "audio" / f"{scene['id']}.mp3"
    com = edge_tts.Communicate(scene["text"], VOICE, rate=RATE, boundary="WordBoundary")
    words = []
    with open(out, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"t": chunk["offset"] / 1e7, "d": chunk["duration"] / 1e7, "w": chunk["text"]})
    toks, k = scene["text"].split(), 0
    norm = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())
    for w in words:
        for k2 in range(k, min(k + 4, len(toks))):
            if norm(toks[k2]).startswith(norm(w["w"])) or norm(w["w"]).startswith(norm(toks[k2])):
                p = toks[k2][-1] if toks[k2][-1] in ".,:?;!" else ""
                w["p"], w["b"], k = p, bool(p), k2 + 1
                break
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)]))
    return {"id": scene["id"], "dur": dur, "words": words}


async def main():
    (HERE / "audio").mkdir(exist_ok=True)
    scenes = json.load(open(HERE / "script.json"))
    timing = [await speak(s) for s in scenes]
    json.dump(timing, open(HERE / "timing.json", "w"), indent=1)
    for t in timing:
        print(f"{t['id']:10s} {t['dur']:5.1f}s")
    print("total", round(sum(t["dur"] for t in timing), 1))


asyncio.run(main())
