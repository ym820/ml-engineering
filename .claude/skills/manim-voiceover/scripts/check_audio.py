"""Flag narration lines that are silent in a rendered video.

Usage (run with the manim env python, which has numpy + soundfile):
    ~/.local/share/uv/tools/manim/bin/python check_audio.py VIDEO.mp4 [vo/timeline.tsv]

timeline.tsv is written by the template's tear_down(); render the video and the
timeline in the same run so the timestamps match.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

video = Path(sys.argv[1])
timeline = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("vo/timeline.tsv")

with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                    "-ar", "16000", tmp.name], check=True)
    x, sr = sf.read(tmp.name)

silent = []
for line in timeline.read_text().splitlines():
    key, t, d = line.split("\t")
    seg = x[int(float(t) * sr):int((float(t) + float(d)) * sr)]
    if seg.size == 0 or np.sqrt(np.mean(seg ** 2)) < 0.01:
        silent.append(key)

n = len(timeline.read_text().splitlines())
print(f"{n} lines checked, silent: {silent or 'none'}")
sys.exit(1 if silent else 0)
