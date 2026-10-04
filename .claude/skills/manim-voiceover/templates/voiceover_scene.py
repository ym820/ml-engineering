"""Narrated ManimCE scene template.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python voiceover_scene.py
2. Draft:  manim -ql voiceover_scene.py Video   (prints HOLD lines for frozen frames)
3. Check:  python scripts/check_audio.py <rendered.mp4>
4. Final:  manim -qh voiceover_scene.py Video
"""
import wave
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow shape animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

# One short line per visual step. Keys are referenced by say().
N = {
    "intro_a": "Here's something that surprised me the first time I saw it.",
    "intro_b": "Take this square, and watch what happens when we cut it into tiles.",
}


def is_text(anim):
    m = anim.mobject
    return isinstance(m, Text) or (isinstance(m, VGroup) and m.submobjects and
                                   all(isinstance(s, Text) for s in m.submobjects))


class Video(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.intro,):
            beat()
            self.wait(0.6)
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
            self.wait(0.3)

    def tear_down(self):
        (VO_DIR / "timeline.tsv").write_text(
            "".join(f"{k}\t{t:.3f}\t{d:.3f}\n" for k, t, d in self.timeline))

    def say(self, key, *steps, pad=0.25):
        """Speak line `key` while playing steps, stretched to span the line.

        A step is an animation or a list of animations played together.
        Text-only steps play at normal speed (capped at TEXT_MAX); shape
        steps absorb the stretch.
        """
        path = VO_DIR / f"{key}.wav"
        with wave.open(str(path)) as w:
            dur = w.getnframes() / w.getframerate()
        steps = [[prepare_animation(a) for a in (s if isinstance(s, list) else [s])]
                 for s in steps]
        text = [all(is_text(a) for a in s) for s in steps]
        times = [max(a.run_time for a in s) for s in steps]
        times = [min(t, TEXT_MAX) if tx else t for t, tx in zip(times, text)]
        text_t = sum(t for t, tx in zip(times, text) if tx)
        other_t = sum(times) - text_t
        k = min(MAX_STRETCH, max(1, (dur - text_t) / other_t)) if other_t else 1
        t0 = self.renderer.time
        # not self.add_sound: it silently drops the sound while a cached
        # animation's skip flag is still set (e.g. after a repeated wait)
        self.renderer.file_writer.add_sound(str(path), t0)
        self.timeline.append((key, t0, dur))
        for s, t, tx in zip(steps, times, text):
            self.play(*s, run_time=t if tx else t * k)
        left = t0 + dur + pad - self.renderer.time
        if left > 1.5:
            print(f"HOLD {key}: {left:.1f}s")  # frozen frame: add a visual step
        if left > 1 / 60:
            self.wait(left)

    def intro(self):
        sq = Square(2, color=BLUE)
        tiles = VGroup(*[Square(0.5) for _ in range(16)]).arrange_in_grid(4, 4, buff=0)
        self.say("intro_a", Create(sq))
        self.say("intro_b", Write(Text("tiles", font_size=36).to_edge(UP)),
                 FadeIn(tiles.move_to(sq)))


if __name__ == "__main__":
    import os
    import subprocess
    import sys

    VO_DIR.mkdir(exist_ok=True)
    env = {**os.environ, "HYPERFRAMES_PYTHON": sys.executable}
    for key, text in N.items():
        wav, txt = VO_DIR / f"{key}.wav", VO_DIR / f"{key}.txt"
        if wav.exists() and txt.exists() and txt.read_text() == f"{VOICE}|{text}":
            continue
        print(key)
        for attempt in range(3):  # the TTS call fails transiently now and then
            if subprocess.run(["npx", "-y", "hyperframes", "tts", text, "-v", VOICE,
                               "-o", str(wav)], env=env, capture_output=True).returncode == 0:
                break
        else:
            sys.exit(f"TTS failed for {key}")
        txt.write_text(f"{VOICE}|{text}")
