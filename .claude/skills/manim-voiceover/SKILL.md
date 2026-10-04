---
name: manim-voiceover
description: Adds TTS narration to ManimCE explainer videos with animations synced to each spoken line, and guides writing narration in a natural 3Blue1Brown voice instead of an AI-sounding one. Use when adding a voiceover or narration to a Manim video, writing or revising a narration script, syncing animation timing to speech, choosing a TTS voice, or debugging missing or out-of-sync audio in a Manim render.
---

# Manim voiceover

Narration drives timing: each line is a WAV clip, and `say(key, *steps)` plays the clip while stretching that line's animations to span it.

Writing the script matters as much as the code. Read [references/writing-narration.md](references/writing-narration.md) before writing any lines. Its guidance supersedes the "Engagement Techniques" in manim-composer.

## Setup (once)

```bash
uv tool install manim --python 3.12 --with kokoro-onnx --with soundfile
```

TTS runs through `npx -y hyperframes tts`, which needs `HYPERFRAMES_PYTHON` pointing at a Python with `kokoro-onnx` (the template sets it from `sys.executable`).

## Workflow

1. Copy [templates/voiceover_scene.py](templates/voiceover_scene.py) next to the video.
2. Write the script into `N` (key → line). One short line per visual step.
3. Write each beat as `self.say("key", step, step, ...)`. A step is one animation or a list played together.
4. Generate clips: `~/.local/share/uv/tools/manim/bin/python <file>.py` (only changed lines regenerate).
5. Draft render: `manim -ql <file>.py Video`. Every `HOLD key: Ns` line is a frozen frame; fix holds over ~3s by adding a visual step, not by slowing animations.
6. Check layout: `ffmpeg -i <draft>.mp4 -vf "fps=1/10,scale=427:-1,tile=5x8" -frames:v 1 sheet.png`, then look at the sheet.
7. Check audio: `~/.local/share/uv/tools/manim/bin/python scripts/check_audio.py <render>.mp4 vo/timeline.tsv`.
8. Final render: `manim -qh` (about 10 min for 6 min of video; run it in the background).
9. Copy the render out of `media/`. Add `media/`, `vo/` and `*.mp4` to `.gitignore`.

## Timing rules (in `say`)

- Shape animations stretch up to `MAX_STRETCH` (4x) to span their line.
- Text-only steps play at normal speed, capped at `TEXT_MAX` (1s). Slow-writing text drags.
- The next line starts only after the current one finishes, plus a 0.25s pad.

## Pitfalls

- **Silent lines after repeated waits.** `Scene.add_sound` returns early while `renderer.skip_animations` is set. That flag stays on after any cached animation, such as an identical `self.wait()` between beats, until the next `play`. The template calls `self.renderer.file_writer.add_sound(path, t0)` instead. Never switch back to `self.add_sound`.
- **Transient TTS failures.** `hyperframes tts` occasionally exits non-zero; the generator retries 3 times.
- **Box sized for placeholder text.** If a `Transform` will put longer text inside a `SurroundingRectangle`, size the box around the final text.
- **Questions without a rising tone.** Kokoro only raises pitch on yes/no questions. See the TTS notes in the reference.

## Voices

Kokoro English IDs: `af_heart`, `af_bella`, `af_nicole`, `af_sky`, `bf_emma`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`. To audition, generate the same line with each into a `voice_samples/` folder and let the user pick by ear.
