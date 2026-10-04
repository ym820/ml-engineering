"""Video 5: Where does memory go?

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python where_does_memory_go.py
2. Render: manim -qh where_does_memory_go.py WhereDoesMemoryGo

Every number comes from training/performance/README.md (Anatomy of Model's Memory
Usage, Additional GPU memory usage, Batch sizes, Gradient Accumulation, Gradient
Checkpointing, Memory-efficient optimizers) and insights/ai-battlefield.md (Tell how
many GPUs do you need in 5 secs). The Llama-3.1-8B totals in GB (16, 144, ~180) are
those sources' bytes-per-parameter multiplied by 8B parameters.
"""
import wave
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
BLUE_D = "#2B7A9E"
YELLOW_T = "#FFFF00"
GOLD_T = "#F0AC5F"
RED_T = "#FC6255"
GREEN_T = "#83C167"
PURPLE_T = "#B48EDA"

# one color per kind of memory, reused in every bar
C_W16, C_W32, C_GRAD, C_M, C_V = BLUE_T, BLUE_D, GREEN_T, YELLOW_T, GOLD_T
C_ACT, C_LOGITS, C_OVER = RED_T, PURPLE_T, GREY_B

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "hk_a": "Take Llama 3.1 8B. In bf16, each of its eight billion parameters takes two "
            "bytes, so the weights come to about 16 gigabytes.",
    "hk_b": "That fits on an 80 gigabyte GPU with lots of room to spare, so you might "
            "expect that training it on that one card would be no problem.",
    "hk_c": "But once you count everything that training actually keeps around, the "
            "total comes to something like 180 gigabytes, more than twice what the card "
            "holds.",
    "hk_d": "So where does all that memory go? Let's take the bar apart, one piece at a "
            "time.",

    "by_a": "Let's start with a single parameter, and count how many bytes training "
            "keeps for it.",
    "by_b": "First there's the weight itself in bf16, which is the two bytes we counted "
            "a moment ago.",
    "by_c": "But mixed precision training also keeps a master copy of every weight in "
            "fp32, which is four more bytes, so the weights alone cost six.",
    "by_d": "Then every parameter needs a gradient, and in many frameworks that's kept "
            "in fp32 too, so that's another four.",
    "by_e": "And AdamW keeps two running statistics for every parameter, one for each "
            "momentum, each in fp32, so that's eight more bytes.",
    "by_f": "Add it all up and you get 18 bytes per parameter, nine times the two bytes "
            "you'd need to just run the model.",
    "by_g": "For our 8 billion parameter model, that's 144 gigabytes, before we've fed "
            "it a single token.",
    "by_h": "And notice that the biggest slice here is the optimizer, at almost half of "
            "the whole thing.",

    "op_a": "That makes the optimizer the obvious place to start cutting.",
    "op_b": "8-bit Adam, from bitsandbytes, stores each of the two states in a single "
            "byte, so eight bytes become two.",
    "op_c": "Optimizers like Adafactor and LION keep less state to begin with, and need "
            "about four bytes.",
    "op_d": "And some courageous people go all the way, and train everything in bf16, "
            "not mixed precision, with something like AnyPrecisionAdamW.",
    "op_e": "Then the weights, the gradients and the optimizer states together come to "
            "just 8 bytes per parameter, instead of 18.",
    "op_f": "The catch is the one from the video on number formats: adding a tiny update "
            "to a big weight in bf16 can lose it entirely, so this needs Kahan summation "
            "or stochastic rounding.",
    "op_g": "And not every optimizer works for every training job, so think of these as "
            "options to try, rather than free savings.",

    "ac_a": "So far, everything has scaled with the number of parameters. Activations "
            "are different, because they scale with how much data you push through.",
    "ac_b": "Let's stay with Llama 3.1 8B, and feed it a batch of one sequence, 32 "
            "thousand tokens long.",
    "ac_c": "The basic unit here is the hidden states tensor, which is batch, by "
            "sequence, by hidden size. In bf16, that comes to a quarter of a gig.",
    "ac_d": "That doesn't sound like much, but computing a single layer's forward pass "
            "creates a whole bunch of intermediate tensors the size of this one.",
    "ac_e": "How many depends on the architecture. Measured on a handful of models, it "
            "ranges from 24 copies for SmolLM2, up to 48 for Gemma.",
    "ac_f": "Llama 3.1 8B makes 28, so one layer's forward needs 28 quarter gigs, which "
            "is 7 gigs.",
    "ac_g": "Now, the backward pass needs these activations to compute the gradients, "
            "so normally every layer has to hold on to its share until backward comes "
            "back around.",
    "ac_h": "With 32 layers, that's 32 times 7, or 224 gigs of activations, nearly "
            "three of our 80 gig cards for this alone.",

    "gc_a": "This is what gradient checkpointing is for. Instead of keeping everything, "
            "each layer keeps only its input, a single copy of the hidden states, and "
            "drops the rest.",
    "gc_b": "When backward reaches a layer, it runs that layer's forward again from the "
            "saved input, to get the activations back just when they're needed.",
    "gc_c": "So now we store 32 quarter gig checkpoints, which is 8 gigs, plus 7 gigs of "
            "working space for whichever layer is being recomputed.",

    "lg_a": "But there's one more tensor, at the very end of the model, and it's easy "
            "to forget.",
    "lg_b": "To compute the loss, the model makes logits over the whole vocabulary for "
            "every token, usually in fp32, and Llama 3's vocabulary has 128 thousand "
            "entries.",
    "lg_c": "That's 4 bytes, times 32 thousand tokens, times 128 thousand, which comes "
            "to 15.7 gigs, about twice as much as all 32 checkpoints put together.",
    "lg_d": "In practice it's often worse, because the logits usually get created in "
            "bf16 first, and then upcast, so for a while you hold both copies.",

    "tt_a": "Adding it all up, with checkpointing we need 7, plus 8, plus 16, or about "
            "31 gigs of activations.",
    "tt_b": "Without it, it's 224 plus 16, which is 240 gigs.",
    "tt_c": "And that gap is why pretty much everybody turns checkpointing on.",

    "cs_a": "Of course, it isn't free. Running the forward pass a second time typically "
            "costs you about 20 to 25 percent of your throughput,",
    "cs_b": "and some recent papers report as much as 30 to 40 percent.",
    "cs_c": "But all that freed memory can go into a bigger batch, which is sometimes "
            "two or even four times bigger than what fit before,",
    "cs_d": "and a GPU that gets more samples per step uses its compute better, so each "
            "step is slower, but the run as a whole usually gets faster.",

    "ov_a": "Then there's the memory you never asked for at all.",
    "ov_b": "The first time PyTorch touches CUDA, loading its kernels can take anywhere "
            "from half a gig to two gigs, and the memory profiler won't even show it.",
    "ov_c": "Initializing torch distributed takes another one to two gigs, and more as "
            "you add GPUs.",
    "ov_d": "And the backend matters too. On an 8 by H200 node, the new nccl2 backend in "
            "PyTorch 2.14 held 5.16 gigs after the first barrier, compared to 1.56 for "
            "the regular nccl.",
    "ov_e": "Finally, as tensors get allocated and freed, memory fragments, so you can "
            "have plenty free in total, but no single gap big enough for the tensor you "
            "need.",
    "ov_f": "Turning on expandable segments in PyTorch's allocator helps a lot here, "
            "especially when the code does a lot of reshaping.",

    "fb_a": "Let's go back to our 80 gig card, and put all of these pieces into one bar.",
    "fb_b": "There are the bf16 weights we started with,",
    "fb_c": "then the fp32 master weights, the gradients, and the two Adam states,",
    "fb_d": "then 31 gigs of activations with checkpointing on, and a few gigs of "
            "overhead.",
    "fb_e": "The weights you'd actually ship are that thin slice at the start, less than "
            "a tenth of what training puts on the GPU.",

    "bs_a": "Once the fixed costs are paid, the batch size is the knob that fills "
            "whatever memory is left, and there are really two of them.",
    "bs_b": "The micro batch size is how many samples one GPU takes in a single forward "
            "call.",
    "bs_c": "The global batch size is how many samples get consumed across all the GPUs "
            "between two optimizer steps.",
    "bs_d": "With data parallelism over 8 GPUs, and a micro batch of 4, that's 32.",
    "bs_e": "And if you want a bigger batch than fits, gradient accumulation runs "
            "several forward and backward passes, adding up their gradients, before "
            "taking a step.",
    "bs_f": "With 4 accumulation steps, the global batch becomes 4 times 8 times 4, or "
            "128, while each GPU only ever holds activations for 4 samples.",
    "bs_g": "It also saves on communication, since gradients only get all-reduced once "
            "per optimizer step. Going from 1 to 8 accumulation steps cuts that traffic "
            "by 8 times.",
    "bs_h": "And one practical tip: if a micro batch of 8 runs out of memory, but 7 "
            "fits, use 7, not 4. Because the batch gets flattened together with the "
            "sequence, its alignment barely affects speed.",

    "im_a": "Finally, here's a way to tell how many GPUs you need in about five seconds.",
    "im_b": "For training, take the parameter count in billions, times 18 bytes, times "
            "1.25 as a very rough allowance for activations, and divide by the GPU's "
            "memory in gigs.",
    "im_c": "For inference, it's the same thing, with 2 bytes instead of 18.",
    "im_d": "So an 80 billion parameter model on 80 gig GPUs needs at least 23 of them "
            "to train,",
    "im_e": "and just 3 to serve.",
    "im_f": "These are minimums, and a bigger batch or a longer sequence will need more.",
    "im_g": "Most of the gap between 23 and 3 is the master weights, gradients and "
            "optimizer states we counted at the start, and later in this series we'll "
            "see what happens when you slice that bar across many GPUs.",
}


def caption(text, size=30, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.5)


def hbar(parts, scale, height=0.6, opacity=0.85):
    """Horizontal stacked bar. parts: [(value, color)]; scale: units per value."""
    return VGroup(*[Rectangle(width=v * scale, height=height, fill_color=c,
                              fill_opacity=opacity, stroke_width=1, stroke_color=BLACK)
                    for v, c in parts]).arrange(RIGHT, buff=0)


def vbar(parts, scale, width=1.2, opacity=0.85):
    """Vertical stacked bar, first part at the bottom."""
    return VGroup(*[Rectangle(width=width, height=v * scale, fill_color=c,
                              fill_opacity=opacity, stroke_width=1, stroke_color=BLACK)
                    for v, c in parts]).arrange(UP, buff=0)


class WhereDoesMemoryGo(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.bytes_per_param, self.optimizers,
                     self.activations, self.logits, self.totals,
                     self.cost, self.overhead, self.full_bar, self.batches,
                     self.instant_math):
            beat()
            self.wait(0.6)
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
            self.wait(0.3)

    def tear_down(self):
        (VO_DIR / "timeline.tsv").write_text(
            "".join(f"{k}\t{t:.3f}\t{d:.3f}\n" for k, t, d in self.timeline))

    def say(self, key, *steps, pad=0.25):
        """Speak a line while playing steps, stretched to span the line.

        A step is an animation or a list of animations played together.
        """
        path = VO_DIR / f"{key}.wav"
        with wave.open(str(path)) as w:
            dur = w.getnframes() / w.getframerate()
        steps = [[prepare_animation(a) for a in (s if isinstance(s, list) else [s])]
                 for s in steps]
        # text-only steps play at normal speed (capped); shapes absorb the stretch
        text = [all(isinstance(a.mobject, Text) or
                    (isinstance(a.mobject, VGroup) and a.mobject.submobjects and
                     all(isinstance(m, Text) for m in a.mobject.submobjects))
                    for a in s) for s in steps]
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
            print(f"HOLD {key}: {left:.1f}s")
        if left > 1 / 60:
            self.wait(left)

    # GB bars for Llama-3.1-8B, shared by the hook and the full-bar beat
    GB = 12.4 / 185
    PARTS_8B = [(16, C_W16), (32, C_W32), (32, C_GRAD), (32, C_M), (32, C_V),
                (33, C_ACT), (3, C_OVER)]

    def gpu_frame(self, y):
        x0 = -6.4
        frame = Rectangle(width=80 * self.GB, height=1.0, stroke_color=WHITE,
                          stroke_width=3).move_to([x0 + 40 * self.GB, y, 0])
        lbl = Text("one 80 GB GPU", font_size=24).next_to(frame, UP, buff=0.15)
        lbl.align_to(frame, LEFT)
        return frame, lbl, x0

    # 0. hook
    def hook(self):
        title = Text("Where does memory go?", font_size=64)
        self.play(Write(title), run_time=1.2)
        self.play(title.animate.scale(0.6).to_edge(UP))
        frame, flbl, x0 = self.gpu_frame(0.6)
        w = hbar([(16, C_W16)], self.GB, height=0.7)
        w.move_to(frame).align_to(frame, LEFT).shift(RIGHT * 0.15)
        wl = Text("bf16 weights: 8B × 2 bytes ≈ 16 GB", font_size=26, color=C_W16)
        wl.next_to(frame, DOWN, buff=0.3).align_to(frame, LEFT)
        self.say("hk_a", [Create(frame), Write(flbl)], [GrowFromEdge(w, LEFT), Write(wl)])
        free = Text("64 GB free", font_size=26, color=GREY_B).move_to(
            frame.get_center() + RIGHT * 0.6)
        room = Rectangle(width=64 * self.GB - 0.3, height=0.7, fill_color=GREEN_T,
                         fill_opacity=0.25, stroke_width=0).next_to(w, RIGHT, buff=0.08)
        self.say("hk_b", [GrowFromEdge(room, LEFT), FadeIn(free)],
                 Indicate(frame, color=GREEN_T, scale_factor=1.03))

        full = hbar(self.PARTS_8B, self.GB, height=0.7)
        full.move_to([0, -1.3, 0]).align_to(w, LEFT)
        tl = Text("training: ~180 GB", font_size=28, color=RED_T)
        tl.next_to(full, DOWN, buff=0.3).align_to(full, LEFT)
        edge = DashedLine(frame.get_corner(UR) + UP * 0.2, [frame.get_right()[0], -2.5, 0],
                          color=WHITE)
        self.say("hk_c", [FadeOut(free), FadeOut(room)],
                 LaggedStart(*[GrowFromEdge(p, LEFT) for p in full], lag_ratio=0.6,
                             run_time=3),
                 [Create(edge), Write(tl)])
        q = Text("?", font_size=60, color=YELLOW_T).next_to(full, RIGHT, buff=0.3)
        self.say("hk_d", FadeIn(q, scale=2),
                 full.animate.arrange(RIGHT, buff=0.12).align_to(w, LEFT))

    # 1. 18 bytes per parameter
    def bytes_per_param(self):
        head = Text("Bytes per parameter", font_size=40).to_edge(UP)
        unit = 0.27
        parts = [(2, C_W16, "bf16 weight", "2"),
                 (4, C_W32, "fp32 master weight", "4"),
                 (4, C_GRAD, "gradient (fp32)", "4"),
                 (4, C_M, "Adam momentum 1 (fp32)", "4"),
                 (4, C_V, "Adam momentum 2 (fp32)", "4")]
        bar = vbar([(v, c) for v, c, _, _ in parts], unit, width=1.4)
        bar.move_to([-2.6, 0, 0]).align_to([0, -3.1, 0], DOWN)
        labels = VGroup(*[Text(f"{n}   {b}", font_size=24, color=c).next_to(seg, RIGHT,
                                                                            buff=0.3)
                          for seg, (_, c, n, b) in zip(bar, parts)])
        cells = VGroup(*[VGroup(*[Line(seg.get_left() + UP * (i * unit - seg.height / 2),
                                       seg.get_right() + UP * (i * unit - seg.height / 2),
                                       stroke_width=0.6, stroke_color=BLACK)
                                  for i in range(1, round(seg.height / unit))])
                         for seg in bar])
        dot = Dot(color=WHITE).move_to([-5.3, 0, 0])
        dl = Text("1 parameter", font_size=24).next_to(dot, DOWN)

        self.say("by_a", Write(head), [FadeIn(dot, scale=2), Write(dl)])
        self.say("by_b", [GrowFromEdge(bar[0], DOWN), FadeIn(cells[0])], Write(labels[0]))
        wb = Brace(bar[:2], LEFT, color=C_W16)
        wbl = Text("6", font_size=28, color=C_W16).next_to(wb, LEFT)
        self.say("by_c", [GrowFromEdge(bar[1], DOWN), FadeIn(cells[1])], Write(labels[1]),
                 [GrowFromCenter(wb), Write(wbl)])
        self.say("by_d", [GrowFromEdge(bar[2], DOWN), FadeIn(cells[2])], Write(labels[2]),
                 Indicate(bar[2], color=C_GRAD, scale_factor=1.05))
        self.say("by_e", [GrowFromEdge(bar[3], DOWN), FadeIn(cells[3]), Write(labels[3])],
                 [GrowFromEdge(bar[4], DOWN), FadeIn(cells[4]), Write(labels[4])])

        inf = vbar([(2, C_W16)], unit, width=1.4).move_to([4.6, 0, 0]).align_to(bar, DOWN)
        tot = Text("18 bytes", font_size=34, color=YELLOW_T).next_to(bar, UP, buff=0.2)
        il = Text("to run it: 2", font_size=24, color=C_W16).next_to(inf, UP)
        tl = Text("to train it", font_size=24).next_to(bar, DOWN, buff=0.15)
        self.play(FadeOut(dot), FadeOut(dl), run_time=0.4)
        self.say("by_f", Write(tot), [GrowFromEdge(inf, DOWN), Write(il)])
        g = Text("× 8B params = 144 GB", font_size=32, color=YELLOW_T)
        g.next_to(tot, RIGHT, buff=0.4)
        self.say("by_g", Write(g), Circumscribe(VGroup(tot, g), color=YELLOW_T),
                 Indicate(bar, color=YELLOW_T, scale_factor=1.03))
        ob = Brace(bar[3:], LEFT, color=C_M)
        obl = Text("8 of 18:\nalmost half", font_size=24, color=C_M).next_to(ob, LEFT)
        self.say("by_h", [GrowFromCenter(ob), Write(obl)],
                 Indicate(VGroup(bar[3], bar[4]), color=C_M, scale_factor=1.05))

    # 2. cheaper optimizers
    def optimizers(self):
        head = Text("Cutting the optimizer", font_size=40).to_edge(UP)
        unit = 0.24
        cols = [("AdamW\n(mixed precision)", [(6, C_W32), (4, C_GRAD), (8, C_M)], "18"),
                ("8-bit Adam", [(6, C_W32), (4, C_GRAD), (2, C_M)], "12"),
                ("Adafactor / LION", [(6, C_W32), (4, C_GRAD), (4, C_M)], "~14"),
                ("all-bf16\nAnyPrecisionAdamW", [(2, C_W16), (2, C_GRAD), (4, C_M)], "8")]
        groups = VGroup()
        for i, (name, parts, total) in enumerate(cols):
            b = vbar(parts, unit, width=1.3)
            b.move_to([-4.8 + i * 3.2, 0, 0]).align_to([0, -2.0, 0], DOWN)
            n = Text(name, font_size=22).next_to(b, DOWN, buff=0.2)
            t = Text(total, font_size=30, color=YELLOW_T).next_to(b, UP, buff=0.15)
            groups.add(VGroup(b, n, t))
        legend = VGroup(*[VGroup(Square(0.22, fill_color=c, fill_opacity=0.85,
                                        stroke_width=0), Text(s, font_size=20))
                          .arrange(RIGHT, buff=0.12)
                          for c, s in ((C_W32, "weights"), (C_GRAD, "grads"),
                                       (C_M, "optimizer states"))])
        legend.arrange(RIGHT, buff=0.5).next_to(head, DOWN, buff=0.25)

        self.say("op_a", Write(head), [FadeIn(groups[0]), FadeIn(legend)],
                 Indicate(groups[0][0][2], color=C_M))
        self.say("op_b", TransformFromCopy(groups[0][0], groups[1][0]),
                 [Write(groups[1][1]), Write(groups[1][2])])
        self.say("op_c", TransformFromCopy(groups[1][0], groups[2][0]),
                 [Write(groups[2][1]), Write(groups[2][2])])
        self.say("op_d", Write(groups[3][1]),
                 LaggedStart(*[GrowFromEdge(p, DOWN) for p in groups[3][0]], lag_ratio=0.7,
                             run_time=2))
        self.say("op_e", Write(groups[3][2]), Indicate(groups[0][2], color=YELLOW_T),
                 Circumscribe(groups[3][2], color=YELLOW_T))
        eq = caption("in bf16:  1.0 + 0.0001 = 1.0", 26, GOLD_T)
        note = caption("needs Kahan summation and/or stochastic rounding", 26, GOLD_T)
        self.say("op_f", Write(eq), Wiggle(groups[3][0][0]), Transform(eq, note),
                 Circumscribe(groups[3], color=GOLD_T))
        self.say("op_g", FadeOut(eq),
                 Write(caption("Not every optimizer trains every model", 26, GREY_B)),
                 LaggedStart(*[Indicate(g[0], color=GREY_B, scale_factor=1.05)
                               for g in groups[1:]], lag_ratio=0.5))

    # 3. activations
    def activations(self):
        head = Text("Activations", font_size=40).to_edge(UP)
        cfg = Text("Llama-3.1-8B   bs = 1   seq = 32,768   hidden = 4096   bf16",
                   font_size=24, color=GREY_B).next_to(head, DOWN)
        pb = Rectangle(width=2.4, height=0.5, fill_color=C_W32, fill_opacity=0.85,
                       stroke_width=0).move_to([-5.2, 0.8, 0], aligned_edge=LEFT)
        pl = Text("weights, grads, optimizer:  ∝ parameters", font_size=24,
                  color=C_W32).next_to(pb, UP, aligned_edge=LEFT)
        ab = Rectangle(width=0.6, height=0.5, fill_color=C_ACT, fill_opacity=0.85,
                       stroke_width=0).move_to([-5.2, -1.0, 0], aligned_edge=LEFT)
        al = Text("activations:  ∝ batch × sequence", font_size=24,
                  color=C_ACT).next_to(ab, UP, aligned_edge=LEFT)
        self.say("ac_a", Write(head), [GrowFromEdge(pb, LEFT), FadeIn(pl)],
                 [GrowFromEdge(ab, LEFT), FadeIn(al)],
                 ab.animate.stretch_to_fit_width(9.5, about_edge=LEFT))
        toks = VGroup(*[Square(0.2, fill_color=C_ACT, fill_opacity=0.7, stroke_width=1,
                               stroke_color=WHITE) for _ in range(40)])
        toks.arrange(RIGHT, buff=0.04).move_to([0, 0.6, 0])
        tl = Text("32,768 tokens", font_size=24).next_to(toks, DOWN)
        self.say("ac_b", FadeOut(VGroup(pb, pl, ab, al)), Write(cfg),
                 LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in toks], lag_ratio=0.05),
                 Write(tl))

        hs = Rectangle(width=2.6, height=0.9, fill_color=C_ACT, fill_opacity=0.7,
                       stroke_color=WHITE).move_to([-4.2, 0.6, 0])
        hs_l = Text("hidden_states", font_size=24).move_to(hs)
        shape = Text("[1, 32768, 4096] × 2 bytes\n= 0.25 GiB", font_size=22,
                     color=YELLOW_T).next_to(hs, DOWN)
        self.say("ac_c", [ReplacementTransform(toks, hs), FadeOut(tl)], Write(hs_l),
                 Write(shape), Circumscribe(shape, color=YELLOW_T))

        copies = VGroup(*[Rectangle(width=0.52, height=0.18, fill_color=C_ACT,
                                    fill_opacity=0.7, stroke_width=1, stroke_color=WHITE)
                          for _ in range(28)]).arrange_in_grid(7, 4, buff=0.06)
        copies.move_to([-0.5, 0.3, 0])
        self.say("ac_d", LaggedStart(*[TransformFromCopy(hs, c) for c in copies],
                                     lag_ratio=0.05, run_time=2.5))

        data = [("SmolLM2-360M", 24), ("Llama-3.1-8B", 28), ("Mistral-7B", 28),
                ("DS-R1-Qwen3-8B", 29.8), ("Qwen3-4B", 38), ("gemma-1.1-2b", 48)]
        # horizontal bars: name, bar, value
        rows = VGroup()
        for name, v in data:
            c = C_ACT if name.startswith("Llama") else GREY_B
            rows.add(VGroup(Text(name, font_size=18),
                            Rectangle(width=v * 0.055, height=0.3, fill_color=c,
                                      fill_opacity=0.85, stroke_width=0),
                            Text(f"{v:g}", font_size=18, color=c)))
        rows.arrange(DOWN, buff=0.22)
        for r in rows:
            r[0].move_to(rows.get_left(), aligned_edge=LEFT).set_y(r.get_y())
        nx = max(r[0].get_right()[0] for r in rows) + 0.2
        for r in rows:
            r[1].move_to([nx, r[0].get_y(), 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.12)
        rows.move_to([4.1, 0.2, 0])
        chart = VGroup(rows, Line(rows[0][1].get_corner(UL) + UP * 0.15,
                                  rows[-1][1].get_corner(DL) + DOWN * 0.15,
                                  stroke_width=2))
        chart.bars = [r[1] for r in rows]
        cl = Text("hidden_states copies per layer", font_size=20).next_to(chart, UP)
        self.say("ac_e", [Create(chart[1]), Write(cl), FadeIn(VGroup(*[r[0] for r in rows]))],
                 [LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.2),
                  FadeIn(VGroup(*[r[2] for r in rows]))])
        per = Text("28 × 0.25 GiB = 7 GiB per layer", font_size=26, color=YELLOW_T)
        per.next_to(copies, DOWN, buff=0.5)
        self.say("ac_f", Indicate(chart.bars[1], color=YELLOW_T), Write(per),
                 Circumscribe(per, color=YELLOW_T))

        self.play(*[FadeOut(m) for m in (hs, hs_l, shape, chart, cl, copies)])
        self.play(per.animate.scale(0.85).next_to(cfg, DOWN, buff=0.25))
        s = 0.3  # units per GiB
        tower = VGroup(*[Rectangle(width=0.26, height=7 * s, fill_color=C_ACT,
                                   fill_opacity=0.75, stroke_width=1, stroke_color=BLACK)
                         for _ in range(32)]).arrange(RIGHT, buff=0.06)
        tower.move_to([0, 0, 0]).align_to([0, -2.6, 0], DOWN)
        base = Line(tower.get_corner(DL) + LEFT * 0.2, tower.get_corner(DR) + RIGHT * 0.2)
        ll = Text("layer 1 … layer 32", font_size=22).next_to(base, DOWN, buff=0.15)
        self.tower, self.base, self.ll, self.per, self.cfg, self.head = (
            tower, base, ll, per, cfg, head)
        sweep = Arrow(tower.get_corner(UL) + UP * 0.4, tower.get_corner(UR) + UP * 0.4,
                      buff=0, color=WHITE, stroke_width=3)
        fw = Text("forward", font_size=22).next_to(sweep, UP, buff=0.05)
        self.say("ac_g", [Create(base), Write(ll)],
                 [GrowArrow(sweep), Write(fw),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in tower], lag_ratio=0.15)],
                 [FadeOut(sweep), FadeOut(fw)])
        tot = Text("32 × 7 = 224 GiB", font_size=34, color=RED_T).next_to(tower, RIGHT,
                                                                           buff=0.3)
        tot.shift(UP * 0.4)
        self.tot = tot
        tb = Brace(tower, UP, color=RED_T)
        self.say("ac_h", Write(tot), GrowFromCenter(tb),
                 Indicate(tower, color=RED_T, scale_factor=1.02))
        self.tb = tb
        self.checkpointing()

    # 4. gradient checkpointing
    def checkpointing(self):
        tower, s = self.tower, 0.3
        self.play(FadeOut(self.tot), FadeOut(self.tb), run_time=0.4)
        head2 = Text("Gradient checkpointing", font_size=40).to_edge(UP)
        flat = VGroup(*[b.copy().stretch_to_fit_height(0.25 * s * 3).align_to(b, DOWN)
                        for b in tower])
        # 0.25 GiB checkpoints are drawn 3x taller than scale so they stay visible
        for f in flat:
            f.set_fill(C_ACT, 1)
        self.say("gc_a", Transform(self.head, head2),
                 LaggedStart(*[Transform(b, f) for b, f in zip(tower, flat)],
                             lag_ratio=0.05, run_time=2.5))
        i = 20
        tall = tower[i].copy().stretch_to_fit_height(7 * s).align_to(tower[i], DOWN)
        tall.set_fill(YELLOW_T, 0.8)
        back = Arrow(tower.get_corner(UR) + UP * 2.4, tower[i].get_top() + UP * 2.4 + RIGHT * 0.1,
                     buff=0, color=WHITE, stroke_width=3)
        bl = Text("backward", font_size=22).next_to(back, UP, buff=0.05)
        rl = Text("recompute\nthis layer", font_size=20, color=YELLOW_T)
        rl.next_to(tall, LEFT, buff=0.15).align_to(tall, UP)
        self.say("gc_b", [GrowArrow(back), Write(bl)], [Transform(tower[i], tall), Write(rl)])
        sums = VGroup(Text("checkpoints: 32 × 0.25 = 8 GiB", font_size=26, color=C_ACT),
                      Text("+ one layer's working memory: 7 GiB", font_size=26,
                           color=YELLOW_T)).arrange(DOWN, aligned_edge=LEFT)
        sums.move_to([1.2, 1.15, 0])
        self.say("gc_c", Write(sums[0]),
                 LaggedStart(*[Indicate(b, color=WHITE) for j, b in enumerate(tower)
                               if j != i], lag_ratio=0.05),
                 Write(sums[1]), Indicate(tower[i], color=YELLOW_T))
        self.sums, self.ckpt_extra = sums, VGroup(back, bl, rl)

    # 5. logits
    def logits(self):
        self.play(*[FadeOut(m) for m in (self.ckpt_extra, self.sums, self.per, self.tower,
                                         self.base, self.ll)])
        head = Text("The logits", font_size=40).to_edge(UP)
        self.play(Transform(self.head, head), run_time=0.6)
        s = 0.3
        ck = VGroup(*[Rectangle(width=1.6, height=0.25 * s, fill_color=C_ACT,
                                fill_opacity=0.9, stroke_width=0.5, stroke_color=BLACK)
                      for _ in range(32)]).arrange(UP, buff=0)
        ck.move_to([-3.0, 0, 0]).align_to([0, -2.6, 0], DOWN)
        ckl = Text("32 checkpoints\n8 GiB", font_size=22, color=C_ACT).next_to(ck, DOWN,
                                                                                buff=0.1)
        self.say("lg_a", [FadeIn(ck), Write(ckl)])
        lg = Rectangle(width=1.6, height=15.7 * s, fill_color=C_LOGITS, fill_opacity=0.85,
                       stroke_width=0).move_to([0.5, 0, 0]).align_to(ck, DOWN)
        lgl = Text("logits (fp32)", font_size=22, color=C_LOGITS).next_to(lg, DOWN, buff=0.1)
        shape = Text("[1, 32768, 128256]\n× 4 bytes", font_size=24).next_to(lg, RIGHT,
                                                                            buff=0.4)
        shape.shift(UP * 1.2)
        self.say("lg_b", Write(lgl), GrowFromEdge(lg, DOWN), Write(shape),
                 Indicate(lg, color=C_LOGITS, scale_factor=1.05))
        val = Text("= 15.7 GiB", font_size=30, color=YELLOW_T).next_to(shape, DOWN,
                                                                       aligned_edge=LEFT)
        ck2 = ck.copy().next_to(ck, UP, buff=0).set_fill(opacity=0.45)
        self.say("lg_c", Write(val), TransformFromCopy(ck, ck2),
                 Indicate(VGroup(ck, ck2), color=WHITE, scale_factor=1.03),
                 Circumscribe(val, color=YELLOW_T))
        dbl = Text("bf16 copy first, then fp32:\n6 bytes per element, not 4", font_size=24,
                   color=GOLD_T).next_to(val, DOWN, buff=0.5, aligned_edge=LEFT)
        ghost = Rectangle(width=0.8, height=15.7 * s / 2, fill_color=C_LOGITS,
                          fill_opacity=0.35, stroke_color=C_LOGITS, stroke_width=1)
        ghost.next_to(lg, RIGHT, buff=0).align_to(lg, DOWN)
        self.say("lg_d", Write(dbl), GrowFromEdge(ghost, LEFT),
                 Indicate(ghost, color=GOLD_T))

    # 6. totals
    def totals(self):
        head = Text("Activation memory, total", font_size=40).to_edge(UP)
        s = 12 / 240
        rows = [("with checkpointing", [(7, YELLOW_T), (8, C_ACT), (16, C_LOGITS)],
                 "7 + 8 + 16 = 31 GiB"),
                ("without", [(224, C_ACT), (16, C_LOGITS)], "224 + 16 = 240 GiB")]
        bars = VGroup()
        for j, (name, parts, txt) in enumerate(rows):
            b = hbar(parts, s, height=0.7).move_to([0, 1.0 - j * 2.2, 0]).to_edge(LEFT,
                                                                                 buff=0.5)
            n = Text(name, font_size=26).next_to(b, UP, buff=0.15).align_to(b, LEFT)
            t = Text(txt, font_size=28, color=YELLOW_T)
            bars.add(VGroup(b, n, t))
        bars[0][2].next_to(bars[0][0], RIGHT, buff=0.3)
        bars[1][2].next_to(bars[1][0], DOWN, buff=0.2).align_to(bars[1][0], RIGHT)
        self.play(Transform(self.head, head), run_time=0.6)
        self.say("tt_a", Write(bars[0][1]),
                 LaggedStart(*[GrowFromEdge(p, LEFT) for p in bars[0][0]], lag_ratio=0.5),
                 Write(bars[0][2]))
        self.say("tt_b", Write(bars[1][1]),
                 LaggedStart(*[GrowFromEdge(p, LEFT) for p in bars[1][0]], lag_ratio=0.5),
                 Write(bars[1][2]))
        self.say("tt_c", Circumscribe(bars[0], color=GREEN_T))

    # 7. the cost of checkpointing
    def cost(self):
        head = Text("What checkpointing costs", font_size=40).to_edge(UP)
        a = Rectangle(width=4, height=0.6, fill_color=BLUE_T, fill_opacity=0.8,
                      stroke_width=0).move_to([-4.5, 1.6, 0], aligned_edge=LEFT)
        b = Rectangle(width=4, height=0.6, fill_color=BLUE_T, fill_opacity=0.8,
                      stroke_width=0).move_to([-4.5, 0.6, 0], aligned_edge=LEFT)
        extra = Rectangle(width=1.0, height=0.6, fill_color=YELLOW_T, fill_opacity=0.8,
                          stroke_width=0).next_to(b, RIGHT, buff=0)
        la = Text("one step", font_size=22).next_to(a, LEFT).shift(RIGHT * 0)
        lb = Text("one step,\ncheckpointed", font_size=22).next_to(b, LEFT)
        VGroup(a, b, extra).shift(RIGHT * 1.0)
        la.next_to(a, LEFT)
        lb.next_to(b, LEFT)
        pc = Text("+20-25% time", font_size=26, color=YELLOW_T).next_to(extra, RIGHT,
                                                                        buff=0.8)
        pc2 = Text("(papers: up to +30-40%)", font_size=22, color=GREY_B).next_to(
            pc, DOWN, aligned_edge=LEFT)
        self.say("cs_a", Write(head), [FadeIn(a), FadeIn(b), Write(la), Write(lb)],
                 [GrowFromEdge(extra, LEFT), Write(pc)])
        more = Rectangle(width=0.6, height=0.6, fill_color=YELLOW_T, fill_opacity=0.3,
                         stroke_color=YELLOW_T, stroke_width=1).next_to(extra, RIGHT, buff=0)
        self.say("cs_b", Write(pc2), GrowFromEdge(more, LEFT))

        mem = lambda y: Rectangle(width=8, height=0.6, stroke_color=WHITE).move_to(
            [0.5, y, 0])
        m1, m2 = mem(-1.2), mem(-2.5)
        m1l = Text("GPU memory", font_size=22).next_to(m1, LEFT)
        m2l = Text("checkpointed", font_size=22).next_to(m2, LEFT)
        fixed1 = Rectangle(width=4.0, height=0.6, fill_color=C_W32, fill_opacity=0.8,
                           stroke_width=0).align_to(m1, LEFT).set_y(m1.get_y())
        act1 = Rectangle(width=3.6, height=0.6, fill_color=C_ACT, fill_opacity=0.8,
                         stroke_width=0).next_to(fixed1, RIGHT, buff=0)
        fixed2 = fixed1.copy().set_y(m2.get_y())
        smp = VGroup(*[Rectangle(width=0.85, height=0.6, fill_color=C_ACT,
                                 fill_opacity=0.8, stroke_width=1.5, stroke_color=BLACK)
                       for _ in range(4)]).arrange(RIGHT, buff=0).next_to(fixed2, RIGHT,
                                                                           buff=0)
        b1 = Text("batch 1", font_size=22).move_to(act1)
        b4 = Text("batch 4", font_size=22).move_to(smp)
        self.say("cs_c", [Create(m1), Create(m2), Write(m1l), Write(m2l), FadeIn(fixed1),
                          FadeIn(fixed2)], [GrowFromEdge(act1, LEFT), Write(b1)],
                 [LaggedStart(*[FadeIn(x, shift=RIGHT * 0.2) for x in smp], lag_ratio=0.3),
                  Write(b4)])
        self.say("cs_d", Circumscribe(smp, color=GREEN_T),
                 Write(caption("slower steps, higher overall throughput", 28, GREEN_T)),
                 Indicate(VGroup(extra, more), color=YELLOW_T))

    # 8. memory you never asked for
    def overhead(self):
        head = Text("Memory you never asked for", font_size=40).to_edge(UP)
        s = 12.0 / 80
        frame = Rectangle(width=80 * s, height=0.8, stroke_color=WHITE).move_to([0, 1.8, 0])
        fl = Text("80 GiB GPU", font_size=22).next_to(frame, UP, buff=0.1).align_to(frame,
                                                                                    LEFT)
        k = Rectangle(width=2 * s, height=0.8, fill_color=C_OVER, fill_opacity=0.9,
                      stroke_width=0).align_to(frame, LEFT).set_y(frame.get_y())
        kd = Rectangle(width=2 * s, height=0.8, fill_color=GREY_D, fill_opacity=0.9,
                       stroke_width=0).next_to(k, RIGHT, buff=0)
        kl = Text("CUDA kernels: 0.5-2 GiB\n(invisible to the profiler)", font_size=22,
                  color=C_OVER).next_to(frame, DOWN, buff=0.2).align_to(frame, LEFT)
        dl = Text("torch.distributed: 1-2 GiB", font_size=22, color=GREY_B)
        dl.next_to(kl, RIGHT, buff=0.8).align_to(kl, UP)
        self.say("ov_a", Write(head), [Create(frame), Write(fl)])
        self.say("ov_b", GrowFromEdge(k, LEFT), Write(kl),
                 Indicate(k, color=WHITE, scale_factor=1.6))
        self.say("ov_c", GrowFromEdge(kd, LEFT), Write(dl))

        s2 = 0.55
        bars = VGroup()
        for j, (name, v, c) in enumerate((("nccl", 1.56, GREY_B), ("nccl2", 5.16, RED_T))):
            r = Rectangle(width=v * s2 * 1.5, height=0.5, fill_color=c, fill_opacity=0.85,
                          stroke_width=0).move_to([-3.5, -0.9 - j * 0.75, 0],
                                                  aligned_edge=LEFT)
            n = Text(name, font_size=22).next_to(r, LEFT)
            t = Text(f"{v} GiB", font_size=22, color=c).next_to(r, RIGHT)
            bars.add(VGroup(r, n, t))
        bh = Text("8x H200, PyTorch 2.14, after first barrier (rank 0)", font_size=20,
                  color=GREY_B).next_to(bars, UP, buff=0.2).align_to(bars, LEFT)
        self.say("ov_d", Write(bh), [GrowFromEdge(bars[0][0], LEFT), Write(bars[0][1:])],
                 [GrowFromEdge(bars[1][0], LEFT), Write(bars[1][1:])],
                 Circumscribe(bars[1], color=RED_T))

        self.play(FadeOut(bars), FadeOut(bh), FadeOut(kl), FadeOut(dl), run_time=0.5)
        strip = Rectangle(width=10, height=0.7, stroke_color=WHITE).move_to([0, -1.0, 0])
        used, gaps = VGroup(), VGroup()
        x = strip.get_left()[0]
        for wu, wg in ((1.1, 0.5), (0.8, 0.6), (1.3, 0.4), (0.7, 0.7), (1.2, 0.5),
                       (0.9, 0.6), (0.5, 0.2)):
            u = Rectangle(width=wu, height=0.7, fill_color=C_W32, fill_opacity=0.8,
                          stroke_width=0).move_to([x + wu / 2, -1.0, 0])
            used.add(u)
            gaps.add(Rectangle(width=wg, height=0.7, stroke_width=0).move_to(
                [x + wu + wg / 2, -1.0, 0]))
            x += wu + wg
        freel = Text("3.5 free in total, in 7 small gaps", font_size=22,
                     color=GREEN_T).next_to(strip, DOWN, buff=0.2)
        want = Rectangle(width=1.5, height=0.7, fill_color=C_ACT, fill_opacity=0.85,
                         stroke_width=0).move_to([0, -2.6, 0])
        wl = Text("needs 1.5 contiguous", font_size=22, color=C_ACT).next_to(want, RIGHT)
        self.say("ov_e", [Create(strip), FadeIn(used)],
                 [LaggedStart(*[Indicate(g.set_fill(GREEN_T, 0.4), color=GREEN_T)
                                for g in gaps], lag_ratio=0.2), Write(freel)],
                 [FadeIn(want, shift=UP * 0.2), Write(wl)],
                 Wiggle(want))
        self.say("ov_f", FadeOut(VGroup(freel, want, wl)), Indicate(strip, color=YELLOW_T),
                 Write(caption("PYTORCH_ALLOC_CONF=expandable_segments:True", 28,
                               YELLOW_T)))

    # 9. the full bar
    def full_bar(self):
        head = Text("Llama-3.1-8B training, one bar", font_size=40).to_edge(UP)
        frame, flbl, x0 = self.gpu_frame(1.2)
        full = hbar(self.PARTS_8B, self.GB, height=0.7)
        full.move_to(frame).align_to([x0 + 0.0, 0, 0], LEFT)
        edge = DashedLine(frame.get_corner(UR) + UP * 0.2,
                          [frame.get_right()[0], -2.8, 0], color=WHITE)
        self.say("fb_a", Write(head), [Create(frame), Write(flbl), Create(edge)])
        names = [("bf16 weights 16", C_W16), ("fp32 master 32", C_W32),
                 ("grads 32", C_GRAD), ("Adam m 32", C_M), ("Adam v 32", C_V),
                 ("activations 33 (31 GiB)", C_ACT), ("overhead ~3", C_OVER)]
        labels = VGroup()
        for i, ((n, c), seg) in enumerate(zip(names, full)):
            t = Text(n, font_size=21, color=c)
            t.next_to(seg, DOWN, buff=0.25 + 0.38 * (i % 4)).align_to(seg, LEFT)
            labels.add(t)
        labels[6].next_to(full[6], UP, buff=0.5).align_to(full[6], RIGHT)
        self.say("fb_b", [GrowFromEdge(full[0], LEFT), FadeIn(labels[0])])
        self.say("fb_c", *[[GrowFromEdge(full[i], LEFT), FadeIn(labels[i])]
                           for i in range(1, 5)])
        self.say("fb_d", [GrowFromEdge(full[5], LEFT), FadeIn(labels[5])],
                 [GrowFromEdge(full[6], LEFT), FadeIn(labels[6])])
        tot = Text("≈ 180 GB (GB = 10⁹ bytes)", font_size=24, color=YELLOW_T)
        tot.to_edge(DOWN, buff=0.6).to_edge(RIGHT, buff=0.5)
        self.say("fb_e", Write(tot),
                 [full[1:].animate.set_opacity(0.25), labels[1:].animate.set_opacity(0.4)],
                 Indicate(full[0], color=C_W16, scale_factor=1.3))

    # 10. batch sizes and gradient accumulation
    def batches(self):
        head = Text("Batch sizes", font_size=40).to_edge(UP)
        gpus = VGroup(*[RoundedRectangle(corner_radius=0.1, width=1.1, height=1.1,
                                         stroke_color=GREEN_T) for _ in range(8)])
        gpus.arrange(RIGHT, buff=0.3).shift(UP * 1.2)
        gl = Text("8 GPUs, data parallel", font_size=22, color=GREEN_T).next_to(gpus, UP)
        samples = VGroup(*[VGroup(*[Dot(radius=0.11, color=BLUE_T) for _ in range(4)])
                           .arrange_in_grid(2, 2, buff=0.15).move_to(g) for g in gpus])
        self.say("bs_a", Write(head), [Create(gpus), Write(gl)])
        mbs = Text("micro batch (MBS) = 4", font_size=26, color=BLUE_T)
        mbs.next_to(gpus, DOWN, buff=0.4)
        self.say("bs_b", FadeIn(samples[0], scale=1.5), Write(mbs))
        gbs = Text("global batch (GBS) = MBS × DP", font_size=28).next_to(mbs, DOWN)
        gb = Brace(gpus, DOWN, color=YELLOW_T).next_to(gpus, DOWN, buff=0.05)
        self.say("bs_c", Write(gbs), GrowFromCenter(gb),
                 LaggedStart(*[Indicate(g, color=YELLOW_T) for g in gpus], lag_ratio=0.2),
                 FadeOut(gb))
        g32 = Text("4 × 8 = 32", font_size=28, color=YELLOW_T).next_to(gbs, DOWN)
        self.say("bs_d", LaggedStart(*[FadeIn(s, scale=1.5) for s in samples[1:]],
                                     lag_ratio=0.15), Write(g32))

        ghosts = VGroup()
        for k in range(1, 4):
            ghosts.add(samples.copy().set_opacity(0.6 - 0.15 * k).shift(DOWN * 0.0))
        gas_l = Text("× GAS: 4 forward/backward passes, then one optimizer step",
                     font_size=24).next_to(g32, DOWN, buff=0.3)
        self.say("bs_e", Write(gas_l),
                 LaggedStart(*[Indicate(samples, color=YELLOW_T) for _ in range(4)],
                             lag_ratio=0.8))
        f = Text("GBS = MBS × DP × GAS = 4 × 8 × 4 = 128", font_size=28, color=YELLOW_T)
        f.move_to(g32)
        self.say("bs_f", FadeOut(gbs), Transform(g32, f),
                 Circumscribe(g32, color=YELLOW_T),
                 Indicate(samples[0], color=BLUE_T, scale_factor=1.3))

        self.play(*[FadeOut(m) for m in (gpus, gl, samples, mbs, g32, gas_l)])

        def lane(gas, y):
            blocks = VGroup(*[Rectangle(width=0.55, height=0.45, fill_color=BLUE_T,
                                        fill_opacity=0.8, stroke_width=1,
                                        stroke_color=BLACK) for _ in range(16)])
            blocks.arrange(RIGHT, buff=0.05).move_to([0.8, y, 0])
            marks = VGroup(*[Triangle(fill_color=RED_T, fill_opacity=1, stroke_width=0)
                             .scale(0.12).rotate(PI).next_to(blocks[i], UP, buff=0.05)
                             .align_to(blocks[i], RIGHT)
                             for i in range(gas - 1, 16, gas)])
            lbl = Text(f"GAS = {gas}", font_size=24).next_to(blocks, LEFT, buff=0.4)
            return VGroup(blocks, marks, lbl)

        l1, l8 = lane(1, 1.3), lane(8, -0.2)
        key = VGroup(Triangle(fill_color=RED_T, fill_opacity=1, stroke_width=0).scale(0.12)
                     .rotate(PI), Text("= gradient all-reduce", font_size=22)).arrange(RIGHT)
        key.next_to(l8, DOWN, buff=0.5)
        self.say("bs_g", [FadeIn(l1[0]), FadeIn(l1[2]), FadeIn(l8[0]), FadeIn(l8[2]),
                          FadeIn(key)],
                 LaggedStart(*[FadeIn(m, shift=DOWN * 0.2) for m in l1[1]], lag_ratio=0.1),
                 LaggedStart(*[FadeIn(m, shift=DOWN * 0.2) for m in l8[1]], lag_ratio=0.5))

        self.play(FadeOut(VGroup(l1, l8, key)), run_time=0.5)
        tip = VGroup(Text("MBS 8 → OOM", font_size=30, color=RED_T),
                     Text("use MBS 7, not 4", font_size=30, color=GREEN_T),
                     Text("batch is flattened with the sequence,\n"
                          "so its alignment barely matters", font_size=24,
                          color=GREY_B)).arrange(DOWN, buff=0.4)
        tip.shift(UP * 1.2)
        r8 = VGroup(*[Dot(radius=0.1, color=RED_T) for _ in range(8)]).arrange(RIGHT)
        r7 = VGroup(*[Dot(radius=0.1, color=GREEN_T) for _ in range(7)]).arrange(RIGHT)
        r8.next_to(tip[0], RIGHT, buff=0.5)
        r7.next_to(tip[1], RIGHT, buff=0.5).align_to(r8, LEFT)
        grid = VGroup(*[Square(0.3, fill_color=BLUE_T, fill_opacity=0.6, stroke_width=1)
                        for _ in range(24)]).arrange_in_grid(3, 8, buff=0.04)
        grid.move_to([0, -2.2, 0])
        gl = Text("[b, s] → [b·s]", font_size=22, color=GREY_B).next_to(grid, LEFT,
                                                                       buff=0.4)
        row = grid.copy().arrange(RIGHT, buff=0.04).move_to(grid)
        self.say("bs_h", [FadeIn(tip[0]), LaggedStart(*[FadeIn(d) for d in r8])],
                 [FadeIn(tip[1]), LaggedStart(*[FadeIn(d) for d in r7])],
                 [FadeIn(tip[2]), FadeIn(grid), FadeIn(gl)],
                 [Transform(grid, row), gl.animate.next_to(row, UP)])

    # 11. GPUs in 5 seconds
    def instant_math(self):
        head = Text("How many GPUs, in 5 seconds", font_size=40).to_edge(UP)
        def formula(name, b, c):
            return VGroup(*[Text(t, font_size=30, color=col) for t, col in
                            ((name, WHITE), ("params_B", WHITE), (f"× {b}", c),
                             ("× 1.25", GOLD_T), ("/ GPU_GB", WHITE))]).arrange(RIGHT)

        tr, inf = formula("training: ", 18, RED_T), formula("inference:", 2, BLUE_T)
        VGroup(tr, inf).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(head, DOWN,
                                                                           buff=0.5)
        x = max(tr[0].get_right()[0], inf[0].get_right()[0]) + 0.3
        for f in (tr, inf):
            f[1:].shift(RIGHT * (x - f[1].get_left()[0]))
        for a, b in zip(tr[1:], inf[1:]):
            b.set_x(a.get_x())
        clock = Circle(radius=0.9, color=WHITE).move_to([0, -1, 0])
        hand = Line(clock.get_center(), clock.get_center() + UP * 0.75, stroke_width=4)
        self.say("im_a", Write(head), [Create(clock), Create(hand)],
                 Rotate(hand, -2 * PI * 5 / 60, about_point=clock.get_center()))
        self.play(FadeOut(clock), FadeOut(hand), run_time=0.4)
        bt = VGroup(*[SurroundingRectangle(m, color=YELLOW_T, buff=0.08) for m in tr[1:]])
        self.say("im_b", Write(tr),
                 LaggedStart(*[Create(b) for b in bt], lag_ratio=1, run_time=4),
                 FadeOut(bt))
        self.say("im_c", Write(inf), Indicate(inf[2], color=BLUE_T, scale_factor=1.3),
                 Circumscribe(VGroup(tr[2], inf[2]), color=YELLOW_T))

        def farm(n, color, cols):
            return VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.42, height=0.42,
                                             stroke_color=color, fill_color=color,
                                             fill_opacity=0.4) for _ in range(n)]
                          ).arrange_in_grid(cols=cols, buff=0.1)

        f1, f2 = farm(23, RED_T, 8), farm(3, BLUE_T, 3)
        f1.move_to([-2.8, -1.4, 0])
        f2.move_to([3.6, -1.4, 0]).align_to(f1, UP)
        t1 = Text("80 × 18 × 1.25 / 80 = 22.5 → 23", font_size=24, color=RED_T)
        t1.next_to(f1, UP, buff=0.25)
        t2 = Text("80 × 2 × 1.25 / 80 = 2.5 → 3", font_size=24, color=BLUE_T)
        t2.next_to(f2, UP, buff=0.25)
        sub = Text("80B model, 80 GB GPUs", font_size=24, color=GREY_B).next_to(
            inf, DOWN, buff=0.3)
        self.say("im_d", Write(sub), Write(t1),
                 LaggedStart(*[FadeIn(g, scale=1.5) for g in f1], lag_ratio=0.08))
        self.say("im_e", Write(t2),
                 LaggedStart(*[FadeIn(g, scale=1.5) for g in f2], lag_ratio=0.3))
        self.say("im_f", Write(caption("minimums: bigger batches and longer sequences "
                                       "need more", 24, GREY_B)),
                 Indicate(VGroup(t1, t2), color=WHITE))
        self.say("im_g", Indicate(f1, color=RED_T, scale_factor=1.05),
                 LaggedStart(*[g.animate.set_fill(YELLOW_T, 0.6).set_stroke(YELLOW_T)
                               for g in f1[3:]], lag_ratio=0.1),
                 Indicate(f2, color=BLUE_T, scale_factor=1.1))


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
