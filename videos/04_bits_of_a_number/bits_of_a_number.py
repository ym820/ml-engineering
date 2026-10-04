"""Video 4: Bits of a number.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python bits_of_a_number.py
2. Render: manim -qh bits_of_a_number.py BitsOfANumber

Facts come from training/dtype.md, inference/README.md (Model Weights) and
training/instabilities/README.md (Numerical instabilities). The rounding
examples (256 + 1, 70000) and the fp4 block are worked out from the formats'
bit layouts; the block values are made up, and the narration says so.
"""
import wave
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
YELLOW_T = "#FFFF00"
RED_T = "#FC6255"
GREEN_T = "#83C167"
GREY_T = "#888888"

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "hk_a": "Here are two very simple sums, done in two different 16 bit formats.",
    "hk_b": "Add one to 256 in fp16, and you get 257, just like you'd expect.",
    "hk_c": "Do the same thing in bf16, and you get 256 back. The one simply disappears.",
    "hk_d": "Now go the other way, and store the number 70,000. bf16 gives you 70,144, "
            "which is close enough,",
    "hk_e": "but fp16 gives you infinity.",
    "hk_f": "Both formats have exactly 16 bits, and each one is wrong in a different "
            "place. To see why, you and I need to look at how those bits get spent.",

    "an_a": "Every floating point number is split into three fields.",
    "an_b": "There's one sign bit, then some exponent bits, and whatever is left goes to "
            "the mantissa.",
    "an_c": "The exponent picks a power of two, so it decides how big or how small a "
            "number can get. That's the range.",
    "an_d": "The mantissa then picks a point between that power of two and the next "
            "one, so it decides how finely you can place a number. That's the precision.",

    "nl_a": "It helps to draw this on a number line. Let's simplify by pretending we "
            "only have 2 mantissa bits.",
    "nl_b": "Between 1 and 2, those two bits give us four evenly spaced values.",
    "nl_c": "Between 2 and 4 we get four values again, but now they're twice as far "
            "apart,",
    "nl_d": "and between 4 and 8, they're twice as far apart again. Every power of two "
            "gets the same number of points, so the gaps grow along with the numbers.",
    "nl_e": "bf16 has 7 mantissa bits, so it puts 128 values between 256 and 512, and "
            "the gap between neighbors there is 2.",
    "nl_f": "257 lands exactly halfway between 256 and 258, and rounding sends it back "
            "down to 256.",
    "nl_g": "fp16 has 10 mantissa bits, so its gap at 256 is only a quarter, and 257 "
            "fits just fine.",
    "nl_h": "But fp16 pays for those mantissa bits by having only 5 exponent bits, so "
            "its largest number is 65,504.",
    "nl_i": "Anything past that overflows to infinity, and 70,000 is past that.",

    "fm_a": "With that in mind, here's the whole family side by side.",
    "fm_b": "fp32 has 8 exponent bits and 23 mantissa bits.",
    "fm_c": "tf32 keeps the same 8 exponent bits but cuts the mantissa down to 10, "
            "which is how it ends up with 19 bits.",
    "fm_d": "fp16 spends its 16 bits on 5 exponent bits and 10 mantissa bits, so it "
            "has the precision of tf32, with a much smaller range.",
    "fm_e": "bf16 makes the opposite choice. It keeps fp32's 8 exponent bits, and so "
            "fp32's range, and only has 7 mantissa bits left over.",
    "fm_f": "And fp8 comes in two flavors. E4M3 has 4 exponent bits and 3 mantissa "
            "bits,",
    "fm_g": "and E5M2 gives up one of those mantissa bits for more range.",

    "hs_a": "The history of training precision is pretty much a walk down this list.",
    "hs_b": "Originally everything was in fp32, and it was very slow.",
    "hs_c": "Then mixed precision combined fp16 with fp32, which sped training up "
            "tremendously.",
    "hs_d": "But fp16 wasn't very stable, and training large language models in it was "
            "extremely difficult.",
    "hs_e": "For example, in the 104B experiments before BLOOM, attention could blow "
            "up when Q times K was computed first and only scaled down afterwards.",
    "hs_f": "The fix was to scale Q and K down before the multiply, so the product "
            "never gets that big.",
    "hs_g": "Then bf16 came along, and slotted into the very same mixed precision "
            "recipe. That made training large models much more stable.",
    "hs_h": "Then came fp8, which is no longer experimental, since DeepSeek-V3 was "
            "trained in it. Still, bf16 mixed precision is the default for most runs.",
    "hs_i": "And Blackwell added fp6 and fp4, which so far are mostly inference "
            "formats.",

    "sp_a": "So why keep dropping bits?",
    "sp_b": "Here are the B200's peak numbers, in TFLOPS, without sparsity.",
    "sp_c": "Going from tf32 to bf16, and from bf16 to fp8, each time the element gets "
            "half as wide, the throughput doubles.",
    "sp_d": "Then the pattern breaks. fp6 runs at the same 4500 as fp8, so it saves you "
            "memory and bandwidth, but no compute.",
    "sp_e": "And fp4 doubles again, to 9000.",
    "sp_f": "On GB300 it's actually 3x, 15,000 versus 5,000 for fp8, so it's always "
            "worth checking the spec for the exact dtype you plan to use.",

    "tf_a": "Notice how far down plain fp32 is.",
    "tf_b": "On an A100, an fp32 matmul runs at 19.5 TFLOPS, while tf32 runs at 156. "
            "That's 8 times faster, for a small loss of precision.",
    "tf_c": "The catch is that tf32 is off by default, so you have to turn it on "
            "yourself, with these two lines.",

    "ac_a": "So low precision is fast. But there's one place where precision has to "
            "stay high, and that's whenever you add things up.",
    "ac_b": "Remember 256 plus 1 in bf16? Now picture a weight sitting at 256,",
    "ac_c": "and the optimizer adds a tiny update to it.",
    "ac_d": "The update is smaller than the gap between neighbors, so the sum rounds "
            "right back to the weight, and the update is simply lost.",
    "ac_e": "That's why training usually keeps fp32 master weights and fp32 optimizer "
            "states.",
    "ac_f": "Gradient accumulation is best done in fp32 too, and for bf16 it's a must.",
    "ac_g": "In reduction collectives, fp16 is ok as long as loss scaling is in place, "
            "but bf16 is only ok in fp32.",
    "ac_h": "Even a well written LayerNorm takes half precision inputs, but adds them "
            "up in fp32 registers before casting the result back down.",
    "ac_i": "And if you really want half precision master weights, Kahan summation or "
            "stochastic rounding can make the tiny updates stick.",

    "f4_a": "Now let's go all the way down, to fp4.",
    "f4_b": "An fp4 element, E2M1, has only 16 bit patterns. Here they all are: plus "
            "or minus 0, a half, 1, 1.5, 2, 3, 4, and 6.",
    "f4_c": "That's far too few to cover a whole tensor with one scale, so fp4 only "
            "works block scaled.",
    "f4_d": "To be clear, I'm making these numbers up, but imagine a small block of "
            "weights like this one.",
    "f4_e": "The block gets one shared scale, picked so that its largest value lands on "
            "6,",
    "f4_f": "and every value then snaps to the nearest of those 16 points, stretched by "
            "the scale.",
    "f4_g": "Now what happens if one value in the block is an outlier?",
    "f4_h": "The scale has to stretch to reach it,",
    "f4_i": "and everything else in the block gets squeezed onto the bottom few levels.",

    "mx_a": "This is exactly where the two fp4 formats differ.",
    "mx_b": "mxfp4 uses blocks of 32 elements, with an 8 bit E8M0 scale, which can only "
            "be a power of two.",
    "mx_c": "That's 4 bits per element, plus 8 bits shared by 32 elements, so 4.25 bits "
            "each.",
    "mx_d": "NVIDIA's nvfp4 uses blocks of 16, with an E4M3 scale that can sit between "
            "powers of two, plus one fp32 scale for the whole tensor.",
    "mx_e": "That works out to 4.5 bits per element.",
    "mx_f": "Smaller blocks mean an outlier distorts the scale of 15 neighbors instead "
            "of 31.",
    "mx_g": "Speed isn't what separates them, since Blackwell runs both at the same "
            "peak.",
    "mx_h": "In NVIDIA's own 8B pretraining comparison, mxfp4 needed about 36 percent "
            "more tokens to reach nvfp4's final loss.",
    "mx_i": "So on Blackwell or newer, nvfp4 is the pick, and mxfp4 is for other "
            "vendors' hardware, or a checkpoint that has to run on more than one.",

    "nm_a": "By now, you can read most dtype names just by looking at them. Take this "
            "one.",
    "nm_b": "float8 is the total width, e4 is 4 exponent bits, and m3 is 3 mantissa "
            "bits.",
    "nm_c": "b11 is the exponent bias,",
    "nm_d": "f means finite values only, so there's no infinity,",
    "nm_e": "n means it has NaNs, but only at the outer range,",
    "nm_f": "and uz means unsigned zero.",

    "sw_a": "The trade between range and precision comes back one last time, when you "
            "switch formats after training.",
    "sw_b": "Run a bf16 trained model in fp16, and it usually fails, because of "
            "overflows past fp16's ceiling.",
    "sw_c": "Going the other way usually works. An fp16 trained model loses a little "
            "in the conversion to bf16, and it's best to finetune it a bit, but it "
            "runs.",
    "sw_d": "So every bit you drop buys you speed, and every format decides whether its "
            "remaining bits go to range or to precision.",
    "sw_e": "The job is knowing which one your numbers need, and keeping the sums wide.",
}


def caption(text, size=30, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.5)


def strip(s, e, m, cell=0.27):
    """A row of bit cells: sign grey, exponent blue, mantissa green."""
    cells = VGroup()
    for n, col in ((s, GREY_T), (e, BLUE_T), (m, GREEN_T)):
        for _ in range(n):
            cells.add(Square(cell, stroke_width=1.2, stroke_color=BLACK,
                             fill_color=col, fill_opacity=0.9))
    return cells.arrange(RIGHT, buff=0)


def numline(lo, hi, step, length, labels=True, size=24, fmt="{:g}"):
    """NumberLine with Text tick labels, so rendering needs no LaTeX."""
    nl = NumberLine(x_range=[lo, hi, step], length=length, include_numbers=False)
    if labels:
        nl.labels = VGroup(*[Text(fmt.format(v), font_size=size).next_to(nl.n2p(v), DOWN, buff=0.2)
                             for v in np.arange(lo, hi + step / 2, step)])
        nl.add(nl.labels)
    return nl


def bar_chart(names, values, ymax, height, width, colors, size=24):
    """Bars on a baseline with Text names and value labels (no LaTeX)."""
    n = len(values)
    slot = width / n
    base = Line(ORIGIN, RIGHT * width, color=GREY_T)
    bars, nm, lab = VGroup(), VGroup(), VGroup()
    for i, (name, v, c) in enumerate(zip(names, values, colors)):
        h = max(v / ymax * height, 0.03)
        b = Rectangle(width=slot * 0.6, height=h, stroke_width=0, fill_color=c, fill_opacity=0.85)
        b.move_to(RIGHT * (slot * (i + 0.5)) + UP * h / 2)
        bars.add(b)
        nm.add(Text(name, font_size=size).next_to(b, DOWN, buff=0.2).set_y(-0.3))
        lab.add(Text(f"{v:g}", font_size=size - 2).next_to(b, UP, buff=0.12))
    g = VGroup(base, bars, nm)
    g.bars, g.names, g.labels = bars, nm, lab
    return g, lab  # move VGroup(g, lab) together; labels are animated separately


FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def q4(x, scale):
    a = abs(x) / scale
    v = min(FP4, key=lambda p: abs(p - a))
    return (1 if x >= 0 else -1) * v * scale


class BitsOfANumber(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.anatomy, self.numberline, self.formats,
                     self.history, self.speed, self.tf32, self.accum, self.fp4,
                     self.mx, self.name, self.switch):
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

    # 0. hook: two 16-bit formats, each wrong in a different place
    def hook(self):
        title = Text("Bits of a number", font_size=64).to_edge(UP)
        hdr = VGroup(Text("fp16", font_size=40, color=BLUE_T),
                     Text("bf16", font_size=40, color=GREEN_T))
        hdr[0].move_to(LEFT * 0.5 + UP * 1.0)
        hdr[1].move_to(RIGHT * 3.5 + UP * 1.0)
        q1 = Text("256 + 1", font_size=40).move_to(LEFT * 4.5 + DOWN * 0.3)
        q2 = Text("store 70000", font_size=40).move_to(LEFT * 4.5 + DOWN * 1.8)
        a11 = Text("257", font_size=44).move_to(hdr[0].get_x() * RIGHT + DOWN * 0.3)
        a12 = Text("256", font_size=44, color=RED_T).move_to(hdr[1].get_x() * RIGHT + DOWN * 0.3)
        a21 = Text("inf", font_size=44, color=RED_T).move_to(hdr[0].get_x() * RIGHT + DOWN * 1.8)
        a22 = Text("70144", font_size=44).move_to(hdr[1].get_x() * RIGHT + DOWN * 1.8)
        sep = Line(LEFT * 6 + UP * 0.45, RIGHT * 5.5 + UP * 0.45, stroke_width=1.5,
                   color=GREY_T)
        bits = VGroup(strip(1, 5, 10, 0.22), strip(1, 8, 7, 0.22))
        bits[0].next_to(hdr[0], UP, buff=0.25)
        bits[1].next_to(hdr[1], UP, buff=0.25)

        def box(m, col):
            return SurroundingRectangle(m, color=col, buff=0.2, corner_radius=0.08)

        r1 = box(VGroup(q1, a12), GREY_T).stretch_to_fit_width(11.4).set_x(-0.25)
        r2 = box(VGroup(q2, a22), GREY_T).stretch_to_fit_width(11.4).set_x(-0.25)
        b11, b12 = box(a11, GREEN_T), box(a12, RED_T)
        b21, b22 = box(a21, RED_T), box(a22, GREEN_T)
        self.say("hk_a", Write(title), [FadeIn(hdr), Create(sep)],
                 [FadeIn(bits[0]), FadeIn(bits[1]), title.animate.set_opacity(0)])
        self.say("hk_b", FadeIn(q1), Create(r1), FadeIn(a11, shift=LEFT * 0.3), Create(b11))
        self.say("hk_c", FadeIn(a12, shift=LEFT * 0.3), Create(b12),
                 Wiggle(VGroup(a12, b12)))
        self.say("hk_d", [FadeOut(r1), FadeIn(q2)], Create(r2),
                 FadeIn(a22, shift=LEFT * 0.3), Create(b22))
        self.say("hk_e", FadeIn(a21, scale=1.5), Create(b21), Wiggle(VGroup(a21, b21)))
        self.say("hk_f", FadeOut(r2),
                 [Circumscribe(bits[0], color=BLUE_T), Circumscribe(bits[1], color=GREEN_T)],
                 [Indicate(bits[0][1:6]), Indicate(bits[1][1:9])])

    # 1. sign / exponent / mantissa
    def anatomy(self):
        s = strip(1, 8, 7, 0.6).shift(UP * 0.8)
        name = Text("bf16", font_size=36).next_to(s, UP, buff=0.4)
        sign, exp, man = s[0], s[1:9], s[9:]
        lbls = VGroup(
            Text("sign", font_size=26, color=GREY_T).next_to(sign, DOWN),
            Text("exponent", font_size=26, color=BLUE_T).next_to(exp, DOWN),
            Text("mantissa", font_size=26, color=GREEN_T).next_to(man, DOWN),
        )
        formula = Text("value = ± 1.mantissa × 2^exponent", font_size=34)
        formula.shift(DOWN * 1.1)
        rng = Text("range", font_size=34, color=BLUE_T).next_to(formula, DOWN, buff=0.6)
        rng.shift(RIGHT * 2.1)
        prec = Text("precision", font_size=34, color=GREEN_T).next_to(formula, DOWN, buff=0.6)
        prec.shift(LEFT * 2.0)

        self.say("an_a", [Write(name), FadeIn(s)])
        self.say("an_b", Indicate(sign, color=WHITE), [Indicate(exp), FadeIn(lbls[0]),
                                                      FadeIn(lbls[1])],
                 [Indicate(man), FadeIn(lbls[2])])
        self.say("an_c", Write(formula), [exp.animate.set_fill(YELLOW_T), FadeIn(rng)],
                 exp.animate.set_fill(BLUE_T))
        self.say("an_d", [man.animate.set_fill(YELLOW_T), FadeIn(prec)],
                 man.animate.set_fill(GREEN_T))

    # 2. spacing doubles every power of two; 256 + 1; fp16 overflow
    def numberline(self):
        head = Text("2 mantissa bits (a toy format)", font_size=34).to_edge(UP)
        nl = numline(0, 8, 1, 12, size=28).shift(DOWN * 0.3)
        groups = []
        for lo, col in ((1, BLUE_T), (2, GREEN_T), (4, YELLOW_T)):
            pts = [lo + i * lo / 4 for i in range(4)]
            groups.append(VGroup(*[Dot(nl.n2p(p), radius=0.09, color=col) for p in pts]))
        end = Dot(nl.n2p(8), radius=0.09, color=YELLOW_T)
        gaps = VGroup(*[
            Text(f"gap {g}", font_size=24, color=c).next_to(nl.n2p(m), UP, buff=0.5)
            for g, m, c in (("¼", 1.5, BLUE_T), ("½", 3, GREEN_T), ("1", 6, YELLOW_T))])

        self.say("nl_a", Write(head), Create(nl))
        self.say("nl_b", LaggedStart(*[GrowFromCenter(d) for d in groups[0]], lag_ratio=0.3),
                 FadeIn(gaps[0]))
        self.say("nl_c", LaggedStart(*[GrowFromCenter(d) for d in groups[1]], lag_ratio=0.3),
                 FadeIn(gaps[1]))
        self.say("nl_d", LaggedStart(*[GrowFromCenter(d) for d in [*groups[2], end]],
                                     lag_ratio=0.3), FadeIn(gaps[2]))
        self.play(*[FadeOut(m) for m in (nl, *groups, end, gaps)], run_time=0.6)

        # zoom in near 256 in bf16
        z = numline(252, 262, 1, 11, size=26).shift(DOWN * 0.3)
        bf = VGroup(*[Dot(z.n2p(v), radius=0.11, color=GREEN_T) for v in range(252, 263, 2)])
        bf_l = Text("bf16: 7 mantissa bits, gap 2 at 256", font_size=30, color=GREEN_T)
        bf_l.move_to(UP * 2.2)
        p = Dot(z.n2p(257), radius=0.11, color=RED_T).shift(UP * 0.9)
        p_l = Text("256 + 1", font_size=26, color=RED_T).next_to(p, UP)
        half = Brace(Line(z.n2p(256), z.n2p(258)), UP, buff=0.35, color=GREY_T)
        self.say("nl_e", Transform(head, Text("Zooming in at 256", font_size=34).to_edge(UP)),
                 Create(z), [FadeIn(bf_l), LaggedStart(*[GrowFromCenter(d) for d in bf],
                                                       lag_ratio=0.15)])
        self.say("nl_f", [FadeIn(p), FadeIn(p_l)], GrowFromCenter(half),
                 [p.animate.move_to(z.n2p(256)), p_l.animate.next_to(z.n2p(256), UP, buff=0.9),
                  FadeOut(half)])
        fp = VGroup(*[Dot(z.n2p(252 + i / 4), radius=0.045, color=BLUE_T)
                      for i in range(41)]).shift(UP * 0.4)
        fp_l = Text("fp16: 10 mantissa bits, gap ¼ at 256", font_size=30, color=BLUE_T)
        fp_l.next_to(z, DOWN, buff=1.0)
        ok = Dot(z.n2p(257), radius=0.08, color=YELLOW_T).shift(UP * 0.4)
        self.say("nl_g", [FadeIn(fp_l), LaggedStart(*[FadeIn(d) for d in fp], lag_ratio=0.03)],
                 Flash(ok, color=YELLOW_T), FadeIn(ok))
        self.play(*[FadeOut(m) for m in (z, bf, bf_l, p, p_l, fp, fp_l, ok)], run_time=0.6)

        # fp16's ceiling
        r = numline(0, 80000, 10000, 11, size=22).shift(DOWN * 0.3)
        span = Rectangle(width=r.n2p(65504)[0] - r.n2p(0)[0], height=0.5, stroke_width=0,
                         fill_color=BLUE_T, fill_opacity=0.5)
        span.move_to(r.n2p(0), aligned_edge=LEFT).shift(UP * 0.5)
        cap = DashedLine(r.n2p(65504) + UP * 1.4, r.n2p(65504) + DOWN * 0.2, color=YELLOW_T)
        cap_l = Text("fp16 max 65504", font_size=26, color=YELLOW_T)
        cap_l.next_to(cap.get_top(), LEFT, buff=0.2)
        x = Dot(r.n2p(70000) + UP * 0.5, color=RED_T)
        x_l = Text("70000 → inf", font_size=26, color=RED_T).next_to(x, UP, buff=0.3)
        x_l.shift(RIGHT * 0.6)
        self.say("nl_h", Transform(head, Text("fp16: 5 exponent bits", font_size=34).to_edge(UP)),
                 Create(r), GrowFromEdge(span, LEFT), [Create(cap), FadeIn(cap_l)])
        self.say("nl_i", FadeIn(x, scale=2), [x.animate.set_color(RED_T).scale(1.6),
                                              FadeIn(x_l)], Indicate(x_l, color=RED_T))

    # 3. the family, aligned by field
    def formats(self):
        head = Text("Where each format spends its bits", font_size=36).to_edge(UP)
        specs = [("fp32", 1, 8, 23), ("tf32", 1, 8, 10), ("fp16", 1, 5, 10),
                 ("bf16", 1, 8, 7), ("fp8 E4M3", 1, 4, 3), ("fp8 E5M2", 1, 5, 2)]
        rows = VGroup()
        for i, (n, s, e, m) in enumerate(specs):
            st = strip(s, e, m)
            st.move_to(LEFT * 4.6 + UP * (1.9 - i * 0.82), aligned_edge=LEFT)
            lab = Text(n, font_size=26).next_to(st, LEFT, buff=0.35)
            lab.align_to(LEFT * 6.6, LEFT)
            cnt = Text(f"{s+e+m} bits  ({e}e {m}m)", font_size=22, color=GREY_B)
            cnt.next_to(st, RIGHT, buff=0.3)
            rows.add(VGroup(lab, st, cnt))
        legend = VGroup(Text("sign", font_size=22, color=GREY_T),
                        Text("exponent = range", font_size=22, color=BLUE_T),
                        Text("mantissa = precision", font_size=22, color=GREEN_T))
        legend.arrange(RIGHT, buff=0.8).to_edge(DOWN, buff=0.35)

        def exp_box(i, e):
            return SurroundingRectangle(rows[i][1][1:1 + e], color=YELLOW_T, buff=0.03)

        self.say("fm_a", Write(head), FadeIn(legend))
        self.say("fm_b", FadeIn(rows[0], shift=RIGHT * 0.3))
        self.say("fm_c", FadeIn(rows[1], shift=RIGHT * 0.3),
                 Indicate(rows[1][1][9:], color=YELLOW_T))
        self.say("fm_d", FadeIn(rows[2], shift=RIGHT * 0.3),
                 Indicate(rows[2][1][1:6], color=RED_T), Indicate(rows[2][1][6:], color=YELLOW_T))
        b0, b3 = exp_box(0, 8), exp_box(3, 8)
        self.say("fm_e", FadeIn(rows[3], shift=RIGHT * 0.3), [Create(b0), Create(b3)],
                 Indicate(rows[3][1][9:], color=RED_T))
        self.say("fm_f", [FadeOut(b0), FadeOut(b3)], FadeIn(rows[4], shift=RIGHT * 0.3))
        self.say("fm_g", FadeIn(rows[5], shift=RIGHT * 0.3),
                 [Circumscribe(rows[5][1][1:6], color=BLUE_T)])

    # 4. how training precision moved down the list
    def history(self):
        head = Text("ML dtype progression", font_size=40).to_edge(UP)
        steps = [("fp32", "very slow", GREY_B),
                 ("fp16 + fp32", "mixed precision: much faster, unstable", BLUE_T),
                 ("bf16 + fp32", "same recipe, fp32's range: stable", GREEN_T),
                 ("fp8", "DeepSeek-V3 trained in it", YELLOW_T),
                 ("fp6 / fp4", "Blackwell, mostly inference", RED_T)]
        nodes = VGroup()
        for i, (n, d, c) in enumerate(steps):
            dot = Dot(radius=0.12, color=c)
            nm = Text(n, font_size=28, color=c)
            ds = Text(d, font_size=20, color=GREY_B)
            g = VGroup(dot, nm, ds)
            dot.move_to(LEFT * 5 + UP * (2.0 - i * 1.0))
            nm.next_to(dot, RIGHT, buff=0.3)
            ds.next_to(nm, RIGHT, buff=0.4)
            nodes.add(g)
        hstrips = VGroup(*[strip(*f, cell=0.09) for f in
                           ((1, 8, 23), (1, 5, 10), (1, 8, 7), (1, 4, 3), (1, 2, 1))])
        for st, g in zip(hstrips, nodes):
            st.move_to(RIGHT * 4.0 + g[0].get_y() * UP, aligned_edge=LEFT)
        spine = Line(nodes[0][0].get_center(), nodes[-1][0].get_center(), color=GREY_T,
                     stroke_width=2)

        self.say("hs_a", Write(head), Create(spine))
        self.say("hs_b", FadeIn(nodes[0], shift=RIGHT * 0.2), Create(hstrips[0]))
        self.say("hs_c", FadeIn(nodes[1], shift=RIGHT * 0.2), Create(hstrips[1]))
        ul = Underline(nodes[1][2], color=RED_T)
        self.say("hs_d", Indicate(nodes[1][2], color=RED_T), Create(ul),
                 Wiggle(hstrips[1][1:6]))

        # the 104B example: scale after vs before the Q·K matmul
        box = RoundedRectangle(corner_radius=0.15, width=8.6, height=2.6,
                               stroke_color=GREY_T, fill_color=BLACK, fill_opacity=0.95)
        box.shift(DOWN * 1.2 + RIGHT * 1.6)
        before = Text("(Q · K) / norm", font_size=30).move_to(box.get_center() + UP * 0.6)
        before_n = Text("product can blow up in fp16 first", font_size=22, color=RED_T)
        before_n.next_to(before, DOWN, buff=0.15)
        after = Text("(Q / √norm) · (K / √norm)", font_size=30, color=GREEN_T)
        after.move_to(box.get_center() + DOWN * 0.4)
        after_n = Text("scaled before the multiply", font_size=22, color=GREEN_T)
        after_n.next_to(after, DOWN, buff=0.15)
        boom = Arrow(DOWN * 0.4, UP * 0.4, color=RED_T, buff=0).next_to(before, RIGHT, buff=0.4)
        fix = Arrow(before_n.get_bottom(), after.get_top(), buff=0.08, color=GREEN_T,
                    stroke_width=3, max_tip_length_to_length_ratio=0.3)
        self.say("hs_e", [FadeIn(box), FadeOut(ul)], Write(before),
                 [FadeIn(before_n), GrowArrow(boom)], Wiggle(boom))
        self.say("hs_f", GrowArrow(fix), Write(after), FadeIn(after_n))
        self.play(FadeOut(VGroup(box, before, before_n, after, after_n, boom, fix)),
                  run_time=0.5)
        self.say("hs_g", FadeIn(nodes[2], shift=RIGHT * 0.2), Create(hstrips[2]),
                 Indicate(hstrips[2][1:9]))
        dflt = SurroundingRectangle(VGroup(nodes[2], hstrips[2]), color=GREEN_T, buff=0.12)
        dflt_l = Text("still the default", font_size=20, color=GREEN_T)
        dflt_l.next_to(dflt, UP, buff=0.05, aligned_edge=RIGHT)
        self.say("hs_h", FadeIn(nodes[3], shift=RIGHT * 0.2), Create(hstrips[3]),
                 [Create(dflt), FadeIn(dflt_l)])
        self.say("hs_i", FadeIn(nodes[4], shift=RIGHT * 0.2), Create(hstrips[4]))

    # 5. B200 throughput by dtype
    def speed(self):
        head = Text("B200 peak TFLOPS (dense)", font_size=36).to_edge(UP)
        data = [("fp32", 80), ("tf32", 1125), ("bf16", 2250), ("fp8", 4500),
                ("fp6", 4500), ("fp4", 9000)]
        chart, labels = bar_chart([n for n, _ in data], [v for _, v in data], 9000, 4.0, 10,
                                  [GREY_T, BLUE_T, BLUE_T, GREEN_T, YELLOW_T, RED_T])
        VGroup(chart, labels).move_to(DOWN * 0.6)
        bars = chart.bars

        def hop(i, j, txt, col):
            a = CurvedArrow(bars[i].get_top() + UP * 0.45, bars[j].get_top() + UP * 0.45,
                            angle=-PI / 3, color=col, stroke_width=3, tip_length=0.18)
            t = Text(txt, font_size=24, color=col).next_to(a, UP, buff=0.05)
            return VGroup(a, t)

        h1, h2 = hop(1, 2, "×2", GREEN_T), hop(2, 3, "×2", GREEN_T)
        h3 = hop(3, 4, "×1", RED_T)
        h4 = hop(3, 5, "×2", GREEN_T).shift(UP * 0.7)
        gb = caption("GB300: fp4 15000 vs fp8 5000 = 3x   →   check the spec for your dtype",
                     24, YELLOW_T)

        self.say("sp_a", Write(head))
        self.say("sp_b", Create(chart, run_time=2), FadeIn(labels))
        self.say("sp_c", Create(h1), Create(h2))
        self.say("sp_d", Create(h3), Indicate(bars[4], color=RED_T))
        self.say("sp_e", Create(h4), Indicate(bars[5], color=YELLOW_T))
        inset, ilab = bar_chart(["fp8", "fp4"], [5000, 15000], 15000, 2.0, 2.2,
                                [GREEN_T, RED_T], size=20)
        ititle = Text("GB300", font_size=24, color=YELLOW_T)
        ig = VGroup(inset, ilab)
        ig.move_to(LEFT * 4.6 + UP * 1.2)
        ititle.next_to(ig, UP, buff=0.2)
        i3 = Text("3x", font_size=32, color=YELLOW_T).next_to(inset.bars[1], RIGHT, buff=0.2)
        self.say("sp_f", [FadeIn(ititle), Create(inset[0]), FadeIn(inset[2])],
                 GrowFromEdge(inset.bars[0], DOWN), GrowFromEdge(inset.bars[1], DOWN),
                 [FadeIn(ilab), FadeIn(i3)], Write(gb))
        self.say("tf_a", [FadeOut(gb), FadeOut(VGroup(ig, ititle, i3)),
                          Circumscribe(VGroup(bars[0], labels[0]), color=YELLOW_T)])

    # 6. TF32
    def tf32(self):
        head = Text("TF32 on A100", font_size=40).to_edge(UP)
        chart, lab = bar_chart(["fp32", "tf32"], [19.5, 156], 156, 3.4, 3.6, [GREY_T, BLUE_T])
        VGroup(chart, lab).move_to(LEFT * 4.6 + DOWN * 0.3)
        x8 = Text("8x", font_size=48, color=YELLOW_T).next_to(chart.bars[1], RIGHT, buff=0.3)
        code = VGroup(
            Text("# off by default", font_size=20, color=GREY_B, font="Monospace"),
            Text("torch.backends.cuda.matmul.allow_tf32 = True", font_size=20,
                 font="Monospace"),
            Text("torch.backends.cudnn.allow_tf32 = True", font_size=20, font="Monospace"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 2.4 + DOWN * 0.2)
        frame = SurroundingRectangle(code, color=GREY_T, buff=0.3, corner_radius=0.1)
        self.say("tf_b", Write(head), [Create(chart[0]), FadeIn(chart.names)],
                 GrowFromEdge(chart.bars[0], DOWN), FadeIn(lab[0]),
                 GrowFromEdge(chart.bars[1], DOWN), FadeIn(lab[1]), FadeIn(x8, scale=1.5))
        self.say("tf_c", Create(frame), FadeIn(code[0]), FadeIn(code[1]), FadeIn(code[2]))

    # 7. accumulate in fp32
    def accum(self):
        head = Text("Keep the sums wide", font_size=40).to_edge(UP)
        z = numline(252, 262, 1, 11, size=26).shift(UP * 0.6)
        bf = VGroup(*[Dot(z.n2p(v), radius=0.1, color=GREEN_T) for v in range(252, 263, 2)])
        w = Dot(z.n2p(256), radius=0.15, color=BLUE_T)
        w_l = Text("weight (bf16)", font_size=24, color=BLUE_T).next_to(w, DOWN, buff=0.5)
        upd = Arrow(z.n2p(256) + UP * 0.7, z.n2p(256.6) + UP * 0.7, buff=0,
                    color=YELLOW_T, stroke_width=4, max_tip_length_to_length_ratio=0.5)
        u_l = Text("+ update", font_size=24, color=YELLOW_T).next_to(upd, UP)
        ghost = Dot(z.n2p(256.6), radius=0.12, color=YELLOW_T)
        lost = Text("update lost", font_size=28, color=RED_T).next_to(z, UP, buff=1.3)

        self.say("ac_a", Write(head), Create(z), FadeIn(bf))
        self.say("ac_b", GrowFromCenter(w), FadeIn(w_l))
        self.say("ac_c", [GrowArrow(upd), FadeIn(u_l)], FadeIn(ghost))
        self.say("ac_d", ghost.animate.move_to(w), [FadeOut(ghost), FadeIn(lost),
                                                    Flash(w, color=RED_T)])

        rules = VGroup(*[
            VGroup(Text(a, font_size=26, color=YELLOW_T), Text(b, font_size=26))
            .arrange(RIGHT, buff=0.3)
            for a, b in (("optimizer step", "fp32 master weights + optimizer states"),
                         ("grad accumulation", "fp32 (a must for bf16)"),
                         ("reductions", "fp16 ok with loss scaling, bf16 only in fp32"),
                         ("LayerNorm", "half-precision in, fp32 accumulators, cast back"),
                         ("half-precision master", "Kahan summation / stochastic rounding"))
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.3)
        for r in rules:
            r[1].align_to(LEFT * 2.1, LEFT)
            r[0].align_to(LEFT * 6.4, LEFT)
        top = VGroup(z, bf, w, w_l, upd, u_l, lost)
        ul = None
        for i, k in enumerate("efghi"):
            new = Underline(rules[i][1], color=YELLOW_T, buff=0.08)
            first = [FadeIn(rules[i], shift=UP * 0.2)]
            if i == 0:
                first.append(FadeOut(top))
            else:
                first.append(FadeOut(ul))
            self.say(f"ac_{k}", first, Indicate(rules[i][0], color=YELLOW_T), Create(new))
            ul = new

    # 8. fp4 and block scaling
    def fp4(self):
        head = Text("fp4 (E2M1): all 16 values", font_size=36).to_edge(UP)
        nl = NumberLine(x_range=[-6, 6, 1], length=12, include_numbers=False).shift(UP * 1.2)
        vals = sorted({s * v for v in FP4 for s in (1, -1)})
        dots = VGroup(*[Dot(nl.n2p(v), radius=0.09, color=YELLOW_T) for v in vals])
        nums = VGroup(*[Text(f"{v:g}", font_size=20).next_to(nl.n2p(v), DOWN, buff=0.2)
                        for v in vals if v >= 0])
        z2 = Text("±0", font_size=20, color=GREY_B).next_to(nl.n2p(0), UP, buff=0.2)

        self.say("f4_a", Write(head), Create(nl))
        self.say("f4_b", LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.12),
                 [FadeIn(nums), FadeIn(z2)])
        self.say("f4_c", Indicate(dots, color=RED_T))
        top = VGroup(nl, dots, nums, z2)
        self.play(top.animate.scale(0.55).to_corner(UR, buff=0.4).shift(DOWN * 0.6),
                  Transform(head, Text("Block scaling", font_size=36).to_edge(UP)),
                  run_time=0.8)

        base_y, x0, dx = -0.9, -5.2, 0.85
        block = [0.11, -0.32, 0.05, 0.27, -0.18, 0.40, 0.08, -0.22]
        outl = list(block)
        outl[5] = 3.0
        axis = Line(LEFT * 5.8 + UP * base_y, RIGHT * 2.0 + UP * base_y, color=GREY_T)

        def bars(vals, k, col=BLUE_T):
            g = VGroup()
            for i, v in enumerate(vals):
                h = max(abs(v) * k, 0.02)
                r = Rectangle(width=0.5, height=h, stroke_width=0, fill_color=col,
                              fill_opacity=0.75)
                r.move_to([x0 + i * dx, base_y + np.sign(v) * h / 2, 0])
                g.add(r)
            return g

        def marks(vals, scale, k):
            return VGroup(*[Line(LEFT * 0.32, RIGHT * 0.32, color=YELLOW_T, stroke_width=5)
                            .move_to([x0 + i * dx, base_y + q4(v, scale) * k, 0])
                            for i, v in enumerate(vals)])

        def levels(scale, k):
            g = VGroup()
            for v in sorted({s * p for p in FP4[1:] for s in (1, -1)}):
                y = base_y + v * scale * k
                if abs(y - base_y) < 2.6:
                    g.add(DashedLine([x0 - 0.5, y, 0], [x0 + 7.5 * dx, y, 0],
                                     stroke_width=1.2, color=YELLOW_T, stroke_opacity=0.45))
            return g

        k1 = 5.0  # 0.4 -> 2 units tall
        s1 = 0.40 / 6
        b1 = bars(block, k1)
        lv1 = levels(s1, k1)
        m1 = marks(block, s1, k1)
        sc1 = Text("scale = 0.4 / 6", font_size=26, color=YELLOW_T).move_to(RIGHT * 4.3 + DOWN * 0.6)
        n_lv1 = len({round(q4(v, s1), 6) for v in block})

        self.say("f4_d", Create(axis), LaggedStart(*[GrowFromEdge(b, DOWN if b.get_y() > base_y else UP)
                                                   for b in b1], lag_ratio=0.15))
        self.say("f4_e", Indicate(b1[5], color=YELLOW_T), [FadeIn(sc1), Create(lv1)])
        lvl_txt1 = Text(f"{n_lv1} distinct levels used", font_size=24, color=GREEN_T)
        lvl_txt1.next_to(sc1, DOWN, buff=0.4)
        self.say("f4_f", LaggedStart(*[FadeIn(m, shift=DOWN * 0.1) for m in m1], lag_ratio=0.1),
                 FadeIn(lvl_txt1))

        k2 = 0.8  # 3.0 -> 2.4 units tall
        s2 = 3.0 / 6
        b2 = bars(outl, k2)
        b2[5].set_fill(RED_T)
        lv2 = levels(s2, k2)
        m2 = marks(outl, s2, k2)
        sc2 = Text("scale = 3.0 / 6", font_size=26, color=RED_T).move_to(sc1)
        n_lv2 = len({round(q4(v, s2), 6) for v in outl if v != 3.0})
        lvl_txt2 = Text(f"{n_lv2} levels left for the other 7", font_size=24, color=RED_T)
        lvl_txt2.move_to(lvl_txt1)
        tall = bars(outl, k1)[5].set_fill(RED_T).stretch_to_fit_height(2.6)
        tall.move_to([x0 + 5 * dx, base_y + 1.3, 0])
        self.say("f4_g", FadeOut(m1), Transform(b1[5], tall))
        self.say("f4_h", [Transform(b1, b2), Transform(lv1, lv2), Transform(sc1, sc2)])
        self.say("f4_i", LaggedStart(*[FadeIn(m, shift=DOWN * 0.1) for m in m2], lag_ratio=0.1),
                 Transform(lvl_txt1, lvl_txt2))

    # 9. mxfp4 vs nvfp4
    def mx(self):
        head = Text("mxfp4 vs nvfp4", font_size=40).to_edge(UP)
        cell = 0.3

        def row(split):
            sq = VGroup(*[Square(cell, stroke_width=1, stroke_color=BLACK, fill_color=BLUE_T,
                                 fill_opacity=0.7) for _ in range(32)])
            sq.arrange(RIGHT, buff=0)
            if split:
                sq[16:].shift(RIGHT * 0.3)
            return sq

        r1, r2 = row(False), row(True)
        r1.move_to(UP * 1.6 + LEFT * 1.6)
        r2.move_to(DOWN * 0.9 + LEFT * 1.6)
        sc1 = Square(cell, stroke_width=1, fill_color=GREY_T, fill_opacity=1)
        sc1.next_to(r1, RIGHT, buff=0.3)
        sc2 = VGroup(Square(cell, fill_color=GREY_T, fill_opacity=1, stroke_width=1)
                     .next_to(r2[15], UP, buff=0.12),
                     Square(cell, fill_color=GREY_T, fill_opacity=1, stroke_width=1)
                     .next_to(r2[31], UP, buff=0.12))
        l1 = Text("mxfp4: 32 per block, E8M0 scale (power of two)", font_size=24)
        l1.next_to(r1, UP, buff=0.3, aligned_edge=LEFT)
        l2 = Text("nvfp4: 16 per block, E4M3 scale + one fp32 per tensor", font_size=24)
        l2.next_to(r2, UP, buff=0.55, aligned_edge=LEFT)
        bits1 = Text("4 + 8/32 = 4.25 bits", font_size=26, color=YELLOW_T).next_to(r1, DOWN, buff=0.3,
                                                                                  aligned_edge=LEFT)
        bits2 = Text("4 + 8/16 = 4.5 bits", font_size=26, color=YELLOW_T).next_to(r2, DOWN, buff=0.3,
                                                                                 aligned_edge=LEFT)

        self.say("mx_a", Write(head))
        self.say("mx_b", [FadeIn(r1), FadeIn(l1)], FadeIn(sc1, shift=LEFT * 0.2))
        self.say("mx_c", Write(bits1), Indicate(sc1, color=YELLOW_T))
        br = VGroup(Brace(r2[:16], DOWN, buff=0.05), Brace(r2[16:], DOWN, buff=0.05))
        self.say("mx_d", [FadeIn(r2), FadeIn(l2)], FadeIn(sc2, shift=DOWN * 0.2),
                 [GrowFromCenter(br[0]), GrowFromCenter(br[1])])
        self.play(FadeOut(br), run_time=0.3)
        self.say("mx_e", Write(bits2))
        o = 5
        hit1 = [r1[i].animate.set_fill(RED_T, 0.35) for i in range(32) if i != o]
        hit2 = [r2[i].animate.set_fill(RED_T, 0.35) for i in range(16) if i != o]
        n1 = Text("31 distorted", font_size=24, color=RED_T).next_to(sc1, RIGHT, buff=0.3)
        n2 = Text("15 distorted", font_size=24, color=RED_T).next_to(r2, RIGHT, buff=0.3)
        self.say("mx_f", [r1[o].animate.set_fill(RED_T, 1), r2[o].animate.set_fill(RED_T, 1)],
                 [*hit1, FadeIn(n1)], [*hit2, FadeIn(n2)])

        grp = VGroup(r1, r2, sc1, sc2, l1, l2, bits1, bits2, n1, n2)
        same = Text("Blackwell: same instruction, same peak for both", font_size=26)
        tok = VGroup(Text("NVIDIA 8B / 1T-token pretraining:", font_size=26),
                     Text("mxfp4 needed ~36% more tokens to match nvfp4's loss", font_size=26,
                          color=YELLOW_T)).arrange(DOWN, buff=0.2)
        src = Text("(NVIDIA's own claim)", font_size=20, color=GREY_B)
        pick = VGroup(Text("Blackwell or newer → nvfp4", font_size=28, color=GREEN_T),
                      Text("other vendors / cross-vendor checkpoint → mxfp4", font_size=28,
                           color=BLUE_T)).arrange(DOWN, buff=0.25)
        self.say("mx_g", grp.animate.scale(0.6).to_edge(UP, buff=1.1),
                 FadeIn(same.shift(DOWN * 0.8 + LEFT * 1.6)))
        tok.next_to(same, DOWN, buff=0.4)
        src.next_to(tok, DOWN, buff=0.15)
        tb, tl = bar_chart(["nvfp4", "mxfp4"], [1.0, 1.36], 1.36, 1.3, 2.4,
                           [GREEN_T, BLUE_T], size=20)
        tl = VGroup(Text("1.00x", font_size=20), Text("~1.36x", font_size=20))
        for b, t in zip(tb.bars, tl):
            t.next_to(b, UP, buff=0.1)
        tg = VGroup(tb, tl)
        tg.next_to(tok, RIGHT, buff=0.5).shift(DOWN * 0.3)
        tt = Text("tokens to same loss", font_size=18, color=GREY_B).next_to(tg, UP, buff=0.15)
        self.say("mx_h", FadeIn(tok[0]), [FadeIn(tok[1]), FadeIn(src)],
                 [Create(tb[0]), FadeIn(tb.names), FadeIn(tt)],
                 GrowFromEdge(tb.bars[0], DOWN), GrowFromEdge(tb.bars[1], DOWN), FadeIn(tl))
        pick.move_to(DOWN * 1.4)
        pbox = SurroundingRectangle(pick, color=GREY_T, buff=0.3, corner_radius=0.1)
        self.say("mx_i", [FadeOut(same), FadeOut(tok), FadeOut(src), FadeOut(tg), FadeOut(tt)],
                 FadeIn(pick[0]), FadeIn(pick[1]), Create(pbox))

    # 10. reading a dtype name
    def name(self):
        head = Text("Reading a dtype name", font_size=40).to_edge(UP)
        t = Text("float8_e4m3b11fnuz", font_size=60, font="Monospace").shift(UP * 1.2)
        parts = [(slice(0, 6), "8 bits total", WHITE),
                 (slice(7, 9), "4 exponent bits", BLUE_T),
                 (slice(9, 11), "3 mantissa bits", GREEN_T),
                 (slice(11, 14), "bias 11", YELLOW_T),
                 (slice(14, 15), "finite only (no inf)", RED_T),
                 (slice(15, 16), "NaNs only at outer range", RED_T),
                 (slice(16, 18), "unsigned zero", RED_T)]
        toks = ["float8", "e4", "m3", "b11", "f", "n", "uz"]
        notes = VGroup()
        for (sl, txt, col), tk in zip(parts, toks):
            notes.add(VGroup(Text(tk, font_size=26, color=col, font="Monospace"),
                             Text(txt, font_size=26, color=col)))
        for i, n in enumerate(notes):
            y = 0.1 - i * 0.55
            n[0].move_to([-3.0, y, 0], aligned_edge=LEFT)
            n[1].move_to([-0.6, y, 0], aligned_edge=LEFT)
        marks = VGroup(*[Underline(t[sl], color=col, buff=0.08) for sl, _, col in parts])

        def show(i):
            sl, _, col = parts[i]
            return [t[sl].animate.set_color(col), Create(marks[i]), FadeIn(notes[i])]

        self.say("nm_a", Write(head), FadeIn(t))
        self.say("nm_b", show(0), show(1), show(2))
        for i, k in ((3, "c"), (4, "d"), (5, "e"), (6, "f")):
            self.say(f"nm_{k}", show(i))

    # 11. switching precision after training, and the close
    def switch(self):
        head = Text("Changing precision after training", font_size=38).to_edge(UP)
        rows = VGroup(
            VGroup(Text("bf16-trained → fp16", font_size=30),
                   Text("usually fails: overflows past 65504", font_size=26, color=RED_T)),
            VGroup(Text("fp16-trained → bf16", font_size=30),
                   Text("usually works: small loss, finetune a bit", font_size=26,
                        color=GREEN_T)),
        )
        for r in rows:
            r.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 1.5 + LEFT * 2)
        f16 = Rectangle(width=3.0, height=0.35, stroke_width=0, fill_color=BLUE_T,
                        fill_opacity=0.7).move_to(LEFT * 4.5 + UP * 1.7, aligned_edge=LEFT)
        b16 = Rectangle(width=10.5, height=0.35, stroke_width=0, fill_color=GREEN_T,
                        fill_opacity=0.7).move_to(LEFT * 4.5 + UP * 0.9, aligned_edge=LEFT)
        f16_l = Text("fp16 range", font_size=22, color=BLUE_T).next_to(f16, LEFT, buff=0.2)
        b16_l = Text("bf16 range", font_size=22, color=GREEN_T).next_to(b16, LEFT, buff=0.2)
        lim = DashedLine(f16.get_right() + UP * 0.4, f16.get_right() + DOWN * 1.2,
                         color=YELLOW_T)
        lim_l = Text("65504", font_size=20, color=YELLOW_T).next_to(lim, UP, buff=0.1)
        big = Dot(b16.get_left() + RIGHT * 4.5, color=WHITE)
        small = Dot(f16.get_left() + RIGHT * 1.6, color=WHITE)
        self.say("sw_a", Write(head), [GrowFromEdge(f16, LEFT), FadeIn(f16_l)],
                 [GrowFromEdge(b16, LEFT), FadeIn(b16_l)], [Create(lim), FadeIn(lim_l)])
        self.say("sw_b", FadeIn(big, scale=2),
                 big.animate.move_to(f16.get_left() + RIGHT * 4.5).set_color(RED_T),
                 Flash(big, color=RED_T), [FadeIn(rows[0][0]), FadeIn(rows[0][1])])
        self.say("sw_c", FadeIn(small, scale=2),
                 small.animate.move_to(b16.get_left() + RIGHT * 1.6).set_color(GREEN_T),
                 [FadeIn(rows[1][0]), FadeIn(rows[1][1])])
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

        strips = VGroup(*[strip(s, e, m, 0.3) for s, e, m in
                          ((1, 8, 23), (1, 8, 7), (1, 4, 3), (1, 2, 1))])
        strips.arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.8)
        speed = Text("fewer bits → more speed", font_size=32).next_to(strips, DOWN, buff=0.6)
        split = VGroup(Text("range", font_size=32, color=BLUE_T),
                       Text("or", font_size=32),
                       Text("precision?", font_size=32, color=GREEN_T)).arrange(RIGHT, buff=0.3)
        split.next_to(speed, DOWN, buff=0.35)
        last = Text("and keep the sums in fp32", font_size=32, color=YELLOW_T)
        last.next_to(split, DOWN, buff=0.35)
        self.say("sw_d", LaggedStart(*[FadeIn(s, shift=RIGHT * 0.3) for s in strips],
                                     lag_ratio=0.3), FadeIn(speed), FadeIn(split))
        self.say("sw_e", Write(last), Circumscribe(last, color=YELLOW_T))


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
