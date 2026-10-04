"""Video 2: Counting FLOPs.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python counting_flops.py
2. Render: manim -qh counting_flops.py CountingFlops

Every number comes from training/performance/README.md (MACs vs FLOP vs FLOPS,
TFLOPS as a performance metric, MFU vs HFU, forward vs backward Execution Speed),
compute/accelerator/README.md (How To Calculate Theoretical TFLOPS, MAMF/MSMF
tables and notes, Why MSMF is well below Theory, Not all accelerators are created
equal) and insights/ai-battlefield.md (TFLOPS, MFU).
"""
import wave
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
YELLOW_T = "#FFFF00"
RED_T = "#FC6255"
GREEN_T = "#83C167"
PURPLE_T = "#9A72AC"

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "hook_a": "Here's a number from a GPU spec sheet. An A100 can do 312 trillion "
              "floating point operations per second in bf16.",
    "hook_b": "When BLOOM was trained on 384 of these, the team got about 150 out of "
              "each one, and that was considered an amazing result.",
    "hook_c": "So less than half of the spec counts as doing well. To see why, and to "
              "predict how long a training run will take, you and I need to count "
              "FLOPs carefully, starting with what a FLOP even is.",

    "mac_a": "A FLOP is a single floating point operation: one add, subtract, "
             "multiply, or divide.",
    "mac_b": "Hardware mostly works in a slightly bigger unit, the multiply-accumulate, "
             "or MAC, which is a times b, plus c. That's a multiply and an add, so one "
             "MAC is two FLOPs.",
    "mac_c": "Now take a matrix multiply, an m by k matrix times a k by n matrix, "
             "giving an m by n output.",
    "mac_d": "Each of those outputs is a dot product of a row and a column, both of "
             "length k, so it takes k MACs.",
    "mac_e": "Do that for every output and you get m times n times k MACs. You can "
             "picture it as a box, with one little cube for each multiply-accumulate,",
    "mac_f": "and since every MAC is two FLOPs, the whole matmul costs two m n k FLOPs. "
             "We'll lean on that formula for the rest of the video.",

    "nm_a": "One annoyance before we go on, which is that the names are a mess.",
    "nm_b": "FLOPS with a capital S usually means operations per second,",
    "nm_c": "but FLOPs with a small s could be a rate or a total count, since it's so "
            "easy to flip the case of that s,",
    "nm_d": "which is why scientific writing often spells out FLOP per second.",
    "nm_e": "In practice you go by context. If there's a division by time it's a rate, "
            "and if someone is talking about how much compute a job needs, it's a count.",

    "sp_a": "So where does a spec number like the H100's 989 TFLOPS come from? "
            "It's the product of a few dials.",
    "sp_b": "The first is the clock, how many times per second the compute units tick, "
            "which for the H100 is 1830 megahertz.",
    "sp_c": "The second is how many fused multiply-adds each tensor core does per tick, "
            "512 for bf16, and a fused multiply-add is the same two FLOPs as a MAC.",
    "sp_d": "The third is how many tensor cores there are. The H100 has 132 SMs with 4 "
            "tensor cores each, so 528.",
    "sp_e": "Multiply them all together and you get 989.4 TFLOPS, which matches the spec.",
    "sp_f": "Here's a nice side effect. For a long time the widely published H100 clock "
            "was 1980 megahertz, but plug that in and you get 1070, about 80 above "
            "NVIDIA's own spec.",
    "sp_g": "Run the formula backwards from 989 and the clock comes out at 1829, so the "
            "formula tells you which clock the vendor actually used. 1830 has since "
            "become the official number.",
    "sp_h": "Keep in mind that the clock is one of the dials. If the chip runs below "
            "its boost clock, the whole product drops with it, and we'll come back "
            "to that.",

    "st_a": "Now for the other side of the division: how many FLOPs does one training "
            "step need?",
    "st_b": "Go back to the matmul. A linear layer with a k by n weight has k times n "
            "parameters, and pushing m tokens through it costs two m k n FLOPs.",
    "st_c": "That's two FLOPs per parameter per token, and the standard estimate treats "
            "the whole model that way, so the forward pass costs about two times "
            "parameters times tokens.",
    "st_d": "The backward pass costs twice as much as the forward, because it computes "
            "gradients with respect to the inputs and with respect to the weights, and "
            "each of those is a matmul as big as the forward one.",
    "st_e": "So forward plus backward is three forwards, or six FLOPs per parameter "
            "per token.",
    "st_f": "And with activation checkpointing, the forward runs again during the "
            "backward, so you pay for four forwards, or eight.",
    "st_g": "Put that together and you get the usual formula for TFLOPS per GPU. Model "
            "size in billions, times four, times two, times sequence length, times "
            "global batch size,",
    "st_h": "divided by the seconds per iteration, the number of GPUs, and a thousand, "
            "which turns billions over seconds into TFLOPS.",
    "st_i": "Let's plug in one example: a 52 billion parameter model, sequence length "
            "2048, global batch 1024, on 64 GPUs, taking 127 seconds per iteration.",
    "st_j": "That comes out to about 107 TFLOPS per GPU. This estimate slightly "
            "under-reports the real number, but it's close enough to steer by.",

    "ce_a": "So far we've used the spec as the ceiling. But nobody reaches it, even "
            "with a single, perfectly shaped matmul and nothing else running.",
    "ce_b": "Take an H200, whose bf16 spec is the same 989 as the H100.",
    "ce_c": "If you sweep through lots of matmul shapes on one GPU and keep the fastest, "
            "you get the maximum achievable matmul FLOPS, or MAMF. For the H200 that's "
            "834, about 84 percent of spec.",
    "ce_d": "And if you keep the matmul running until the chip is saturated, it settles "
            "lower again, at 755. That's the maximum sustainable matmul FLOPS, MSMF, and "
            "it's the one to use for training, which is one long saturated run.",
    "ce_e": "Why would the same matmul get slower the longer it runs? Watch what the "
            "chip's power and clock do. This is a B200, and the shape of the curves is "
            "a sketch.",
    "ce_f": "During a short burst, the B200 draws around 300 watts and holds its full "
            "boost clock of 1965 megahertz.",
    "ce_g": "But a matmul that really fills every SM draws the full 1000 watt power "
            "limit, and to stay inside it, the clock settles down to around 1450.",
    "ce_h": "Scale the 2250 TFLOPS spec by that drop, roughly 1425 over 1965, and you "
            "get a ceiling of about 1630,",
    "ce_i": "and the measured MSMF of 1429 is in that ballpark, with the rest going to "
            "real kernel overhead. So nothing is broken, the chip is just running at "
            "its power limit.",

    "mf_a": "Now we can hold a real training run up to a ceiling. Divide the TFLOPS you "
            "achieved by the spec TFLOPS, and that's your utilization.",
    "mf_b": "Say you measured 400 TFLOPS on an H100. That's 400 over 989, about "
            "40 percent.",
    "mf_c": "But which FLOPs did you count? With checkpointing the forward runs twice, "
            "and the hardware really did do that work.",
    "mf_d": "Counting the recompute gives you hardware FLOPS utilization, or HFU, "
            "with the factor of four.",
    "mf_e": "Leaving it out gives model FLOPS utilization, or MFU, with the factor of "
            "three, which only counts the work the model itself needs.",
    "mf_f": "So with recompute on, MFU comes out a little below HFU, and you can see it "
            "in the numbers Megatron-LM published for A100s.",
    "mf_g": "For their 175 billion parameter model, that's 51.4 percent MFU against "
            "52.8 percent HFU.",
    "mf_h": "And as a rule of thumb, with NVIDIA GPUs, if you're above 50 percent MFU on "
            "a multi-node run with a large model, you're already doing fantastic.",

    "dy_a": "Here's where the counting pays off. Once you pick the model size and how "
            "many tokens to train on, you know the total FLOPs before the run even "
            "starts.",
    "dy_b": "Divide that by the hardware's TFLOPS and you get a time. Say it comes out to "
            "604,800 seconds, which is 7 days.",
    "dy_c": "Multiply by what that hardware costs per day, and you can compare offers "
            "from different providers.",
    "dy_d": "But those 7 days assumed the spec TFLOPS. At 50 percent MFU, the same "
            "run takes 14.",
    "dy_e": "Knowing what's achievable also tells you when to stop tuning. BLOOM went "
            "from under 100 TFLOPS to 150 in a few weeks,",
    "dy_f": "and at that point the team knew there wasn't much more to find, so they "
            "stopped optimizing and started training.",

    "cm_a": "You'll see MFU numbers in lots of papers, and it's tempting to line them up "
            "against each other. Usually, though, they don't compare.",
    "cm_b": "The first question is how they counted the FLOPs. There are many versions "
            "of the formula around, some slightly off and some very off, and a compiler "
            "can optimize operations away.",
    "cm_c": "The second is what they timed: the whole iteration back to back, including "
            "the DataLoader and logging, or only the forward and backward.",
    "cm_d": "If either one differs, the numbers aren't comparable, and most papers just "
            "report an MFU without saying how they got it.",
    "cm_e": "What does work is comparing your own setup with itself, before and after a "
            "change, counting the same way every time.",

    "lt_a": "One more thing the spec sheet can't tell you is that two GPUs with the same "
            "model number aren't the same chip.",
    "lt_b": "Tiny manufacturing differences mean some dies need more voltage to hold a "
            "given clock, which makes more heat, which pulls their clock down sooner "
            "under load.",
    "lt_c": "Add uneven cooling, where the GPUs nearer the cold air run cooler, and on a "
            "single 8 GPU node you can see 5 percent or more between them.",
    "lt_d": "When they train together, the slowest one sets the pace for all the others.",
    "lt_e": "So measure all of them at the same time, and use the slowest one's number "
            "as your ceiling.",

    "end_a": "So the next time you see a TFLOPS number, the useful question is which one "
             "it is: the spec, the burst, the sustained rate, or what a real run "
             "actually got.",
}


def caption(text, size=30, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.5)


def is_text(anim):
    m = anim.mobject
    return isinstance(m, Text) or (isinstance(m, VGroup) and m.submobjects and
                                   all(isinstance(s, Text) for s in m.submobjects))


def grid(rows, cols, s, color, opacity=0.25):
    g = VGroup(*[Square(s, stroke_width=1.5, stroke_color=color, fill_color=color,
                        fill_opacity=opacity) for _ in range(rows * cols)])
    return g.arrange_in_grid(rows, cols, buff=0)


def hbar(width, color, h=0.5, opacity=0.8):
    return Rectangle(width=width, height=h, fill_color=color, fill_opacity=opacity,
                     stroke_width=0)


class CountingFlops(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.macs, self.names, self.spec, self.step,
                     self.ceilings, self.mfu, self.days, self.compare, self.lottery,
                     self.ending):
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

    # 0. hook: 150 of 312
    def hook(self):
        chip = RoundedRectangle(corner_radius=0.2, width=2.4, height=2.4,
                                stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=1)
        pins = VGroup(*[Line(ORIGIN, RIGHT * 0.25, stroke_color=GREY_B)
                        .move_to(chip.get_left() + LEFT * 0.12 + UP * (y - 0.9))
                        for y in np.linspace(0, 1.8, 6)])
        pins.add(*[p.copy().move_to(chip.get_right() + RIGHT * 0.12 + UP * p.get_y())
                   for p in list(pins)])
        chip = VGroup(chip, pins, Text("A100", font_size=34)).shift(LEFT * 4.6)
        full_w = 7
        outline = Rectangle(width=full_w, height=0.7, stroke_color=WHITE)
        outline.move_to(RIGHT * 1.6 + UP * 0.4)
        spec = hbar(full_w, BLUE_T, 0.7, 0.25).move_to(outline)
        spec_l = Text("spec: 312 TFLOPS bf16", font_size=28, color=BLUE_T)
        spec_l.next_to(outline, UP, aligned_edge=LEFT)
        got = hbar(full_w * 150 / 312, GREEN_T, 0.7).align_to(outline, LEFT)
        got.set_y(outline.get_y())
        got_l = Text("BLOOM: ~150 per GPU", font_size=28, color=GREEN_T)
        got_l.next_to(got, DOWN, aligned_edge=LEFT)
        gap = Rectangle(width=full_w - got.width, height=0.7, stroke_width=0)
        gap.align_to(outline, RIGHT).set_y(outline.get_y())
        br = Brace(gap, DOWN, color=RED_T)
        q = Text("where does this go?", font_size=26, color=RED_T).next_to(br, DOWN)
        title = Text("Counting FLOPs", font_size=56).to_edge(UP)

        self.say("hook_a", FadeIn(chip, shift=RIGHT * 0.3),
                 [Create(outline), FadeIn(spec), Write(spec_l)])
        x384 = Text("× 384", font_size=34, color=GREY_B).next_to(chip, DOWN)
        self.say("hook_b", FadeIn(x384, shift=UP * 0.2),
                 GrowFromEdge(got, LEFT, run_time=2.5), Write(got_l))
        self.say("hook_c", [GrowFromCenter(br, run_time=1.5), Write(q)], Write(title))

    # 1. MAC vs FLOP, and the matmul box
    def macs(self):
        head = Text("MAC vs FLOP", font_size=40).to_edge(UP)
        ops = VGroup(*[Text(o, font_size=48) for o in "+−×÷"]).arrange(RIGHT, buff=0.6)
        flop = VGroup(Text("1 FLOP =", font_size=34), ops).arrange(RIGHT, buff=0.5)
        flop.shift(UP * 1.2)
        mac = VGroup(Text("1 MAC  =", font_size=34), Text("a × b", font_size=34),
                     Text("+ c", font_size=34)).arrange(RIGHT, buff=0.3)
        mac.next_to(flop, DOWN, buff=0.8).align_to(flop, LEFT)
        br_m = Brace(mac[1], DOWN, color=BLUE_T)
        br_a = Brace(mac[2], DOWN, color=GREEN_T)
        l_m = Text("multiply", font_size=24, color=BLUE_T).next_to(br_m, DOWN, buff=0.1)
        l_a = Text("add", font_size=24, color=GREEN_T).next_to(br_a, DOWN, buff=0.1)
        eq = Text("=  2 FLOPs", font_size=34, color=YELLOW_T).next_to(mac, RIGHT, buff=0.4)
        self.say("mac_a", Write(head), Write(flop[0]),
                 LaggedStart(*[FadeIn(o, scale=1.5) for o in ops], lag_ratio=0.25))
        self.say("mac_b", Write(mac), [GrowFromCenter(br_m), FadeIn(l_m)],
                 [GrowFromCenter(br_a), FadeIn(l_a)], Write(eq),
                 Circumscribe(eq, color=YELLOW_T))
        self.play(FadeOut(VGroup(flop, mac, eq, br_m, br_a, l_m, l_a)), run_time=0.6)

        s = 0.42
        m, k, n = 4, 5, 3
        A = grid(m, k, s, BLUE_T)
        B = grid(k, n, s, GREEN_T)
        C = grid(m, n, s, YELLOW_T, 0.15)
        A.move_to(LEFT * 2.2 + DOWN * 1.4)
        C.next_to(A, RIGHT, buff=0.5)
        B.next_to(C, UP, buff=0.5)
        la = Text("m × k", font_size=26, color=BLUE_T).next_to(A, LEFT)
        lb = Text("k × n", font_size=26, color=GREEN_T).next_to(B, RIGHT)
        lc = Text("m × n", font_size=26, color=YELLOW_T).next_to(C, RIGHT)
        self.say("mac_c", [Create(A), Write(la)], [Create(B), Write(lb)],
                 [Create(C), Write(lc)])

        i, j = 1, 2
        row = VGroup(*[A[i * k + c] for c in range(k)])
        col = VGroup(*[B[r * n + j] for r in range(k)])
        cell = C[i * n + j]
        kl = Text("k MACs per output", font_size=28, color=YELLOW_T)
        kl.next_to(VGroup(A, B, C), RIGHT, buff=0.9).shift(UP * 0.3)
        self.say("mac_d", [row.animate.set_fill(BLUE_T, 0.9), col.animate.set_fill(GREEN_T, 0.9)],
                 [cell.animate.set_fill(YELLOW_T, 0.9), Write(kl)])

        # a box of MACs: the m×n output stacked k deep, oblique projection
        layers = VGroup()
        off = np.array([0.22, 0.16, 0])
        base = C.copy().set_fill(YELLOW_T, 0.25)
        for d in range(k):
            layers.add(base.copy().shift(off * (k - 1 - d)))
        layers.move_to(RIGHT * 0.2 + DOWN * 0.2)
        box_l = Text("m · n · k MACs", font_size=32).next_to(layers, DOWN, buff=0.4)
        flops = Text("= 2mnk FLOPs", font_size=40, color=YELLOW_T)
        flops.next_to(layers, RIGHT, buff=0.8)
        self.say("mac_e", FadeOut(VGroup(A, B, la, lb, lc, kl)),
                 [C.animate.move_to(layers[-1]), FadeOut(lc)],
                 [LaggedStart(*[FadeIn(l, shift=-off) for l in layers[:-1][::-1]],
                              lag_ratio=0.3), Write(box_l)])
        self.say("mac_f", Write(flops), Circumscribe(flops, color=YELLOW_T, run_time=2))

    # 2. naming
    def names(self):
        head = Text("FLOP, FLOPS, FLOPs, FLOP/s", font_size=40).to_edge(UP)
        rows = [("FLOP", "one operation", WHITE),
                ("FLOPS", "operations per second (usually)", BLUE_T),
                ("FLOPs", "a rate or a count", RED_T),
                ("FLOP/s", "per second, unambiguous", GREEN_T)]
        tbl = VGroup()
        for a, b, c in rows:
            tbl.add(VGroup(Text(a, font_size=34, color=c), Text(b, font_size=30)))
        for r in tbl:
            r[1].next_to(r[0], RIGHT, buff=0.4)
        tbl.arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(UP * 0.3)
        left_x = max(r[0].get_right()[0] for r in tbl) + 0.6
        for r in tbl:
            r[1].align_to(np.array([left_x, 0, 0]), LEFT)
        rule = VGroup()
        for a, b in (("÷ time", "rate"), ("\"compute a job needs\"", "count")):
            ta, tb = Text(a, font_size=28), Text(b, font_size=28, color=YELLOW_T)
            arr = Arrow(LEFT * 0.5, RIGHT * 0.5, buff=0, color=YELLOW_T)
            rule.add(VGroup(ta, arr, tb).arrange(RIGHT, buff=0.25))
        rule.arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=0.6)
        self.say("nm_a", Write(head), FadeIn(tbl[0]))
        self.say("nm_b", FadeIn(tbl[1], shift=UP * 0.2), Circumscribe(tbl[1], color=BLUE_T))
        s_box = SurroundingRectangle(tbl[2][0][-1], color=RED_T, buff=0.06)
        self.say("nm_c", FadeIn(tbl[2], shift=UP * 0.2), Create(s_box),
                 Wiggle(tbl[2][0]), Circumscribe(tbl[2][1], color=RED_T))
        self.say("nm_d", FadeIn(tbl[3], shift=UP * 0.2), Circumscribe(tbl[3], color=GREEN_T))
        self.say("nm_e", Write(rule[0][0]), GrowArrow(rule[0][1]), Write(rule[0][2]),
                 Write(rule[1][0]), GrowArrow(rule[1][1]), Write(rule[1][2]))

    # 3. where the spec number comes from
    def spec(self):
        head = Text("Where 989 TFLOPS comes from", font_size=40).to_edge(UP)

        def dial(label, value, frac, color):
            arc = Arc(radius=1.0, start_angle=PI, angle=-PI, stroke_color=GREY_B,
                      stroke_width=6)
            fill = Arc(radius=1.0, start_angle=PI, angle=-PI * frac, stroke_color=color,
                       stroke_width=6)
            hub = Dot(arc.get_arc_center(), color=WHITE)
            needle = Line(hub.get_center(), hub.get_center() + 0.85 * RIGHT,
                          stroke_color=WHITE, stroke_width=4)
            needle.rotate(PI - PI * frac, about_point=hub.get_center())
            lbl = Text(label, font_size=22, color=color).next_to(arc, UP, buff=0.2)
            val = Text(value, font_size=30).next_to(hub, DOWN, buff=0.25)
            return VGroup(arc, fill, needle, hub, lbl, val)

        clock = dial("clock", "1830 MHz", 1830 / 2500, BLUE_T)
        fma = dial("FMAs / clock / core", "512", 0.5, GREEN_T)
        cores = dial("tensor cores", "528", 528 / 700, PURPLE_T)
        x1 = Text("×", font_size=40)
        x2 = Text("× 2 ×", font_size=40, color=YELLOW_T)
        row = VGroup(clock, x1, fma, x2, cores).arrange(RIGHT, buff=0.45).shift(UP * 0.6)
        fma_note = Text("1 FMA = 2 FLOPs", font_size=22, color=YELLOW_T)
        fma_note.next_to(x2, DOWN, buff=0.3)
        sm_note = Text("132 SMs × 4", font_size=22, color=GREY_B).next_to(cores, DOWN, buff=0.1)
        res = Text("= 989.4 TFLOPS  ✓", font_size=40, color=GREEN_T)
        res.next_to(row, DOWN, buff=1.2)
        formula = Text("1830 MHz × 512 × 2 × 528", font_size=30).next_to(res, UP, buff=0.25)

        self.say("sp_a", Write(head), Create(VGroup(clock[0], fma[0], cores[0]), run_time=2))
        self.say("sp_b", [Create(clock[1], run_time=2), FadeIn(clock[2:4]), Write(clock[4])],
                 Write(clock[5]))
        self.say("sp_c", [FadeIn(x1), Create(fma[1]), FadeIn(fma[2:4]), Write(fma[4])],
                 Write(fma[5]), [FadeIn(x2), Write(fma_note)])
        self.say("sp_d", [Create(cores[1], run_time=2), FadeIn(cores[2:4]), Write(cores[4])],
                 Write(cores[5]), FadeIn(sm_note))
        self.say("sp_e", LaggedStart(*[Indicate(d) for d in (clock, fma, cores)],
                                     lag_ratio=0.4), Write(formula), Write(res),
                 Circumscribe(res, color=GREEN_T))

        hub = clock[3].get_center()
        d_ang = -PI * (1980 - 1830) / 2500
        new_fill = Arc(radius=1.0, start_angle=PI, angle=-PI * 1980 / 2500,
                       stroke_color=RED_T, stroke_width=6).move_arc_center_to(hub)
        bad_val = Text("1980 MHz", font_size=30, color=RED_T).move_to(clock[5])
        bad_f = Text("1980 MHz × 512 × 2 × 528", font_size=30).move_to(formula)
        bad_res = Text("= 1070.5 TFLOPS  ✗  (spec: 989)", font_size=40, color=RED_T)
        bad_res.move_to(res)
        self.say("sp_f", [Rotate(clock[2], d_ang, about_point=hub, run_time=2),
                          Transform(clock[1], new_fill, run_time=2),
                          Transform(clock[5], bad_val)],
                 [Transform(formula, bad_f), Transform(res, bad_res)],
                 Circumscribe(res, color=RED_T, run_time=1.5))

        inv = Text("989 / (512 × 2 × 528)  =  1829 MHz", font_size=34, color=YELLOW_T)
        inv.move_to(VGroup(formula, res))
        good_fill = Arc(radius=1.0, start_angle=PI, angle=-PI * 1830 / 2500,
                        stroke_color=BLUE_T, stroke_width=6).move_arc_center_to(hub)
        back = CurvedArrow(inv.get_left() + LEFT * 0.1, clock.get_bottom() + DOWN * 0.15,
                           color=YELLOW_T, angle=-PI / 3)
        official = Text("1830 MHz: now official", font_size=24, color=GREEN_T)
        official.next_to(inv, DOWN, buff=0.4)
        self.say("sp_g", FadeOut(VGroup(formula, res)), Write(inv), Create(back),
                 [Rotate(clock[2], -d_ang, about_point=hub, run_time=2),
                  Transform(clock[1], good_fill, run_time=2),
                  Transform(clock[5], Text("1830 MHz", font_size=30).move_to(clock[5]))],
                 [FadeOut(back), Write(official)])
        self.say("sp_h", Circumscribe(clock, color=BLUE_T, run_time=2),
                 Wiggle(clock[2], rotation_angle=0.08 * TAU))

    # 4. FLOPs per training step
    def step(self):
        head = Text("FLOPs per training step", font_size=40).to_edge(UP)
        eq0 = VGroup(Text("TFLOPS  =", font_size=36), Text("FLOPs", font_size=36,
                                                            color=YELLOW_T),
                     Text("/  time", font_size=36)).arrange(RIGHT, buff=0.3)
        self.say("st_a", Write(head), Write(eq0), Circumscribe(eq0[1], color=YELLOW_T,
                                                               run_time=1.5))
        self.play(FadeOut(eq0), run_time=0.5)

        X = Rectangle(width=1.6, height=2.0, fill_color=BLUE_T, fill_opacity=0.3,
                      stroke_color=BLUE_T)
        W = Rectangle(width=1.2, height=1.6, fill_color=GREEN_T, fill_opacity=0.3,
                      stroke_color=GREEN_T)
        at = Text("@", font_size=40)
        mm = VGroup(X, at, W).arrange(RIGHT, buff=0.4).shift(LEFT * 3 + UP * 1.6)
        xl = Text("m tokens", font_size=24, color=BLUE_T).next_to(X, DOWN)
        wl = Text("k·n params", font_size=24, color=GREEN_T).next_to(W, DOWN)
        cost = Text("2·m·k·n FLOPs", font_size=32, color=YELLOW_T)
        cost.next_to(mm, RIGHT, buff=0.9)
        per = Text("= 2 × params × tokens", font_size=32, color=YELLOW_T)
        per.next_to(cost, DOWN, buff=0.3).align_to(cost, LEFT)
        self.say("st_b", [DrawBorderThenFill(X), Write(xl)], FadeIn(at),
                 [DrawBorderThenFill(W), Write(wl)], Write(cost))

        u = 1.8
        fwd = VGroup(hbar(u, BLUE_T, 0.7), Text("forward", font_size=22))
        bi = VGroup(hbar(u, RED_T, 0.7), Text("∂ inputs", font_size=22))
        bw = VGroup(hbar(u, RED_T, 0.7, 0.6), Text("∂ weights", font_size=22))
        rc = VGroup(hbar(u, YELLOW_T, 0.7, 0.6), Text("recompute", font_size=22,
                                                      color=BLACK))
        blocks = VGroup(fwd, bi, bw, rc)
        for b in blocks:
            b[1].move_to(b[0])
        blocks.arrange(RIGHT, buff=0.05).to_edge(DOWN, buff=1.3).to_edge(LEFT, buff=1.0)
        two = VGroup(*[Text("2", font_size=26, color=GREY_B).next_to(b, UP, buff=0.12)
                       for b in blocks])
        self.say("st_c", Write(per), Circumscribe(per, color=YELLOW_T),
                 [GrowFromEdge(fwd, LEFT, run_time=1.5), FadeIn(two[0])])
        bwd_br = Brace(VGroup(bi, bw), DOWN, color=RED_T)
        bwd_l = Text("backward = 2 × forward", font_size=24, color=RED_T)
        bwd_l.next_to(bwd_br, DOWN, buff=0.1)
        self.say("st_d", [GrowFromEdge(bi, LEFT), FadeIn(two[1])],
                 [GrowFromEdge(bw, LEFT), FadeIn(two[2])],
                 [GrowFromCenter(bwd_br), Write(bwd_l)])
        six = Text("×3 → 6 per param per token", font_size=28, color=WHITE)
        line6 = Line(fwd.get_corner(UL) + UP * 0.6, bw.get_corner(UR) + UP * 0.6)
        six.next_to(line6, UP, buff=0.1).align_to(line6, LEFT)
        self.say("st_e", [Create(line6), Write(six)])
        eight = Text("×4 → 8", font_size=28, color=YELLOW_T)
        eight.next_to(rc, UP, buff=0.6).shift(RIGHT * 0.6)
        line8 = Line(fwd.get_corner(UL) + UP * 1.3, rc.get_corner(UR) + UP * 1.3,
                     color=YELLOW_T)
        eight.next_to(line8, UP, buff=0.1).align_to(line8, RIGHT)
        self.say("st_f", [GrowFromEdge(rc, LEFT), FadeIn(two[3])],
                 [Create(line8), Write(eight)])
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head], run_time=0.6)

        num = VGroup(*[Text(t, font_size=34) for t in
                       ("size_B", "× 4", "× 2", "× seqlen", "× GBS")]).arrange(RIGHT, buff=0.2)
        bar = Line(LEFT * 3.6, RIGHT * 3.6)
        den = Text("sec_per_iter × GPUs × 1e3", font_size=34)
        frac = VGroup(num, bar, den).arrange(DOWN, buff=0.2)
        lhs = Text("TFLOPS per GPU  =", font_size=34, color=YELLOW_T)
        f = VGroup(lhs, frac).arrange(RIGHT, buff=0.4).shift(UP * 1.2)
        self.say("st_g", Write(lhs), *[Write(t) for t in num],
                 Circumscribe(num[1:3], color=YELLOW_T))
        b_note = Text("1e9 params / 1e12 = 1/1e3", font_size=22, color=GREY_B)
        b_note.next_to(den, DOWN, buff=0.15)
        self.say("st_h", Create(bar), Write(den), FadeIn(b_note))

        ex_num = VGroup(*[Text(t, font_size=32, color=BLUE_T) for t in
                          ("52", "× 4", "× 2", "× 2048", "× 1024")]).arrange(RIGHT, buff=0.2)
        ex_bar = Line(LEFT * 3.2, RIGHT * 3.2)
        ex_den = VGroup(*[Text(t, font_size=32, color=BLUE_T) for t in
                          ("127", "× 64", "× 1e3")]).arrange(RIGHT, buff=0.2)
        ex = VGroup(ex_num, ex_bar, ex_den).arrange(DOWN, buff=0.2)
        ex.next_to(f, DOWN, buff=0.9).align_to(frac, LEFT)
        ex_res = Text("≈ 107 TFLOPS", font_size=40, color=GREEN_T).next_to(ex, RIGHT, buff=0.5)
        self.say("st_i", FadeOut(b_note), *[Write(t) for t in ex_num], Create(ex_bar),
                 *[Write(t) for t in ex_den])
        sc = 4.0 / 120
        est_b = hbar(107 * sc, GREEN_T, 0.4).to_edge(DOWN, buff=1.4).to_edge(LEFT, buff=3)
        real_b = Rectangle(width=115 * sc, height=0.4, stroke_color=GREY_B)
        real_b.next_to(est_b, DOWN, buff=0.15, aligned_edge=LEFT)
        est_l = Text("estimate", font_size=22, color=GREEN_T).next_to(est_b, LEFT)
        real_l = Text("real (a bit higher)", font_size=22, color=GREY_B).next_to(real_b, RIGHT)
        self.say("st_j", Write(ex_res), Circumscribe(ex_res, color=GREEN_T),
                 [GrowFromEdge(est_b, LEFT), FadeIn(est_l)],
                 [Create(real_b), FadeIn(real_l)])

    # 5. three ceilings, and why sustained is lower
    def ceilings(self):
        head = Text("Three ceilings", font_size=40).to_edge(UP)
        sc = 8.0 / 1000
        names = [("H200 spec", 989, GREY_B, ""), ("MAMF", 834, BLUE_T, "84%"),
                 ("MSMF", 755, GREEN_T, "76%")]
        rows = VGroup()
        for name, v, c, pct in names:
            b = hbar(v * sc, c, 0.6)
            lab = Text(name, font_size=28, color=c)
            val = Text(f"{v}" + (f"  ({pct})" if pct else ""), font_size=26)
            rows.add(VGroup(lab, b, val))
        rows.arrange(DOWN, buff=0.5, aligned_edge=LEFT).shift(UP * 0.3)
        for r in rows:
            r[0].move_to(np.array([-5.0, r.get_y(), 0]))
            r[1].next_to(np.array([-3.8, r.get_y(), 0]), RIGHT, buff=0)
            r[2].next_to(r[1], RIGHT, buff=0.2)
        gpu = RoundedRectangle(corner_radius=0.15, width=2.6, height=2.6, stroke_color=GREY_B)
        mm = grid(6, 6, 0.32, BLUE_T, 0.5).move_to(gpu)
        lone = Text("one GPU, one perfectly shaped matmul", font_size=26, color=GREY_B)
        lone.next_to(gpu, DOWN)
        self.say("ce_a", Write(head), Create(gpu, run_time=1.5),
                 LaggedStart(*[FadeIn(c) for c in mm], lag_ratio=0.03, run_time=2),
                 Write(lone))
        self.play(FadeOut(VGroup(gpu, mm, lone)), run_time=0.5)
        self.say("ce_b", [Write(rows[0][0]), GrowFromEdge(rows[0][1], LEFT, run_time=2),
                          Write(rows[0][2])])
        mamf_n = Text("short burst, best shape", font_size=22, color=BLUE_T)
        msmf_n = Text("saturated at the power limit: use for training", font_size=22,
                      color=GREEN_T)
        dims = [(1.4, 0.5), (0.6, 1.0), (1.0, 1.0), (1.6, 0.4), (0.8, 0.7), (1.2, 0.9)]
        sweep = VGroup(*[Rectangle(width=w, height=h, stroke_color=BLUE_T, fill_color=BLUE_T,
                                   fill_opacity=0.3) for w, h in dims])
        sweep.arrange(RIGHT, buff=0.35).to_edge(DOWN, buff=0.5)
        sweep_l = Text("shape sweep", font_size=22, color=BLUE_T).next_to(sweep, UP, buff=0.1)
        best = sweep[5]
        self.say("ce_c", [LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in sweep],
                                      lag_ratio=0.25, run_time=2), FadeIn(sweep_l)],
                 [best.animate.set_fill(YELLOW_T, 0.8).set_stroke(YELLOW_T)],
                 [Write(rows[1][0]), GrowFromEdge(rows[1][1], LEFT)],
                 [Write(rows[1][2]),
                  Write(mamf_n.next_to(rows[1][1], DOWN, buff=0.08, aligned_edge=LEFT))])
        msmf_bar = rows[2][1]
        start = hbar(834 * sc, GREEN_T, 0.6).move_to(msmf_bar, aligned_edge=LEFT)
        self.say("ce_d", FadeOut(VGroup(sweep, sweep_l)),
                 [Write(rows[2][0]), GrowFromEdge(start, LEFT)],
                 Transform(start, msmf_bar, run_time=2),
                 [Write(rows[2][2]),
                  Write(msmf_n.next_to(rows[2][1], DOWN, buff=0.08, aligned_edge=LEFT))],
                 Circumscribe(msmf_n, color=GREEN_T, run_time=1.5))
        self.play(FadeOut(VGroup(rows, mamf_n, msmf_n, start)), run_time=0.6)

        def axes(y0, y1, ylab, color):
            a = Axes(x_range=[0, 10, 1], y_range=[y0, y1, (y1 - y0) / 2], x_length=8,
                     y_length=2.0, tips=False,
                     axis_config={"include_ticks": False},
                     y_axis_config={"include_numbers": False})
            yl = Text(ylab, font_size=22, color=color).next_to(a.y_axis, LEFT, buff=0.2)
            return VGroup(a, yl)

        clk = axes(1000, 2100, "SM clock\n(MHz)", BLUE_T).shift(UP * 1.0 + RIGHT * 0.6)
        pwr = axes(0, 1100, "power\n(W)", RED_T).shift(DOWN * 1.9 + RIGHT * 0.6)
        tl = Text("time →", font_size=20, color=GREY_B).next_to(pwr[0].x_axis, DOWN, buff=0.1)
        tl.align_to(pwr[0].x_axis, RIGHT)
        sketch = Text("B200, sketch", font_size=22, color=GREY_B).next_to(head, DOWN)
        self.say("ce_e", Write(sketch), [Create(clk, run_time=2), Create(pwr, run_time=2),
                                         FadeIn(tl)])

        ca, pa = clk[0], pwr[0]

        def path(a, pts, color):
            return VMobject(color=color, stroke_width=4).set_points_smoothly(
                [a.c2p(x, y) for x, y in pts])

        c1 = path(ca, [(0, 1965), (1.5, 1965), (3, 1965)], BLUE_T)
        p1 = path(pa, [(0, 295), (1.5, 295), (3, 295)], RED_T)
        c1l = Text("~1965 MHz boost", font_size=22, color=BLUE_T).next_to(
            ca.c2p(1.8, 1965), UP, buff=0.1)
        p1l = Text("~295 W", font_size=22, color=RED_T).next_to(ca.c2p(0, 0), UP)
        p1l.next_to(pa.c2p(1.5, 295), UP, buff=0.1)
        self.say("ce_f", [Create(c1, run_time=2), Create(p1, run_time=2)],
                 [Write(c1l), Write(p1l)])

        c2 = path(ca, [(3, 1965), (3.6, 1850), (4.4, 1550), (5.2, 1460), (6.5, 1455),
                       (10, 1455)], BLUE_T)
        p2 = path(pa, [(3, 295), (3.4, 700), (4.0, 960), (4.6, 978), (10, 978)], RED_T)
        tdp = DashedLine(pa.c2p(0, 1000), pa.c2p(10, 1000), color=YELLOW_T)
        tdp_l = Text("1000 W limit", font_size=20, color=YELLOW_T).next_to(
            pa.c2p(10, 1000), RIGHT, buff=0.1)
        c2l = Text("~1455 MHz", font_size=22, color=BLUE_T).next_to(
            ca.c2p(8, 1455), UP, buff=0.1)
        p2l = Text("~978 W", font_size=22, color=RED_T).next_to(pa.c2p(8, 978), DOWN,
                                                                buff=0.1)
        self.say("ce_g", [Create(tdp), FadeIn(tdp_l)], [Create(c2), Create(p2)],
                 [Write(c2l), Write(p2l)])
        self.play(FadeOut(VGroup(clk, pwr, tl, c1, p1, c1l, p1l, c2, p2, tdp, tdp_l,
                                 c2l, p2l)), run_time=0.6)

        est = Text("2250 × 1425 / 1965  ≈  1632 TFLOPS", font_size=36, color=YELLOW_T)
        est.shift(UP * 0.6)
        est_l = Text("spec × sustained clock / boost clock", font_size=24, color=GREY_B)
        est_l.next_to(est, UP)
        meas = Text("measured MSMF:  1429 TFLOPS", font_size=36, color=GREEN_T)
        meas.next_to(est, DOWN, buff=0.7)
        ovh = Text("the rest: real-kernel overhead", font_size=24, color=GREY_B)
        ovh.next_to(meas, DOWN)
        bsc = 5.0 / 2250
        spec_b = hbar(2250 * bsc, GREY_B, 0.45).to_edge(DOWN, buff=1.9).to_edge(LEFT, buff=3.2)
        spec_bl = Text("2250", font_size=22).next_to(spec_b, LEFT)
        est_b = hbar(1632 * bsc, YELLOW_T, 0.45).move_to(spec_b, aligned_edge=LEFT)
        est_bl = Text("≈1632", font_size=22, color=YELLOW_T).next_to(est_b, RIGHT)
        meas_b = hbar(1429 * bsc, GREEN_T, 0.45)
        meas_b.next_to(spec_b, DOWN, buff=0.2, aligned_edge=LEFT)
        meas_bl = Text("1429", font_size=22, color=GREEN_T).next_to(meas_b, RIGHT)
        meas.scale(0.85).next_to(est, DOWN, buff=0.4)
        ovh.next_to(meas_b, DOWN, buff=0.25).align_to(meas_b, LEFT)
        self.say("ce_h", Write(est_l), [GrowFromEdge(spec_b, LEFT), FadeIn(spec_bl)],
                 Write(est), [Transform(spec_b, est_b, run_time=2), FadeOut(spec_bl),
                              FadeIn(est_bl)])
        self.say("ce_i", Write(meas), [GrowFromEdge(meas_b, LEFT, run_time=2),
                                       FadeIn(meas_bl)],
                 FadeIn(ovh), Circumscribe(meas, color=GREEN_T, run_time=1.5))

    # 6. MFU vs HFU
    def mfu(self):
        head = Text("MFU vs HFU", font_size=40).to_edge(UP)
        util = Text("utilization  =  achieved TFLOPS / spec TFLOPS", font_size=32)
        util.next_to(head, DOWN, buff=0.6)
        ex = Text("H100:  400 / 989  ≈  40%", font_size=32, color=YELLOW_T)
        ex.next_to(util, DOWN, buff=0.4)
        g_out = Rectangle(width=8, height=0.55, stroke_color=GREY_B).next_to(ex, DOWN, buff=0.5)
        g_l = Text("spec 989", font_size=22, color=GREY_B).next_to(g_out, RIGHT)
        g_in = hbar(8 * 400 / 989, YELLOW_T, 0.55).move_to(g_out, aligned_edge=LEFT)
        g_il = Text("400", font_size=22, color=BLACK).move_to(g_in)
        self.say("mf_a", Write(head), Write(util), [Create(g_out, run_time=1.5), FadeIn(g_l)],
                 Circumscribe(util, color=WHITE))
        self.say("mf_b", Write(ex), [GrowFromEdge(g_in, LEFT, run_time=2), FadeIn(g_il)])

        u = 1.8
        cols = [(BLUE_T, "forward", 0.8), (RED_T, "∂ inputs", 0.8),
                (RED_T, "∂ weights", 0.6), (YELLOW_T, "recompute", 0.6)]
        blocks = VGroup()
        for c, t, o in cols:
            b = VGroup(hbar(u, c, 0.7, o), Text(t, font_size=22,
                                                 color=BLACK if c == YELLOW_T else WHITE))
            b[1].move_to(b[0])
            blocks.add(b)
        blocks.arrange(RIGHT, buff=0.05).shift(DOWN * 0.8)
        self.say("mf_c", LaggedStart(*[GrowFromEdge(b, LEFT) for b in blocks],
                                     lag_ratio=0.3),
                 Indicate(blocks[3], color=YELLOW_T))
        hb = Brace(blocks, UP, color=YELLOW_T)
        hl = Text("HFU: all of it (×4)", font_size=26, color=YELLOW_T).next_to(hb, UP, buff=0.1)
        mb = Brace(blocks[:3], DOWN, color=GREEN_T)
        ml = Text("MFU: what the model needs (×3)", font_size=26, color=GREEN_T)
        ml.next_to(mb, DOWN, buff=0.1)
        self.say("mf_d", FadeOut(VGroup(ex, g_out, g_l, g_in, g_il)),
                 [GrowFromCenter(hb), Write(hl)])
        self.say("mf_e", [GrowFromCenter(mb), Write(ml)])
        self.play(FadeOut(VGroup(util, blocks, hb, hl, mb, ml)), run_time=0.6)

        data = [("22B", "41.5%", "43.7%"), ("175B", "51.4%", "52.8%"),
                ("530B", "56.0%", "57.0%"), ("1T", "56.3%", "57.0%")]
        tbl = Table([list(r) for r in data],
                    col_labels=[Text(s) for s in ("model", "MFU", "HFU")],
                    element_to_mobject=lambda s: Text(s),
                    include_outer_lines=False, line_config={"stroke_width": 1})
        tbl.scale(0.6).next_to(head, DOWN, buff=0.5)
        src = Text("Megatron-LM, A100-80GB", font_size=22, color=GREY_B).next_to(tbl, DOWN)
        self.say("mf_f", [FadeIn(tbl.get_col_labels()), Create(tbl.get_horizontal_lines())],
                 [LaggedStart(*[FadeIn(r) for r in tbl.get_rows()[1:]], lag_ratio=0.2),
                  FadeIn(src)])
        r = tbl.get_rows()[2]
        self.say("mf_g", r.animate.set_color(YELLOW_T),
                 Circumscribe(r, color=YELLOW_T, run_time=2))
        scale = NumberLine(x_range=[0, 100, 10], length=9, include_numbers=False)
        scale.shift(DOWN * 0.6)
        nums = VGroup(*[Text(f"{v}%", font_size=20).next_to(scale.n2p(v), DOWN)
                        for v in (0, 25, 50, 75, 100)])
        zone = Rectangle(width=scale.n2p(70)[0] - scale.n2p(50)[0], height=0.5,
                         fill_color=GREEN_T, fill_opacity=0.6, stroke_width=0)
        zone.move_to(scale.n2p(50), aligned_edge=LEFT)
        zl = Text("fantastic", font_size=24, color=GREEN_T).next_to(zone, UP)
        self.say("mf_h", FadeOut(VGroup(tbl, src)), [Create(scale), FadeIn(nums)],
                 [GrowFromEdge(zone, LEFT), FadeIn(zl)],
                 Write(caption("NVIDIA, multi-node, large model: >50% MFU is "
                               "fantastic", 28, GREEN_T)))

    # 7. FLOPs to days to money
    def days(self):
        head = Text("From FLOPs to days", font_size=40).to_edge(UP)
        tot = VGroup(*[Text(t, font_size=32) for t in
                       ("total FLOPs  ≈", "6", "× params", "× tokens")]).arrange(RIGHT, buff=0.2)
        tot.next_to(head, DOWN, buff=0.6)
        known = Text("known before the run starts", font_size=24, color=GREY_B)
        known.next_to(tot, DOWN, buff=0.15)
        self.say("dy_a", Write(head), *[Write(t) for t in tot],
                 Circumscribe(tot, color=WHITE, run_time=1.5), FadeIn(known))

        div = Text("total FLOPs / TFLOPS  =  604,800 s  =  7 days", font_size=32,
                   color=YELLOW_T).next_to(known, DOWN, buff=0.6)
        day = 0.62
        ticks = VGroup(*[Rectangle(width=day, height=0.55, stroke_color=WHITE,
                                   stroke_width=1.5, fill_color=BLUE_T, fill_opacity=0.6)
                         for _ in range(7)]).arrange(RIGHT, buff=0)
        ticks.to_edge(LEFT, buff=1.2).shift(DOWN * 1.4)
        tl = Text("at spec TFLOPS", font_size=24, color=BLUE_T).next_to(ticks, UP,
                                                                         aligned_edge=LEFT)
        self.say("dy_b", Write(div), [LaggedStart(*[FadeIn(t) for t in ticks],
                                                  lag_ratio=0.2), Write(tl)])
        money = Text("× $ per day  →  compare providers", font_size=28, color=GREEN_T)
        money.next_to(ticks, RIGHT, buff=0.5)
        provs = VGroup()
        for name, w in (("A", 1.6), ("B", 2.2), ("C", 1.2)):
            provs.add(VGroup(Text(name, font_size=22), hbar(w, GREEN_T, 0.3)).arrange(RIGHT))
        provs.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(money, DOWN, buff=0.25)
        provs.align_to(money, LEFT)
        self.say("dy_c", Write(money),
                 LaggedStart(*[GrowFromEdge(p[1], LEFT) for p in provs], lag_ratio=0.3),
                 FadeIn(VGroup(*[p[0] for p in provs])))

        more = VGroup(*[t.copy().set_fill(RED_T, 0.6) for t in ticks])
        more.next_to(ticks, DOWN, buff=0.5).align_to(ticks, LEFT)
        more2 = VGroup(*[t.copy().set_fill(RED_T, 0.6) for t in ticks])
        more2.next_to(more, RIGHT, buff=0)
        ml = Text("at 50% MFU: 14 days", font_size=24, color=RED_T)
        ml.next_to(more2, RIGHT, buff=0.3)
        self.say("dy_d", FadeOut(VGroup(money, provs)),
                 [LaggedStart(*[FadeIn(t) for t in more], lag_ratio=0.1)],
                 [LaggedStart(*[FadeIn(t) for t in more2], lag_ratio=0.1), Write(ml)])
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head], run_time=0.6)

        ax = Axes(x_range=[0, 5, 1], y_range=[0, 200, 50], x_length=7, y_length=3.4,
                  tips=False, axis_config={"include_ticks": False},
                  y_axis_config={"include_numbers": False})
        ax.shift(DOWN * 0.3 + LEFT * 1)
        ynums = VGroup(*[Text(str(v), font_size=22).next_to(ax.c2p(0, v), LEFT)
                         for v in (100, 150)])
        yl = Text("TFLOPS / GPU", font_size=22).next_to(ax.y_axis, UP)
        xl = Text("weeks of tuning →", font_size=22, color=GREY_B).next_to(ax.x_axis, DOWN)
        tune = VMobject(color=BLUE_T, stroke_width=4).set_points_smoothly(
            [ax.c2p(x, y) for x, y in [(0, 90), (1, 105), (2, 125), (3, 142), (3.6, 150)]])
        flat = Line(ax.c2p(3.6, 150), ax.c2p(5, 150), color=GREEN_T, stroke_width=4)
        cap = DashedLine(ax.c2p(0, 150), ax.c2p(5, 150), color=GREY_B, stroke_width=2)
        go = Text("start training", font_size=26, color=GREEN_T).next_to(
            ax.c2p(3.6, 150), UP, buff=0.5)
        arr = Arrow(go.get_bottom(), ax.c2p(3.6, 152), buff=0.05, color=GREEN_T,
                    stroke_width=3)
        bl = Text("BLOOM, A100 (shape sketched)", font_size=22, color=GREY_B)
        bl.next_to(xl, DOWN, buff=0.15)
        self.say("dy_e", [Create(ax), FadeIn(yl), FadeIn(xl), FadeIn(bl), FadeIn(ynums)],
                 Create(tune))
        self.say("dy_f", Create(cap), [FadeIn(go), GrowArrow(arr)], Create(flat))

    # 8. why MFU across papers doesn't compare
    def compare(self):
        head = Text("Comparing MFU across papers", font_size=40).to_edge(UP)
        papers = VGroup()
        for name in ("paper A", "paper B", "paper C"):
            p = Rectangle(width=1.6, height=2.0, stroke_color=GREY_B)
            lines = VGroup(*[Line(LEFT * 0.55, RIGHT * 0.55, stroke_color=GREY_C,
                                  stroke_width=2) for _ in range(4)]).arrange(DOWN, buff=0.22)
            lines.move_to(p).shift(UP * 0.2)
            m = Text("MFU ?", font_size=22, color=YELLOW_T).next_to(lines, DOWN, buff=0.2)
            papers.add(VGroup(p, lines, m, Text(name, font_size=20).next_to(p, UP, buff=0.1)))
        papers.arrange(RIGHT, buff=0.8).shift(UP * 0.6)
        neq = VGroup(*[Text("≠", font_size=40, color=RED_T).move_to(
            (papers[i].get_right() + papers[i + 1].get_left()) / 2) for i in range(2)])
        self.say("cm_a", Write(head), LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in papers],
                                                  lag_ratio=0.3), FadeIn(neq))
        self.play(VGroup(papers, neq).animate.scale(0.55).to_edge(LEFT, buff=0.6).shift(UP * 0.4),
                  run_time=0.6)

        q1 = Text("1. How were the FLOPs counted?", font_size=30, color=BLUE_T)
        q1.move_to(RIGHT * 1.6 + UP * 1.6)
        variants = VGroup(Text("formula A", font_size=22), Text("formula B", font_size=22),
                          Text("compiler removed ops", font_size=22)).arrange(RIGHT, buff=0.4)
        variants.set_color(GREY_B).next_to(q1, DOWN, buff=0.3)
        self.say("cm_b", Write(q1), LaggedStart(*[FadeIn(v) for v in variants],
                                                lag_ratio=0.3),
                 LaggedStart(*[Circumscribe(v, color=BLUE_T) for v in variants],
                             lag_ratio=0.5))

        q2 = Text("2. What was timed?", font_size=30, color=GREEN_T)
        q2.next_to(variants, DOWN, buff=0.7).align_to(q1, LEFT)
        segs = VGroup()
        for t, w, c in (("data", 1.2, GREY_B), ("fwd", 1.4, BLUE_T),
                        ("bwd", 2.4, RED_T), ("log", 0.8, GREY_B)):
            s = VGroup(hbar(w, c, 0.5, 0.6), Text(t, font_size=20))
            s[1].move_to(s[0])
            segs.add(s)
        segs.arrange(RIGHT, buff=0.04).next_to(q2, DOWN, buff=0.7).align_to(q1, LEFT)
        full = Brace(segs, DOWN, buff=0.1)
        full_l = Text("whole iteration", font_size=20).next_to(full, DOWN, buff=0.05)
        part = Brace(segs[1:3], UP, buff=0.1, color=YELLOW_T)
        part_l = Text("fwd + bwd only", font_size=20, color=YELLOW_T).next_to(part, UP, buff=0.05)
        self.say("cm_c", Write(q2), LaggedStart(*[GrowFromEdge(s, LEFT) for s in segs],
                                                lag_ratio=0.2),
                 [GrowFromCenter(full), FadeIn(full_l)], [GrowFromCenter(part), FadeIn(part_l)])
        cross = Cross(papers, stroke_color=RED_T, stroke_width=5)
        self.say("cm_d", Create(cross, run_time=2), Indicate(papers, color=RED_T))
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head], run_time=0.6)

        before = VGroup(Text("your run, before", font_size=28), Text("45%", font_size=40))
        after = VGroup(Text("your run, after", font_size=28),
                       Text("49%", font_size=40, color=GREEN_T))
        for g in (before, after):
            g.arrange(DOWN)
        pair = VGroup(before, after).arrange(RIGHT, buff=2.5)
        ar = Arrow(before.get_right(), after.get_left(), color=GREEN_T)
        same = Text("same counting, same timing", font_size=26, color=GREY_B)
        same.next_to(pair, DOWN, buff=0.6)
        note = Text("(made-up numbers)", font_size=20, color=GREY_B).next_to(same, DOWN)
        self.say("cm_e", FadeIn(before), [GrowArrow(ar), FadeIn(after)],
                 [Write(same), FadeIn(note)])

    # 9. silicon lottery
    def lottery(self):
        head = Text("Silicon lottery", font_size=40).to_edge(UP)

        def chip(lbl):
            r = RoundedRectangle(corner_radius=0.15, width=1.8, height=1.8,
                                 stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=1)
            return VGroup(r, Text(lbl, font_size=28).move_to(r))

        a, b = chip("H100"), chip("H100")
        pair = VGroup(a, Text("≠", font_size=56, color=RED_T), b).arrange(RIGHT, buff=0.8)
        pair.shift(UP * 0.9)
        self.say("lt_a", Write(head), FadeIn(a), [FadeIn(pair[1]), FadeIn(b)])
        chain = VGroup(Text("more voltage", font_size=28), Text("→", font_size=28),
                       Text("more heat", font_size=28, color=RED_T), Text("→", font_size=28),
                       Text("lower clock", font_size=28, color=BLUE_T)).arrange(RIGHT, buff=0.3)
        chain.next_to(pair, DOWN, buff=0.9)
        self.say("lt_b", LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in chain],
                                     lag_ratio=0.35))
        self.play(FadeOut(VGroup(pair, chain)), run_time=0.6)

        # illustrative spread of ~5% across 8 GPUs, not measured values
        rel = [1.00, 0.985, 0.97, 0.99, 0.955, 0.98, 0.965, 0.95]
        h = 3.6
        base_y = -2.6
        bars = VGroup()
        for i, v in enumerate(rel):
            bar = Rectangle(width=0.8, height=h * v, fill_color=BLUE_T,
                            fill_opacity=0.8, stroke_width=0)
            bar.move_to(np.array([-4.2 + i * 1.2, base_y, 0]), aligned_edge=DOWN)
            bars.add(bar)
        labels = VGroup(*[Text(f"GPU{i}", font_size=20).next_to(b, DOWN, buff=0.1)
                          for i, b in enumerate(bars)])
        fan = Text("cold air →", font_size=24, color=BLUE_T).next_to(bars, LEFT, buff=0.2)
        fan.shift(UP * 0.6)
        spread = Text("5%+ spread on one node (bar heights illustrative)", font_size=24,
                      color=GREY_B).next_to(head, DOWN, buff=0.3)
        self.say("lt_c", [LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1),
                          FadeIn(labels)], [FadeIn(fan), Write(spread)])
        lo = min(range(8), key=lambda i: rel[i])
        top = bars[lo].get_top()[1]
        line = DashedLine(np.array([-4.8, top, 0]), np.array([4.8, top, 0]), color=RED_T)
        self.say("lt_d", [Indicate(bars[lo], color=RED_T), bars[lo].animate.set_fill(RED_T)],
                 Create(line),
                 [b.animate.stretch_to_fit_height(top - base_y).align_to(bars[lo], DOWN)
                  .set_fill(opacity=0.4)
                  for i, b in enumerate(bars) if i != lo])
        self.say("lt_e", LaggedStart(*[Indicate(b, color=YELLOW_T) for b in bars],
                                     lag_ratio=0.0, run_time=1.5),
                 Circumscribe(bars[lo], color=RED_T, run_time=1.5), Write(caption("benchmark all GPUs at once; the slowest is your "
                                       "ceiling", 28, YELLOW_T).shift(DOWN * 0.15)))

    # 10. which TFLOPS?
    def ending(self):
        head = Text("Which TFLOPS is it?", font_size=44).to_edge(UP)
        sc = 8.0 / 1000
        data = [("spec", 989, GREY_B), ("burst (MAMF, H200)", 834, BLUE_T),
                ("sustained (MSMF, H200)", 755, GREEN_T), ("a real run (H100)", 400,
                                                           YELLOW_T)]
        rows = VGroup()
        for name, v, c in data:
            r = VGroup(hbar(v * sc, c, 0.55), Text(f"{name}:  {v}", font_size=26, color=c))
            rows.add(r)
        rows.arrange(DOWN, buff=0.75, aligned_edge=LEFT).shift(DOWN * 0.2)
        for r in rows:
            r[1].next_to(r[0], UP, buff=0.08, aligned_edge=LEFT)
        rows.to_edge(LEFT, buff=1.2)
        self.say("end_a", Write(head),
                 LaggedStart(*[AnimationGroup(GrowFromEdge(r[0], LEFT), FadeIn(r[1]))
                               for r in rows], lag_ratio=0.5))


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
