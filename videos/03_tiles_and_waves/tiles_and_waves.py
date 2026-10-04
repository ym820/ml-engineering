"""Video 3: Tiles and waves.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python tiles_and_waves.py
2. Render: manim -qh tiles_and_waves.py TilesAndWaves

Every number comes from training/performance/README.md
(Vector and matrix size divisibility -> Final recommendations for model sizing).
"""
import wave
from math import ceil
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
YELLOW_T = "#FFFF00"
RED_T = "#FC6255"
GREEN_T = "#83C167"

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "title_a": "Let's start with a puzzle.",
    "title_b": "Take two matrix multiplies that are almost exactly the same size, like "
               "these two. You'd expect them to take about the same time on a GPU, but "
               "one of them can take twice as long as the other.",
    "title_c": "The reason is that a GPU doesn't really compute smoothly across a matrix. "
               "It works in discrete chunks, and once you and I can see those chunks, the "
               "2x stops being mysterious.",

    "tiles_a": "To see the chunks, let's look at the output of a single matrix multiply.",
    "tiles_b": "The GPU chops this output up into tiles,",
    "tiles_c": "and each tile gets handed to its own thread block, which is responsible "
               "for computing just that one piece.",
    "tiles_d": "Those thread blocks then get scheduled onto the GPU's streaming "
               "multiprocessors, which we'll just call SMs,",
    "tiles_e": "and you can think of each SM as working on one block at a time.",
    "tiles_f": "For a sense of scale, an A100 has 108 of these SMs, and an H100 has 132.",

    "tq_a": "This picture already hides the first catch.",
    "tq_b": "The kernel picks its tile size from a few fixed options, but nothing forces "
            "your matrix to be a whole number of tiles wide. Here, the edge of the matrix "
            "lands partway through the last column.",
    "tq_c": "So does the GPU give that last column of tiles a discount?",
    "tq_d": "Unfortunately not. Each of those tiles still runs in full on its SM, and "
            "only part of what it computes actually gets used.",
    "tq_e": "This is what's called tile quantization. Since the work comes in whole "
            "tiles, the number of blocks you pay for is the ceiling of m over the tile "
            "height, times the ceiling of n over the tile width.",
    "tq_f": "There's a related rule at the level of the tensor cores, which are only "
            "fully used when the dimensions are multiples of 8 elements for fp16 on a "
            "V100, or 64 on an A100.",

    "wq_a": "The second catch happens one level up, when the blocks get scheduled.",
    "wq_b": "An A100 has 108 SMs, so it can run 108 thread blocks at the same time.",
    "wq_c": "Now imagine your matmul needs 109 of them.",
    "wq_d": "The first 108 all go out together, and we'll call that one wave.",
    "wq_e": "That leaves one block left over,",
    "wq_f": "and it has to run by itself in a second wave, which takes almost as long as "
            "the first one did.",
    "wq_g": "So you've paid for nearly twice the time, in exchange for less than one "
            "percent more work.",
    "wq_h": "Since a wave costs about the same whether it's full or nearly empty, what "
            "really sets the time is the number of waves, and this effect is called "
            "wave quantization.",

    "st_a": "Here's a question that falls right out of this. If I gave you a GPU with "
            "more SMs, would you get more TFLOPS?",
    "st_b": "This isn't hypothetical, because B300 ships in at least two versions, one "
            "with 148 SMs and one with 152. They're the same silicon running at the same "
            "clock, and the bigger one just has 2.7 percent more SMs.",
    "st_c": "If time is set by the number of waves, then with C thread blocks, the "
            "speedup is the ceiling of C over 148, divided by the ceiling of C over 152.",
    "st_d": "A natural guess would be that this hovers around 1.027, the ratio of the "
            "SM counts.",
    "st_e": "But when you actually plot it, you get this staircase instead.",
    "st_148": "At 148 blocks, both chips finish in a single wave, so there's no "
              "difference at all.",
    "st_152": "Add just four more blocks, and the 148 SM chip needs a second, nearly "
              "empty wave, which makes the bigger chip twice as fast.",
    "st_296": "At 296 they're tied again, because 296 fills 148 SMs exactly twice.",
    "st_304": "And at 304, the bigger chip comes out one and a half times faster.",
    "st_f": "Let's put those four points in a table,",
    "st_g": "and add a shape you'd actually run, the MLP up-projection of Llama 3.1 8B "
            "at 8K tokens.",
    "st_h": "Its output is 8192 by 14336, and cutting that into 128 by 256 tiles gives "
            "64 times 56, or 3584 blocks.",
    "st_i": "That shape gets a speedup of 1.042, which is actually a bit above the SM "
            "ratio, not below it.",
    "st_j": "It's only for really big matmuls that the speedup settles down to that 1.027.",
    "st_k": "So 2.7 percent more SMs can buy you nothing at all, or twice the speed, "
            "depending entirely on the shape of your matrices.",

    "pk_a": "This also says something about the TFLOPS number on a spec sheet, which is "
            "an aggregate over the whole chip rather than a per-SM rate.",
    "pk_b": "B300 is rated at 2250 TFLOPS in bf16. So should the 152 SM version get the "
            "same rating?",
    "pk_c": "Well, if you divide 2250 by 148, each SM is worth about 15.2,",
    "pk_d": "and multiplying that by 152 gives you 2311.",
    "pk_e": "NVIDIA publishes one set of numbers for B300, though, so two real versions "
            "can't both match it.",

    "sw_a": "Let's see how this plays out in a real model.",
    "sw_b": "Models like LLaMA use a SwiGLU MLP, which has three weight matrices instead "
            "of the usual two. To keep the parameter count the same, the SwiGLU paper "
            "suggests making the MLP eight-thirds of the hidden size instead of four "
            "times.",
    "sw_c": "With a hidden size of 4096, eight-thirds of that comes out to 10922.",
    "sw_d": "Here's how a matmul of that size runs on an H100, next to a few nearby sizes.",
    "sw_e": "10922 gets 273 TFLOPS, while its neighbors all land around 395.",
    "sw_f": "Look at 10944, which is only 22 bigger, and runs 46 percent faster.",
    "sw_g": "And Llama 2 7B uses 11008, which turns out to be one of the best performing "
            "sizes in its range.",
    "sw_h": "So eight-thirds is really just a suggestion, and it's worth searching nearby "
            "for a size the hardware likes.",

    "hd_a": "The same kind of thinking applies to attention heads.",
    "hd_b": "It's most efficient to keep the head size, h over a, as large as you can "
            "without hurting accuracy,",
    "hd_c": "and having more powers of 2 in it helps.",
    "hd_d": "The good news is that if you use flash attention, it takes care of these "
            "head sizing constraints for you,",
    "hd_e": "and all it asks is that h over a is large enough to saturate the GPU.",

    "ck_a": "If we put all of this together, we get a short checklist for sizing a model.",
    "ck_0": "Make the vocab size divisible by 64.",
    "ck_1": "Make the micro batch size as large as you can.",
    "ck_2": "Keep b times s, h over a, and h over t divisible by a power of 2.",
    "ck_3": "Make sure b times a, over t, comes out to an integer.",
    "ck_4": "Keep the tensor parallel degree, t, as small as possible.",
    "ck_5": "And for SwiGLU, search near eight-thirds of h for the fastest size.",
    "ck_b": "And whenever a shape surprises you, count your blocks and divide by your "
            "SM count, and you'll know where you stand on the staircase.",
}


def caption(text, size=30):
    return Text(text, font_size=size).to_edge(DOWN, buff=0.5)


class TilesAndWaves(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.title, self.tiles, self.tile_quant, self.waves,
                     self.staircase, self.peak, self.swiglu, self.heads,
                     self.checklist):
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

    # 0. hook
    def title(self):
        t = Text("Tiles and waves", font_size=72)
        sz = 0.4

        def mat(cols_exact, x):
            m = Rectangle(width=cols_exact * sz, height=4 * sz, stroke_color=WHITE,
                          fill_color=BLUE_T, fill_opacity=0.5).move_to(RIGHT * x + UP * 0.6)
            g = VGroup(*[Square(sz, stroke_width=1.5, stroke_color=YELLOW_T)
                         for _ in range(4 * ceil(cols_exact))])
            g.arrange_in_grid(4, ceil(cols_exact), buff=0).align_to(m, UL)
            return m, g

        a, ga = mat(6, -3)
        b, gb = mat(6.2, 3)
        bar_a = Rectangle(width=1.5, height=0.35, fill_color=GREEN_T, fill_opacity=0.8,
                          stroke_width=0).next_to(a, DOWN, buff=0.6).align_to(a, LEFT)
        bar_b = Rectangle(width=3.0, height=0.35, fill_color=RED_T, fill_opacity=0.8,
                          stroke_width=0).next_to(b, DOWN, buff=0.6).align_to(b, LEFT)
        la = Text("time 1x", font_size=24).next_to(bar_a, RIGHT)
        lb = Text("time 2x", font_size=24).next_to(bar_b, DOWN, aligned_edge=LEFT)
        self.say("title_a", Write(t))
        self.say("title_b", t.animate.scale(0.6).to_edge(UP),
                 [DrawBorderThenFill(a), DrawBorderThenFill(b)],
                 [GrowFromEdge(bar_a, LEFT), FadeIn(la)],
                 [GrowFromEdge(bar_b, LEFT), FadeIn(lb)])
        self.say("title_c", [Create(ga), Create(gb)],
                 Indicate(gb[ceil(6.2) - 1::ceil(6.2)], color=RED_T))

    # 1. output tiles -> thread blocks -> SMs
    def tiles(self):
        rows, cols, sz = 3, 4, 0.8
        out = Rectangle(width=cols * sz, height=rows * sz, stroke_color=WHITE,
                        fill_color=BLUE_T, fill_opacity=0.5).shift(LEFT * 3.5 + UP * 0.3)
        lbl = Text("output of a matmul", font_size=26).next_to(out, UP)
        tiles = VGroup(*[Square(sz, stroke_color=WHITE, stroke_width=2,
                                fill_color=BLUE_T, fill_opacity=0.5)
                         for _ in range(rows * cols)])
        tiles.arrange_in_grid(rows, cols, buff=0).move_to(out)
        self.say("tiles_a", [DrawBorderThenFill(out), Write(lbl)])
        self.add(tiles)
        self.remove(out)
        self.say("tiles_b", tiles.animate.arrange_in_grid(rows, cols, buff=0.08).move_to(out))

        note = Text("each tile = one thread block", font_size=26, color=YELLOW_T)
        note.next_to(tiles, DOWN)
        self.say("tiles_c", [LaggedStart(*[Indicate(s, color=YELLOW_T) for s in tiles],
                                         lag_ratio=0.08), Write(note)])

        sms = VGroup(*[RoundedRectangle(corner_radius=0.08, width=0.9, height=0.9,
                                        stroke_color=GREEN_T)
                       for _ in range(rows * cols)])
        sms.arrange_in_grid(3, 4, buff=0.15).shift(RIGHT * 3.3 + UP * 0.3)
        sm_lbl = Text("SMs", font_size=26, color=GREEN_T).next_to(sms, UP)
        self.say("tiles_d", [Create(sms), Write(sm_lbl)])
        self.say("tiles_e", LaggedStart(*[tiles[i].copy().animate.move_to(sms[i]).scale(0.85)
                                          for i in range(rows * cols)], lag_ratio=0.1,
                                        run_time=2.5))
        self.say("tiles_f", Write(caption("A100: 108 SMs     H100: 132 SMs")))

    # 2. tile quantization
    def tile_quant(self):
        head = Text("Tile quantization", font_size=40).to_edge(UP)
        sz, rows, full_cols, frac = 1.0, 3, 3, 0.35
        grid = VGroup()
        for r in range(rows):
            for c in range(full_cols + 1):
                sq = Square(sz, stroke_color=WHITE, stroke_width=2)
                sq.move_to(np.array([c * sz, -r * sz, 0]))
                grid.add(sq)
        grid.move_to(LEFT * 2.5)
        useful, wasted = VGroup(), VGroup()
        for r in range(rows):
            for c in range(full_cols + 1):
                sq = grid[r * (full_cols + 1) + c]
                if c < full_cols:
                    useful.add(sq.copy().set_fill(BLUE_T, 0.5).set_stroke(width=0))
                else:
                    u = Rectangle(width=sz * frac, height=sz, fill_color=BLUE_T,
                                  fill_opacity=0.5, stroke_width=0)
                    u.align_to(sq, LEFT).align_to(sq, UP)
                    w = Rectangle(width=sz * (1 - frac), height=sz, fill_color=RED_T,
                                  fill_opacity=0.6, stroke_width=0)
                    w.align_to(sq, RIGHT).align_to(sq, UP)
                    useful.add(u)
                    wasted.add(w)
        edge = DashedLine(grid.get_corner(UL) + RIGHT * (full_cols + frac) * sz,
                          grid.get_corner(DL) + RIGHT * (full_cols + frac) * sz,
                          color=YELLOW_T)
        n_lbl = Text("n", font_size=28, color=YELLOW_T).next_to(edge, UP)
        last_col = VGroup(*[grid[r * (full_cols + 1) + full_cols] for r in range(rows)])
        w_lbl = Text("full work,\npartial output", font_size=26, color=RED_T)
        w_lbl.next_to(grid, RIGHT, buff=0.5)
        formula = Text("blocks = ceil(m/Tm) × ceil(n/Tn)", font_size=30)
        formula.next_to(grid, DOWN, buff=0.6).set_x(0)
        tc = VGroup(
            Text("Tensor cores want dimensions in multiples of:", font_size=26),
            Text("V100 fp16: 8 elements      A100: 64 elements", font_size=26,
                 color=YELLOW_T),
        ).arrange(DOWN).to_edge(DOWN, buff=0.4)

        self.say("tq_a", Write(head))
        self.say("tq_b", Create(grid), [FadeIn(useful), Create(edge), Write(n_lbl)])
        self.say("tq_c", Indicate(last_col, color=YELLOW_T))
        self.say("tq_d", FadeIn(wasted), Write(w_lbl))
        self.say("tq_e", Write(formula), Circumscribe(last_col, color=RED_T))
        self.say("tq_f", [formula.animate.shift(UP * 0.2), Write(tc[0])], Write(tc[1]))

    # 3. wave quantization
    def waves(self):
        head = Text("Wave quantization", font_size=40).to_edge(UP)
        cols, rows, s = 12, 9, 0.32
        slots = VGroup(*[Square(s, stroke_color=GREEN_T, stroke_width=1.5)
                         for _ in range(cols * rows)])
        slots.arrange_in_grid(rows, cols, buff=0.06).shift(LEFT * 3.4 + DOWN * 0.2)
        lbl = Text("A100: 108 SMs", font_size=26, color=GREEN_T).next_to(slots, UP)
        demand = Text("matmul needs 109 thread blocks", font_size=28, color=YELLOW_T)
        demand.shift(RIGHT * 3.2 + UP * 2)
        blocks = VGroup(*[Square(s * 0.85, stroke_width=0, fill_color=BLUE_T,
                                 fill_opacity=0.9).move_to(sl) for sl in slots])
        spare = blocks[0].copy().next_to(demand, DOWN, buff=0.3)

        wave_len = 2.4
        w1 = Rectangle(width=wave_len, height=0.5, fill_color=BLUE_T, fill_opacity=0.8,
                       stroke_width=1).move_to(RIGHT * 0.8 + DOWN * 0.3, aligned_edge=LEFT)
        w1_lbl = Text("wave 1: 108/108", font_size=22).next_to(w1, DOWN)
        w2 = Rectangle(width=wave_len, height=0.5, stroke_color=RED_T,
                       stroke_width=2).next_to(w1, RIGHT, buff=0)
        w2_used = Rectangle(width=wave_len / 108 * 4, height=0.5, fill_color=BLUE_T,
                            fill_opacity=0.8, stroke_width=0).align_to(w2, LEFT).set_y(w2.get_y())
        w2_lbl = Text("wave 2: 1/108", font_size=22, color=RED_T).next_to(w2, DOWN)
        arrow = DoubleArrow(w1.get_corner(UL) + UP * 0.3, w2.get_corner(UR) + UP * 0.3,
                            buff=0, color=WHITE, stroke_width=2)
        tl = Text("~2x the time, <1% more work", font_size=24).next_to(arrow, UP)

        self.say("wq_a", Write(head))
        self.say("wq_b", [Create(slots), Write(lbl)])
        self.say("wq_c", Write(demand))
        self.say("wq_d", LaggedStart(*[FadeIn(b, shift=DOWN * 0.3) for b in blocks],
                                     lag_ratio=0.01, run_time=1.5),
                 [GrowFromEdge(w1, LEFT), Write(w1_lbl)])
        self.say("wq_e", FadeOut(blocks), FadeIn(spare, scale=1.5))
        self.say("wq_f", spare.animate.move_to(slots[0]),
                 [GrowFromEdge(w2, LEFT), FadeIn(w2_used), Write(w2_lbl)])
        self.say("wq_g", [GrowFromCenter(arrow), Write(tl)])
        self.say("wq_h", Write(caption(
            "A wave costs the same full or not: time ∝ ceil(blocks / SMs)", 28)))

    # 4. more SMs: the staircase
    def staircase(self):
        head = Text("Do more SMs give more TFLOPS?", font_size=38).to_edge(UP)
        sub = Text("B300 ships with 148 and 152 SMs: 2.7% more", font_size=26,
                   color=GREY_B).next_to(head, DOWN)
        formula = Text("speedup = ceil(C/148) / ceil(C/152)", font_size=28,
                       color=YELLOW_T).next_to(sub, DOWN)

        ax = Axes(x_range=[0, 340, 50], y_range=[0.9, 2.1, 0.5],
                  x_length=10, y_length=3.6, tips=False,
                  axis_config={"include_numbers": True, "font_size": 22},
                  y_axis_config={"numbers_to_include": [1.0, 1.5, 2.0]})
        ax.to_edge(DOWN, buff=0.6)
        xl = Text("thread blocks C", font_size=22).next_to(ax.x_axis, DOWN, buff=0.35)

        def sp(c):
            return ceil(c / 148) / ceil(c / 152)

        pts = []
        for c in range(1, 341):
            pts += [ax.c2p(c - 1, sp(c)), ax.c2p(c, sp(c))]
        curve = VMobject(color=BLUE_T, stroke_width=3).set_points_as_corners(pts)
        ratio = DashedLine(ax.c2p(0, 152 / 148), ax.c2p(340, 152 / 148), color=GREEN_T)
        rl = Text("SM ratio 1.027", font_size=20, color=GREEN_T).next_to(
            ax.c2p(60, 152 / 148), UP, buff=0.1)

        self.say("st_a", Write(head))
        chips = VGroup()
        for n, col in ((148, BLUE_T), (152, GREEN_T)):
            sq = VGroup(*[Square(0.16, stroke_width=1, stroke_color=col,
                                 fill_color=col, fill_opacity=0.4) for _ in range(n)])
            sq.arrange_in_grid(cols=19, buff=0.04, flow_order="rd")
            chips.add(VGroup(sq, Text(f"{n} SMs", font_size=26, color=col).next_to(sq, UP)))
        chips.arrange(RIGHT, buff=1.5, aligned_edge=UP).shift(DOWN * 0.6)
        extra = chips[1][0][148:]
        self.say("st_b", FadeIn(chips[0]), FadeIn(chips[1]),
                 extra.animate.set_fill(YELLOW_T, 1).set_stroke(YELLOW_T),
                 [FadeOut(chips), FadeIn(sub)])
        self.say("st_c", Write(formula), Circumscribe(formula, color=YELLOW_T))
        self.say("st_d", [Create(ax), FadeIn(xl)], [Create(ratio), FadeIn(rl)])
        self.say("st_e", Create(curve, run_time=4, rate_func=linear))

        dots = VGroup()
        for c, txt in [(148, "148: 1.000x"), (152, "152: 2.000x"),
                       (296, "296: 1.000x"), (304, "304: 1.500x")]:
            d = Dot(ax.c2p(c, sp(c)), color=YELLOW_T)
            t = Text(txt, font_size=20, color=YELLOW_T)
            t.next_to(d, {148: LEFT, 152: RIGHT, 296: LEFT, 304: RIGHT}[c], buff=0.15)
            if sp(c) < 1.4:
                t.shift(UP * 0.3)
            g = VGroup(d, t)
            dots.add(g)
            self.say(f"st_{c}", FadeIn(g, scale=1.5), Flash(d, color=YELLOW_T))
        self.play(*[FadeOut(m) for m in (ax, xl, curve, ratio, rl, dots)])

        rows = [("148", "1", "1", "1.000x", "4 SMs idle on the bigger part"),
                ("152", "2", "1", "2.000x", "2x from 2.7% more SMs"),
                ("296", "2", "2", "1.000x", "2 x 148 fills 148 exactly"),
                ("304", "3", "2", "1.500x", "crossover again"),
                ("3584", "25", "24", "1.042x", "Llama-3.1-8B MLP up-proj, 8K tokens"),
                ("50000", "338", "329", "1.027x", "asymptote at the SM ratio")]
        tbl = Table([list(r) for r in rows],
                    col_labels=[Text(s) for s in ("blocks", "@148", "@152", "speedup", "")],
                    element_to_mobject=lambda s: Text(s),
                    include_outer_lines=False, line_config={"stroke_width": 1})
        tbl.scale(0.42).next_to(formula, DOWN, buff=0.4)
        trs = tbl.get_rows()
        note = caption("8192 x 14336 output over a 128 x 256 tile = 64 x 56 = 3584 blocks", 24)
        moral = caption("2.7% more SMs buys anywhere from nothing to 2x, on shape alone", 28)
        moral.set_color(YELLOW_T)

        self.say("st_f", [FadeIn(tbl.get_col_labels()), Create(tbl.get_horizontal_lines())],
                 LaggedStart(*[FadeIn(trs[i]) for i in range(1, 5)], lag_ratio=0.3))
        self.say("st_g", FadeIn(trs[5]), Indicate(trs[5][4], color=YELLOW_T))
        self.say("st_h", Write(note), Circumscribe(trs[5][0], color=YELLOW_T))
        self.say("st_i", trs[5].animate.set_color(YELLOW_T))
        self.say("st_j", FadeIn(trs[6]))
        self.say("st_k", FadeOut(note), Write(moral))

    # 5. published peak is an aggregate
    def peak(self):
        head = Text("A published peak is an aggregate", font_size=40).to_edge(UP)
        a = VGroup(Text("B300  148 SMs", font_size=30, color=BLUE_T),
                   Text("2250 TFLOPS bf16", font_size=30)).arrange(DOWN)
        b = VGroup(Text("B300  152 SMs", font_size=30, color=GREEN_T),
                   Text("? TFLOPS bf16", font_size=30)).arrange(DOWN)
        left = VGroup(SurroundingRectangle(a, color=BLUE_T, buff=0.3), a).shift(LEFT * 3.5 + UP)
        right = VGroup(SurroundingRectangle(VGroup(b, Text("2311 TFLOPS bf16", font_size=30).move_to(b[1])), color=GREEN_T, buff=0.3), b).shift(RIGHT * 3.5 + UP)
        per_sm = Text("2250 / 148 = 15.20 per SM", font_size=30, color=YELLOW_T)
        per_sm.shift(DOWN * 0.6)
        scaled = Text("15.20 × 152 = 2311", font_size=30, color=YELLOW_T)
        scaled.next_to(per_sm, DOWN)

        self.say("pk_a", Write(head))
        self.say("pk_b", FadeIn(left), FadeIn(right), Indicate(b[1], color=GREEN_T))
        self.say("pk_c", Write(per_sm))
        self.say("pk_d", Write(scaled),
                 Transform(b[1], Text("2311 TFLOPS bf16", font_size=30).move_to(b[1])))
        self.say("pk_e", Write(caption(
            "One spec sheet for B300: two real SKUs can't both match it", 28)))

    # 6. SwiGLU 8/3 trap
    def swiglu(self):
        head = Text("SwiGLU's 8/3 trap", font_size=40).to_edge(UP)
        sub = Text("d_ff = 8/3 × 4096 = 10922     (H100, h = 4096)", font_size=26,
                   color=GREY_B).next_to(head, DOWN)
        data = [("10922", 272.73), ("10944", 398.38), ("10848", 395.62),
                ("10880", 395.52), ("10912", 395.16), ("11008", 395.01)]
        chart = BarChart([v for _, v in data], bar_names=[n for n, _ in data],
                         y_range=[0, 500, 100], y_length=3.6, x_length=9,
                         bar_colors=[RED_T] + [BLUE_T] * 5,
                         y_axis_config={"font_size": 22},
                         x_axis_config={"font_size": 22})
        chart.next_to(sub, DOWN, buff=0.4)
        yl = Text("TFLOPS", font_size=22).next_to(chart.y_axis, UP)
        labels = chart.get_bar_labels(font_size=20)
        b0, b1 = chart.bars[0], chart.bars[1]
        arr = Arrow(b0.get_top() + UP * 0.45, b1.get_top() + UP * 0.45, buff=0.05,
                    color=YELLOW_T, path_arc=-1.2)
        gain = Text("+22 → +46%", font_size=26, color=YELLOW_T).next_to(arr, UP, buff=0.05)
        llama = Text("Llama-2-7B picked 11008", font_size=24, color=GREEN_T)
        llama.next_to(chart.bars[5], UP, buff=0.5).shift(LEFT * 0.6)

        self.say("sw_a", Write(head))
        def mats(n, lbl):
            return VGroup(*[VGroup(Rectangle(width=1.3, height=0.8, stroke_color=BLUE_T),
                                   Text(lbl, font_size=22)) for _ in range(n)]
                          ).arrange(RIGHT, buff=0.2)

        std = mats(2, "h × 4h")
        glu = mats(3, "h × 8/3·h")
        std_l = Text("standard MLP", font_size=24, color=GREY_B)
        glu_l = Text("SwiGLU MLP", font_size=24, color=GREY_B)
        std_p = Text("2 × 4h² = 8h²", font_size=26, color=YELLOW_T)
        glu_p = Text("3 × 8/3·h² = 8h²", font_size=26, color=YELLOW_T)
        mlp = VGroup(VGroup(std_l, std, std_p).arrange(DOWN),
                     VGroup(glu_l, glu, glu_p).arrange(DOWN)).arrange(RIGHT, buff=1.2)
        for g in (std, glu):
            for m in g:
                m[1].move_to(m[0])
        self.say("sw_b", [FadeIn(std_l), FadeIn(std)], [FadeIn(glu_l), FadeIn(glu)],
                 Write(std_p), Write(glu_p))
        self.say("sw_c", FadeOut(mlp), FadeIn(sub))
        self.say("sw_d", [Create(chart, run_time=2), FadeIn(yl)])
        self.say("sw_e", FadeIn(labels[0]), FadeIn(labels[1:]))
        self.say("sw_f", [Create(arr), Write(gain)], Indicate(chart.bars[1], color=YELLOW_T))
        self.say("sw_g", [Indicate(chart.bars[5], color=GREEN_T), FadeIn(llama)])
        self.say("sw_h", Write(caption(
            "8/3 is a suggestion: search nearby for a shape the GPU likes", 28)))

    # 7. heads
    def heads(self):
        head = Text("Attention heads", font_size=40).to_edge(UP)
        lines = VGroup(
            Text("Keep h/a (head size) as large as accuracy allows", font_size=30),
            Text("More powers of 2 in h/a helps", font_size=30),
            Text("Flash attention takes care of head-sizing constraints", font_size=30),
            Text("only need h/a large enough to saturate the GPU", font_size=30,
                 color=YELLOW_T),
        ).arrange(DOWN, buff=0.45)
        self.say("hd_a", Write(head))
        for i, l in enumerate(lines):
            self.say("hd_" + "bcde"[i], FadeIn(l, shift=UP * 0.2))

    # 8. checklist
    def checklist(self):
        head = Text("Model sizing checklist", font_size=40).to_edge(UP)
        items = ["Vocab size divisible by 64",
                 "Micro-batch size as large as possible",
                 "b*s, h/a, h/t divisible by a power of 2",
                 "(b*a)/t is an integer",
                 "t as small as possible",
                 "SwiGLU: search for the fastest d_ff near 8/3*h"]
        rows = VGroup()
        for it in items:
            box = Square(0.35, stroke_color=GREEN_T)
            rows.add(VGroup(box, Text(it, font_size=28)).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(head, DOWN, buff=0.6)
        legend = Text("b micro-batch   s seq len   h hidden   a heads   t tensor parallel",
                      font_size=20, color=GREY_B).to_edge(DOWN, buff=0.4)
        self.say("ck_a", Write(head), FadeIn(legend))
        for i, r in enumerate(rows):
            check = Text("✓", font_size=30, color=GREEN_T).move_to(r[0])
            self.say(f"ck_{i}", FadeIn(r, shift=RIGHT * 0.2), FadeIn(check, scale=1.5))
            r.add(check)
        self.wait(0.5)
        self.play(FadeOut(rows), FadeOut(legend))
        self.say("ck_b", Write(Text("Count your blocks. Divide by your SMs.",
                                    font_size=34, color=YELLOW_T)))


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
