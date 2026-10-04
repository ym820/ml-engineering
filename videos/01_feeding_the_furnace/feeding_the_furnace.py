"""Video 1: Feeding the furnace.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python feeding_the_furnace.py
2. Render: manim -qh feeding_the_furnace.py FeedingTheFurnace

Every number comes from insights/ai-battlefield.md (Can you feed the furnace fast
enough?, Moving bits, Network) and training/performance/README.md (Anatomy of
Model's Operations, torch.compile -> Where it pays off). The roofline section is
added material: the repo never draws one, so it is derived from the repo's H100
numbers (989 TFLOPS bf16, 3.35 TBps HBM) and labeled as such on screen.
"""
import wave
from math import log10
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
YELLOW_T = "#FFFF00"
RED_T = "#FC6255"
GREEN_T = "#83C167"
ORANGE_T = "#FF862F"
GREY_T = "#888888"

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "h_a": "Here are two measurements taken on the same GPU, an NVIDIA B200, using the "
           "same compiler, torch dot compile.",
    "h_b": "On Llama 3.2 1B, with a sequence length of 512, compiling makes a training "
           "step about six times faster.",
    "h_c": "On Llama 3.1 8B, with a sequence length of 8192, the compiled step actually "
           "gets a little slower, running at about 0.94 times the speed of eager mode.",
    "h_d": "It's the same tool on the same chip, with wildly different results. To see "
           "why, it helps to think a little less about how fast a GPU can compute, and a "
           "little more about how fast it can be fed.",

    "f_a": "Picture a steam locomotive.",
    "f_b": "The engine itself can be wonderful, but somebody has to shovel coal into the "
           "firebox, and that person is called the fireman.",
    "f_c": "If the fireman can't shovel fast enough, it doesn't matter how great the "
           "engine is, because the train just won't move fast.",
    "f_d": "And that's pretty much the state of ML hardware today. The bottleneck is in "
           "moving bits around, and not in computing on them.",
    "f_e": "Accelerators get roughly twice as fast every two years, but memory and "
           "networks haven't kept that pace, and both already hold compute back.",

    "p_a": "You can see that gap in NVIDIA's own GPUs.",
    "p_b": "Starting from the V100, fp16 compute went up two and a half times on the "
           "A100, eight times on the H100, eighteen times on the B200, and thirty-two "
           "times on Rubin.",
    "p_c": "Now let's put intra-node bandwidth on the same plot. That's NVLink, which is "
           "how the GPUs inside one machine talk to each other.",
    "p_d": "For the first few generations, it only grew by about 150 gigabytes per "
           "second each time, going from 150, to 300, to 450.",
    "p_e": "NVLink 5 doubled that, so it catches up a little, but by Rubin you have "
           "twelve times the bandwidth to feed thirty-two times the compute.",
    "p_f": "And between machines it's worse. Most network cards there do 100 or 200 "
           "gigabits per second, which is only 12 and a half or 25 gigabytes per "
           "second, since a byte is eight bits.",

    "o_a": "So which parts of a transformer actually feel this? It turns out a "
           "transformer only does three kinds of operations.",
    "o_b": "First, there are tensor contractions, which are the matrix multiplies in the "
           "linear layers and inside attention.",
    "o_c": "Then there are statistical normalizations, like softmax and layer norm, "
           "which do a reduction and then apply the result back over the tensor.",
    "o_d": "And everything else is element-wise: biases, dropout, activations, and "
           "residual connections.",
    "o_e": "Only the first group is compute-heavy. The other two read a tensor from "
           "memory, do a tiny bit of arithmetic on each element, and write it right "
           "back.",
    "o_f": "This way of splitting things up comes from a paper with a pretty telling "
           "title: Data Movement Is All You Need.",

    "r_a": "To put a number on compute-heavy, I'm going to step outside the repo for a "
           "moment and borrow an idea called arithmetic intensity, which is just how "
           "many FLOPs you do for each byte you move.",
    "r_b": "An H100 can do 989 teraflops in bf16, and its memory delivers 3.35 terabytes "
           "per second.",
    "r_c": "Divide one by the other and you get about 295. So unless an operation does "
           "around 295 FLOPs for every byte it pulls from memory, the memory is the "
           "limit, and not the math.",
    "r_d": "If you plot the best possible speed against arithmetic intensity, you get "
           "this shape, which people call a roofline. On the left you're limited by "
           "bandwidth, and on the right you're limited by compute.",
    "r_e": "An element-wise op, say multiplying a bf16 tensor by a constant, reads two "
           "bytes and writes two bytes for every single FLOP, so it sits way down here, "
           "at a quarter of a FLOP per byte.",
    "r_f": "A big square matmul is the opposite. With 4096 by 4096 matrices in bf16, it "
           "does over a thousand FLOPs per byte, so it sits right up against the "
           "compute roof.",

    "w_a": "Memory is only one of the places bits have to move, though, so let's draw a "
           "whole cluster.",
    "w_b": "Inside each GPU, data travels between high bandwidth memory, or HBM, and the "
           "streaming multiprocessors that do the math.",
    "w_c": "The GPUs inside a node talk to each other over NVLink,",
    "w_d": "nodes talk to each other through their network cards,",
    "w_e": "and the training data comes from storage, through the CPU and the "
           "DataLoader workers.",
    "w_f": "Every one of these arrows is a place where the GPU can end up sitting idle, "
           "waiting for coal.",
    "w_g": "If you only have one GPU and the model fits on it, the HBM arrow is the only "
           "one you need to worry about.",
    "w_h": "Once you shard the model across GPUs, the network joins in. Good frameworks "
           "hide a lot of it by overlapping communication with compute, but if the "
           "comms take longer than the compute, you're still waiting on them.",
    "w_i": "Storage matters mostly for feeding the DataLoader, which adds very little "
           "overhead with enough workers, and for saving checkpoints, where the GPUs "
           "sit idle unless the save is asynchronous.",
    "w_j": "Later videos dig into each of these. For now, let's stay on that very first "
           "arrow, between HBM and the compute.",

    "fu_a": "There's not much you can do about HBM bandwidth itself, since it is what it "
            "is. What you can change is how many times your code crosses it.",
    "fu_b": "Imagine a chain of three element-wise ops, say adding a bias, then an "
            "activation, and then dropout.",
    "fu_c": "In eager mode, each op is its own kernel. The first one reads the tensor "
            "from HBM, does its little bit of math, and writes the result back.",
    "fu_d": "Then the second one reads the whole thing again, and writes it back again,",
    "fu_e": "and the third one does the same. That's three round trips, for very little "
            "arithmetic.",
    "fu_f": "A fused kernel reads the tensor once, does all three ops while the data is "
            "sitting right next to the compute, and writes it back once.",
    "fu_g": "Flash attention does this by hand for attention, and fusing chains of small "
            "ops like these is the bulk of what torch dot compile does to a transformer.",
    "fu_h": "Notice what fusion doesn't help, though. The big matmuls already go to "
            "vendor libraries running near the hardware limit, so there's very little "
            "left there for a compiler to win.",

    "pay_a": "Which brings us back to those two measurements.",
    "pay_b": "The 1B model at sequence 512 takes only a few milliseconds per step, and "
             "that time is dominated by kernel launch overhead and by the "
             "bandwidth-bound normalizations and element-wise ops we just looked at.",
    "pay_c": "So for a forward and backward pass, default compile is 3.9 times faster.",
    "pay_d": "And the reduce-overhead mode, which replays the whole step as CUDA graphs "
             "instead of launching thousands of kernels one at a time, gets all the way "
             "to 6.",
    "pay_e": "The 8B model at sequence 8192 takes hundreds of milliseconds per step, and "
             "almost all of that time sits in big matmuls that already run at cuBLAS "
             "peak.",
    "pay_f": "Every compiled mode lands at about 0.94. Compilation didn't fail here. "
             "There was simply nothing for it to remove.",
    "pay_g": "On top of that, the first compiled step took about 47 seconds on the 8B "
             "model, and with no steady-state saving, that time never pays back. On the "
             "1B model, compiling breaks even after roughly 600 steps.",
    "pay_h": "So the speedup you can expect tracks how bandwidth-bound or launch-bound "
             "your step is. It's the same tool, with a different bottleneck underneath.",

    "c_a": "The repo's advice from all of this is to research the whole machine, and not "
           "just its engine.",
    "c_b": "It even floats a slightly crazy idea. An older GPU might do just fine if you "
           "can actually feed it as fast as it computes,",
    "c_c": "and if you can get three of them for the cost of one next generation GPU, "
           "you might finish training sooner, and for less.",
    "c_d": "We'll keep coming back to this furnace throughout the series, every time "
           "something leaves the GPU waiting for coal.",
}


def is_text(anim):
    """Text appearing (Write/FadeIn); emphasis on text (Indicate...) may stretch."""
    if not isinstance(anim, (Write, FadeIn)):
        return False
    m = anim.mobject
    return isinstance(m, Text) or (isinstance(m, VGroup) and m.submobjects and
                                   all(isinstance(s, Text) for s in m.submobjects))


def caption(text, size=28, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.45)


def box(label, w, h, color, size=24, fill=0.25):
    r = RoundedRectangle(corner_radius=0.1, width=w, height=h, stroke_color=color,
                         fill_color=color, fill_opacity=fill)
    return VGroup(r, Text(label, font_size=size).move_to(r))


def locomotive():
    """A side view facing right. Returns a dict of named parts plus 'all'."""
    p = {}
    p["boiler"] = RoundedRectangle(corner_radius=0.35, width=4.2, height=1.4,
                                   stroke_color=WHITE, fill_color="#3a3a4a",
                                   fill_opacity=1).move_to([0.6, 0.15, 0])
    p["smokebox"] = Rectangle(width=0.35, height=1.5, stroke_color=WHITE,
                              fill_color="#222230", fill_opacity=1).move_to([2.6, 0.15, 0])
    p["stack"] = Polygon([1.85, 0.85, 0], [2.25, 0.85, 0], [2.4, 1.7, 0], [1.7, 1.7, 0],
                         stroke_color=WHITE, fill_color="#222230", fill_opacity=1)
    p["dome"] = Arc(radius=0.35, start_angle=0, angle=PI, stroke_color=WHITE,
                    fill_color="#3a3a4a", fill_opacity=1).move_to([0.4, 1.0, 0])
    p["cab"] = Rectangle(width=1.7, height=2.5, stroke_color=WHITE, fill_color="#5a3030",
                         fill_opacity=1).move_to([-2.35, 0.45, 0])
    p["roof"] = Line([-3.4, 1.75, 0], [-1.3, 1.75, 0], stroke_width=8, color=WHITE)
    p["window"] = Rectangle(width=0.9, height=0.75, stroke_color=WHITE, fill_color=BLACK,
                            fill_opacity=1).move_to([-2.35, 0.95, 0])
    p["firebox"] = Rectangle(width=0.55, height=0.55, stroke_color=ORANGE_T,
                             fill_color=ORANGE_T, fill_opacity=0.9).move_to([-1.75, -0.35, 0])
    p["frame"] = Rectangle(width=6.6, height=0.25, stroke_color=WHITE, fill_color="#222230",
                           fill_opacity=1).move_to([-0.4, -0.85, 0])
    wheels = VGroup()
    for x, r in ((-2.6, 0.4), (-0.9, 0.55), (0.4, 0.55), (1.7, 0.55)):
        w = VGroup(Circle(r, stroke_color=WHITE, stroke_width=4, fill_color="#222230",
                          fill_opacity=1),
                   Line(UP * r, DOWN * r, stroke_width=2), Line(LEFT * r, RIGHT * r,
                                                                stroke_width=2))
        wheels.add(w.move_to([x, -1.55 + r, 0]))
    p["wheels"] = wheels
    p["rod"] = Line([-0.9, -1.0, 0], [1.7, -1.0, 0], stroke_width=5, color=GREY_B)
    p["tender"] = Rectangle(width=2.1, height=1.4, stroke_color=WHITE, fill_color="#5a3030",
                            fill_opacity=1).move_to([-4.6, -0.15, 0])
    coal = VGroup(*[Dot([-5.45 + 0.17 * i + 0.08 * (j % 2), 0.62 + 0.13 * j, 0],
                        radius=0.09, color="#1b1b1b").set_stroke(GREY_D, 1)
                    for j in range(3) for i in range(10 - 2 * j)])
    p["coal"] = coal.shift(RIGHT * 0.17 * 0)
    tw = VGroup(*[Circle(0.35, stroke_color=WHITE, stroke_width=4, fill_color="#222230",
                         fill_opacity=1).move_to([x, -1.2, 0]) for x in (-5.2, -4.0)])
    p["tender_wheels"] = tw
    # fireman, standing in the cab, facing the firebox
    head = Circle(0.17, stroke_color=WHITE, fill_color=WHITE, fill_opacity=1)
    head.move_to([-2.75, 0.85, 0])
    body = Line([-2.75, 0.68, 0], [-2.75, -0.05, 0], color=WHITE, stroke_width=5)
    legs = VGroup(Line([-2.75, -0.05, 0], [-2.95, -0.65, 0], color=WHITE, stroke_width=5),
                  Line([-2.75, -0.05, 0], [-2.55, -0.65, 0], color=WHITE, stroke_width=5))
    shovel = VGroup(Line([-2.75, 0.45, 0], [-2.2, 0.0, 0], color=WHITE, stroke_width=5),
                    Line([-2.2, 0.0, 0], [-1.95, -0.25, 0], color=GREY_B, stroke_width=4),
                    Polygon([-2.05, -0.2, 0], [-1.85, -0.38, 0], [-1.75, -0.22, 0],
                            [-1.92, -0.08, 0], stroke_color=GREY_B, fill_color=GREY_B,
                            fill_opacity=1))
    p["fireman"] = VGroup(head, body, legs, shovel)
    p["shovel"] = shovel
    p["shoulder"] = Dot([-2.75, 0.45, 0], radius=0.01).set_opacity(0)
    p["all"] = VGroup(p["tender"], p["coal"], p["tender_wheels"], p["frame"], p["boiler"],
                      p["smokebox"], p["stack"], p["dome"], p["cab"], p["roof"],
                      p["window"], p["firebox"], p["fireman"], p["wheels"], p["rod"],
                      p["shoulder"])
    return p


class FeedingTheFurnace(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.furnace, self.pipes, self.ops, self.roofline,
                     self.machine, self.fusion, self.payoff, self.coda):
            beat()
            self.wait(0.6)
            self.clear_updaters()
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
            self.wait(0.3)

    def clear_updaters(self):
        for m in self.mobjects:
            m.clear_updaters()

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

    # 0. hook: same tool, opposite results
    def hook(self):
        chip = Text("NVIDIA B200  ·  torch.compile  ·  bf16", font_size=30, color=GREY_B)
        chip.to_edge(UP, buff=0.8)
        scale = 1.05  # units per 1x

        def row(name, val, label, color, y):
            n = Text(name, font_size=28).move_to([-4.6, y, 0])
            base = np.array([-2.6, y, 0])
            bar = Rectangle(width=val * scale, height=0.6, fill_color=color,
                            fill_opacity=0.85, stroke_width=0)
            bar.move_to(base, aligned_edge=LEFT)
            v = Text(label, font_size=32, color=color).next_to(bar, RIGHT)
            return n, bar, v

        one = Line([-2.6 + scale, 1.6, 0], [-2.6 + scale, -2.0, 0], color=WHITE,
                   stroke_width=2).set_opacity(0.6)
        one_l = Text("eager = 1x", font_size=22, color=GREY_B).next_to(one, DOWN)
        a = row("Llama-3.2-1B\nseq 512", 6.0, "~6x faster", GREEN_T, 0.6)
        b = row("Llama-3.1-8B\nseq 8192", 0.94, "0.94x", RED_T, -1.0)
        self.say("h_a", Write(chip), [Create(one), FadeIn(one_l)],
                 Indicate(chip, color=YELLOW_T, scale_factor=1.05))
        self.say("h_b", FadeIn(a[0]), [GrowFromEdge(a[1], LEFT), FadeIn(a[2])],
                 Indicate(a[2], color=GREEN_T))
        slower = Text("slower than not compiling", font_size=24, color=RED_T)
        slower.next_to(b[2], RIGHT, buff=0.4)
        self.say("h_c", FadeIn(b[0]), [GrowFromEdge(b[1], LEFT), FadeIn(b[2])],
                 Indicate(one, color=YELLOW_T), [FadeIn(slower), Indicate(b[2], color=RED_T)])
        title = Text("Feeding the furnace", font_size=64)
        self.say("h_d", Circumscribe(VGroup(a[1], b[1]), color=YELLOW_T),
                 FadeOut(VGroup(chip, one, one_l, slower, *a, *b)), Write(title),
                 Indicate(title, color=ORANGE_T, scale_factor=1.05))

    # 1. the locomotive
    def furnace(self):
        p = locomotive()
        loco = p["all"]
        rig = VGroup(loco).scale(0.8).shift(UP * 0.4)
        rail_y = rig.get_bottom()[1] - 0.05
        rail = Line([-8, rail_y, 0], [8, rail_y, 0], color=GREY_B, stroke_width=4)
        ties = VGroup(*[Rectangle(width=0.35, height=0.12, fill_color="#6b4a2b",
                                  fill_opacity=1, stroke_width=0)
                        .move_to([x, rail_y - 0.1, 0]) for x in np.arange(-8, 8.1, 0.8)])
        speed = ValueTracker(2.4)

        def roll(m, dt):
            for t in m:
                t.shift(LEFT * speed.get_value() * dt)
                if t.get_x() < -8:
                    t.shift(RIGHT * 16.8)
        ties.add_updater(roll)

        def spin(m, dt):
            for w in m:
                w.rotate(-speed.get_value() * dt / 0.44)
        p["wheels"].add_updater(spin)
        p["tender_wheels"].add_updater(spin)

        stack_top = p["stack"].get_top()
        puffs = VGroup()

        def puff():
            c = Circle(0.18, stroke_width=0, fill_color=GREY_C, fill_opacity=0.7)
            c.move_to(stack_top + UP * 0.1)
            puffs.add(c)
            return c.animate.shift(UP * 1.1 + LEFT * 1.6).scale(2.5).set_opacity(0)

        self.say("f_a", [FadeIn(rail), FadeIn(ties), DrawBorderThenFill(loco, run_time=2)],
                 LaggedStart(*[puff() for _ in range(4)], lag_ratio=0.35))
        fireman = Text("the fireman", font_size=26, color=YELLOW_T)
        fireman.next_to(p["cab"], UP, buff=0.5).shift(LEFT * 0.6)
        arr = Arrow(fireman.get_bottom(), p["fireman"][0].get_top(), buff=0.1,
                    color=YELLOW_T, stroke_width=3)
        fb = Text("firebox", font_size=22, color=ORANGE_T)
        fb.next_to(p["boiler"], DOWN, buff=0.15).align_to(p["boiler"], LEFT).shift(DOWN * 0.9)
        fb_arr = Arrow(fb.get_top(), p["firebox"].get_bottom(), buff=0.05, color=ORANGE_T,
                       stroke_width=3)
        src, dst = p["coal"].get_top() + RIGHT * 0.3, p["firebox"].get_center()
        sh = p["shoulder"].get_center()

        def scoop():
            lump = Dot(src, radius=0.08, color=GREY_D)
            path = ArcBetweenPoints(src, dst, angle=-PI / 2)
            return Succession(
                Rotate(p["shovel"], 0.5, about_point=sh, run_time=0.4),
                AnimationGroup(MoveAlongPath(lump, path, run_time=0.6),
                               Rotate(p["shovel"], -0.5, about_point=sh, run_time=0.6)),
                AnimationGroup(FadeOut(lump, run_time=0.2),
                               Flash(p["firebox"], color=ORANGE_T, run_time=0.4)))

        self.say("f_b", [Write(fireman), GrowArrow(arr)], [FadeIn(fb), GrowArrow(fb_arr)],
                 scoop(), scoop())
        slow = Text("can't keep up  →  train crawls", font_size=28, color=RED_T)
        slow.to_edge(UP, buff=0.4)
        self.say("f_c", [FadeOut(fireman), FadeOut(arr), FadeOut(fb), FadeOut(fb_arr)],
                 [speed.animate.set_value(0.25),
                  p["firebox"].animate.set_fill(opacity=0.25), Write(slow)],
                 Rotate(p["shovel"], 0.5, about_point=sh, run_time=1.5))
        thesis = VGroup(Text("The bottleneck is moving bits,", font_size=34),
                        Text("not computing on them.", font_size=34, color=YELLOW_T))
        thesis.arrange(DOWN).to_edge(UP, buff=0.35)
        self.say("f_d", FadeOut(slow), Write(thesis[0]), Write(thesis[1]),
                 [Indicate(p["firebox"], color=ORANGE_T),
                  Rotate(p["shovel"], -0.5, about_point=sh)])
        rates = VGroup(Text("compute: ~2x every 2 years", font_size=26, color=GREEN_T),
                       Text("memory, network: not keeping pace", font_size=26,
                            color=RED_T)).arrange(RIGHT, buff=0.8)
        rates.to_edge(DOWN, buff=0.15)
        self.say("f_e", [FadeOut(VGroup(rail, ties)), FadeIn(rates[0])], FadeIn(rates[1]),
                 Indicate(p["fireman"], color=RED_T))

    # 2. compute outgrows the pipes
    def pipes(self):
        head = Text("Compute outgrows the pipes", font_size=40).to_edge(UP)
        gpus = ["V100", "A100", "H100", "B200", "Rubin"]
        comp = [1, 2.5, 8, 18, 32]
        link = [1, 2, 3, 6, 12]
        gbps = [150, 300, 450, 900, 1800]
        ax = Axes(x_range=[0, 4.6, 1], y_range=[0, 35, 5], x_length=9, y_length=4.6,
                  tips=False, axis_config={"stroke_width": 2},
                  y_axis_config={"include_numbers": True, "font_size": 22,
                                 "numbers_to_include": [0, 10, 20, 30]})
        ax.shift(DOWN * 0.4 + LEFT * 0.6)
        xl = VGroup(*[Text(g, font_size=22).next_to(ax.c2p(i, 0), DOWN)
                      for i, g in enumerate(gpus)])
        yl = Text("speedup vs V100", font_size=22).next_to(ax.y_axis, UP).shift(RIGHT * 0.6)

        def curve(vals, color):
            dots = VGroup(*[Dot(ax.c2p(i, v), color=color) for i, v in enumerate(vals)])
            line = VMobject(color=color, stroke_width=4).set_points_as_corners(
                [ax.c2p(i, v) for i, v in enumerate(vals)])
            return line, dots

        cl, cd = curve(comp, YELLOW_T)
        ll, ld = curve(link, BLUE_T)
        cvals = VGroup(*[Text(f"{v:g}x", font_size=20, color=YELLOW_T)
                         .next_to(cd[i], UL, buff=0.08) for i, v in enumerate(comp)])
        lvals = VGroup(*[Text(f"{g} GBps", font_size=18, color=BLUE_T)
                         .next_to(xl[i], DOWN, buff=0.08) for i, g in enumerate(gbps)])
        ctag = Text("fp16 TFLOPS", font_size=24, color=YELLOW_T).next_to(cd[-1], RIGHT)
        ltag = Text("NVLink GBps", font_size=24, color=BLUE_T).next_to(ld[-1], RIGHT)
        self.say("p_a", Write(head), [Create(ax), FadeIn(xl), FadeIn(yl)])
        self.say("p_b", Create(cl, run_time=2.5), [FadeIn(cd), FadeIn(cvals), FadeIn(ctag)])
        self.say("p_c", [Create(ll, run_time=2), FadeIn(ld), FadeIn(ltag)])
        self.say("p_d", LaggedStart(*[FadeIn(lvals[i], scale=1.4) for i in range(3)],
                                    lag_ratio=0.5))
        gap = DoubleArrow(ld[-1].get_center(), cd[-1].get_center(), buff=0.15,
                          color=RED_T, stroke_width=3)
        gl = Text("32x vs 12x", font_size=26, color=RED_T).next_to(gap, RIGHT)
        self.say("p_e", LaggedStart(*[FadeIn(lvals[i], scale=1.4) for i in (3, 4)],
                                    lag_ratio=0.5), [GrowFromCenter(gap), Write(gl)])
        nic = VGroup(Text("between nodes: NICs at 100 / 200 Gbps", font_size=26),
                     Text("= 12.5 / 25 GBps   (1 byte = 8 bits)", font_size=26,
                          color=RED_T)).arrange(DOWN, buff=0.15)
        nic.to_edge(DOWN, buff=0.15)
        self.say("p_f", [ax.animate.shift(UP * 0.5), *[m.animate.shift(UP * 0.5) for m in
                          (xl, yl, cl, cd, ll, ld, cvals, lvals, ctag, ltag, gap, gl)]],
                 Write(nic[0]), Write(nic[1]), Circumscribe(nic[1], color=RED_T))

    # 3. three kinds of ops
    def ops(self):
        head = Text("Three kinds of operations", font_size=40).to_edge(UP)
        M, S, E = "M", "S", "E"
        rows = [[("LayerNorm", S), ("QKV proj", M), ("Q·Kᵀ", M), ("softmax", S),
                 ("dropout", E), ("·V", M), ("out proj", M)],
                [("residual", E), ("LayerNorm", S), ("MLP up", M), ("bias", E),
                 ("GeLU", E), ("MLP down", M), ("residual", E)]]
        color = {M: BLUE_T, S: YELLOW_T, E: RED_T}
        blocks = VGroup()
        kinds = []
        for r in rows:
            g = VGroup()
            for name, k in r:
                g.add(box(name, 1.6, 0.75, GREY_T, size=18, fill=0.15))
                kinds.append(k)
            g.arrange(RIGHT, buff=0.18)
            blocks.add(g)
        blocks.arrange(DOWN, buff=0.5).shift(UP * 0.9)
        flat = [b for g in blocks for b in g]
        arrows = VGroup(*[Arrow(flat[i].get_right(), flat[i + 1].get_left(), buff=0.02,
                                stroke_width=2, max_tip_length_to_length_ratio=0.5,
                                color=GREY_B)
                          for i in range(len(flat) - 1) if i != 6])
        blk_l = Text("one transformer block", font_size=22, color=GREY_B)
        blk_l.next_to(blocks, UP, buff=0.2)

        def paint(k):
            return [flat[i][0].animate.set_stroke(color[k]).set_fill(color[k], 0.35)
                    for i in range(len(flat)) if kinds[i] == k]

        legend = VGroup(
            Text("1. tensor contractions (matmuls)", font_size=26, color=BLUE_T),
            Text("2. statistical normalizations (reductions)", font_size=26,
                 color=YELLOW_T),
            Text("3. element-wise ops", font_size=26, color=RED_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(blocks, DOWN, buff=0.6)
        self.say("o_a", Write(head),
                 [LaggedStart(*[FadeIn(b) for b in flat], lag_ratio=0.08),
                  FadeIn(arrows), FadeIn(blk_l)])
        self.say("o_b", paint(M), FadeIn(legend[0]))
        self.say("o_c", paint(S), FadeIn(legend[1]),
                 LaggedStart(*[Indicate(flat[i], color=YELLOW_T, scale_factor=1.1)
                               for i in range(len(flat)) if kinds[i] == S], lag_ratio=0.3))
        self.say("o_d", paint(E), FadeIn(legend[2]))
        heavy = Text("compute-bound", font_size=24, color=BLUE_T)
        light = Text("memory-bound", font_size=24, color=RED_T)
        heavy.next_to(legend[0], RIGHT, buff=0.5)
        light.next_to(VGroup(legend[1], legend[2]), RIGHT, buff=0.5)
        brace = Brace(VGroup(legend[1], legend[2]), RIGHT, color=GREY_B)
        light.next_to(brace, RIGHT)
        self.say("o_e", FadeIn(heavy), [GrowFromCenter(brace), FadeIn(light)],
                 LaggedStart(*[Indicate(flat[i], color=RED_T, scale_factor=1.1)
                               for i in range(len(flat)) if kinds[i] != M],
                             lag_ratio=0.1))
        self.say("o_f", Write(caption('"Data Movement Is All You Need" (Ivanov et al., 2020)',
                                      24, GREY_B)),
                 Circumscribe(VGroup(brace, light), color=RED_T))

    # 4. roofline (added material, derived from the repo's H100 numbers)
    def roofline(self):
        head = Text("Arithmetic intensity", font_size=40).to_edge(UP)
        tag = Text("added material: derived, not drawn in the repo", font_size=20,
                   color=ORANGE_T).next_to(head, DOWN, buff=0.12)
        defn = Text("FLOPs done  /  bytes moved", font_size=32, color=YELLOW_T)
        defn.next_to(tag, DOWN, buff=0.4)
        gist = Text("how much math you get out of each trip to memory", font_size=24,
                    color=GREY_B).next_to(defn, DOWN, buff=0.2)
        self.say("r_a", Write(head), FadeIn(tag), Write(defn),
                 Circumscribe(defn, color=YELLOW_T), FadeIn(gist, shift=UP * 0.2),
                 Indicate(gist, color=WHITE, scale_factor=1.05))
        h100 = VGroup(Text("H100:  989 TFLOPS bf16", font_size=30),
                      Text("HBM:  3.35 TBps", font_size=30)).arrange(DOWN)
        h100.next_to(gist, DOWN, buff=0.4)
        self.say("r_b", Write(h100[0]), Indicate(h100[0], color=BLUE_T), Write(h100[1]),
                 Indicate(h100[1], color=RED_T))
        ridge = Text("989 / 3.35 ≈ 295 FLOP per byte", font_size=32, color=YELLOW_T)
        ridge.next_to(h100, DOWN, buff=0.4)
        sides = VGroup(Text("below 295: memory is the limit", font_size=24, color=RED_T),
                       Text("above 295: math is the limit", font_size=24, color=BLUE_T))
        sides.arrange(RIGHT, buff=0.8).next_to(ridge, DOWN, buff=0.35)
        self.say("r_c", Write(ridge), Circumscribe(ridge, color=YELLOW_T),
                 FadeIn(sides[0], shift=UP * 0.2), FadeIn(sides[1], shift=UP * 0.2),
                 Indicate(sides[0], color=RED_T, scale_factor=1.05),
                 Indicate(sides[1], color=BLUE_T, scale_factor=1.05))

        # log10 coordinates shifted by +1, so the axes cross at (0.1, 0.1)
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 4.5, 1], x_length=9, y_length=4.4,
                  tips=False, axis_config={"stroke_width": 2})
        ax.shift(DOWN * 0.75)
        lx = lambda v: log10(v) + 1
        xt = VGroup(*[Text(s, font_size=20).next_to(ax.c2p(i, 0), DOWN)
                      for i, s in enumerate(["0.1", "1", "10", "100", "1000", "10000"])])
        yt = VGroup(*[Text(s, font_size=20).next_to(ax.c2p(0, i), LEFT)
                      for i, s in enumerate(["0.1", "1", "10", "100", "1000"])])
        xl = Text("FLOP per byte (log)", font_size=22).next_to(xt, DOWN, buff=0.15)
        yl = Text("TFLOPS (log)", font_size=22).next_to(ax.y_axis, UP)
        roof = VMobject(stroke_width=5, color=WHITE).set_points_as_corners([
            ax.c2p(0, lx(0.335)), ax.c2p(lx(295), lx(989)), ax.c2p(5, lx(989))])
        mem = Text("bandwidth-bound", font_size=22, color=RED_T)
        mem.rotate(0.5).move_to(ax.c2p(1.5, 2.7))
        cmp_ = Text("compute-bound", font_size=22, color=BLUE_T).move_to(ax.c2p(4.4, 4.3))
        rdot = Dot(ax.c2p(lx(295), lx(989)), color=YELLOW_T)
        rlab = Text("295", font_size=20, color=YELLOW_T).next_to(rdot, UP, buff=0.12)
        self.say("r_d", [FadeOut(VGroup(defn, gist, h100, ridge, sides)),
                         Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl)],
                 Create(roof, run_time=2), [FadeIn(rdot), FadeIn(rlab)],
                 [FadeIn(mem), FadeIn(cmp_)])
        e = Dot(ax.c2p(lx(0.25), lx(3.35 * 0.25)), color=RED_T, radius=0.11)
        el = Text("element-wise: 1 FLOP / 4 bytes\n= 0.25 → 0.84 TFLOPS", font_size=20,
                  color=RED_T).next_to(e, RIGHT, buff=0.5).shift(DOWN * 0.4)
        floor_ = DashedLine(ax.c2p(lx(0.25), 0), e.get_center(), color=RED_T,
                            stroke_width=2)
        self.say("r_e", FadeIn(e, scale=2), FadeIn(el), [Create(floor_), Flash(e, color=RED_T)],
                 Indicate(el, color=RED_T, scale_factor=1.05))
        m = Dot(ax.c2p(lx(1365), lx(989)), color=BLUE_T, radius=0.11)
        ml = Text("4096³ matmul: 2n³ FLOPs / 6n² bytes\n= n/3 ≈ 1365", font_size=20,
                  color=BLUE_T).next_to(m, DOWN, buff=1.6).align_to(ax, RIGHT)
        mfloor = DashedLine(ax.c2p(lx(1365), 0), m.get_center(), color=BLUE_T,
                            stroke_width=2)
        self.say("r_f", FadeIn(m, scale=2), FadeIn(ml), [Create(mfloor), Flash(m, color=BLUE_T)],
                 Indicate(roof, color=YELLOW_T, scale_factor=1.0))

    # 5. where bits move
    def machine(self):
        head = Text("Where bits move", font_size=40).to_edge(UP, buff=0.3)
        nodes, gpus_all, hbm_arrows, nvl, cpus = VGroup(), [], VGroup(), VGroup(), VGroup()
        for cx in (-3.5, 3.5):
            frame = RoundedRectangle(corner_radius=0.2, width=6.0, height=4.3,
                                     stroke_color=GREY_B).move_to([cx, 0.35, 0])
            nl = Text("node", font_size=20, color=GREY_B).next_to(frame, UP, buff=0.05)
            nl.align_to(frame, LEFT).shift(RIGHT * 0.15)
            gs = VGroup()
            for i in range(4):
                shell = RoundedRectangle(corner_radius=0.08, width=1.25, height=1.7,
                                         stroke_color=GREEN_T)
                hbm = Rectangle(width=1.0, height=0.38, stroke_width=0, fill_color=BLUE_T,
                                fill_opacity=0.6)
                sms = Rectangle(width=1.0, height=0.38, stroke_width=0, fill_color=GREEN_T,
                                fill_opacity=0.6)
                hbm.move_to(shell.get_top() + DOWN * 0.32)
                sms.move_to(shell.get_bottom() + UP * 0.32)
                ht = Text("HBM", font_size=16).move_to(hbm)
                st = Text("SMs", font_size=16).move_to(sms)
                a = DoubleArrow(hbm.get_bottom(), sms.get_top(), buff=0.03, stroke_width=3,
                                tip_length=0.12, color=YELLOW_T)
                g = VGroup(shell, hbm, sms, ht, st, a)
                gs.add(g)
                hbm_arrows.add(a)
            gs.arrange(RIGHT, buff=0.18).move_to([cx, 1.25, 0])
            gpus_all.append(gs)
            sw = Rectangle(width=5.4, height=0.3, fill_color=BLUE_T, fill_opacity=0.35,
                           stroke_color=BLUE_T).move_to([cx, -0.05, 0])
            swl = Text("NVLink", font_size=18).move_to(sw)
            links = VGroup(*[Line(g.get_bottom(), [g.get_x(), sw.get_top()[1], 0],
                                  color=BLUE_T, stroke_width=4) for g in gs])
            nvl.add(VGroup(sw, swl, links))
            cpu = box("CPU + DataLoader", 3.2, 0.6, GREY_T, size=18)
            cpu.move_to([cx, -1.15, 0])
            cpus.add(cpu)
            nodes.add(VGroup(frame, nl))
        nic = DoubleArrow(nodes[0][0].get_right() + DOWN * 0.4,
                          nodes[1][0].get_left() + DOWN * 0.4, buff=0, color=RED_T,
                          stroke_width=5, tip_length=0.18)
        nicl = Text("NICs", font_size=18, color=RED_T).next_to(nic, UP, buff=0.05)
        storage = box("storage", 2.4, 0.55, ORANGE_T, size=20).move_to([0, -2.6, 0])
        feeds = VGroup(*[Arrow(storage.get_corner(side), c.get_bottom(), buff=0.08,
                               color=ORANGE_T, stroke_width=3)
                         for side, c in zip((UL, UR), cpus)])
        loads = VGroup(*[Arrow(c.get_top(), sw[0].get_bottom(), buff=0.05, color=ORANGE_T,
                               stroke_width=3, max_tip_length_to_length_ratio=0.4)
                         for c, sw in zip(cpus, nvl)])
        gpus = VGroup(*gpus_all)
        hbm_parts = VGroup(*[VGroup(g[1], g[2], g[3], g[4]) for gs in gpus_all for g in gs])

        self.say("w_a", Write(head), [FadeIn(nodes), FadeIn(VGroup(*[VGroup(*[g[0] for g in gs])
                                                                        for gs in gpus_all]))])
        self.say("w_b", FadeIn(hbm_parts), LaggedStart(*[GrowFromCenter(a) for a in hbm_arrows],
                                                       lag_ratio=0.05))
        self.say("w_c", LaggedStart(*[Create(n) for n in nvl], lag_ratio=0.3))
        self.say("w_d", [GrowFromCenter(nic), FadeIn(nicl)])
        self.say("w_e", [FadeIn(storage), FadeIn(cpus)], [GrowArrow(f) for f in feeds] +
                 [GrowArrow(l) for l in loads])
        every = VGroup(hbm_arrows, VGroup(*[n[2] for n in nvl]), nic, feeds, loads)
        self.say("w_f", LaggedStart(*[Indicate(x, color=RED_T, scale_factor=1.05)
                                      for x in every], lag_ratio=0.25))

        everything = VGroup(nodes, gpus, nvl, nic, nicl, storage, cpus, feeds, loads)
        one = gpus_all[0][0]

        def focus(*keep, rest=everything):
            anims = [rest.animate.set_opacity(0.15)]
            for k in keep:
                anims.append(k.animate.set_opacity(1))
            return anims

        g1 = Text("one GPU, model fits: only HBM matters", font_size=24, color=YELLOW_T)
        g1.to_edge(DOWN, buff=0.15)
        self.play(everything.animate.set_opacity(0.15), run_time=0.5)
        self.say("w_g", [one.animate.set_opacity(1), FadeIn(g1)],
                 Indicate(one[5], color=YELLOW_T, scale_factor=1.5))
        self.play(everything.animate.set_opacity(1), run_time=0.5)
        # set_opacity(1) also filled hollow shapes; restore the outlines
        for gs in gpus_all:
            for g in gs:
                g[0].set_fill(opacity=0)
                g[1].set_fill(opacity=0.6)
                g[2].set_fill(opacity=0.6)
        for n in nodes:
            n[0].set_fill(opacity=0)
        for n in nvl:
            n[0].set_fill(opacity=0.35)
        for c in cpus:
            c[0].set_fill(opacity=0.25)
        storage[0].set_fill(opacity=0.25)
        lanes = VGroup(
            Text("compute", font_size=20, color=GREEN_T),
            Text("comms", font_size=20, color=RED_T)).arrange(DOWN, aligned_edge=LEFT,
                                                              buff=0.2)
        lanes.move_to([-6.0, -3.35, 0])
        comp = VGroup(*[Rectangle(width=1.4, height=0.25, fill_color=GREEN_T,
                                  fill_opacity=0.8, stroke_width=0) for _ in range(3)])
        comp.arrange(RIGHT, buff=0.1).next_to(lanes[0], RIGHT, buff=0.3)
        comm = VGroup(*[Rectangle(width=2.1, height=0.25, fill_color=RED_T,
                                  fill_opacity=0.8, stroke_width=0) for _ in range(3)])
        comm.arrange(RIGHT, buff=0.1).next_to(lanes[1], RIGHT, buff=0.3).align_to(comp, LEFT)
        comm.shift(RIGHT * 0.3)
        exposed = Text("comms > compute: GPUs wait", font_size=20, color=RED_T)
        exposed.next_to(comm, RIGHT, buff=0.3)
        self.say("w_h", [FadeOut(g1), Indicate(nic, color=RED_T),
                         *[Indicate(n[2], color=BLUE_T) for n in nvl]],
                 [FadeIn(lanes), FadeIn(comp, lag_ratio=0.3)],
                 FadeIn(comm, lag_ratio=0.3), FadeIn(exposed))
        st = Text("DataLoader: cheap with enough workers   ·   checkpoints: GPUs idle unless async",
                  font_size=20, color=ORANGE_T).to_edge(DOWN, buff=0.15)
        self.say("w_i", FadeOut(VGroup(lanes, comp, comm, exposed)),
                 Indicate(cpus, color=ORANGE_T),
                 [Indicate(storage, color=ORANGE_T), *[Indicate(f, color=ORANGE_T)
                                                        for f in feeds], FadeIn(st)])
        self.say("w_j", FadeOut(st), [everything.animate.set_opacity(0.15),
                                      hbm_arrows.animate.set_opacity(1)],
                 LaggedStart(*[Indicate(a, color=YELLOW_T, scale_factor=1.6)
                               for a in hbm_arrows], lag_ratio=0.05))

    # 6. fusion
    def fusion(self):
        head = Text("Crossing HBM less often", font_size=40).to_edge(UP, buff=0.3)
        g_hbm = Rectangle(width=6, height=0.6, fill_color=BLUE_T, fill_opacity=0.5,
                          stroke_color=BLUE_T).move_to(UP * 1.4)
        g_sm = Rectangle(width=6, height=0.6, fill_color=GREEN_T, fill_opacity=0.5,
                         stroke_color=GREEN_T).move_to(DOWN * 1.4)
        g_arr = DoubleArrow(g_hbm.get_bottom(), g_sm.get_top(), buff=0.1, color=YELLOW_T)
        g_txt = VGroup(Text("HBM", font_size=24).move_to(g_hbm),
                       Text("SMs", font_size=24).move_to(g_sm),
                       Text("bandwidth: fixed", font_size=24, color=GREY_B)
                       .next_to(g_arr, LEFT),
                       Text("crossings: up to you", font_size=24, color=YELLOW_T)
                       .next_to(g_arr, RIGHT))
        generic = VGroup(g_hbm, g_sm, g_arr, g_txt)
        self.say("fu_a", Write(head), [FadeIn(g_hbm), FadeIn(g_sm), FadeIn(g_txt[:2]),
                                       GrowFromCenter(g_arr)],
                 FadeIn(g_txt[2]), FadeIn(g_txt[3]), Indicate(g_arr, color=YELLOW_T))
        self.play(FadeOut(generic), run_time=0.6)

        def lane(cx, title):
            hbm = Rectangle(width=5.6, height=0.6, fill_color=BLUE_T, fill_opacity=0.5,
                            stroke_color=BLUE_T).move_to([cx, 1.9, 0])
            sm = Rectangle(width=5.6, height=0.6, fill_color=GREEN_T, fill_opacity=0.5,
                           stroke_color=GREEN_T).move_to([cx, -1.5, 0])
            ht = Text("HBM", font_size=22).move_to(hbm)
            st = Text("SMs (compute)", font_size=22).move_to(sm)
            t = Text(title, font_size=28, color=YELLOW_T).next_to(hbm, UP, buff=0.2)
            return VGroup(hbm, sm, ht, st, t)

        L, R = lane(-3.5, "eager: one kernel per op"), lane(3.5, "fused: one kernel")
        chain = VGroup(*[box(s, 1.3, 0.5, RED_T, size=20) for s in ("+ bias", "GeLU",
                                                                    "dropout")])
        chain.arrange(RIGHT, buff=0.35).move_to([0, 0.2, 0])
        carr = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.05,
                              stroke_width=3, color=GREY_B) for i in range(2)])
        self.say("fu_b", FadeIn(chain, lag_ratio=0.3), FadeIn(carr))

        def tensor(cx):
            return Rectangle(width=0.9, height=0.35, fill_color=YELLOW_T, fill_opacity=0.9,
                             stroke_width=0).move_to([cx, 1.9, 0])

        count = Integer(0, font_size=30, color=RED_T)
        cnt_l = Text("HBM round trips:", font_size=24, color=RED_T)
        cnt = VGroup(cnt_l, count).arrange(RIGHT).move_to([-3.5, -2.6, 0])

        self.play(FadeOut(VGroup(chain, carr)), FadeIn(L), run_time=0.8)
        ops_l = [chain[i].copy().scale(0.8) for i in range(3)]
        xs = [-5.2, -3.5, -1.8]
        for o, x in zip(ops_l, xs):
            o.move_to([x, 0.2, 0])
        self.add(cnt)

        def trip(i):
            t = tensor(xs[i])
            down = t.animate.move_to([xs[i], -1.5, 0])
            return t, [FadeIn(t), FadeIn(ops_l[i])], down

        t0, a0, d0 = trip(0)
        self.say("fu_c", a0, d0, Indicate(ops_l[0], color=RED_T),
                 [t0.animate.move_to([xs[0], 1.9, 0]), count.animate.set_value(1)])
        t1, a1, d1 = trip(1)
        self.say("fu_d", a1, d1, [t1.animate.move_to([xs[1], 1.9, 0]),
                                  count.animate.set_value(2)])
        t2, a2, d2 = trip(2)
        self.say("fu_e", a2, d2, [t2.animate.move_to([xs[2], 1.9, 0]),
                                  count.animate.set_value(3)],
                 Circumscribe(cnt, color=RED_T))

        fused = VGroup(*[chain[i].copy().scale(0.7) for i in range(3)]).arrange(RIGHT,
                                                                             buff=0.08)
        fbox = SurroundingRectangle(fused, color=YELLOW_T, buff=0.1)
        fk = VGroup(fused, fbox).move_to([3.5, 0.2, 0])
        tf = tensor(3.5)
        count2 = Integer(0, font_size=30, color=GREEN_T)
        cnt2 = VGroup(Text("HBM round trips:", font_size=24, color=GREEN_T), count2)
        cnt2.arrange(RIGHT).move_to([3.5, -2.6, 0])
        self.say("fu_f", [FadeIn(R), FadeIn(fk), FadeIn(tf), FadeIn(cnt2)],
                 tf.animate.move_to([3.5, -1.5, 0]),
                 LaggedStart(*[Indicate(f, color=YELLOW_T) for f in fused], lag_ratio=0.3),
                 [tf.animate.move_to([3.5, 1.9, 0]), count2.animate.set_value(1)])
        tools = Text("flash attention (by hand)   ·   torch.compile (automatic)",
                     font_size=26, color=YELLOW_T).to_edge(DOWN, buff=0.2)
        self.say("fu_g", Write(tools), Circumscribe(fk, color=YELLOW_T),
                 Indicate(tools, color=YELLOW_T, scale_factor=1.05))
        mm = Text("matmuls: already near peak in vendor GEMM libraries", font_size=26,
                  color=BLUE_T).to_edge(DOWN, buff=0.2)
        mm2 = Text("little left for a compiler to win there", font_size=22, color=GREY_B)
        mm2.next_to(mm, UP, buff=0.12)
        self.say("fu_h", [FadeOut(tools), FadeIn(mm)], Indicate(mm, color=BLUE_T),
                 FadeIn(mm2), Circumscribe(VGroup(cnt, cnt2), color=GREY_B))

    # 7. payoff: the two compile tables
    def payoff(self):
        head = Text("Same tool, different bottleneck", font_size=40).to_edge(UP, buff=0.3)
        sub = Text("forward+backward step, B200, bf16, batch 1, speedup vs eager",
                   font_size=22, color=GREY_B).next_to(head, DOWN, buff=0.12)
        self.say("pay_a", Write(head), FadeIn(sub))
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 8, 1], x_length=10, y_length=4.4,
                  tips=False, x_axis_config={"stroke_width": 0},
                  y_axis_config={"include_numbers": True, "font_size": 22,
                                 "numbers_to_include": [0, 2, 4, 6]})
        ax.shift(DOWN * 0.55)
        one = DashedLine(ax.c2p(0, 1), ax.c2p(8, 1), color=WHITE, stroke_width=2)
        one_l = Text("eager = 1x", font_size=16, color=GREY_B).next_to(ax.c2p(4.3, 1), UP,
                                                                       buff=0.05)
        modes = ["compile()", "reduce-\noverhead", "max-\nautotune"]
        cols = [BLUE_T, GREEN_T, YELLOW_T]

        def group(x0, vals):
            g = VGroup()
            for i, (v, c) in enumerate(zip(vals, cols)):
                x = x0 + i * 1.1
                bar = Rectangle(width=ax.x_length / 8 * 0.95, height=ax.y_length / 8 * v,
                                fill_color=c, fill_opacity=0.85, stroke_width=0)
                bar.move_to(ax.c2p(x, 0), aligned_edge=DOWN)
                val = Text(f"{v}x", font_size=22, color=c).next_to(bar, UP, buff=0.08)
                lab = Text(modes[i], font_size=16, color=GREY_B).next_to(ax.c2p(x, 0), DOWN,
                                                                         buff=0.12)
                g.add(VGroup(bar, val, lab))
            return g

        a = group(1, [3.9, 6.0, 6.1])
        b = group(5.4, [0.94, 0.94, 0.93])
        an = Text("Llama-3.2-1B, seq 512", font_size=24).next_to(a, DOWN, buff=0.15)
        bn = Text("Llama-3.1-8B, seq 8192", font_size=24).next_to(b, DOWN, buff=0.15)
        an.set_y(-3.65)
        bn.set_y(-3.65)
        why_a = Text("ms-scale steps:\nlaunch + bandwidth bound", font_size=20,
                     color=GREEN_T).move_to(ax.c2p(2.1, 7.4))
        self.say("pay_b", [Create(ax), Create(one), FadeIn(one_l), FadeIn(an)],
                 FadeIn(why_a), Indicate(an, color=GREEN_T), Circumscribe(why_a, color=GREEN_T))
        self.say("pay_c", [GrowFromEdge(a[0][0], DOWN), FadeIn(a[0][1]), FadeIn(a[0][2])])
        self.say("pay_d", [GrowFromEdge(a[1][0], DOWN), FadeIn(a[1][1]), FadeIn(a[1][2])],
                 [GrowFromEdge(a[2][0], DOWN), FadeIn(a[2][1]), FadeIn(a[2][2])])
        why_b = Text("100s-of-ms steps:\nmatmuls at cuBLAS peak", font_size=20,
                     color=RED_T).move_to(ax.c2p(6.5, 4.0))
        self.say("pay_e", FadeIn(bn), FadeIn(why_b), Indicate(bn, color=RED_T),
                 Circumscribe(why_b, color=RED_T))
        self.say("pay_f", [GrowFromEdge(g[0], DOWN) for g in b] +
                 [FadeIn(g[1]) for g in b] + [FadeIn(g[2]) for g in b],
                 Indicate(VGroup(*[g[1] for g in b]), color=RED_T),
                 Circumscribe(VGroup(*[g[0] for g in b]), color=RED_T))
        cold = VGroup(Text("first compiled step: 47s (8B), 22s (1B)", font_size=20),
                      Text("8B: never pays back", font_size=20, color=RED_T),
                      Text("1B: breaks even in ~600 steps", font_size=20, color=GREEN_T))
        cold.arrange(DOWN, buff=0.12).move_to(ax.c2p(6.5, 5.6))
        self.say("pay_g", [FadeOut(why_b), FadeIn(cold[0])], FadeIn(cold[1]),
                 Indicate(VGroup(*[g[0] for g in b]), color=RED_T), FadeIn(cold[2]),
                 Indicate(VGroup(*[g[0] for g in a]), color=GREEN_T))
        moral = Text("speedup tracks how bandwidth- or launch-bound the step is",
                     font_size=24, color=YELLOW_T).move_to(sub)
        self.say("pay_h", FadeOut(sub), FadeIn(moral), Circumscribe(moral, color=YELLOW_T),
                 [Indicate(VGroup(*[g[0] for g in a]), color=GREEN_T),
                  Indicate(VGroup(*[g[0] for g in b]), color=RED_T)])

    # 8. coda
    def coda(self):
        q = Text("Research the whole machine,\nnot just its engine.", font_size=44)
        self.say("c_a", Write(q), Circumscribe(q, color=YELLOW_T))
        new = box("next-gen GPU", 2.6, 1.6, GREEN_T, size=24).move_to([-4.6, -0.8, 0])
        olds = VGroup(*[box("older GPU", 1.7, 1.1, BLUE_T, size=20) for _ in range(3)])
        olds.arrange(RIGHT, buff=0.25).move_to([2.6, -0.8, 0])
        eq = Text("same cost", font_size=26, color=YELLOW_T).move_to([-1.9, -0.8, 0])
        fed = Text("fed as fast as it computes", font_size=24, color=GREY_B)
        fed.next_to(olds, DOWN, buff=0.3)
        self.say("c_b", q.animate.scale(0.7).to_edge(UP), FadeIn(olds[0]), FadeIn(fed))
        self.say("c_c", [FadeIn(new), FadeIn(eq)], FadeIn(olds[1:], lag_ratio=0.4),
                 Write(Text("→ maybe sooner, and cheaper", font_size=26, color=YELLOW_T)
                       .next_to(fed, DOWN)))
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        p = locomotive()
        loco = p["all"].scale(0.8).shift(UP * 0.2)
        stack_top = p["stack"].get_top()

        def puff():
            c = Circle(0.18, stroke_width=0, fill_color=GREY_C, fill_opacity=0.7)
            c.move_to(stack_top + UP * 0.1)
            self.add(c)
            return c.animate.shift(UP * 1.1 + LEFT * 1.6).scale(2.5).set_opacity(0)

        self.say("c_d", FadeIn(loco),
                 [Flash(p["firebox"], color=ORANGE_T),
                  LaggedStart(*[puff() for _ in range(5)], lag_ratio=0.3)],
                 Write(Text("Feeding the furnace", font_size=40).to_edge(DOWN, buff=0.6)))


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
