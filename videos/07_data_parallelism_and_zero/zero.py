"""Video 7: Data parallelism and ZeRO.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python zero.py
2. Render: manim -qh zero.py Zero

Every number comes from training/model-parallelism/README.md (Data Parallelism, ZeRO,
ZeRO with multiple replicas, Parallelism network collectives, Inter-node speed
requirements to use ZeRO), network/README.md (Single node training, Comms and compute
overlap) and training/performance/README.md (Anatomy of Model's Memory Usage, Gradient
Accumulation). The per-stage memory picture is schematic: the repo has no per-stage
formulas (those are in the ZeRO paper, arXiv 1910.02054).
"""
import wave
from pathlib import Path

from manim import *
from manim.animation.animation import prepare_animation

BLUE_T = "#58C4DD"
YELLOW_T = "#FFFF00"
RED_T = "#FC6255"
GREEN_T = "#83C167"
ORANGE_T = "#FF862F"
PURPLE_T = "#9A72AC"

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "hook_a": "Here's a model being trained on four GPUs with plain data parallelism.",
    "hook_b": "Every GPU holds its own copy of the weights, the gradients, and the Adam "
              "optimizer states, which in mixed precision comes to 18 bytes for every "
              "parameter.",
    "hook_c": "But if you look across the row, all four copies are exactly the same, so "
              "we're paying for those 18 bytes four times over.",
    "hook_d": "ZeRO is what you get when you stop doing that, and the interesting part is "
              "what it costs you on the network instead.",

    "ddp_a": "Before we get rid of the copies, it's worth being clear about why plain D D P "
             "keeps them.",
    "ddp_b": "Each GPU gets a different slice of the batch, and runs forward and backward "
             "on it with its own full copy of the model,",
    "ddp_c": "so each one ends up with its own gradients, and they disagree, because each "
             "GPU saw different data.",
    "ddp_d": "Before anyone can take an optimizer step, the gradients get averaged with an "
             "all-reduce, and everybody walks away with the same result.",
    "ddp_e": "An all-reduce is a reduce-scatter followed by an all-gather, so every GPU "
             "ends up sending about twice the size of the gradients.",
    "ddp_f": "To be exact, it's two times n minus one, over n, which is 1.75 for eight "
             "GPUs, and already very close to 2 by 64.",

    "ov_a": "That would be a lot of waiting if it all happened at the end of the step, "
            "but it doesn't have to.",
    "ov_b": "Backward runs from the last layer to the first, and as soon as the last "
            "layer has its gradients, they can go out on the network,",
    "ov_c": "while the layer before it is still busy computing its own backward.",
    "ov_d": "As long as the communication keeps up with the compute, it hides almost "
            "completely behind it.",
    "ov_e": "But if the comms take longer than the compute, the GPU has to sit idle "
            "waiting for data, and the part that doesn't overlap is called exposed "
            "communication.",

    "bp_a": "Now, back to those identical copies.",
    "bp_b": "The repo this series is built on describes ZeRO with a backpacking trip. "
            "Imagine three people hiking together.",
    "bp_c": "The D D P way is for each of them to carry their own tent, their own stove, "
            "and their own axe.",
    "bp_d": "The ZeRO way is for one person to carry the tent, one the stove, and one "
            "the axe,",
    "bp_e": "and each night they share what they have with the others,",
    "bp_f": "and in the morning everyone packs up just their own piece again.",

    "toy_a": "Let's make that concrete with a tiny model, with three layers, A, B and C, "
             "and just three parameters in each.",
    "toy_b": "With three GPUs, ZeRO gives each one a single row. GPU zero holds a-zero, "
             "b-zero and c-zero, and the other two GPUs hold the rest.",
    "toy_c": "Each GPU still gets its own slice of the batch, just like in D D P, and the "
             "inputs have no idea anything is different.",
    "toy_d": "When the inputs reach layer A, GPU zero only has a-zero, but to compute "
             "anything it needs all three pieces.",
    "toy_e": "So it gets a-one from GPU one and a-two from GPU two, while the other GPUs "
             "do the same thing at the same time. That's an all-gather.",
    "toy_f": "Now every GPU has the full layer A, and runs its forward,",
    "toy_g": "and as soon as that's done, it throws away the pieces it borrowed, and "
             "keeps only its own.",
    "toy_h": "Then the same thing happens for layer B, then layer C,",
    "toy_i": "and then backward, from C back to A.",
    "toy_j": "In practice the next layer's pieces are prefetched while the current one is "
             "computing, so the gathering mostly hides behind compute, the same overlap "
             "trick as before.",

    "stg_a": "ZeRO comes in three stages, depending on how much of those 18 bytes you "
             "shard.",
    "stg_b": "Stage one shards the optimizer states, which is the biggest piece,",
    "stg_c": "stage two shards the gradients as well,",
    "stg_d": "and stage three shards the parameters themselves, which is the version we "
             "just walked through.",
    "stg_e": "I'm keeping this picture schematic on purpose. The ZeRO paper has the exact "
             "memory formulas for each stage, which go beyond what the repo covers.",

    "cm_a": "So what does each stage cost on the network?",
    "cm_b": "D D P does a single all-reduce, which we said is about two times the "
            "parameters.",
    "cm_c": "Stages one and two split that same work into a reduce-scatter of the "
            "gradients, so each GPU only gets the part it owns, and then an all-gather of "
            "the updated parameters.",
    "cm_d": "That adds up to the same two times, so the memory you save in those stages "
            "doesn't cost any extra traffic.",
    "cm_e": "Stage three has to gather the weights twice, once before forward and once "
            "before backward, plus the reduce-scatter, so it moves three times the "
            "parameters, one and a half times as much as D D P.",
    "cm_f": "For a 10 billion parameter model in b f sixteen, that's 40 gigabytes per step "
            "for D D P, and 60 for ZeRO-3.",

    "nw_a": "Whether that extra traffic hurts depends on whether the network can keep up "
            "with the compute.",
    "nw_b": "The comms time is roughly that multiplier, times the bytes per parameter, "
            "times the parameter count, divided by the network bandwidth.",
    "nw_c": "The compute time is the number of passes, times two, times the parameters, "
            "times the tokens in a batch, divided by what all the GPUs together can do.",
    "nw_d1": "Let's plug in a real run. IDEFICS 80 B was trained on 512 A one hundreds,",
    "nw_d2": "over a 340 gigabit E F A network, which works out to 42.5 gigabytes per "
             "second.",
    "nw_e": "With ZeRO-3, the comms come out to three times two times 80, over 42.5, "
            "which is about 11 seconds.",
    "nw_f1": "For compute, say the GPUs reach about 250 teraflops when they're not "
             "waiting, which is roughly 80 percent of the A one hundred's peak.",
    "nw_f2": "With activation recompute, that's four passes, and with a batch of 3584 "
             "sequences of 1024 tokens, it comes out to about 18 seconds.",
    "nw_g": "So the comms are shorter than the compute, and on paper they could mostly "
            "hide behind it.",
    "nw_h": "But the measured step took about 49 seconds, and the run got 90 teraflops "
            "per GPU, while Megatron style T P plus P P plus D P was getting more than 150.",
    "nw_i": "The estimate is in the right ballpark, but the gap means more bottlenecks "
            "were hiding in there, and the source is upfront that it needs more "
            "investigation.",
    "nw_j": "Now imagine the network were five times slower. The comms jump to 56 "
            "seconds, three times the compute, and the GPUs spend most of the step "
            "waiting.",
    "nw_k": "And five times faster gets you 2 seconds, which basically disappears behind "
            "the compute.",

    "v_a": "The DeepSpeed team saw this directly, training a 176 billion parameter model "
           "on 384 V one hundreds.",
    "v_b": "With 100 gigabit InfiniBand, they got under 20 teraflops per GPU,",
    "v_c": "with 200 to 400 gigabit, around 30 to 40,",
    "v_d": "and with 800 gigabit, more than 40. Those are the same GPUs, and the network "
           "alone made at least a two times difference.",
    "v_e": "And those were V one hundreds. An H one hundred is four to eight times faster "
           "at half precision, so its network has to be that much faster to keep up.",

    "g_a": "Gradient accumulation makes this worse, in a way you might not expect.",
    "g_b": "In D D P, accumulating over eight micro-batches means one all-reduce instead "
           "of eight, so the network overhead drops eight times.",
    "g_c": "ZeRO-1 keeps the full gradients on every GPU, so accumulation costs it no "
           "extra traffic either.",
    "g_d": "But in ZeRO-2 and ZeRO-3, the weights have to be gathered again for every "
           "micro-batch,",
    "g_e": "and since there's nowhere to keep the in-between gradients, they get reduced "
           "every time too. So the comms grow with the number of accumulation steps.",

    "sc_a": "Two more problems show up at scale.",
    "sc_b": "The first is that the shards get so small that with 1024 GPUs you might fit "
            "a micro-batch of 32 on each, which is a global batch of 32 thousand, "
            "probably far more than you want.",
    "sc_c": "The second is that ZeRO normally spreads one copy of the model across every "
            "GPU in the cluster,",
    "sc_d": "so every gather crosses the slow network between nodes, even though the GPUs "
            "inside a node are connected by something much faster.",
    "sc_e": "ZeRO plus plus, with its hierarchical partitioning, and F S D P's hybrid "
            "shard, fix that by sharding the model inside each node, and keeping a full "
            "replica per node.",
    "sc_f": "Now both all-gathers ride the fast links inside the node, and only the "
            "reduce-scatter of the gradients crosses the slow network.",
    "sc_g": "The price is memory, which goes up with the number of nodes. And it doesn't "
            "really fix the batch size, though with less free memory the micro-batch "
            "tends to shrink anyway.",

    "end_a": "So ZeRO really is just D D P that stops carrying the same tent four times,",
    "end_b": "and whether that's a good trade comes down to one comparison you can do on "
             "a napkin, the comms time against the compute time.",
    "end_c": "Next time, we'll stop reassembling the model on the fly, and actually cut "
             "it apart.",
}


def caption(text, size=28, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.5)


def head(text, size=40):
    return Text(text, font_size=size).to_edge(UP)


SEGS = [("weights", 6, BLUE_T), ("grads", 4, GREEN_T), ("Adam states", 8, ORANGE_T)]


def mem_bar(u=0.17, w=0.7, shard=(), n=4):
    """Stacked bytes-per-param bar, bottom to top: weights, grads, Adam states."""
    segs = VGroup()
    for i, (_, b, c) in enumerate(SEGS):
        h = u * b / (n if i in shard else 1)
        segs.add(Rectangle(width=w, height=h, stroke_width=1, stroke_color=BLACK,
                           fill_color=c, fill_opacity=0.9))
    segs.arrange(UP, buff=0)
    return segs


def gpu(label, w=1.5, h=4.0, size=22):
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=GREY_B)
    return VGroup(box, Text(label, font_size=size, color=GREY_B).next_to(box, UP, buff=0.1))


def lane_block(width, color, label=None, h=0.5, size=18):
    r = Rectangle(width=width, height=h, stroke_width=1, stroke_color=BLACK,
                  fill_color=color, fill_opacity=0.85)
    if label is None:
        return VGroup(r)
    return VGroup(r, Text(label, font_size=size, color=BLACK).move_to(r))


class Zero(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.ddp, self.overlap, self.backpack, self.toy,
                     self.stages, self.comms, self.network, self.v100, self.gas,
                     self.scale, self.outro):
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

    # 0. hook: four identical copies
    def hook(self):
        gpus = VGroup(*[gpu(f"GPU {i}") for i in range(4)]).arrange(RIGHT, buff=0.6)
        gpus.shift(LEFT * 1.6 + DOWN * 0.3)
        bars = VGroup(*[mem_bar().move_to(g[0]) for g in gpus])
        legend = VGroup()
        for name, b, c in reversed(SEGS):
            sw = Square(0.3, fill_color=c, fill_opacity=0.9, stroke_width=0)
            legend.add(VGroup(sw, Text(f"{name}  {b} B", font_size=24)).arrange(RIGHT))
        legend.add(Text("= 18 bytes / param", font_size=26, color=YELLOW_T))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(gpus, RIGHT, buff=0.7)
        brace = Brace(bars[0], LEFT, color=YELLOW_T)
        title = Text("Data parallelism and ZeRO", font_size=48).to_edge(UP)

        self.say("hook_a", Create(gpus), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                                                     lag_ratio=0.2))
        self.say("hook_b", GrowFromCenter(brace),
                 LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in legend], lag_ratio=0.3))
        self.say("hook_c", FadeOut(brace),
                 LaggedStart(*[Indicate(b, color=YELLOW_T, scale_factor=1.08) for b in bars],
                             lag_ratio=0.25),
                 [b.animate.set_opacity(0.35) for b in bars[1:]])
        self.say("hook_d", Write(title), bars[1:].animate.set_opacity(0.08),
                 Indicate(bars[0], color=YELLOW_T, scale_factor=1.05))

    # 1. DDP
    def ddp(self):
        h = head("DDP: replicate everything")
        gpus = VGroup(*[gpu(f"GPU {i}", w=1.6, h=1.6) for i in range(4)]).arrange(RIGHT, buff=1.2)
        gpus.shift(UP * 0.6)
        models = VGroup(*[Text("model", font_size=22).move_to(g[0]) for g in gpus])
        xs = VGroup(*[Text(f"x{i}", font_size=28, color=YELLOW_T).next_to(g, UP, buff=0.5)
                      for i, g in enumerate(gpus)])
        heights = [0.6, 1.0, 0.45, 0.8]
        cols = [BLUE_T, GREEN_T, ORANGE_T, PURPLE_T]
        grads = VGroup(*[Rectangle(width=0.5, height=hh, fill_color=c, fill_opacity=0.9,
                                   stroke_width=0).next_to(g[0], DOWN, buff=0.3)
                         .align_to(g[0].get_bottom() + DOWN * 1.5, DOWN)
                         for g, hh, c in zip(gpus, heights, cols)])
        glabels = VGroup(*[Text(f"g{i}", font_size=22).next_to(b, DOWN, buff=0.1)
                           for i, b in enumerate(grads)])
        avg_h = sum(heights) / 4
        avg = Rectangle(width=0.5, height=avg_h, fill_color=WHITE, fill_opacity=0.9,
                        stroke_width=0)
        avgs = VGroup(*[avg.copy().move_to(b).align_to(b, DOWN) for b in grads])
        center = gpus.get_center() + DOWN * 2.2

        self.say("ddp_a", Write(h), [Create(gpus), FadeIn(models)])
        arrows = VGroup(*[Arrow(x.get_bottom(), g.get_top(), buff=0.1, stroke_width=3,
                                color=YELLOW_T) for x, g in zip(xs, gpus)])
        self.say("ddp_b", LaggedStart(*[FadeIn(x, shift=DOWN * 0.3) for x in xs], lag_ratio=0.2),
                 LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                 LaggedStart(*[Indicate(m, color=WHITE) for m in models], lag_ratio=0.2))
        self.say("ddp_c", LaggedStart(*[GrowFromEdge(b, DOWN) for b in grads], lag_ratio=0.2),
                 FadeIn(glabels))
        ar = Text("all-reduce (average)", font_size=26, color=YELLOW_T).move_to(center + DOWN * 0.4)
        self.say("ddp_d", [g.animate.move_to(center).set_opacity(0.6) for g in grads] +
                 [FadeOut(glabels), Write(ar)],
                 [ReplacementTransform(grads, avgs), FadeOut(ar)])

        # all-reduce = reduce-scatter (each keeps one quarter) + all-gather (share quarters)
        quarters = VGroup()
        for b in avgs:
            q = VGroup(*[Rectangle(width=0.5, height=avg_h / 4, fill_color=c, fill_opacity=0.9,
                                   stroke_width=1, stroke_color=BLACK) for c in cols])
            quarters.add(q.arrange(UP, buff=0).move_to(b))
        eq = VGroup(Text("all-reduce", font_size=28), Text("=", font_size=28),
                    Text("reduce-scatter", font_size=28, color=ORANGE_T),
                    Text("+", font_size=28), Text("all-gather", font_size=28, color=BLUE_T))
        eq.arrange(RIGHT, buff=0.2)
        wire = Text("≈ 2 × params on the wire", font_size=28, color=YELLOW_T)
        VGroup(eq, wire).arrange(DOWN).to_edge(DOWN, buff=0.4)
        self.say("ddp_e", [FadeIn(quarters), FadeOut(avgs), FadeIn(eq[:2])],
                 [FadeIn(eq[2])] + [q[j].animate.set_opacity(0.12)
                                    for i, q in enumerate(quarters) for j in range(4) if j != i],
                 [FadeIn(eq[3:])] + [q[j].animate.set_opacity(0.9)
                                     for i, q in enumerate(quarters) for j in range(4) if j != i],
                 Write(wire))
        exact = Text("exactly 2(n-1)/n:   n=8 → 1.75     n=64 → 1.97", font_size=28,
                     color=YELLOW_T).move_to(wire)
        self.say("ddp_f", Transform(wire, exact),
                 LaggedStart(*[Indicate(g[0], color=YELLOW_T, scale_factor=1.05) for g in gpus],
                             lag_ratio=0.25),
                 Circumscribe(wire, color=YELLOW_T))

    # 2. comms/compute overlap and exposed comms
    def overlap(self):
        h = head("Overlap")
        x0 = -5.0
        lbl_c = Text("compute", font_size=22).move_to([x0 - 1.1, 1.6, 0])
        lbl_m = Text("comms", font_size=22).move_to([x0 - 1.1, 0.9, 0])
        cw, mw = 2.2, 1.6
        comp = VGroup(*[lane_block(cw, BLUE_T, f"bwd L{4 - i}") for i in range(4)])
        comp.arrange(RIGHT, buff=0).move_to([x0, 1.6, 0], aligned_edge=LEFT)
        comm = VGroup(*[lane_block(mw, ORANGE_T, f"grads L{4 - i}") for i in range(4)])
        for i, m in enumerate(comm):
            m.move_to([comp[i].get_right()[0], 0.9, 0], aligned_edge=LEFT)
        axis = Arrow([x0, 0.4, 0], [6.2, 0.4, 0], buff=0, stroke_width=2,
                     max_tip_length_to_length_ratio=0.02)
        tl = Text("time →", font_size=18, color=GREY_B).next_to(axis, DOWN, buff=0.05).align_to(axis, RIGHT)

        self.say("ov_a", Write(h), [FadeIn(lbl_c), FadeIn(lbl_m), GrowArrow(axis), FadeIn(tl)])
        self.say("ov_b", FadeIn(comp[0], shift=RIGHT * 0.2), FadeIn(comm[0], shift=RIGHT * 0.2))
        self.say("ov_c", FadeIn(comp[1], shift=RIGHT * 0.2),
                 LaggedStart(*[FadeIn(VGroup(comp[i], comm[i - 1] if i > 1 else VGroup()),
                                      shift=RIGHT * 0.2) for i in range(2, 4)], lag_ratio=0.5),
                 FadeIn(comm[3], shift=RIGHT * 0.2))
        ok = Text("comms ≤ compute: almost fully hidden", font_size=24, color=GREEN_T)
        ok.next_to(axis, DOWN, buff=0.45)
        self.say("ov_d", Write(ok), Indicate(comm[:3], color=GREEN_T, scale_factor=1.05))

        # iteration-level: comms longer than DL + compute
        y_c, y_m = -1.5, -2.2
        l2c = Text("compute", font_size=22).move_to([x0 - 1.1, y_c, 0])
        l2m = Text("comms", font_size=22).move_to([x0 - 1.1, y_m, 0])
        row_c, row_m, idles = VGroup(), VGroup(), VGroup()
        x = x0
        for it in range(2):
            dl = lane_block(0.5, GREY_B, "DL").move_to([x, y_c, 0], aligned_edge=LEFT)
            cp = lane_block(2.6, BLUE_T, "compute").next_to(dl, RIGHT, buff=0)
            idle = lane_block(1.4, RED_T, "idle").next_to(cp, RIGHT, buff=0)
            idle[0].set_fill(opacity=0.35)
            cm = lane_block(4.5, ORANGE_T, "comms").move_to([x, y_m, 0], aligned_edge=LEFT)
            row_c.add(dl, cp)
            idles.add(idle)
            row_m.add(cm)
            x += 4.5
        exp = Text("exposed communication", font_size=24, color=RED_T).next_to(idles, DOWN, buff=1.0)
        exp.set_x(0)
        self.say("ov_e", [FadeIn(l2c), FadeIn(l2m), FadeIn(row_c), FadeIn(row_m)],
                 FadeIn(idles), Write(exp))

    # 3. backpacking metaphor
    def backpack(self):
        h = head("Sharing the load")
        people = VGroup()
        for name in "ABC":
            body = VGroup(Circle(0.3, color=WHITE), Line(DOWN * 0.3, DOWN * 1.2),
                          Line(DOWN * 0.6, DOWN * 0.6 + LEFT * 0.4),
                          Line(DOWN * 0.6, DOWN * 0.6 + RIGHT * 0.4),
                          Line(DOWN * 1.2, DOWN * 1.7 + LEFT * 0.3),
                          Line(DOWN * 1.2, DOWN * 1.7 + RIGHT * 0.3))
            people.add(VGroup(body, Text(name, font_size=26).next_to(body, DOWN)))
        people.arrange(RIGHT, buff=3.0).shift(UP * 0.8 + LEFT * 0.6)

        def tent():
            return VGroup(Triangle(color=GREEN_T, fill_opacity=0.8).scale(0.3),
                          Text("tent", font_size=18))

        def stove():
            return VGroup(Square(0.45, color=ORANGE_T, fill_opacity=0.8),
                          Text("stove", font_size=18))

        def axe():
            return VGroup(VGroup(Line(DOWN * 0.3, UP * 0.3, color=GREY_B, stroke_width=6),
                                 Rectangle(width=0.25, height=0.18, color=BLUE_T,
                                           fill_opacity=0.9).shift(UP * 0.25 + RIGHT * 0.1)),
                          Text("axe", font_size=18))

        makers = [tent, stove, axe]

        def pack(i, j):
            it = makers[j]()
            it[1].next_to(it[0], DOWN, buff=0.05)
            return it.scale(0.9).next_to(people[i][0], RIGHT, buff=0.2).shift(
                UP * (0.6 - 0.75 * j))

        full = VGroup(*[VGroup(*[pack(i, j) for j in range(3)]) for i in range(3)])
        ddp = Text("DDP: everyone carries everything", font_size=28).to_edge(DOWN, buff=0.8)
        zero = Text("ZeRO: each carries one piece", font_size=28, color=YELLOW_T).move_to(ddp)
        night = Text("night: share", font_size=28, color=BLUE_T).move_to(ddp)
        morning = Text("morning: pack only your own piece", font_size=28,
                       color=YELLOW_T).move_to(ddp)

        mini = VGroup(*[mem_bar(u=0.12, w=0.5) for _ in range(4)]).arrange(RIGHT, buff=0.4)
        self.say("bp_a", Write(h), LaggedStart(*[FadeIn(m, shift=UP * 0.2) for m in mini],
                                               lag_ratio=0.2))
        self.say("bp_b", FadeOut(mini),
                 LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in people], lag_ratio=0.3))
        self.say("bp_c", LaggedStart(*[FadeIn(f) for f in full], lag_ratio=0.3), Write(ddp))
        own = VGroup(*[full[i][i] for i in range(3)])
        others = VGroup(*[full[i][j] for i in range(3) for j in range(3) if i != j])
        self.say("bp_d", [FadeOut(others), Transform(ddp, zero)],
                 Indicate(own, color=YELLOW_T))
        shared = VGroup()
        anims = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    c = own[j].copy()
                    shared.add(c)
                    anims.append(c.animate.move_to(pack(i, j)))
        self.say("bp_e", Transform(ddp, night),
                 LaggedStart(*anims, lag_ratio=0.1))
        self.say("bp_f", [FadeOut(shared, shift=UP * 0.3), Transform(ddp, morning)])

    # 4. toy example: 3 layers x 3 params on 3 GPUs
    def toy(self):
        h = head("A tiny model under ZeRO-3")

        def cell(t, c=WHITE):
            return Text(t, font_size=28, color=c)

        cols = [BLUE_T, GREEN_T, ORANGE_T]
        hdr = VGroup(*[cell(f"L{l}", GREY_B) for l in "abc"])
        grid = VGroup(*[VGroup(*[cell(f"{l}{i}", cols[k]) for k, l in enumerate("abc")])
                        for i in range(3)])
        table = VGroup(hdr, *grid)
        for row in table:
            row.arrange(RIGHT, buff=0.7)
        table.arrange(DOWN, buff=0.35)
        self.say("toy_a", Write(h), FadeIn(hdr), LaggedStart(*[FadeIn(r) for r in grid],
                                                             lag_ratio=0.3))

        boxes = VGroup(*[gpu(f"GPU {i}", w=3.0, h=1.4) for i in range(3)])
        boxes.arrange(RIGHT, buff=0.9).shift(UP * 0.5)
        rows_t = []
        for i in range(3):
            hd = VGroup(*[cell(f"L{l}", GREY_B).scale(0.8) for l in "abc"]).arrange(RIGHT, buff=0.55)
            r = grid[i].copy().arrange(RIGHT, buff=0.55)
            VGroup(hd, r).arrange(DOWN, buff=0.15).move_to(boxes[i][0])
            for k in range(3):
                r[k].set_x(hd[k].get_x())
            rows_t.append((hd, r))
        hdrs = VGroup(*[t[0] for t in rows_t])
        self.say("toy_b", [Create(boxes), FadeOut(hdr), FadeIn(hdrs)],
                 LaggedStart(*[grid[i].animate.become(rows_t[i][1]) for i in range(3)],
                             lag_ratio=0.4),
                 Circumscribe(grid[0], color=YELLOW_T),
                 Circumscribe(VGroup(grid[1], grid[2]), color=GREY_B))
        xs = VGroup(*[Text(f"x{i}", font_size=28, color=YELLOW_T).next_to(b, UP, buff=0.5)
                      for i, b in enumerate(boxes)])
        xarrows = VGroup(*[Arrow(x.get_bottom(), b.get_top(), buff=0.08, stroke_width=3,
                                 color=YELLOW_T) for x, b in zip(xs, boxes)])
        self.say("toy_c", LaggedStart(*[FadeIn(x, shift=DOWN * 0.3) for x in xs], lag_ratio=0.25),
                 LaggedStart(*[GrowArrow(a) for a in xarrows], lag_ratio=0.25))

        def gather(k):
            targets, anims = VGroup(), []
            for i in range(3):
                tg = VGroup(*[cell(f"{'abc'[k]}{j}", cols[k]) for j in range(3)])
                tg.arrange(RIGHT, buff=0.35).next_to(boxes[i][0], DOWN, buff=0.4)
                movers = VGroup()
                for j in range(3):
                    m = grid[j][k].copy()
                    movers.add(m)
                    anims.append(m.animate.move_to(tg[j]))
                targets.add(movers)
            return targets, anims

        need = VGroup(*[Text(f"a{j}", font_size=28, color=BLUE_T if j == 0 else RED_T)
                        for j in range(3)]).arrange(RIGHT, buff=0.35)
        need.next_to(boxes[0][0], DOWN, buff=0.4)
        need[1:].set_opacity(0.35)
        self.say("toy_d", Circumscribe(grid[0][0], color=YELLOW_T, run_time=1.5),
                 FadeIn(need), Indicate(need[1:], color=RED_T))
        ga, anims = gather(0)
        ag = Text("all-gather", font_size=28, color=YELLOW_T).to_edge(DOWN, buff=1.2)
        self.say("toy_e", [FadeOut(need), LaggedStart(*anims, lag_ratio=0.08)], Write(ag))
        fwd = Text("forward La", font_size=28, color=BLUE_T).move_to(ag)
        self.say("toy_f", [Transform(ag, fwd)] +
                 [Indicate(g, color=WHITE, scale_factor=1.15) for g in ga],
                 [Circumscribe(g, color=BLUE_T) for g in ga])
        keep = [ga[i][i] for i in range(3)]
        drop = VGroup(*[ga[i][j] for i in range(3) for j in range(3) if i != j])
        self.say("toy_g", [FadeOut(drop, shift=DOWN * 0.3), FadeOut(ag)],
                 [Indicate(k, color=YELLOW_T, scale_factor=1.3, run_time=1.5) for k in keep],
                 FadeOut(VGroup(*keep)))

        gb, anims_b = gather(1)
        gc, anims_c = gather(2)
        step = Text("forward Lb", font_size=28, color=GREEN_T).to_edge(DOWN, buff=1.2)
        step_c = Text("forward Lc", font_size=28, color=ORANGE_T).move_to(step)
        self.say("toy_h", [LaggedStart(*anims_b, lag_ratio=0.04), FadeIn(step)],
                 FadeOut(gb),
                 [LaggedStart(*anims_c, lag_ratio=0.04), Transform(step, step_c)],
                 FadeOut(gc))
        bwd = Text("backward: Lc → Lb → La", font_size=28, color=PURPLE_T).move_to(step)
        hl = [SurroundingRectangle(VGroup(*[grid[i][k] for i in range(3)]), color=PURPLE_T,
                                   buff=0.12) for k in (2, 1, 0)]
        self.say("toy_i", Transform(step, bwd),
                 *[Succession(Create(r), FadeOut(r)) for r in hl])
        lc = Text("compute", font_size=20).move_to([-6.4, -2.0, 0])
        lg = Text("gather", font_size=20).move_to([-6.4, -2.65, 0])
        cl = VGroup(*[lane_block(2.0, c, f"fwd L{l}", h=0.45) for l, c in
                      zip("abc", cols)]).arrange(RIGHT, buff=0).move_to([-4.0, -2.0, 0],
                                                                        aligned_edge=LEFT)
        gl = VGroup(*[lane_block(1.4, c, f"gather L{l}", h=0.45, size=16) for l, c in
                      zip("abc", cols)])
        gl[0].move_to([cl[0].get_left()[0] - 1.4, -2.65, 0], aligned_edge=LEFT)
        for i in (1, 2):
            gl[i].move_to([cl[i - 1].get_left()[0], -2.65, 0], aligned_edge=LEFT)
        pf = Text("prefetch: gather the next layer while computing this one", font_size=24,
                  color=YELLOW_T).to_edge(DOWN, buff=0.3)
        self.say("toy_j", [FadeOut(step), FadeIn(lc), FadeIn(lg)],
                 LaggedStart(*[FadeIn(VGroup(g, c), shift=RIGHT * 0.2) for g, c in zip(gl, cl)],
                             lag_ratio=0.5),
                 Write(pf), Indicate(gl[1:], color=YELLOW_T, scale_factor=1.05))

    # 5. stages, as the memory bar sliced
    def stages(self):
        h = head("ZeRO stages: what gets sharded")
        u = 0.2
        gpus = VGroup(*[gpu(f"GPU {i}", w=1.3, h=4.2) for i in range(4)]).arrange(RIGHT, buff=0.5)
        gpus.shift(LEFT * 2.3 + DOWN * 0.4)
        base = gpus[0][0].get_bottom()[1] + 0.25

        def bars(shard):
            g = VGroup()
            for b in gpus:
                m = mem_bar(u=u, w=0.7, shard=shard)
                m.move_to([b[0].get_x(), 0, 0]).align_to([0, base, 0], DOWN)
                g.add(m)
            return g

        cur = bars(())
        ghosts = VGroup(*[DashedVMobject(Rectangle(width=0.7, height=u * 18).move_to(b),
                                         num_dashes=30).set_stroke(GREY_B, 1.5)
                          for b in cur])
        legend = VGroup()
        for name, b, c in reversed(SEGS):
            sw = Square(0.3, fill_color=c, fill_opacity=0.9, stroke_width=0)
            legend.add(VGroup(sw, Text(name, font_size=24)).arrange(RIGHT))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(gpus, RIGHT, buff=0.8).shift(UP * 1.2)
        tags = VGroup(
            Text("DDP: nothing sharded", font_size=26),
            Text("ZeRO-1: Adam states ÷ 4", font_size=26, color=ORANGE_T),
            Text("ZeRO-2: + grads ÷ 4", font_size=26, color=GREEN_T),
            Text("ZeRO-3: + weights ÷ 4", font_size=26, color=BLUE_T),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(legend, DOWN, buff=0.6, aligned_edge=LEFT)

        self.say("stg_a", Write(h), [Create(gpus), FadeIn(cur), FadeIn(legend)],
                 [FadeIn(ghosts), FadeIn(tags[0])])
        self.add(ghosts)
        for key, shard, i in (("stg_b", (2,), 1), ("stg_c", (2, 1), 2),
                              ("stg_d", (2, 1, 0), 3)):
            nxt = bars(shard)
            self.say(key, Transform(cur, nxt), FadeIn(tags[i], shift=RIGHT * 0.2))
        self.say("stg_e", Write(caption("schematic: exact per-stage formulas are in the "
                                        "ZeRO paper (arXiv 1910.02054)", 22, GREY_B)),
                 LaggedStart(*[Circumscribe(VGroup(g, c), color=GREY_B)
                               for g, c in zip(ghosts, cur)], lag_ratio=0.3, run_time=2))

    # 6. comms per stage
    def comms(self):
        h = head("What each stage costs on the network")
        rows = [("DDP", "1 all-reduce", "2×"),
                ("ZeRO-1/2", "reduce-scatter grads + all-gather params", "2×"),
                ("ZeRO-3", "2 all-gather weights (fwd, bwd) + reduce-scatter grads", "3×")]
        tbl = Table([list(r) for r in rows],
                    col_labels=[Text(s) for s in ("", "collectives per step", "× params")],
                    element_to_mobject=lambda s: Text(s),
                    include_outer_lines=False, line_config={"stroke_width": 1})
        tbl.scale(0.42).next_to(h, DOWN, buff=0.4)
        trs = tbl.get_rows()

        # wire bytes per step, in units of the parameter size
        u, x0 = 2.0, -3.0
        ys = [-0.9, -1.75, -2.6]
        names = ["DDP", "ZeRO-1/2", "ZeRO-3"]
        labels = VGroup(*[Text(n, font_size=24).next_to([x0, y, 0], LEFT, buff=0.3)
                          for n, y in zip(names, ys)])

        def seg(n, color, label, y, start):
            r = lane_block(n * u, color, label, h=0.55, size=17)
            return r.move_to([x0 + start * u, y, 0], aligned_edge=LEFT)

        ddp = seg(2, PURPLE_T, "all-reduce", ys[0], 0)
        z12 = VGroup(seg(1, ORANGE_T, "reduce-scatter", ys[1], 0),
                     seg(1, BLUE_T, "all-gather", ys[1], 1))
        z3 = VGroup(seg(1, BLUE_T, "all-gather", ys[2], 0),
                    seg(1, BLUE_T, "all-gather", ys[2], 1),
                    seg(1, ORANGE_T, "reduce-scatter", ys[2], 2))
        ticks = VGroup(*[DashedLine([x0 + k * u, -0.5, 0], [x0 + k * u, -2.9, 0],
                                    stroke_width=1, color=GREY_B) for k in range(4)])
        tl = VGroup(*[Text(f"{k}×", font_size=20, color=GREY_B).move_to([x0 + k * u, -3.1, 0])
                      for k in range(4)])

        self.say("cm_a", Write(h), [FadeIn(trs[0]), Create(tbl.get_horizontal_lines())],
                 [Create(ticks), FadeIn(tl)])
        self.say("cm_b", [FadeIn(trs[1]), FadeIn(labels[0])], GrowFromEdge(ddp, LEFT))
        self.say("cm_c", [FadeIn(trs[2][:2]), FadeIn(labels[1])],
                 GrowFromEdge(z12[0], LEFT, run_time=1.5), GrowFromEdge(z12[1], LEFT, run_time=1.5))
        self.say("cm_d", FadeIn(trs[2][2]), Indicate(VGroup(ddp, z12), color=YELLOW_T,
                                                     scale_factor=1.03, run_time=1.5),
                 Circumscribe(VGroup(trs[1][2], trs[2][2]), color=YELLOW_T))
        self.say("cm_e", [FadeIn(trs[3]), FadeIn(labels[2])],
                 LaggedStart(*[GrowFromEdge(p, LEFT) for p in z3], lag_ratio=0.6),
                 trs[3][2].animate.set_color(RED_T))
        gb = VGroup(Text("40 GB", font_size=24, color=YELLOW_T).next_to(ddp, RIGHT),
                    Text("40 GB", font_size=24, color=YELLOW_T).next_to(z12, RIGHT),
                    Text("60 GB", font_size=24, color=RED_T).next_to(z3, RIGHT))
        ex = Text("for 10B params × 2 bytes (bf16)", font_size=22, color=YELLOW_T)
        ex.to_edge(DOWN, buff=0.2)
        self.say("cm_f", FadeIn(ex), LaggedStart(*[FadeIn(g, shift=LEFT * 0.2) for g in gb],
                                                 lag_ratio=0.4))

    # 7. will the network keep up: IDEFICS-80B
    def network(self):
        h = head("Will the network keep up?")
        f1 = Text("comms = mult × bytes × params_B / GBps", font_size=28, color=ORANGE_T)
        f2 = Text("compute = passes × 2 × params_B × seqlen × GBS / (GPUs × 1e3 × TFLOPS)",
                  font_size=24, color=BLUE_T)
        VGroup(f1, f2).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(h, DOWN, buff=0.4)

        s, x0 = 0.13, -3.0

        def bar(sec, color, label, y):
            r = Rectangle(width=sec * s, height=0.45, fill_color=color, fill_opacity=0.85,
                          stroke_width=0).move_to([x0, y, 0], aligned_edge=LEFT)
            l = Text(label, font_size=22).next_to([x0, y, 0], LEFT, buff=0.2)
            v = Text(f"{sec} s", font_size=22).next_to(r, RIGHT, buff=0.15)
            return VGroup(r, l, v)

        comp = bar(18, BLUE_T, "compute", -0.6)
        comm = bar(11, ORANGE_T, "ZeRO-3 comms", -1.2)
        meas = bar(49, RED_T, "measured step", -1.8)
        slow = bar(56, RED_T, "comms, 5× slower", -2.6)
        fast = bar(2, GREEN_T, "comms, 5× faster", -3.2)
        lanes = VGroup(*[Rectangle(width=56 * s, height=0.45, stroke_color=GREY_D,
                                   stroke_width=1).move_to([x0, y, 0], aligned_edge=LEFT)
                         for y in (-0.6, -1.2)])
        q = VGroup(*[Text("?", font_size=26, color=GREY_B).move_to(l) for l in lanes])
        ql = VGroup(comp[1].copy(), comm[1].copy())

        self.say("nw_a", Write(h), [Create(lanes), FadeIn(ql), FadeIn(q)])
        self.say("nw_b", Write(f1), Circumscribe(f1, color=ORANGE_T, run_time=2))
        self.say("nw_c", Write(f2), Circumscribe(f2, color=BLUE_T, run_time=2))

        run = VGroup(Text("IDEFICS-80B   512 × A100", font_size=26, color=YELLOW_T),
                     Text("340 Gbps EFA = 42.5 GBps", font_size=26, color=YELLOW_T))
        run.arrange(RIGHT, buff=0.8).next_to(f2, DOWN, buff=0.35)
        chips = VGroup(*[Square(0.09, stroke_width=0, fill_color=GREEN_T, fill_opacity=0.8)
                         for _ in range(512)]).arrange_in_grid(8, 64, buff=0.03)
        chips.move_to([0, -1.9, 0])
        self.say("nw_d1", [FadeOut(lanes), FadeOut(q), FadeOut(ql), Write(run[0])],
                 LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.004, run_time=2))
        wires = VGroup(*[Line(chips.get_left() + RIGHT * 0.1 + UP * dy,
                              chips.get_right() + LEFT * 0.1 + UP * dy, color=ORANGE_T,
                              stroke_width=2) for dy in (-0.45, 0.45)])
        self.say("nw_d2", Write(run[1]), [Create(w) for w in wires],
                 Circumscribe(run[1], color=YELLOW_T))
        c1 = Text("comms = 3 × 2 × 80 / 42.5 ≈ 11 s", font_size=28, color=ORANGE_T)
        c1.move_to(f1, aligned_edge=LEFT)
        self.say("nw_e", [FadeOut(chips), FadeOut(wires), Transform(f1, c1)],
                 GrowFromEdge(comm, LEFT))
        tf = Text("250 TFLOPS ≈ 80% of 312 peak", font_size=26, color=BLUE_T)
        tf.move_to(run[1])
        self.say("nw_f1", [FadeOut(run[1]), FadeIn(tf)], Circumscribe(tf, color=BLUE_T, run_time=2))
        c2 = Text("compute = 4 × 2 × 80 × 1024 × 3584 / (512 × 1e3 × 250) ≈ 18 s",
                  font_size=24, color=BLUE_T).move_to(f2, aligned_edge=LEFT)
        self.say("nw_f2", Transform(f2, c2), Circumscribe(f2, color=BLUE_T, run_time=1.5),
                 GrowFromEdge(comp, LEFT, run_time=2))
        self.say("nw_g", Indicate(comm, color=WHITE, scale_factor=1.05),
                 Circumscribe(VGroup(comp, comm), color=GREEN_T))
        tf2 = Text("90 TFLOPS on ZeRO-3   vs   150+ on Megatron TP+PP+DP", font_size=24,
                   color=RED_T).move_to(VGroup(run[0], tf))
        self.say("nw_h", GrowFromEdge(meas, LEFT), [FadeOut(run[0]), FadeOut(tf), FadeIn(tf2)],
                 Circumscribe(tf2, color=RED_T, run_time=2))
        gap = BraceBetweenPoints(meas[0].get_corner(DR) + LEFT * 20 * s, meas[0].get_corner(DR),
                                 DOWN, color=YELLOW_T)
        gl = Text("≈ 20 s beyond comms + compute", font_size=20, color=YELLOW_T)
        gl.next_to(gap, DOWN, buff=0.05)
        self.say("nw_i", GrowFromCenter(gap), FadeIn(gl), Indicate(meas, color=YELLOW_T,
                                                                   scale_factor=1.03))
        self.say("nw_j", [FadeOut(gap), FadeOut(gl)], GrowFromEdge(slow, LEFT),
                 Indicate(slow, color=RED_T, scale_factor=1.03))
        self.say("nw_k", GrowFromEdge(fast, LEFT), Indicate(comp, color=WHITE, scale_factor=1.05))

    # 8. DeepSpeed 176B on V100
    def v100(self):
        h = head("176B on 384 V100s (DeepSpeed)")
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 50, 10], x_length=8, y_length=4.2, tips=False,
                  axis_config={"include_numbers": False, "include_ticks": False},
                  y_axis_config={"include_ticks": True})
        ax.shift(DOWN * 0.1)
        ynums = VGroup(*[Text(str(v), font_size=20).next_to(ax.c2p(0, v), LEFT, buff=0.15)
                         for v in range(0, 51, 10)])
        yl = Text("TFLOPS / GPU", font_size=22).next_to(ax.y_axis, UP)
        specs = [("100 Gbps", 0, 20, "< 20", RED_T), ("200-400 Gbps", 30, 40, "30-40", ORANGE_T),
                 ("800 Gbps", 40, 46, "40+", GREEN_T)]
        groups = []
        for i, (name, lo, hi, txt, c) in enumerate(specs):
            x = i + 0.5
            w = ax.c2p(0.55, 0)[0] - ax.c2p(0, 0)[0]
            if lo:
                solid = Rectangle(width=w, height=ax.c2p(0, lo)[1] - ax.c2p(0, 0)[1],
                                  fill_color=c, fill_opacity=0.85, stroke_width=0)
                solid.move_to(ax.c2p(x, 0), aligned_edge=DOWN)
                rng = Rectangle(width=w, height=ax.c2p(0, hi)[1] - ax.c2p(0, lo)[1],
                                fill_color=c, fill_opacity=0.35, stroke_width=0)
                rng.next_to(solid, UP, buff=0)
                b = VGroup(solid, rng)
            else:
                b = VGroup(Rectangle(width=w, height=ax.c2p(0, hi)[1] - ax.c2p(0, 0)[1],
                                     fill_color=c, fill_opacity=0.35, stroke_width=0)
                           .move_to(ax.c2p(x, 0), aligned_edge=DOWN))
            lab = Text(name, font_size=22).next_to(ax.c2p(x, 0), DOWN)
            val = Text(txt, font_size=24).next_to(b, UP, buff=0.1)
            groups.append(VGroup(b, lab, val))
        self.say("v_a", Write(h), [Create(ax, run_time=2), FadeIn(ynums), FadeIn(yl)])
        for key, g in zip(("v_b", "v_c"), groups):
            self.say(key, [GrowFromEdge(g[0], DOWN), FadeIn(g[1])], FadeIn(g[2]))
        g = groups[2]
        two = DoubleArrow(ax.c2p(0.5, 20) + RIGHT * 0.9, ax.c2p(0.5, 40) + RIGHT * 0.9, buff=0,
                          color=YELLOW_T)
        twol = Text("≥ 2×", font_size=24, color=YELLOW_T).next_to(two, RIGHT, buff=0.1)
        self.say("v_d", [GrowFromEdge(g[0], DOWN), FadeIn(g[1])], FadeIn(g[2]),
                 [GrowFromCenter(two), FadeIn(twol)])
        note = caption("H100 is 4-8× a V100 at half precision → network must be 4-8× faster",
                       26, YELLOW_T)
        self.say("v_e", Write(note), Indicate(groups[0], color=RED_T),
                 Circumscribe(note, color=YELLOW_T))

    # 9. gradient accumulation
    def gas(self):
        h = head("Gradient accumulation (GAS = 4 here)")
        x0 = -3.5
        rows = [("DDP", "end"), ("ZeRO-1", "end"), ("ZeRO-2 / ZeRO-3", "each")]
        lanes = []
        for i, (name, mode) in enumerate(rows):
            y = 1.6 - 1.6 * i
            lab = Text(name, font_size=24).next_to([x0, y, 0], LEFT, buff=0.3)
            mbs = VGroup(*[lane_block(1.6, BLUE_T, f"mb {k + 1}") for k in range(4)])
            comms = VGroup()
            if mode == "end":
                mbs.arrange(RIGHT, buff=0).move_to([x0, y, 0], aligned_edge=LEFT)
                comms.add(lane_block(1.6, ORANGE_T, "comms").next_to(mbs, RIGHT, buff=0))
            else:
                for k, m in enumerate(mbs):
                    m.move_to([x0 + 2.2 * k, y, 0], aligned_edge=LEFT)
                    comms.add(lane_block(0.6, ORANGE_T).next_to(m, RIGHT, buff=0))
            lanes.append((lab, mbs, comms))
        empty = VGroup(*[DashedVMobject(Rectangle(width=8.6, height=0.5).move_to(
            [x0, 1.6 - 1.6 * i, 0], aligned_edge=LEFT), num_dashes=40).set_stroke(GREY_B, 1)
            for i in range(3)])
        self.say("g_a", Write(h), Create(empty, run_time=2))
        for key, (lab, mbs, comms) in zip(("g_b", "g_c"), lanes[:2]):
            self.say(key, [FadeIn(lab), FadeIn(mbs, shift=RIGHT * 0.2)], FadeIn(comms))
        lab, mbs, comms = lanes[2]
        self.say("g_d", [FadeIn(lab), FadeIn(mbs, shift=RIGHT * 0.2)],
                 LaggedStart(*[FadeIn(c) for c in comms], lag_ratio=0.3))
        note = caption("ZeRO-2/3 comms × GAS     ZeRO-1: no extra comms     DDP: ÷ GAS",
                       26, YELLOW_T)
        self.say("g_e", Indicate(comms, color=RED_T, run_time=2), Write(note))

    # 10. scale: batch size and hierarchical sharding
    def scale(self):
        h = head("At scale")
        gbs = Text("GBS = 1024 GPUs × MBS 32 = 32k", font_size=32, color=YELLOW_T)
        self.say("sc_a", Write(h))
        grid = VGroup(*[Square(0.1, stroke_width=0, fill_color=GREEN_T, fill_opacity=0.8)
                        for _ in range(1024)]).arrange_in_grid(16, 64, buff=0.04)
        grid.shift(UP * 0.6)
        gbs.next_to(grid, DOWN, buff=0.6)
        self.say("sc_b", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=0.002, run_time=2.5),
                 grid.animate.set_fill(YELLOW_T, 0.8), Write(gbs), Circumscribe(gbs, color=YELLOW_T))

        nodes = VGroup()
        for n in range(2):
            gs = VGroup(*[RoundedRectangle(corner_radius=0.08, width=1.0, height=1.0,
                                           stroke_color=GREY_B) for _ in range(4)])
            gs.arrange_in_grid(2, 2, buff=0.25)
            box = SurroundingRectangle(gs, color=WHITE, buff=0.3, corner_radius=0.1)
            nodes.add(VGroup(box, gs, Text(f"node {n}", font_size=22).next_to(box, UP, buff=0.1)))
        nodes.arrange(RIGHT, buff=2.6).shift(DOWN * 0.5)
        lbl1 = VGroup(*[Text(f"1/8", font_size=22, color=BLUE_T).move_to(g)
                        for nd in nodes for g in nd[1]])
        fast = VGroup(*[Text("fast links", font_size=20, color=GREEN_T).next_to(nd[0], DOWN)
                        for nd in nodes])
        slow = DoubleArrow(nodes[0][0].get_right(), nodes[1][0].get_right() * [0, 1, 1] +
                           nodes[1][0].get_left() * [1, 0, 0], buff=0.1, color=RED_T)
        slow_l = Text("slow", font_size=22, color=RED_T).next_to(slow, UP, buff=0.1)
        self.say("sc_c", [FadeOut(grid), gbs.animate.scale(0.7).next_to(h, DOWN)],
                 Create(nodes), FadeIn(lbl1))
        cross = VGroup(*[Line(nodes[0][1][i].get_center(), nodes[1][1][j].get_center(),
                              color=RED_T, stroke_width=1.5, stroke_opacity=0.6)
                         for i in range(4) for j in range(4)])
        self.say("sc_d", [GrowFromCenter(slow), FadeIn(slow_l), FadeIn(fast)],
                 LaggedStart(*[Create(c) for c in cross], lag_ratio=0.03))
        lbl2 = VGroup(*[Text("1/4", font_size=22, color=BLUE_T).move_to(g)
                        for nd in nodes for g in nd[1]])
        rep = VGroup(*[Text("full replica", font_size=20, color=YELLOW_T).next_to(nd[0], DOWN)
                       for nd in nodes])
        hy = Text("ZeRO++ hpZ  /  FSDP HYBRID_SHARD", font_size=28, color=YELLOW_T)
        hy.to_edge(DOWN, buff=0.4)
        self.say("sc_e", [FadeOut(cross), Write(hy)],
                 [Transform(lbl1, lbl2), Transform(fast, rep)],
                 [Circumscribe(nd[0], color=YELLOW_T, run_time=1.5) for nd in nodes])
        intra = VGroup()
        for nd in nodes:
            g = nd[1]
            for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
                intra.add(Line(g[a].get_center(), g[b].get_center(), color=GREEN_T,
                               stroke_width=4, buff=0.5))
        ag = Text("all-gathers", font_size=20, color=GREEN_T).next_to(nodes[0][0], LEFT, buff=0.2)
        rs = Text("reduce-scatter\nonly", font_size=20, color=RED_T).next_to(slow, DOWN, buff=0.15)
        self.say("sc_f", [Create(intra, run_time=2), FadeIn(ag)], Transform(slow_l, rs),
                 Indicate(slow, color=RED_T))
        mem = Text("cost:\nmemory\n× nodes", font_size=22, color=ORANGE_T)
        mem.next_to(nodes[1][0], RIGHT, buff=0.2)
        reps_bars = VGroup(*[mem_bar(u=0.1, w=0.35, shard=(0, 1, 2)).next_to(nd[0], LEFT if i == 0
                                                                              else RIGHT, buff=0.2)
                             for i, nd in enumerate(nodes)])
        mem.next_to(reps_bars[1], RIGHT, buff=0.15)
        self.say("sc_g", [ag.animate.next_to(reps_bars[0], LEFT, buff=0.15),
                          GrowFromEdge(reps_bars, DOWN, run_time=2)],
                 FadeIn(mem, shift=LEFT * 0.2), Indicate(gbs, color=GREY_B, run_time=2))

    # 11. outro
    def outro(self):
        bars = VGroup(*[mem_bar(u=0.2, w=0.8) for _ in range(4)]).arrange(RIGHT, buff=0.5).shift(UP * 0.8)
        sharded = VGroup(*[mem_bar(u=0.2, w=0.8, shard=(0, 1, 2)) for _ in range(4)]).arrange(RIGHT, buff=0.5)
        for s, b in zip(sharded, bars):
            s.move_to(b).align_to(b, DOWN)
        self.say("end_a", LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.15),
                 Transform(bars, sharded))
        ineq = Text("comms_time  ≤  compute_time", font_size=44, color=YELLOW_T)
        ineq.move_to(DOWN * 2.0)
        self.say("end_b", Write(ineq), Circumscribe(ineq, color=YELLOW_T, run_time=1.5))
        nxt = Text("Next: cutting the model", font_size=32, color=GREY_B).to_edge(DOWN, buff=0.6)
        mat = Rectangle(width=3, height=2, fill_color=BLUE_T, fill_opacity=0.5,
                        stroke_color=WHITE).move_to(UP * 0.6)
        cols4 = VGroup(*[Rectangle(width=0.75, height=2, fill_color=BLUE_T, fill_opacity=0.5,
                                   stroke_color=WHITE) for _ in range(4)]).arrange(RIGHT, buff=0)
        cols4.move_to(mat)
        self.say("end_c", [FadeOut(bars), FadeOut(ineq), FadeIn(nxt, shift=UP * 0.2),
                           FadeIn(mat)],
                 [FadeIn(cols4), FadeOut(mat)],
                 cols4.animate.arrange(RIGHT, buff=0.4).move_to(mat))


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
