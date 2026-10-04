"""Video 6: Collectives.

1. Voiceover (Kokoro TTS, cached in vo/, regenerated when a line or VOICE changes):
   ~/.local/share/uv/tools/manim/bin/python collectives.py
2. Render: manim -qh collectives.py Collectives

Every number comes from network/comms.md, network/README.md (Glossary algbw/busbw,
SHARP) and training/model-parallelism/README.md (Parallelism network collectives).
No LaTeX needed: all labels are Text.
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
ORANGE_T = "#FF862F"
TEAL_T = "#5CD0B3"
PINK_T = "#D147BD"
GPU_COLORS = [BLUE_T, GREEN_T, YELLOW_T, RED_T, PURPLE_T, ORANGE_T, TEAL_T, PINK_T]

VOICE = "af_heart"
VO_DIR = Path(__file__).parent / "vo"
MAX_STRETCH = 4  # slow animations by at most this much to span their line
TEXT_MAX = 1.0  # seconds; text appearing slower than this drags

N = {
    "h_a": "Imagine eight GPUs training the same model with data parallelism. Each one "
           "just ran backward on its own slice of the batch, so each one is holding a "
           "different set of gradients.",
    "h_b": "Before anyone can take an optimizer step, every GPU needs the sum of all eight.",
    "h_c": "So how much data does each GPU have to push onto the wire to make that happen?",
    "h_d": "You might guess it grows with the number of GPUs, since there are more "
           "gradients to collect. But done the right way, each GPU sends 1.75 times the "
           "size of its gradients,",
    "h_e": "and even with a thousand GPUs, it would never send more than twice.",
    "h_f": "To see where that comes from, you and I need a small vocabulary of "
           "communication patterns, and once we have it, the costs turn out to be "
           "pretty predictable.",

    "k_a": "The simplest kind of communication has exactly one sender and one receiver.",
    "k_b": "One GPU calls send, and the other calls a matching receive. Pipeline "
           "parallelism works this way, handing activations from one stage to the next.",
    "k_c": "Collectives involve a whole group of GPUs, and every one of them makes the "
           "same call together.",
    "k_d": "And there's a third kind, called one-sided. Here one GPU writes straight into "
           "another's memory, without the other side posting a receive at all.",
    "k_e": "That's useful when the access is sparse, like routing a token to the one GPU "
           "that holds its expert. PyTorch 2.14 exposes this as an experimental window "
           "over NCCL's put and get.",
    "k_f": "For the rest of this video, we'll stay with the middle group, the collectives.",

    "v_a": "Let's set up four GPUs, each with a few slots of memory, and color each piece "
           "of data by the GPU it started on.",
    "bc_a": "The simplest collective is broadcast. One GPU has a tensor,",
    "bc_b": "and afterwards, every GPU has a copy of it.",
    "sc_a": "Scatter starts the same way, except the tensor is cut into pieces,",
    "sc_b": "and each GPU receives just its own piece.",
    "ga_a": "Gather is scatter run backwards. Every GPU holds one piece,",
    "ga_b": "and all of them get collected onto a single GPU.",
    "ag_a": "If every GPU wants the full collection, and not just one of them, that's "
            "all-gather.",
    "ag_b": "This is what ZeRO uses to gather the sharded weights just before forward "
            "and backward.",
    "rd_a": "Now for the ones that do arithmetic. In a reduce, every GPU holds its own "
            "version of the same tensor,",
    "rd_b": "and they get combined, usually summed, onto one GPU.",
    "ar_a": "All-reduce leaves that sum on every GPU,",
    "ar_b": "and this is the one from our opening example. DDP uses it to sum the "
            "gradients across all of its ranks.",
    "rs_a": "Reduce-scatter starts with every GPU holding a full set of gradients, cut "
            "into shards,",
    "rs_b": "but each GPU ends up with the sum of just one shard. ZeRO uses this for its "
            "gradients, since each GPU only updates its own shard anyway.",
    "aa_a": "And finally, all-to-all, where every GPU has a separate piece for every "
            "other GPU,",
    "aa_b": "and they all swap at once, a bit like transposing a matrix. Ulysses sequence "
            "parallelism uses it to switch between sharding by tokens and sharding by "
            "heads, and mixture of experts uses it to send tokens to their experts.",

    "dc_a": "There's a useful relationship hiding in this list. Put a reduce-scatter and "
            "an all-gather back to back,",
    "dc_b": "and you've got an all-reduce. The first phase leaves each GPU with one fully "
            "summed shard, and the second phase shares those shards with everyone.",

    "rr_a": "Let's watch this happen on a ring, which is one of the ways NCCL does it. "
            "Each GPU only ever sends to its neighbor on the right.",
    "rr_b": "Each GPU's gradients are split into four chunks, and the colored stripes in "
            "each chunk show whose gradients have been added in so far.",
    "rr_c": "On the first step, every GPU sends one chunk to its neighbor, who adds it to "
            "its own copy.",
    "rr_d": "On the second step, each GPU passes along the chunk it just added to, so now "
            "some chunks hold three contributions.",
    "rr_e": "After the third step, every GPU holds one chunk with all four contributions "
            "in it, and that's a reduce-scatter.",
    "rr_f": "Then the all-gather phase passes those finished chunks around the ring for "
            "three more steps,",
    "rr_g": "and now every GPU has the full sum.",

    "cn_a": "So let's count what each GPU sent. On every step, it sent one chunk, which is "
            "a quarter of the gradients.",
    "cn_b": "That's three quarters for the reduce-scatter, and another three quarters for "
            "the all-gather.",
    "cn_c": "In general, with n GPUs, that's two times n minus one, over n, times the "
            "payload.",
    "cn_d": "At eight GPUs, that comes out to 1.75, our number from the start,",
    "cn_e": "and as n grows, it creeps up toward two, but never gets past it.",
    "cn_f": "This also explains why a reduce-scatter on its own moves half as much as an "
            "all-reduce. It's simply missing the all-gather half.",

    "rb_a": "The ring has one more trick, and it's easiest to see with a broadcast.",
    "rb_b": "Say GPU zero has N bytes to send to three others, over links with bandwidth "
            "B. The naive way passes the whole message down the line, one hop at a time.",
    "rb_c": "Each hop takes N over B, so with k GPUs you wait k minus one times N over B, "
            "and most of the links sit idle the whole time.",
    "rb_d": "Instead, let's split the message into S smaller chunks, and forward each "
            "chunk the moment it arrives.",
    "rb_e": "Now the links are busy at the same time. Each step moves only N over S B, "
            "and it takes S plus k minus 2 steps.",
    "rb_f": "Here that's six quarter steps, which is half the time of the naive way.",
    "rb_g": "And if the chunks are tiny, so that S is much larger than k, the total comes "
            "out to about N over B, as if the other GPUs weren't even there.",

    "tr_a": "A ring does have a weakness, though. A message has to pass through every GPU "
            "in turn, so the number of hops, and the latency that comes with them, grows "
            "with k.",
    "tr_b": "A tree fans the message out instead, so the hops only grow with the log "
            "of k.",
    "tr_c": "The catch is that half the nodes in a binary tree are leaves, and a leaf "
            "only ever receives, so its outgoing bandwidth goes to waste.",
    "tr_d": "So NCCL builds two trees over the same GPUs, arranged so that no GPU is an "
            "interior node in both.",
    "tr_e": "Every GPU gets to send in one of the two trees, and running both at once "
            "gets the full bandwidth back.",
    "tr_f": "So a ring keeps the higher peak bandwidth, a tree keeps the lower latency, "
            "and for a fixed message size, enough GPUs will make the tree the faster one.",

    "pr_a": "The algorithm decides who sends what to whom. Separately, NCCL picks a "
            "protocol, which decides how the receiver knows that the data has arrived.",
    "pr_b": "Simple sends large chunks, and uses a memory fence to be sure each chunk is "
            "complete. That gets close to peak bandwidth, but costs about six "
            "microseconds per hop.",
    "pr_c": "LL, for low latency, pairs every four bytes of data with a four byte flag, "
            "written together, so the receiver can go the moment the flag lands. That's "
            "about one microsecond per hop, but it only reaches 25 to 50 percent of peak.",
    "pr_d": "LL128 uses the same flag trick on 128 byte lines, 120 bytes of data and 8 of "
            "flag, which gets about 95 percent of peak at around two microseconds.",
    "pr_e": "The catch is that those 128 byte writes must never be split or reordered, "
            "so in practice you see it on NVLink, inside a node.",

    "bw_a": "So when you benchmark a collective, which bandwidth should you look at?",
    "bw_b": "The obvious one is algbw, the payload size divided by the time it took.",
    "bw_c": "But during an all-reduce, each GPU actually sent nearly twice the payload, "
            "and that factor changes with the number of ranks, so algbw doesn't line up "
            "with your link speed.",
    "bw_d": "busbw fixes this by multiplying in the collective's correction factor. "
            "That's two times n minus one, over n, for all-reduce, and n minus one over "
            "n for all-gather and reduce-scatter.",
    "bw_e": "On an eight GPU H200 node, a 16 gigabyte all-reduce measured an algbw of "
            "275.6.",
    "bw_f": "Times 1.75, that's a busbw of 482.3.",
    "bw_g": "And that's strange, because the NVLink spec on this node is only 450 "
            "gigabytes per second.",

    "sh_a": "The reason is SHARP. Newer NVSwitches have their own arithmetic units, so "
            "the switch can do the reduction itself.",
    "sh_b": "Each GPU sends its data up to the switch, once,",
    "sh_c": "the switch adds it all up,",
    "sh_d": "and then sends the result back out to everyone. That's N plus one sends, "
            "instead of the ring's 2N.",
    "sh_e": "busbw still assumes ring traffic, so it reports more than the wire can "
            "carry. With SHARP switched off, the same node measures 367.6, or 82 "
            "percent of the spec.",
    "sh_f": "You don't get SHARP everywhere, though. On H200, NCCL only switches to it "
            "above four GPUs, and the gain ramps up from there, to about 1.3x at eight.",
    "sh_g": "On B200, the switch happens above five GPUs instead.",
    "sh_h": "And it only helps all-reduce. All-gather and reduce-scatter on the same "
            "node stay on the ring, at around 80 percent of spec.",

    "en_a": "So distributed training really does get by with a small vocabulary.",
    "en_b": "DDP all-reduces its gradients,",
    "en_c": "ZeRO-3 all-gathers weights twice and reduce-scatters gradients once,",
    "en_d": "pipeline stages send and receive,",
    "en_e": "and sequence and expert parallelism go all-to-all.",
    "en_f": "And since an all-reduce costs about twice the payload, and an all-gather or "
            "reduce-scatter about once, you can add up the bill before you ever launch "
            "the job.",
}


def caption(text, size=28, color=WHITE):
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.45)


def cell(label, colors, w=1.1, h=0.5, fs=22):
    """A memory slot. One color: data from that GPU. Several: a partial or full sum."""
    rect = Rectangle(width=w, height=h, stroke_color=WHITE, stroke_width=1.5)
    if len(colors) == 1:
        rect.set_fill(colors[0], 0.85)
        stripes = VGroup()
    else:
        stripes = VGroup(*[Rectangle(width=w / len(colors), height=h, stroke_width=0,
                                     fill_color=c, fill_opacity=0.85) for c in colors])
        stripes.arrange(RIGHT, buff=0).move_to(rect)
    t = Text(label, font_size=fs, color=BLACK, weight=BOLD).move_to(rect)
    return VGroup(stripes, rect, t)


class Gpus:
    """Four GPUs side by side, each a column of four memory slots."""

    def __init__(self, n=4, y=-0.35, gap=3.0, slot_h=0.5, slot_w=1.1):
        self.n = n
        self.slot_h, self.slot_w = slot_h, slot_w
        xs = [(i - (n - 1) / 2) * gap for i in range(n)]
        self.boxes = VGroup(*[RoundedRectangle(corner_radius=0.15, width=slot_w + 0.5,
                                               height=4 * slot_h + 5 * 0.15 + 0.2,
                                               stroke_color=GREY_B).move_to([x, y, 0])
                              for x in xs])
        self.labels = VGroup(*[Text(f"GPU{i}", font_size=24, color=GPU_COLORS[i])
                               .next_to(b, UP, buff=0.12) for i, b in enumerate(self.boxes)])
        self.group = VGroup(self.boxes, self.labels)

    def pos(self, g, s):
        b = self.boxes[g]
        return b.get_top() + DOWN * (0.25 + self.slot_h / 2 + s * (self.slot_h + 0.15))

    def cell(self, g, s, label, colors):
        return cell(label, colors, self.slot_w, self.slot_h).move_to(self.pos(g, s))


class Collectives(Scene):
    def construct(self):
        self.timeline = []
        for beat in (self.hook, self.kinds, self.vocab, self.decompose, self.ring,
                     self.count, self.ring_broadcast, self.tree, self.protocols,
                     self.busbw, self.sharp, self.sharp_numbers, self.outro):
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

    # 0. hook: 8 GPUs, each with its own gradients
    def hook(self):
        n = 8
        boxes = VGroup(*[RoundedRectangle(corner_radius=0.12, width=1.25, height=1.3,
                                          stroke_color=GREY_B) for _ in range(n)])
        boxes.arrange(RIGHT, buff=0.3).shift(UP * 1.2)
        lbls = VGroup(*[Text(f"GPU{i}", font_size=20, color=GPU_COLORS[i])
                        .next_to(b, UP, buff=0.1) for i, b in enumerate(boxes)])
        grads = VGroup(*[cell(f"g{i}", [GPU_COLORS[i]], 0.95, 0.5, 20).move_to(b)
                         for i, b in enumerate(boxes)])
        sums = VGroup(*[cell("Σg", GPU_COLORS, 0.95, 0.5, 20).move_to(b) for b in boxes])
        q = Text("How much does each GPU send?", font_size=36).shift(DOWN * 0.9)
        guess = Text("grows with the number of GPUs?", font_size=28, color=GREY_B)
        guess.next_to(q, DOWN, buff=0.35)
        ans = Text("1.75 × its gradients", font_size=44, color=YELLOW_T).move_to(guess)
        ans.shift(DOWN * 0.2)
        bound = Text("≤ 2× for any number of GPUs", font_size=30, color=GREEN_T)
        bound.next_to(ans, DOWN, buff=0.35)
        title = Text("Collectives", font_size=72)

        self.say("h_a", [FadeIn(boxes), FadeIn(lbls)],
                 LaggedStart(*[FadeIn(g, shift=DOWN * 0.2) for g in grads], lag_ratio=0.15))
        self.say("h_b", LaggedStart(*[ReplacementTransform(g, s)
                                      for g, s in zip(grads, sums)], lag_ratio=0.1))
        gbar = Rectangle(width=0.5, height=0.35, fill_color=GREY_B, fill_opacity=0.7,
                         stroke_width=0).next_to(guess, DOWN, buff=0.3).align_to(guess, LEFT)
        gbar_long = gbar.copy().stretch_to_fit_width(guess.width).align_to(guess, LEFT)
        cross = Cross(VGroup(guess, gbar_long), stroke_color=RED_T, stroke_width=5)
        names = VGroup(*[Text(t, font_size=24, color=GREY_B) for t in
                         ("broadcast", "scatter", "gather", "all-gather", "reduce",
                          "all-reduce", "reduce-scatter", "all-to-all")])
        names.arrange_in_grid(2, 4, buff=(0.6, 0.3)).shift(DOWN * 1.4)
        self.say("h_c", Write(q), LaggedStart(*[Indicate(s) for s in sums], lag_ratio=0.1))
        self.say("h_d", [FadeIn(guess), FadeIn(gbar)], Transform(gbar, gbar_long),
                 Create(cross), [FadeOut(VGroup(guess, gbar, cross)), FadeIn(ans, scale=1.3)])
        self.say("h_e", FadeIn(bound, shift=UP * 0.2), Circumscribe(bound, color=GREEN_T))
        self.say("h_f", [FadeOut(VGroup(boxes, lbls, sums, q, ans, bound)), Write(title)],
                 title.animate.scale(0.7).shift(UP * 0.6),
                 LaggedStart(*[FadeIn(n, shift=UP * 0.2) for n in names], lag_ratio=0.15))

    # 1. point-to-point vs collective vs one-sided
    def kinds(self):
        cols = [-4.6, 0, 4.6]
        heads = VGroup(*[Text(t, font_size=30).move_to([x, 2.8, 0]) for t, x in
                         zip(("point-to-point", "collective", "one-sided"), cols)])

        def gpu(x, y, i, s=0.8):
            return VGroup(RoundedRectangle(corner_radius=0.1, width=s, height=s,
                                           stroke_color=GPU_COLORS[i]),
                          Text(f"{i}", font_size=22, color=GPU_COLORS[i])).move_to([x, y, 0])

        a, b = gpu(cols[0] - 1, 1.2, 0), gpu(cols[0] + 1, 1.2, 1)
        arr = Arrow(a.get_right(), b.get_left(), buff=0.1, color=WHITE)
        pkt = Square(0.25, fill_color=BLUE_T, fill_opacity=1, stroke_width=0).move_to(a)
        sr = VGroup(Text("send", font_size=22).next_to(a, DOWN),
                    Text("recv", font_size=22).next_to(b, DOWN))
        pp = Text("pipeline parallel:\nactivations to\nthe next stage", font_size=22,
                  color=GREY_B).move_to([cols[0], -0.6, 0])

        ring = VGroup(*[gpu(cols[1] + 1.1 * np.cos(t), 0.8 + 1.1 * np.sin(t), i, 0.6)
                        for i, t in enumerate([PI / 2, 0, -PI / 2, PI])])
        same = VGroup(*[Text("all_reduce()", font_size=16, color=YELLOW_T)
                        .next_to(g, DOWN if i == 2 else UP, buff=0.05)
                        for i, g in enumerate(ring)])
        links = VGroup(*[Line(ring[i].get_center(), ring[j].get_center(), stroke_width=1.5,
                              color=GREY_B, buff=0.35)
                         for i in range(4) for j in range(i + 1, 4)])

        c, d = gpu(cols[2] - 1, 1.2, 0), gpu(cols[2] + 1, 1.2, 1)
        mem = VGroup(*[Square(0.18, stroke_width=1, stroke_color=GREY_B) for _ in range(4)])
        mem.arrange(RIGHT, buff=0.04).next_to(d, DOWN, buff=0.2)
        win = Text("window", font_size=18, color=GREY_B).next_to(mem, DOWN, buff=0.1)
        put = Text("put", font_size=22).next_to(c, DOWN)
        pkt2 = Square(0.18, fill_color=BLUE_T, fill_opacity=1, stroke_width=0).move_to(c)
        moe = Text("sparse access:\none token to\none expert", font_size=22,
                   color=GREY_B).move_to([cols[2], -0.6, 0])
        exp = Text("PyTorch 2.14: experimental\nnccl2 window over Put/Get", font_size=20,
                   color=GREY_B).move_to([cols[2], -2.1, 0])
        dim = [heads[0], heads[2], a, b, arr, sr, pp, c, d, mem, win, put, pkt2, moe, exp]

        self.say("k_a", Write(heads[0]), [FadeIn(a), FadeIn(b)])
        self.add(pkt)
        self.say("k_b", [GrowArrow(arr), FadeIn(sr)], pkt.animate.move_to(b), FadeIn(pp))
        self.say("k_c", Write(heads[1]), [FadeIn(ring), Create(links)], FadeIn(same))
        self.say("k_d", Write(heads[2]), [FadeIn(c), FadeIn(d), FadeIn(mem), FadeIn(win)],
                 [FadeIn(put), pkt2.animate.move_to(mem[2])])
        pkt3 = pkt2.copy().move_to(c)
        self.say("k_e", FadeIn(moe), Circumscribe(mem[2], color=BLUE_T),
                 pkt3.animate.move_to(mem[0]), FadeIn(exp),
                 Indicate(VGroup(c, d), color=WHITE))
        self.say("k_f", [m.animate.set_opacity(0.25) for m in dim],
                 Circumscribe(VGroup(heads[1], ring, same), color=YELLOW_T))

    # 2. the vocabulary on four GPUs
    def collective(self, gp, name, key_a, key_b, before, after, moves, user, reduce=False,
                   prev=None, extra=()):
        title = Text(name, font_size=40).to_edge(UP, buff=0.35)
        cells = {p: gp.cell(*p, *spec) for p, spec in before.items()}
        intro = [FadeIn(VGroup(*cells.values()))]
        intro.insert(0, ReplacementTransform(prev, title) if prev else Write(title))
        self.say(key_a, *intro)

        fly = {}
        for src, dst in moves:
            fly.setdefault(dst, []).append(cells[src].copy())
        travel = [c.animate.move_to(gp.pos(*dst)) for dst, cs in fly.items() for c in cs]
        merge, final, used = [], [], set()
        for p, spec in after.items():
            new = gp.cell(*p, *spec)
            final.append(new)
            srcs = list(fly.get(p, []))
            if p in before and before[p] == spec and not srcs:
                final[-1] = cells[p]
                used.add(p)
                continue
            if p in before and reduce:
                srcs.append(cells[p])
                used.add(p)
            if srcs:
                merge.append(ReplacementTransform(VGroup(*srcs), new))
            else:
                merge.append(FadeIn(new))
        merge += [FadeOut(c) for p, c in cells.items() if p not in used]
        cap = caption(user, 26, GREY_B)
        self.say(key_b, LaggedStart(*travel, lag_ratio=0.05), merge, FadeIn(cap), *extra)
        self.play(FadeOut(VGroup(*final)), FadeOut(cap), run_time=0.5)
        return title

    def vocab(self):
        gp = Gpus()
        self.say("v_a", FadeIn(gp.group),
                 LaggedStart(*[Indicate(l, color=GPU_COLORS[i])
                               for i, l in enumerate(gp.labels)], lag_ratio=0.2))
        C = GPU_COLORS
        rng = range(4)
        t = self.collective(
            gp, "broadcast", "bc_a", "bc_b", {(0, 0): ("A", [C[0]])},
            {(g, 0): ("A", [C[0]]) for g in rng}, [((0, 0), (g, 0)) for g in range(1, 4)],
            "dist.broadcast: copy one tensor to every rank")
        t = self.collective(
            gp, "scatter", "sc_a", "sc_b", {(0, s): (f"A{s}", [C[0]]) for s in rng},
            {**{(0, s): (f"A{s}", [C[0]]) for s in rng},
             **{(g, g): (f"A{g}", [C[0]]) for g in range(1, 4)}},
            [((0, g), (g, g)) for g in range(1, 4)],
            "dist.scatter: piece i goes to rank i", prev=t)
        diag = {(g, g): (f"x{g}", [C[g]]) for g in rng}
        t = self.collective(
            gp, "gather", "ga_a", "ga_b", diag,
            {**diag, **{(0, s): (f"x{s}", [C[s]]) for s in rng}},
            [((g, g), (0, g)) for g in range(1, 4)],
            "dist.gather: every rank's piece lands on one rank", prev=t)
        t = self.collective(
            gp, "all-gather", "ag_a", "ag_b", diag,
            {(g, s): (f"x{s}", [C[s]]) for g in rng for s in rng},
            [((s, s), (g, s)) for s in rng for g in rng if g != s],
            "used by ZeRO / FSDP: gather sharded weights before forward and backward",
            prev=t)
        t = self.collective(
            gp, "reduce", "rd_a", "rd_b", {(g, 0): (f"g{g}", [C[g]]) for g in rng},
            {**{(g, 0): (f"g{g}", [C[g]]) for g in range(1, 4)}, (0, 0): ("Σ", C[:4])},
            [((g, 0), (0, 0)) for g in range(1, 4)],
            "dist.reduce: sum (or avg, max, ...) onto one rank", reduce=True, prev=t)
        t = self.collective(
            gp, "all-reduce", "ar_a", "ar_b", {(g, 0): (f"g{g}", [C[g]]) for g in rng},
            {(g, 0): ("Σ", C[:4]) for g in rng},
            [((s, 0), (g, 0)) for s in rng for g in rng if g != s],
            "used by DDP: sum the gradients across all ranks", reduce=True, prev=t)
        t = self.collective(
            gp, "reduce-scatter", "rs_a", "rs_b",
            {(g, s): ("ABCD"[s], [C[g]]) for g in rng for s in rng},
            {(g, g): ("Σ" + "ABCD"[g], C[:4]) for g in rng},
            [((s, g), (g, g)) for s in rng for g in rng if g != s],
            "used by ZeRO / FSDP: each rank keeps the summed gradients of its own shard",
            reduce=True, prev=t)
        self.collective(
            gp, "all-to-all", "aa_a", "aa_b",
            {(g, s): (f"{g}→{s}", [C[g]]) for g in rng for s in rng},
            {(s, g): (f"{g}→{s}", [C[g]]) for g in rng for s in rng},
            [((g, s), (s, g)) for g in rng for s in rng if g != s],
            "used by Ulysses sequence parallelism and MoE expert parallelism", prev=t,
            extra=[LaggedStart(*[Indicate(b, color=GPU_COLORS[i])
                                 for i, b in enumerate(gp.boxes)], lag_ratio=0.2)])

    # 3. all-reduce = reduce-scatter + all-gather
    def decompose(self):
        C = GPU_COLORS

        def panel(state, title):
            g = VGroup()
            for i in range(4):
                col = VGroup(*[cell(*state(i, s), w=0.6, h=0.4, fs=16) if state(i, s)
                               else Rectangle(width=0.6, height=0.4, stroke_color=GREY_D,
                                              stroke_width=1)
                               for s in range(4)]).arrange(DOWN, buff=0.06)
                g.add(col)
            g.arrange(RIGHT, buff=0.1)
            return VGroup(Text(title, font_size=24).next_to(g, UP), g)

        a = panel(lambda i, s: ("ABCD"[s], [C[i]]), "start")
        b = panel(lambda i, s: ("ΣABCD"[s + 1], C[:4]) if s == i else None, "one sum each")
        c = panel(lambda i, s: ("ΣABCD"[s + 1], C[:4]), "full sum everywhere")
        VGroup(a, b, c).arrange(RIGHT, buff=1.8).shift(DOWN * 0.2)
        ar1 = Arrow(a.get_right(), b.get_left(), buff=0.15)
        ar2 = Arrow(b.get_right(), c.get_left(), buff=0.15)
        l1 = Text("reduce-\nscatter", font_size=22, color=YELLOW_T).next_to(ar1, DOWN)
        l2 = Text("all-\ngather", font_size=22, color=YELLOW_T).next_to(ar2, DOWN)
        brace = Brace(VGroup(a, b, c), DOWN, buff=0.3)
        bl = Text("= all-reduce", font_size=32, color=GREEN_T).next_to(brace, DOWN)
        head = Text("all-reduce = reduce-scatter + all-gather", font_size=36).to_edge(UP)
        self.say("dc_a", FadeIn(a), [GrowArrow(ar1), FadeIn(l1)], FadeIn(b),
                 [GrowArrow(ar2), FadeIn(l2)], FadeIn(c))
        self.say("dc_b", [GrowFromCenter(brace), FadeIn(bl)],
                 Indicate(b[1], color=YELLOW_T), Indicate(c[1], color=YELLOW_T),
                 Write(head))

    # 4. ring reduce-scatter then all-gather
    def ring(self):
        gp = Gpus(y=-0.1)
        C = GPU_COLORS
        title = Text("Ring all-reduce", font_size=40).to_edge(UP, buff=0.3)
        arrows = VGroup(*[Arrow(gp.boxes[i].get_right(), gp.boxes[i + 1].get_left(),
                                buff=0.1, color=GREY_B) for i in range(3)])
        wrap = CurvedArrow(gp.boxes[3].get_bottom() + DOWN * 0.05,
                           gp.boxes[0].get_bottom() + DOWN * 0.05, angle=-0.5,
                           color=GREY_B)
        rows = VGroup(*[Text("ABCD"[s], font_size=24, color=GREY_B)
                        .next_to(gp.boxes[0], LEFT, buff=0.25).set_y(gp.pos(0, s)[1])
                        for s in range(4)])
        contrib = [[{g} for _ in range(4)] for g in range(4)]
        cells = [[gp.cell(g, s, "ABCD"[s], [C[g]]) for s in range(4)] for g in range(4)]

        self.say("rr_a", Write(title), [FadeIn(gp.group), Create(arrows), Create(wrap)],
                 LaggedStart(*[ShowPassingFlash(a.copy().set_color(YELLOW_T), time_width=0.6)
                               for a in [*arrows, wrap]], lag_ratio=0.6))
        self.say("rr_b", [FadeIn(VGroup(*[c for col in cells for c in col])), FadeIn(rows)],
                 Indicate(cells[1][2], color=WHITE))

        def step(chunk_of, full):
            """Every GPU g sends chunk chunk_of(g) to g+1, which adds (or copies) it."""
            fly, upd = [], []
            for g in range(4):
                c, d = chunk_of(g), (g + 1) % 4
                cp = cells[g][c].copy()
                arc = -0.6 if g == 3 else 0
                fly.append(cp.animate(path_arc=arc).move_to(gp.pos(d, c)))
                contrib[d][c] = set(range(4)) if full else contrib[d][c] | contrib[g][c]
                new = gp.cell(d, c, "ABCD"[c], [C[x] for x in sorted(contrib[d][c])])
                upd.append((cp, d, c, new))
            merge = []
            for cp, d, c, new in upd:
                merge.append(ReplacementTransform(VGroup(cp, cells[d][c]), new))
                cells[d][c] = new
            return [fly, merge]

        self.say("rr_c", *step(lambda g: g % 4, False))
        self.say("rr_d", *step(lambda g: (g - 1) % 4, False))
        s3 = step(lambda g: (g - 2) % 4, False)
        self.say("rr_e", *s3)
        boxes = VGroup(*[SurroundingRectangle(cells[g][(g + 1) % 4], color=WHITE, buff=0.05)
                         for g in range(4)])
        self.play(Create(boxes), run_time=0.6)
        ag = [step(lambda g, t=t: (g + 1 - t) % 4, True) for t in range(3)]
        # all-gather: three steps under one line
        steps = [FadeOut(boxes)] + [s for st in ag[:2] for s in st]
        self.say("rr_f", *steps)
        self.say("rr_g", *ag[2])

    # 5. counting the bytes
    def count(self):
        head = Text("What does each GPU send?", font_size=38).to_edge(UP)
        per = Text("each step: 1 chunk = 1/4 of the payload", font_size=30).shift(UP * 1.8)
        chunks = VGroup(*[Rectangle(width=0.9, height=0.45, fill_color=BLUE_T,
                                    fill_opacity=0.8, stroke_color=WHITE)
                          for _ in range(6)]).arrange(RIGHT, buff=0.1).shift(UP * 0.7)
        rs = Text("reduce-scatter: 3/4", font_size=24, color=YELLOW_T)
        ag = Text("all-gather: 3/4", font_size=24, color=GREEN_T)
        rs.next_to(chunks[:3], DOWN)
        ag.next_to(chunks[3:], DOWN)
        tot = Text("total: 1.5 × payload  (n = 4)", font_size=30).next_to(VGroup(rs, ag),
                                                                          DOWN, buff=0.35)
        gen = Text("per-GPU traffic = 2(n−1)/n × payload", font_size=34, color=YELLOW_T)
        gen.next_to(tot, DOWN, buff=0.4)

        self.say("cn_a", Write(head), Write(per), FadeIn(chunks[0], shift=RIGHT * 0.3))
        self.say("cn_b", [LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in chunks[1:3]],
                                      lag_ratio=0.5), FadeIn(rs)],
                 [LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in chunks[3:]],
                              lag_ratio=0.5), FadeIn(ag)], FadeIn(tot))
        self.say("cn_c", Write(gen), Circumscribe(gen, color=YELLOW_T))

        self.play(FadeOut(VGroup(per, chunks, rs, ag, tot)),
                  gen.animate.next_to(head, DOWN, buff=0.3), run_time=0.8)
        ax = Axes(x_range=[0, 64, 8], y_range=[0, 2.2, 0.5], x_length=9.5, y_length=3.6,
                  tips=False).shift(DOWN * 1.1)
        xl = Text("GPUs n", font_size=22).next_to(ax.x_axis.get_end(), UP, buff=0.15)
        ticks = VGroup(*[Text(str(v), font_size=18).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
                         for v in (8, 16, 32, 64)])
        two = DashedLine(ax.c2p(0, 2), ax.c2p(64, 2), color=GREEN_T)
        twol = Text("2", font_size=22, color=GREEN_T).next_to(two, LEFT, buff=0.1)
        one = Text("1", font_size=20).next_to(ax.c2p(0, 1), LEFT, buff=0.1)
        dots = VGroup(*[Dot(ax.c2p(n, 2 * (n - 1) / n), radius=0.05, color=BLUE_T)
                        for n in range(2, 65)])
        d8 = Dot(ax.c2p(8, 1.75), color=YELLOW_T, radius=0.09)
        l8 = Text("n = 8: 1.75", font_size=24, color=YELLOW_T).next_to(d8, DR, buff=0.1)
        self.say("cn_d", [Create(ax), FadeIn(xl), FadeIn(ticks), FadeIn(one)],
                 LaggedStart(*[FadeIn(d) for d in dots[:7]], lag_ratio=0.2),
                 [FadeIn(d8, scale=2), FadeIn(l8)])
        self.say("cn_e", LaggedStart(*[FadeIn(d) for d in dots[7:]], lag_ratio=0.05),
                 [Create(two), FadeIn(twol)])
        self.play(FadeOut(VGroup(ax, xl, ticks, two, twol, one, dots, d8, l8)), run_time=0.6)
        cmp = VGroup(
            VGroup(Text("all-reduce", font_size=28),
                   Rectangle(width=2 * 3.5 * 7 / 8, height=0.45, fill_color=BLUE_T,
                             fill_opacity=0.8, stroke_width=0),
                   Text("2(n−1)/n", font_size=24)).arrange(RIGHT, buff=0.3),
            VGroup(Text("reduce-scatter", font_size=28),
                   Rectangle(width=3.5 * 7 / 8, height=0.45, fill_color=YELLOW_T,
                             fill_opacity=0.8, stroke_width=0),
                   Text("(n−1)/n", font_size=24)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.6)
        for r in cmp:
            r[1].align_to(cmp[0][1], LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.3)
        self.say("cn_f", FadeIn(cmp[0]), FadeIn(cmp[1]))

    # 6. pipelined ring broadcast
    def ring_broadcast(self):
        head = Text("Ring broadcast", font_size=40).to_edge(UP, buff=0.3)
        k, S = 4, 4
        xs = [-4.5, -1.5, 1.5, 4.5]
        gpus = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=1.6, height=0.9,
                                                stroke_color=GPU_COLORS[i]),
                               Text(f"GPU{i}", font_size=22, color=GPU_COLORS[i]))
                        .move_to([x, 1.9, 0]) for i, x in enumerate(xs)])
        links = VGroup(*[Arrow(gpus[i].get_right(), gpus[i + 1].get_left(), buff=0.08,
                               color=GREY_B) for i in range(3)])
        lab = VGroup(Text("N bytes", font_size=22).next_to(gpus[0], UP, buff=0.12),
                     Text("bandwidth B per link", font_size=20, color=GREY_B)
                     .next_to(links[1], UP, buff=0.5))

        # timelines: one unit = N/B
        unit = 3.0
        t0x = -4.2
        axis = Line([t0x, -2.6, 0], [t0x + 3 * unit + 0.2, -2.6, 0], color=GREY_B)
        tl = VGroup(*[Text(s, font_size=18, color=GREY_B).next_to([t0x + i * unit, -2.6, 0],
                                                                  DOWN, buff=0.1)
                      for i, s in enumerate(("0", "N/B", "2N/B", "3N/B"))])
        naive_l = Text("naive", font_size=22).next_to([t0x, -1.2, 0], LEFT, buff=0.2)
        pipe_l = Text("pipelined", font_size=22).next_to([t0x, -1.95, 0], LEFT, buff=0.2)

        msg = Rectangle(width=1.2, height=0.45, fill_color=BLUE_T, fill_opacity=0.9,
                        stroke_width=0).move_to(gpus[0].get_bottom() + DOWN * 0.5)
        naive_bars = VGroup(*[Rectangle(width=unit, height=0.35, fill_color=BLUE_T,
                                        fill_opacity=0.7, stroke_color=WHITE, stroke_width=1)
                              .move_to([t0x + (i + 0.5) * unit, -1.2, 0]) for i in range(3)])
        hops = []
        for i in range(1, 4):
            hops.append([msg.animate.move_to(gpus[i].get_bottom() + DOWN * 0.5),
                         GrowFromEdge(naive_bars[i - 1], LEFT)])
        form1 = Text("(k−1) · N/B", font_size=26, color=YELLOW_T).next_to(naive_bars, RIGHT)

        self.say("rb_a", Write(head), [FadeIn(gpus), GrowArrow(links[0]), GrowArrow(links[1]),
                                       GrowArrow(links[2])])
        self.say("rb_b", FadeIn(lab), FadeIn(msg), [Create(axis), FadeIn(tl), FadeIn(naive_l)], *hops[:2])
        self.say("rb_c", hops[2], Write(form1),
                 Indicate(VGroup(links[0], links[1]), color=RED_T))

        # pipelined: S chunks, each hop N/(SB)
        slots = [VGroup(*[Square(0.28, stroke_color=GREY_B, stroke_width=1)
                          for _ in range(S)]).arrange(RIGHT, buff=0.04)
                 .next_to(gpus[g], DOWN, buff=0.25) for g in range(k)]
        chunks = VGroup(*[Square(0.28, fill_color=BLUE_T, fill_opacity=0.9, stroke_width=0)
                          .move_to(slots[0][j]) for j in range(S)])
        sub = unit / S
        pbars = []
        self.say("rb_d", [FadeOut(msg), FadeIn(VGroup(*slots[1:])), FadeIn(slots[0]),
                          FadeIn(chunks), FadeIn(pipe_l)],
                 FadeIn(Text("S chunks", font_size=20).next_to(slots[0], DOWN, buff=0.1)))
        steps = []
        for t in range(1, S + k - 1):
            anims = []
            for j in range(S):
                src, dst = t - 1 - j, t - j
                if 0 <= src and dst <= k - 1:
                    anims.append(chunks[j].copy().animate.move_to(slots[dst][j]))
            bar = Rectangle(width=sub, height=0.35, fill_color=GREEN_T, fill_opacity=0.7,
                            stroke_color=WHITE, stroke_width=1)
            bar.move_to([t0x + (t - 0.5) * sub, -1.95, 0])
            pbars.append(bar)
            steps.append(anims + [GrowFromEdge(bar, LEFT)])
        form2 = Text("(S+k−2) · N/(SB)", font_size=26, color=GREEN_T)
        form2.next_to(VGroup(*pbars), RIGHT)
        self.say("rb_e", *steps[:4])
        self.say("rb_f", *steps[4:], Write(form2))
        lim = Text("S ≫ k  ⇒  time ≈ N/B", font_size=34, color=YELLOW_T)
        lim.move_to([3.2, -0.55, 0])
        self.say("rb_g", FadeIn(lim, shift=UP * 0.2), Circumscribe(lim, color=YELLOW_T),
                 Indicate(VGroup(*pbars), color=GREEN_T))

    # 7. ring vs (double binary) tree
    def tree(self):
        def node(r):
            return VGroup(Circle(0.2, fill_color=GREY_B, fill_opacity=1, stroke_width=0),
                          Text(str(r), font_size=18, color=BLACK))

        head = Text("Ring vs tree", font_size=40).to_edge(UP, buff=0.3)
        chain = VGroup(*[node(i) for i in range(8)])
        chain.arrange(RIGHT, buff=0.55).shift(UP * 1.9)
        ce = VGroup(*[Line(chain[i].get_right(), chain[i + 1].get_left(), color=GREY_B)
                      for i in range(7)])
        msg = Dot(chain[0].get_center(), color=YELLOW_T, radius=0.1)
        hop_r = Text("ring: k−1 = 7 hops", font_size=24, color=YELLOW_T)
        hop_r.next_to(chain, RIGHT, buff=0.3)

        # tree 1: interior = odd ranks; tree 2 = mirror (r -> 7-r): interior = even
        e1 = [(7, 3), (3, 1), (3, 5), (1, 0), (1, 2), (5, 4), (5, 6)]
        e2 = [(7 - a, 7 - b) for a, b in e1]
        d1 = {7: 0, 3: 1, 1: 2, 5: 2, 0: 3, 2: 3, 4: 3, 6: 3}

        def build(edges, depth, x0, color):
            nodes = {}
            for r in range(8):
                nodes[r] = node(r).move_to([x0 + r * 0.6, 0.2 - depth[r] * 0.8, 0])
            es = VGroup(*[Line(nodes[a].get_center(), nodes[b].get_center(), buff=0.2,
                               color=color) for a, b in edges])
            return VGroup(es, VGroup(*nodes.values())), nodes

        t1, n1 = build(e1, d1, -2.1, BLUE_T)
        d2 = {7 - r: d for r, d in d1.items()}
        t2, n2 = build(e2, d2, 1.2, GREEN_T)
        msg2 = Dot(n1[7].get_center(), color=YELLOW_T, radius=0.1)
        hop_t = Text("tree: log₂k = 3 hops", font_size=24, color=YELLOW_T)
        hop_t.next_to(t1, RIGHT, buff=0.4)
        scale = Text("k = 512:  ring 511 hops,  tree 9", font_size=26).to_edge(DOWN, buff=0.5)

        self.say("tr_a", Write(head), [FadeIn(chain), Create(ce)],
                 Succession(*[msg.animate(rate_func=linear, run_time=0.3)
                              .move_to(chain[i].get_center()) for i in range(1, 8)]),
                 FadeIn(hop_r))
        self.say("tr_b", FadeIn(t1),
                 Succession(*[msg2.animate.move_to(n1[r].get_center()) for r in (3, 5, 6)]),
                 FadeIn(hop_t), FadeIn(scale))
        self.play(FadeOut(VGroup(chain, ce, msg, hop_r, msg2, hop_t, scale)),
                  t1.animate.shift(LEFT * 2 + UP * 1.4), run_time=0.8)
        leaves1 = VGroup(*[n1[r] for r in (0, 2, 4, 6)])
        lt = Text("leaves only receive", font_size=24, color=RED_T).next_to(t1, DOWN, buff=0.3)
        self.say("tr_c", [l[0].animate.set_fill(RED_T) for l in leaves1], FadeIn(lt),
                 Indicate(leaves1, color=RED_T))
        t2.shift(RIGHT * 1.0 + UP * 1.4)
        ts = VGroup(Text("tree 1", font_size=24, color=BLUE_T).next_to(t1, UP, buff=0.2),
                    Text("tree 2 (mirrored)", font_size=24, color=GREEN_T)
                    .next_to(t2, UP, buff=0.2))
        self.say("tr_d", [FadeOut(lt), *[n1[r][0].animate.set_fill(GREY_B) for r in (0, 2, 4, 6)],
                          *[n1[r][0].animate.set_fill(BLUE_T) for r in (1, 3, 5, 7)],
                          FadeIn(ts[0])],
                 [FadeIn(t2), FadeIn(ts[1])],
                 [n2[r][0].animate.set_fill(GREEN_T) for r in (0, 2, 4, 6)])
        row = VGroup(*[VGroup(Square(0.55, fill_color=BLUE_T if r % 2 else GREEN_T,
                                     fill_opacity=0.8, stroke_color=WHITE),
                              Text(str(r), font_size=22, color=BLACK)) for r in range(8)])
        row.arrange(RIGHT, buff=0.12).shift(DOWN * 1.9)
        rl = Text("each GPU sends in the tree where it's interior", font_size=24)
        rl.next_to(row, DOWN, buff=0.25)
        self.say("tr_e", LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in row], lag_ratio=0.1),
                 FadeIn(rl))
        self.play(FadeOut(VGroup(row, rl)), run_time=0.5)
        ax = Axes(x_range=[0, 64, 8], y_range=[0, 6, 1], x_length=6.5, y_length=3.2,
                  tips=False).shift(DOWN * 0.6 + LEFT * 2.2)
        axl = VGroup(Text("GPUs k", font_size=20).next_to(ax.x_axis.get_end(), UP, buff=0.1),
                     Text("time, fixed message", font_size=20)
                     .next_to(ax.y_axis.get_end(), RIGHT, buff=0.1))
        ill = Text("(illustrative)", font_size=18, color=GREY_B).next_to(ax, DOWN, buff=0.1)
        ring_c = ax.plot(lambda k: 1 + 0.07 * k, x_range=[2, 64], color=YELLOW_T)
        tree_c = ax.plot(lambda k: 2.2 + 0.3 * np.log2(k), x_range=[2, 64], color=GREEN_T)
        tbl = VGroup(Text("ring: higher peak bandwidth", font_size=26, color=YELLOW_T),
                     Text("tree: lower latency,\nwins at large k", font_size=26,
                          color=GREEN_T)).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        tbl.next_to(ax, RIGHT, buff=0.6)
        self.say("tr_f", [FadeOut(VGroup(t1, t2, ts)), Create(ax), FadeIn(axl), FadeIn(ill)],
                 [Create(ring_c), FadeIn(tbl[0], shift=UP * 0.2)],
                 [Create(tree_c), FadeIn(tbl[1], shift=UP * 0.2)])

    # 8. protocols
    def protocols(self):
        head = Text("Protocols: how the receiver knows", font_size=38).to_edge(UP, buff=0.3)
        x0, W = -4.0, 7.6  # strip spans 128 bytes
        byte = W / 128

        def strip(parts):
            g = VGroup()
            for n, col in parts:
                g.add(Rectangle(width=n * byte, height=0.6, fill_color=col, fill_opacity=0.85,
                                stroke_color=BLACK, stroke_width=1))
            return g.arrange(RIGHT, buff=0)

        def row(name, s, stats, y):
            s.move_to([x0 + W / 2, y, 0])
            n = Text(name, font_size=28).next_to(s, LEFT, buff=0.3)
            st = Text(stats, font_size=20, line_spacing=0.8).next_to(s, RIGHT, buff=0.3)
            return n, s, st

        simple = strip([(128, BLUE_T)])
        fence = Line(UP * 0.5, DOWN * 0.5, color=RED_T, stroke_width=8)
        rs = row("Simple", simple, "near peak\n~6 µs/hop", 1.6)
        fence.next_to(simple, RIGHT, buff=0.04)
        rs[2].next_to(fence, RIGHT, buff=0.25)
        fl = Text("memory fence", font_size=18, color=RED_T).next_to(fence, DOWN, buff=0.1)
        ll = row("LL", strip([(4, BLUE_T), (4, YELLOW_T)] * 16), "25-50% of peak\n~1 µs/hop", 0)
        l128 = row("LL128", strip([(120, BLUE_T), (8, YELLOW_T)]), "~95% of peak\n~2 µs/hop", -1.6)
        legend = VGroup(Square(0.25, fill_color=BLUE_T, fill_opacity=0.85, stroke_width=0),
                        Text("data", font_size=20),
                        Square(0.25, fill_color=YELLOW_T, fill_opacity=0.85, stroke_width=0),
                        Text("flag", font_size=20)).arrange(RIGHT, buff=0.15)
        legend.to_edge(DOWN, buff=0.4)
        nv = Text("128-byte writes must not be split or reordered → NVLink, intra-node",
                  font_size=22, color=GREY_B).next_to(l128[1], DOWN, buff=0.3)

        snd = VGroup(RoundedRectangle(corner_radius=0.1, width=1.6, height=1.0, stroke_color=BLUE_T),
                     Text("sender", font_size=22)).move_to(LEFT * 3)
        rcv = VGroup(RoundedRectangle(corner_radius=0.1, width=1.6, height=1.0, stroke_color=GREEN_T),
                     Text("receiver", font_size=22)).move_to(RIGHT * 3)
        lnk = Arrow(snd.get_right(), rcv.get_left(), buff=0.1, color=GREY_B)
        bits = VGroup(*[Square(0.25, fill_color=BLUE_T, fill_opacity=0.9, stroke_width=0)
                        for _ in range(4)]).arrange(RIGHT, buff=0.05).move_to(snd)
        qm = Text("all here yet?", font_size=24, color=YELLOW_T).next_to(rcv, UP)
        demo = VGroup(snd, rcv, lnk, bits, qm)
        self.say("pr_a", Write(head), [FadeIn(snd), FadeIn(rcv), GrowArrow(lnk)],
                 LaggedStart(*[b.animate.move_to(rcv.get_center() + DOWN * 0.25 +
                                                 RIGHT * (i - 1.5) * 0.3)
                               for i, b in enumerate(bits)], lag_ratio=0.3),
                 FadeIn(qm), Indicate(rcv, color=YELLOW_T))
        self.play(FadeOut(demo), run_time=0.5)
        self.say("pr_b", [FadeIn(rs[0]), GrowFromEdge(simple, LEFT)], [Create(fence), FadeIn(fl)],
                 FadeIn(rs[2]))
        self.say("pr_c", [FadeIn(ll[0]), FadeIn(legend)],
                 LaggedStart(*[FadeIn(p) for p in ll[1]], lag_ratio=0.05), FadeIn(ll[2]))
        self.say("pr_d", FadeIn(l128[0]), GrowFromEdge(l128[1], LEFT),
                 Indicate(l128[1][1], color=YELLOW_T, scale_factor=1.5), FadeIn(l128[2]),
                 Circumscribe(VGroup(ll[2], l128[2]), color=YELLOW_T))
        self.say("pr_e", FadeIn(nv, shift=UP * 0.2), Circumscribe(l128[1][1], color=YELLOW_T),
                 Wiggle(l128[1]))

    # 9. algbw vs busbw
    def busbw(self):
        head = Text("algbw vs busbw", font_size=40).to_edge(UP, buff=0.3)
        alg = Text("algbw = payload / time", font_size=30).shift(UP * 2.1)
        busb = Text("busbw = algbw × correction factor", font_size=30, color=YELLOW_T)
        busb.next_to(alg, DOWN, buff=0.3)
        facs = VGroup(Text("all-reduce: 2(n−1)/n", font_size=24),
                      Text("all-gather, reduce-scatter: (n−1)/n", font_size=24)
                      ).arrange(RIGHT, buff=0.8).next_to(busb, DOWN, buff=0.25)
        why = Text("each GPU sends ~2× payload in an all-reduce", font_size=24,
                   color=GREY_B).next_to(alg, DOWN, buff=0.3)

        x0, sc = -3.6, 7 / 500  # GBps -> units
        y1, y2 = -0.9, -1.9

        def bar(v, y, col):
            return Rectangle(width=v * sc, height=0.55, fill_color=col, fill_opacity=0.85,
                             stroke_width=0).move_to([x0 + v * sc / 2, y, 0])

        b1, b2 = bar(275.58, y1, BLUE_T), bar(482.26, y2, YELLOW_T)
        l1 = Text("algbw", font_size=24).next_to(b1, LEFT).set_x(x0 - 0.8)
        l2 = Text("busbw", font_size=24).next_to(b2, LEFT).set_x(x0 - 0.8)
        v1 = Text("275.6 GBps", font_size=22).next_to(b1, RIGHT)
        v2 = Text("× 1.75 = 482.3", font_size=22).next_to(b2, RIGHT)
        sub = Text("8× H200, all-reduce, 16GiB payload", font_size=22, color=GREY_B)
        sub.next_to(b1, UP, buff=0.3).set_x(0)
        spec = DashedLine([x0 + 450 * sc, y1 + 0.6, 0], [x0 + 450 * sc, y2 - 0.6, 0],
                          color=RED_T)
        sl = Text("NVLink 4 spec: 450 GBps", font_size=22, color=RED_T)
        sl.next_to(spec, DOWN, buff=0.1)

        self.say("bw_a", Write(head))
        pay = Rectangle(width=2.0, height=0.45, fill_color=BLUE_T, fill_opacity=0.85,
                        stroke_width=0).move_to([-2.0, -0.6, 0], aligned_edge=LEFT)
        wire = Rectangle(width=3.5, height=0.45, fill_color=YELLOW_T, fill_opacity=0.85,
                         stroke_width=0).move_to([-2.0, -1.5, 0], aligned_edge=LEFT)
        pl = Text("payload", font_size=22).next_to(pay, LEFT)
        wl = Text("sent per GPU", font_size=22).next_to(wire, LEFT)
        nl = Text("n = 8: 1.75×", font_size=22, color=YELLOW_T).next_to(wire, RIGHT)
        alts = [(2, "n = 2: 1×"), (4, "n = 4: 1.5×"), (8, "n = 8: 1.75×")]
        clock = Arrow([-2.0, -0.05, 0], [0.0, -0.05, 0], buff=0, color=GREY_B,
                      stroke_width=3)
        cl = Text("time", font_size=20, color=GREY_B).next_to(clock, RIGHT, buff=0.1)
        self.say("bw_b", Write(alg), [GrowFromEdge(pay, LEFT), FadeIn(pl)],
                 [GrowArrow(clock), FadeIn(cl)])
        self.say("bw_c", FadeIn(why), [GrowFromEdge(wire, LEFT), FadeIn(wl), FadeIn(nl)],
                 *[[wire.animate.stretch_to_fit_width(2 * 2 * (n - 1) / n).align_to(pay, LEFT),
                    Transform(nl, Text(t, font_size=22, color=YELLOW_T).next_to(
                        [-2.0 + 2 * 2 * (n - 1) / n, -1.5, 0], RIGHT))]
                   for n, t in alts])
        fx = Arrow(pay.get_right() + RIGHT * 0.1, wire.get_right() + RIGHT * 0.1 + UP * 0.1,
                   buff=0, color=YELLOW_T, path_arc=-1.2)
        self.say("bw_d", [FadeOut(why), Write(busb)], FadeIn(facs), Create(fx),
                 Indicate(wire, color=YELLOW_T))
        self.play(FadeOut(VGroup(pay, wire, pl, wl, nl, clock, cl, fx)), run_time=0.5)
        self.say("bw_e", FadeIn(sub), [FadeIn(l1), GrowFromEdge(b1, LEFT)], FadeIn(v1))
        self.say("bw_f", [FadeIn(l2), GrowFromEdge(b2, LEFT)], FadeIn(v2))
        self.say("bw_g", [Create(spec), FadeIn(sl)], Indicate(b2, color=RED_T))

    # 10. SHARP: the switch does the reduction
    def sharp(self):
        head = Text("SHARP: reduce inside the switch", font_size=38).to_edge(UP, buff=0.3)
        sw = RoundedRectangle(corner_radius=0.15, width=4.4, height=1.2, stroke_color=WHITE)
        sw.shift(UP * 1.3)
        swl = Text("NVSwitch", font_size=24).next_to(sw, LEFT, buff=0.2)
        alu = VGroup(Circle(0.3, color=YELLOW_T), Text("+", font_size=36, color=YELLOW_T))
        alu.move_to(sw.get_right() + LEFT * 0.6)
        xs = [-3.3, -1.1, 1.1, 3.3]
        gpus = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=1.5, height=1.0,
                                                stroke_color=GPU_COLORS[i]),
                               Text(f"GPU{i}", font_size=20, color=GPU_COLORS[i])
                               .shift(UP * 0.3)).move_to([x, -1.4, 0])
                        for i, x in enumerate(xs)])
        links = VGroup(*[Line(g.get_top(), [g.get_x() * 0.4, sw.get_bottom()[1], 0],
                              color=GREY_B) for g in gpus])
        data = VGroup(*[cell(f"g{i}", [GPU_COLORS[i]], 0.8, 0.35, 16)
                        .move_to(g.get_center() + DOWN * 0.15) for i, g in enumerate(gpus)])
        copies = [d.copy() for d in data]
        ups = [c.animate.move_to(sw.get_center() + LEFT * 1.6 + RIGHT * 0.85 * i)
               for i, c in enumerate(copies)]
        tot = cell("Σ", GPU_COLORS[:4], 1.2, 0.45, 20).move_to(sw.get_center() + LEFT * 0.4)
        self.say("sh_a", Write(head), [Create(sw), FadeIn(swl), FadeIn(gpus), Create(links)],
                 [FadeIn(alu, scale=1.5), FadeIn(data)])
        self.say("sh_b", LaggedStart(*ups, lag_ratio=0.15))
        self.say("sh_c", [ReplacementTransform(VGroup(*copies), tot),
                          Indicate(alu, scale_factor=1.4)])
        downs = [tot.copy().animate.scale(0.8).move_to(g.get_center() + DOWN * 0.15)
                 for g in gpus]
        cnt = VGroup(Text("ring all-reduce: 2N sends", font_size=26, color=GREY_B),
                     Text("SHARP: N+1 sends", font_size=26, color=YELLOW_T)
                     ).arrange(RIGHT, buff=1.0).to_edge(DOWN, buff=0.35)
        self.say("sh_d", [*downs, FadeOut(data)], FadeIn(cnt[0]), FadeIn(cnt[1]))

    def sharp_numbers(self):
        head = Text("8× H200 all-reduce, 16GiB", font_size=36).to_edge(UP, buff=0.3)
        x0, sc = -3.0, 7 / 500

        def bar(v, y, col, name, txt):
            b = Rectangle(width=v * sc, height=0.5, fill_color=col, fill_opacity=0.85,
                          stroke_width=0).move_to([x0 + v * sc / 2, y, 0])
            n = Text(name, font_size=22).next_to(b, LEFT).set_x(x0 - 1.0)
            t = Text(txt, font_size=20).next_to(b, RIGHT)
            return VGroup(n, b, t)

        a = bar(482.26, 1.0, YELLOW_T, "SHARP", "482.3 (107%)")
        r = bar(367.61, 0.0, BLUE_T, "ring", "367.6 (82%)")
        spec = DashedLine([x0 + 450 * sc, 1.6, 0], [x0 + 450 * sc, -0.5, 0], color=RED_T)
        sl = Text("450", font_size=20, color=RED_T).next_to(spec, UP, buff=0.05)
        note = Text("busbw assumes ring traffic (2N); SHARP sends N+1", font_size=22,
                    color=GREY_B).next_to(r, DOWN, buff=0.3).set_x(0)
        self.say("sh_e", Write(head), [FadeIn(a), Create(spec), FadeIn(sl)], FadeIn(r),
                 FadeIn(note))
        self.play(FadeOut(VGroup(a, r, spec, sl, note)), run_time=0.6)

        gains = [(4, 1.00), (5, 1.14), (6, 1.19), (7, 1.26), (8, 1.29)]
        base = -1.6
        cols = VGroup()
        for i, (n, g) in enumerate(gains):
            x = -4.5 + i * 1.4
            h = max((g - 1) * 10, 0.03)
            b = Rectangle(width=0.9, height=h, fill_color=GREEN_T, fill_opacity=0.85,
                          stroke_width=0).move_to([x, base + h / 2, 0])
            v = Text(f"{g:.2f}x", font_size=22).next_to(b, UP, buff=0.1)
            lab = Text(str(n), font_size=22).next_to([x, base, 0], DOWN, buff=0.15)
            cols.add(VGroup(b, v, lab))
        ax = Line([-5.3, base, 0], [1.8, base, 0], color=GREY_B)
        xl = Text("GPUs in the all-reduce", font_size=20, color=GREY_B)
        xl.next_to(ax, DOWN, buff=0.55)
        ttl = Text("gain from SHARP on H200 (8GiB)", font_size=26).move_to([-1.7, 2.2, 0])
        thr = DashedLine([-3.8, base - 0.1, 0], [-3.8, 1.6, 0], color=YELLOW_T)
        self.play(Transform(head, Text("When does SHARP kick in?", font_size=36)
                            .to_edge(UP, buff=0.3)), run_time=0.6)
        b200 = Text("B200: above 5 GPUs\n(1.23x at 8)", font_size=26, color=ORANGE_T)
        b200.move_to([4.6, 0.3, 0])
        thr2 = DashedLine([-2.4, base - 0.1, 0], [-2.4, 1.6, 0], color=ORANGE_T)
        h200 = Text("H200", font_size=20, color=YELLOW_T).next_to(thr, UP, buff=0.05)
        self.say("sh_f", [FadeIn(ttl), Create(ax), FadeIn(xl)],
                 LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.25),
                 [Create(thr), FadeIn(h200)])
        self.say("sh_g", [FadeIn(b200, shift=LEFT * 0.2), Create(thr2)],
                 Indicate(cols[1], color=ORANGE_T))
        self.play(FadeOut(VGroup(cols, ax, xl, ttl, thr, thr2, h200, b200)), run_time=0.6)
        rows = [("all-reduce, NVLS", "480.0", "107%", YELLOW_T),
                ("all-reduce, ring", "367.2", "82%", WHITE),
                ("all-gather", "361.4", "80%", WHITE),
                ("reduce-scatter", "362.9", "81%", WHITE)]
        tbl = VGroup(*[VGroup(Text(a, font_size=26, color=c), Text(b + " GBps", font_size=26, color=c),
                              Text(p, font_size=26, color=c)) for a, b, p, c in rows])
        for t in tbl:
            t[0].move_to([-2.0, 0, 0], aligned_edge=RIGHT)
            t[1].move_to([1.2, 0, 0])
            t[2].move_to([3.6, 0, 0])
        tbl.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        for t in tbl:
            t[0].set_x(-2.0 - t[0].width / 2)
            t[1].set_x(1.2)
            t[2].set_x(3.6)
        tbl.move_to(ORIGIN)
        sub = Text("8× H200, 16GiB, busbw as % of 450", font_size=22, color=GREY_B)
        sub.next_to(tbl, DOWN, buff=0.45)
        self.say("sh_h", [FadeIn(tbl[:2]), FadeIn(sub)],
                 LaggedStart(FadeIn(tbl[2]), FadeIn(tbl[3]), lag_ratio=0.4))

    # 11. outro: strategies -> collectives
    def outro(self):
        head = Text("The vocabulary of distributed training", font_size=38).to_edge(UP, buff=0.4)
        rows = [("DDP", "all-reduce (grads)", "2× params"),
                ("ZeRO-3", "2× all-gather (weights) + reduce-scatter (grads)", "3× params"),
                ("PP", "send / recv (activations)", ""),
                ("SP, MoE", "all-to-all", "")]
        grp = VGroup()
        for i, (a, b, c) in enumerate(rows):
            y = 1.6 - i * 1.0
            r = VGroup(Text(a, font_size=26, color=GPU_COLORS[i]).move_to([-4.6, y, 0]),
                       Text(b, font_size=22).move_to([0.6, y, 0]),
                       Text(c, font_size=24, color=YELLOW_T).move_to([5.6, y, 0]))
            r[0].set_x(-4.3 - r[0].width / 2)
            r[1].set_x(-3.9 + r[1].width / 2)
            if c:
                r[2].set_x(6.4 - r[2].width / 2)
            grp.add(r)
        moral = Text("all-reduce ≈ 2× payload,  all-gather / reduce-scatter ≈ 1×",
                     font_size=28, color=YELLOW_T).to_edge(DOWN, buff=0.7)
        ul = Line(LEFT * 4, RIGHT * 4, color=YELLOW_T).next_to(head, DOWN, buff=0.15)
        self.say("en_a", Write(head), Create(ul))
        self.say("en_b", FadeIn(grp[0][:2], shift=RIGHT * 0.2))
        self.say("en_c", FadeIn(grp[1][:2], shift=RIGHT * 0.2), Circumscribe(grp[1][1], color=GPU_COLORS[1]))
        self.say("en_d", FadeIn(grp[2][:2], shift=RIGHT * 0.2))
        self.say("en_e", FadeIn(grp[3][:2], shift=RIGHT * 0.2))
        self.say("en_f", [FadeIn(grp[0][2]), FadeIn(grp[1][2])], Write(moral),
                 Circumscribe(VGroup(grp[0][2], grp[1][2]), color=YELLOW_T),
                 Circumscribe(moral, color=YELLOW_T))


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
