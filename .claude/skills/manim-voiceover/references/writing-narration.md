# Writing narration that explains (and doesn't sound like AI)

Two primary sources, both 3Blue1Brown transcripts:
- [What makes a great math explanation?](https://www.youtube.com/watch?v=cDofhN-RJqg) (SoME2 results): Grant's advice
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc): what his narration actually sounds like

Quotes below are from the auto-captions of those videos.

## Grant's advice

**Motivation, macro scale: hook with something concrete, then get moving.**
- "Motivation is critical, but it doesn't have to take long, and often what actually keeps the viewer engaged is to get right to the point and leave any commentary about broader themes and connections to the end."
- "If you can, motivate using clear examples, not sweeping statements or promises of what is to come."
- Good hooks he cites: a specific puzzling observation (a plane's pitch falling), a tangible problem (how panorama stitching works), a "nerd sniping" puzzle.

**Motivation, micro scale: every new idea needs a reason to be there.**
- Show the idea before the equation, so the equation "arrives only once it's articulating something that already exists at least loosely in the viewer's mind".
- Derive the definition from properties you'd want it to have, instead of handing it "down from on high".
- "Start with a naive but flawed solution, and then progressively refine it." Each flaw motivates the next idea.

**Clarity.**
- "If motivating a lesson determines how much attention and focus the viewer is willing to give you, clarity determines how quickly you burn through that focus."
- Keep "one or two examples front and center", play with them, push them into edge cases, "before general rules are presented".
- When making a general point, show a concrete example on screen, without "overemphasizing its importance".
- Music, if any, stays "decidedly in the background" during technical parts.

**Novelty and memorability.** A unique perspective matters more than a unique style. Memorable means a question that's fun to think about or an aha moment that sticks.

## What his narration sounds like

From the attention video. Copy these habits:

| Habit | Example |
|---|---|
| "You and I" as co-investigators | "in this chapter you and I will dig into what this attention mechanism is" |
| Say when you're simplifying | "let's simplify by pretending that tokens are always just words" |
| Correct yourself in the open | "Actually, that's not quite true. They also encode the position of the word." |
| Name the difficulty up front | "a lot of people find the attention mechanism... very confusing, so don't worry if it takes some time" |
| Examples before machinery | "before we dive into the computational details... it's worth thinking about a couple examples" |
| Defer, explicitly | "There's a lot more to say about... but right now, all you need to know is..." |
| Admit what's made up | "To be clear, I'm making up this example... just to illustrate" |
| "Imagine / consider / you could imagine" | "Imagine, for example, that the text you input is most of an entire mystery novel" |
| Plain, slightly informal words | "a lot of", "pretty", "much, much richer", "playing the deep learning game" |

Sentences are full and flowing, with clauses, and they read like someone thinking out loud. They are not chopped into punchy fragments.

## AI tells to cut

These showed up in our first draft of video 3. Each one sounds scripted:

| Tell | Draft line | Rewrite |
|---|---|---|
| Fragment triplets | "Same silicon, same clock. Just 2.7 percent more SMs." | "They're the same silicon running at the same clock, and one just has 2.7 percent more SMs." |
| Drumroll colons / labels | "The moral: eight-thirds is a suggestion, not a law." | "So eight-thirds is really just a suggestion, and it's worth searching nearby for a size the hardware likes." |
| "It's not X. It's Y." | "it's not a line at all. It's a staircase." | "but when you actually plot it, you get this staircase." |
| Announcing the excitement | "Which raises a fun question." / "Now let's watch this bite in a real model." | Ask the question directly, or just show the model. |
| Question, then a one-word slap | "...get a discount? Nope." | "...get a discount? Unfortunately not: it still runs in full." |
| Grand opener | "a lot of odd-looking advice about model sizes starts to make sense" | Open on the specific observation instead (two matmul sizes, the timings). |

Other things to watch for: "Here's the thing", "Let's dive in", "game-changer", recaps of what was just shown, and every line ending on a punchline.

## Process

1. Find the single concrete example (or two) that the whole video can keep coming back to.
2. Write the hook as that example's puzzling observation, two or three lines at most.
3. For each new idea, write down the question it answers before writing the line that introduces it.
4. Read the script out loud. If a line sounds like a slide title or a tweet, rewrite it as something you'd say to a friend at a whiteboard.
5. Keep facts strictly to the sources. A conversational tone is not license to add claims.

## TTS notes (Kokoro)

- Yes/no questions rise at the end ("Do more SMs give you more TFLOPS?"). Questions starting with what/how/why fall, the same as in normal speech, so a "?" changes nothing there. Measured: ~240Hz rising vs ~155Hz falling for the same yes/no sentence with "?" vs ".".
- Write numbers the way they should be spoken if Kokoro reads them oddly (e.g. "eight-thirds" rather than "8/3").
- Keep each line to one visual step. Long lines leave the screen frozen.
