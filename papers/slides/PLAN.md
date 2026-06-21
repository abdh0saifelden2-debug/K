# Talk decks v2 — production plan

A concrete plan, grounded in researched presentation craft, to rebuild the four
paper decks properly: an audience script, intuitive examples, our real figures **and**
tests, and a pleasant, accessible palette. Implemented by `build_slides.py` (v2).

## What was wrong with v1 (and why)

v1 used PowerPoint's default **phrase-headline + bullet-list** — the exact anti-pattern
Michael Alley's research identifies as the biggest weakness of technical talks. It had
no audience script, no intuitive examples, didn't show the tests, and used a plain white
body with no designed palette. This plan fixes all five.

## The craft (researched, with sources)

- **Assertion–Evidence** (Alley, Penn State). Every content slide is a **one-sentence
  assertion** (the message, top-left, ~28 pt bold sans-serif) supported by **visual
  evidence** — a figure, graph, diagram, or result — **not** a bullet list. Secondary
  detail goes into the **speaker notes**, not the slide. Keep on-slide text to 1–2 lines,
  ≤4 items, with generous whitespace.
- **ABT narrative** (Olson, "And · But · Therefore"). Each talk is structured as
  *"we know A **and** B, **but** here's the problem, **therefore** we did X"* — for focus
  and a real story arc, not a topic dump.
- **Write-to-speak script.** The slide shows the *insight*; the **script** (in the notes
  pane + a `SCRIPT.md`) carries the *explanation*, a transition, and one **analogy** per
  hard idea. Conversational, short sentences, rhetorical questions, breathing space.
- **Color (60-30-10 + WCAG).** 60% soft background · 30% structure/headings · 10% accent
  for key results. 3–4 colors max; blue/green/teal/gray read as professional and survive
  projectors. Contrast ≥ 4.5:1 (we use a softened near-black on warm off-white, not stark
  black-on-white). Never encode meaning by color alone — always pair with a label.

## Design system (applied by the generator)

- **Canvas** 16:9. **Background**: soft warm off-white `#F6F7F9` with a thin per-paper
  accent rule (pleasant, projector-safe, screenshots well). **Ink** `#1E232B` (softened).
- **Per-paper identity colour** (the 30% structure / 10% accent), one hue each:
  P1 navy `#1F3B73` · P2 teal `#0F6E6E` · P3 plum `#5B2C83` · P4 forest `#145A32`.
- **Status colours** (always with a text label): verified `#1B7F37` · falsified `#B3261A`
  · open `#B57A00`.
- **Type**: assertion headline 26–28 pt bold; on-slide support 16–18 pt; captions 12 pt.
- **Slide model**: assertion headline (top, left-justified, on a soft tint band) → one
  **dominant visual** (a real figure **or** a built "evidence" panel of metric cards /
  a small result table for the *tests*) → 1-line caption → status chip when relevant.
- **Evidence = figures AND tests.** Where a slide makes a claim we back it with either the
  real plot (PNG from the repo) or a result callout showing the actual numbers
  (e.g. "median 6 km vs 222 km · r = 0.970 · 10/10 tests pass").
- **Speaker notes** on every slide carry the script paragraph; a `SCRIPT.md` mirrors it.

## Narrative per deck (ABT) and the analogy that opens it

- **P1 — closure repair.** *Turbulence models close small scales with a simple
  down-gradient rule **and** it's convenient, **but** that rule secretly throws away the
  flow's memory, **therefore** we show it's the memoryless limit of an exact memory kernel
  and put the memory back.* Analogy: a smoothing/averaging app that erases the echo in a
  room — fine until the echo is the signal.
- **P2 — melt ceiling.** *We worry turbulence speeds up basal melt **and** it sounds
  plausible, **but** when we test it the melt is capped by a thin conductive skin,
  **therefore** turbulence doesn't raise net melt — there's a ceiling.* Analogy: a thick
  wool blanket (conduction) sets how fast heat crosses, no matter how hard the wind blows
  outside.
- **P3 — parity break.** *Scallops on the bed drift downstream **and** we'd like our model
  to capture it, **but** a symmetric (down-gradient) model literally can't point
  downstream, **therefore** migration is a clean, measurable test the standard closure
  fails.* Analogy: a perfectly symmetric clock with no "tick vs tock" can't tell you which
  way time runs.
- **P4 — falsifiable forecasts.** *Subglacial hydrology drives the ice **and** we have
  predictions, **but** most aren't checkable on real data, **therefore** we make three and
  hold each to verify/falsify on open Antarctic data.* Analogy: three bets, and we show our
  hand on every one — including the one we lost.

## Reference-first checkpoint

Build **Paper 4** (the richest) as the full reference deck — new palette,
assertion-evidence slides, embedded figures + test callouts, complete speaker script —
confirm the direction, then roll the identical system across Papers 1–3. (This avoids
v1's mistake of mass-producing before the style is right.)

## Deliverables

`paperN_*.pptx` (assertion-evidence, scripted notes, figures + tests, pleasant palette) ·
`paperN_SCRIPT.md` (the audience script) · PDF previews · `build_slides.py` v2 for cheap
iteration.
