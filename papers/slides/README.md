# Talk decks for Papers 1–4 (v2)

One `.pptx` per manuscript + a matching `*_SCRIPT.md` (the audience script), built by
[`build_slides.py`](build_slides.py) (engine in [`deck_kit.py`](deck_kit.py)).
The full rationale and design system is in [`PLAN.md`](PLAN.md).

Rebuild everything (cheap, one pass):

```bash
python build_slides.py     # -> paperN_*.pptx + paperN_*_SCRIPT.md  (needs python-pptx, Pillow)
```

## What's here

| File | What it is |
|---|---|
| `PLAN.md` | the production plan + researched craft + design system |
| `paperN_*.pptx` | the deck (assertion-evidence; figures + test results; scripted notes) |
| `paperN_*_SCRIPT.md` | the speaker script, one block per slide (also in the deck's notes pane) |
| `build_slides.py` | per-paper content (assertions, evidence, analogies, scripts) |
| `deck_kit.py` | the style engine (palette, layout, cards, notes, SCRIPT export) |

## The style, in one paragraph

Each content slide is a **one-sentence assertion** supported by **visual evidence** — a
real figure or a built result/test panel — not a bullet list (Alley's assertion-evidence
method). Each deck is an **And-But-Therefore** story (Olson) that opens with one
**analogy**. The **script** lives in the speaker notes (and `*_SCRIPT.md`), written to be
*spoken*. Colour follows **60-30-10** on a soft off-white background with one per-paper
identity hue and WCAG-safe contrast; **verified / falsified / open** is shown with a
colour **and** a text label. See `PLAN.md` for the details and sources.

## Before you present

- Replace `[your name] · [affiliation] · [ORCID]` on each title slide.
- Read from `*_SCRIPT.md` or the notes pane; let the slide carry the visual, say the rest.
- ~9–12 slides ≈ a 10–15 min talk. To restyle, edit the palette/layout in `deck_kit.py`
  and rerun.
