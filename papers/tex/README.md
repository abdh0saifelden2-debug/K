# LaTeX builds of the manuscripts

> **Source of truth (updated 2026-07):** the `.tex` files in this folder are now
> **canonical** for submission content. Since commit `c2e8784` the NR-promotion
> paragraphs, Paper 1's §5c/§5d/§5e additions, and the abstract trims were made
> in the `.tex` (and `cryosphere/`, `prfluids/` variants) **only**; the Markdown
> drafts in `../` are the historical sources and no longer carry the latest text.
> Regenerating a `.tex` from its `.md` therefore **destroys content** — the
> Makefile's automatic md→tex rule has been removed; use
> `make regen-from-md P=<paper>` only if you really mean it.

The Markdown drafts in `../` (`paper1_closure_theory.md` … `paper4b_sliding_law_field_test.md`)
were the original sources. `md2tex.py` converts a draft into a standalone LaTeX
file that compiles with **XeLaTeX** + `preamble.sty`.

## Rebuild

```sh
make            # compile all six .tex to .pdf (2 passes each)
make arxiv      # package per-paper arXiv source tarballs into arxiv/ (upload set only)
make clean      # remove LaTeX aux/log artifacts
```

or manually, per paper:

```sh
xelatex -interaction=nonstopmode paper1_closure_theory.tex   # run twice
```

## arXiv upload set

`make arxiv` packages **P1, P2, P3, P4a, P4b** (tex + `preamble.sty` + the
figures each paper references). The **P4 omnibus**
(`paper4_subglacial_hydrology_forecasts`) is *superseded for submission* by its
single-thesis children 4a/4b (see the provenance note at its top) and is
deliberately excluded — uploading both the omnibus and its children would be
self-overlapping submissions under arXiv moderation.

## Requirements

- TeX Live with `xelatex`, `fontspec`, `unicode-math`, `fvextra`, `booktabs`,
  `tabularx`, `enumitem`, `microtype`, `hyperref`.
- Fonts: Latin Modern (Roman + Math) and **DejaVu Sans Mono** (Unicode-rich
  monospace used for code spans / verbatim).

## What the converter handles

- Claim tags (`[VERIFIED]`, `[DERIVED]`, `[HYP]`, `[FALSIFIED]`, `[LIT]`,
  `[MEASURED]`, …) → colour-coded macros, and `[cite: …]` slots → citation marks
  (no references are invented).
- Inline mathematics written in Unicode (Greek, operators, relations, arrows,
  sub/superscripts, hats/bars/dots, `√`, matrices) → real LaTeX math.
- Code/paths/commands → breakable `\texttt`; pipe tables (pipes inside math
  spans are respected) → `booktabs`/`tabularx`; fenced blocks → wrapped
  `fvextra` verbatim; soft-wrapped paragraphs/lists/quotes are re-joined before
  inline parsing; straight quotes → typographic quotes.
