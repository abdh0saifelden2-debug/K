# LaTeX builds of the four manuscripts

The Markdown drafts in `../` (`paper1_closure_theory.md` … `paper4_subglacial_hydrology_forecasts.md`)
are the source of truth. `md2tex.py` converts a draft into a standalone LaTeX
file that compiles with **XeLaTeX** + `preamble.sty`.

## Rebuild

```sh
make            # regenerate all four .tex and compile to .pdf (2 passes each)
make clean      # remove LaTeX aux/log artifacts
```

or manually, per paper:

```sh
python3 md2tex.py ../paper1_closure_theory.md paper1_closure_theory.tex
xelatex -interaction=nonstopmode paper1_closure_theory.tex   # run twice
```

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
