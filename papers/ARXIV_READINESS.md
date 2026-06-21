# arXiv readiness review — Papers 1–4

A practical assessment of (a) how arXiv submission actually works, and (b) whether
the four manuscripts in `papers/` are ready to upload and fairly represent the
research. Honest verdict up front, then the evidence.

## Verdict

**Scientifically: plausibly arXiv-worthy.** The four papers contain original,
self-contained results with real-data validation and explicit falsifications — the
kind of contribution arXiv's `physics.flu-dyn`, `physics.geo-ph`, and
`physics.ao-ph` communities accept.

**Mechanically: now content-ready (updated 2026-06-15, PR #1).** The two hard blockers
and the should-fix items below are resolved without touching the science — bylines added,
complete Crossref-verified bibliographies installed, key figures embedded, internal jargon
neutralized. Remaining before upload: (a) package an arXiv source tarball plus a xelatex
font-compatibility check, and (b) **endorsement**, now the gating external step (see below).

## How arXiv submission works (the process)

- **Account + endorsement.** Submission is free, but a first-time submitter (or a
  submitter new to a category) must be **endorsed**. As of **21 Jan 2026** an academic
  email alone no longer qualifies a new author: you need either (1) an institutional email
  **plus** a prior arXiv paper in the same subject class, or (2) a **personal endorsement**
  from an established author in that category's endorsement domain (`physics.flu-dyn` and
  `physics.geo-ph` are separate domains). arXiv staff cannot endorse. Endorsement is *not*
  peer review — it certifies the work is topical and the author is a real member of the field.
- **Self-submission.** Authors submit their own work (third-party submission is
  restricted). No anonymous submissions.
- **Upload LaTeX source** (preferred over PDF-only): arXiv recompiles it. Include
  ancillary files (data, code pointers) rather than appending large non-text blocks.
- **Moderation.** Every submission is screened by volunteer subject-matter
  moderators for topicality, originality/novelty/significance, completeness, neutral
  scholarly language, and no misrepresented data. They may reclassify the category.
- **Metadata.** Title + author list + abstract are entered separately and must match
  the PDF; check them carefully before the final submit.
- **Revisions** are *replacements* of the same arXiv ID (v1, v2, …) — never a new
  submission.

## Format bar arXiv enforces

Title and authorship (no anonymous) · **complete references** · code/data links must
resolve to a **public** repo · machine-readable · single-spaced · 10–14 pt type ·
≥1″ margins · **no slides/posters in the article body** (use ancillary files).

## Where the K papers stand (item by item)

| Item | Status | Note |
|---|---|---|
| Type size / margins / single-spaced | [PASS] | `article` 11 pt, 1″ margins (`preamble.sty`). |
| LaTeX source builds | [PASS] | `papers/tex/*.tex` compiles with xelatex; PDFs build clean. |
| Public code/data | [PASS] | repo is **public** (`github.com/abdh0saifelden2-debug/K`); each paper has a `Reproduce` section. |
| Novelty / scholarly interest | [PASS] | substantial, with honest falsifications; fits `physics.flu-dyn` / `physics.geo-ph`. |
| Author + affiliation | [PASS] | Byline added (Abdelrahman Saifelden, AIET) to all four via the `md2tex` template. ORCID optional, not yet supplied. |
| Complete references | [PASS] | Prose name-lists replaced with full bibliographies — 18/12/10/16 entries for Papers 1–4, each verified against Crossref (title, journal, volume, pages, year, DOI). |
| Embedded figures | [PASS] | Key plots embedded with captions — Paper 1: 4 (256² benchmark + solver), Paper 2: 1 (double-diffusive Nu), Paper 4: 2 (RTN on Bedmap2 + BedMachine cross-check). Paper 3's referenced plots were JSON-only; the dangling refs were removed. |
| Neutral language | [PASS] | Paper 4's "strict claim-tag discipline" replaced with plain wording; internal `[VERIFIED]/[HYP]` tags already stripped. |
| Abstract block / date | [WARN] minor | Abstract is a running section, not `\begin{abstract}`; `\date{}` is empty. |
| Reader-standalone-ness | [WARN] minor | Very dense; heavy inline code + Unicode math, long titles, many internal `§`/`NR` cross-references. Readable, but reviewer-dense — a gentler intro and fewer internal pointers would help it stand alone. |
| Endorsement | [FAIL] gating | The binding external step (21 Jan 2026 policy): needs a personal endorsement from an established `physics.flu-dyn`/`physics.geo-ph` author, since the author has no prior in-category arXiv paper. |

## Does it "represent the research" well?

Mostly yes — the verify/falsify discipline and the public, reproducible repo are
genuine strengths, and the falsifications are presented honestly rather than buried.
Two cautions: (1) a reader cannot reproduce the central claims **without** the repo,
so cite the exact commit and tag a release (`git tag`) for permanence; (2) the
internal framing (claim tags, NR cross-references) occasionally leaks through and
should be smoothed so each paper stands on its own.

## Recommended path to upload (smallest correct set)

1. Add author, affiliation, ORCID; set a date; wrap the abstract in `\begin{abstract}`.
2. Build a real BibTeX bibliography (full entries for the already-named works).
3. Embed the 2–4 key figures per paper (the PNGs already exist in the repo), or
   remove figure references that won't be shown.
4. Neutralize the "claim-tag discipline" phrasing in Paper 4's abstract.
5. Tag a release / pin the commit, and use full GitHub URLs in the Reproduce sections.
6. Secure an endorser in the chosen category; submit LaTeX source; verify metadata.

Items 1–5 are mechanical and do not change any result; I can do them on request.
