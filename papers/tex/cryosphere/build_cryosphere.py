#!/usr/bin/env python3
"""Build a manuscript in the Copernicus (The Cryosphere) LaTeX class.

Usage: python3 build_cryosphere.py [paper_stem]   (default: paper3_scallop_parity_break)

Reuses the repo's md2tex.py math/markup conversion, then post-processes the body
into the `copernicus` document class (pdflatex): maps the first section to
\\introduction, strips manual section numbers, swaps booktabs rules for \\hline,
Verbatim->verbatim, \\code->\\texttt, drops decorative rules, and assembles the
Copernicus front/back matter + a thebibliography. Keeps the abdh repo URL.

Each source .md must use the section layout: `## Abstract`, `## 1. Introduction`,
... numbered sections ..., `## Data and code availability`, `## References`.
"""
from __future__ import annotations
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEXDIR = os.path.dirname(HERE)                      # papers/tex
sys.path.insert(0, TEXDIR)
import md2tex  # noqa: E402

CORR_EMAIL = "Abdulrahman.Saifelden22011411@aiet.edu.eg"  # author's AIET account
RUNNING_AUTHOR = "A. Saifelden"
AFFIL = "Alexandria Higher Institute of Engineering and Technology (AIET), Alexandria, Egypt"

# Per-paper short running titles (full titles are auto-extracted from the `# ` line).
PAPERS = {
    "paper3_scallop_parity_break":
        "A temperature-free morphological test of interface heat transport",
    "paper2_subglacial_melt_ceiling":
        "An area-partition ceiling on scalloped basal melt",
    "paper4a_hydraulic_memory_kernel":
        "A hydraulic Mori--Zwanzig kernel for subglacial surge lag",
    "paper4b_sliding_law_field_test":
        "A field-measurable regularized-Coulomb sliding law",
}

STEM = sys.argv[1] if len(sys.argv) > 1 else "paper3_scallop_parity_break"
if STEM not in PAPERS:
    raise SystemExit(f"unknown paper stem {STEM!r}; choose from {sorted(PAPERS)}")
RUNNING_TITLE = PAPERS[STEM]
SRC = os.path.join(os.path.dirname(TEXDIR), f"{STEM}.md")
DST = os.path.join(HERE, f"{STEM}.tex")


def slice_between(md, start_pat, end_pat):
    s = re.search(start_pat, md)
    e = re.search(end_pat, md[s.end():]) if s else None
    if not s:
        return ""
    return md[s.end(): s.end() + e.start()] if e else md[s.end():]


def postprocess_body(tex: str) -> str:
    # drop decorative horizontal rules (markdown '---')
    tex = re.sub(r"^\\medskip\\hrule\\medskip\s*$", "", tex, flags=re.M)
    # first section -> \introduction
    tex = re.sub(r"\\section\*\{\s*1\.\s*Introduction\s*\}", r"\\introduction", tex, count=1)
    # remaining numbered sections -> \section{Title} (strip "N. ")
    tex = re.sub(r"\\section\*\{\s*\d+\.\s*([^}]*)\}", r"\\section{\1}", tex)
    tex = re.sub(r"\\subsection\*\{", r"\\subsection{", tex)
    tex = re.sub(r"\\subsubsection\*\{", r"\\subsubsection{", tex)
    # booktabs rules -> \hline (copernicus.cls has no booktabs)
    for r in ("toprule", "midrule", "bottomrule"):
        tex = tex.replace("\\" + r, "\\hline")
    # fancyvrb Verbatim -> built-in verbatim
    tex = tex.replace(r"\begin{Verbatim}", r"\begin{verbatim}")
    tex = tex.replace(r"\end{Verbatim}", r"\end{verbatim}")
    # custom \code macro -> \texttt (keeps nested \allowbreak{})
    tex = tex.replace(r"\code{", r"\texttt{")
    # collapse 3+ blank lines
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


# Unicode that pdflatex+inputenc(utf8) does not define out of the box. md2tex
# converts these in prose/math, but a few leak through inside \texttt and code
# listings; map them to LaTeX outside verbatim and to ASCII inside verbatim
# (where macro expansion would print literally).
_UNI_MATH = {
    "\U0001D4A6": r"\ensuremath{\mathcal{K}}",  # 𝒦 script K
    "\u2093": r"\ensuremath{{}_{x}}",            # ₓ subscript x
    "\u03c4": r"\ensuremath{\tau}",               # τ
    "\u03c6": r"\ensuremath{\varphi}",            # φ
}
_UNI_ASCII = {
    "\U0001D4A6": "K", "\u2093": "x", "\u03c4": "tau", "\u03c6": "phi",
}


def _math_dash_safe(seg: str) -> str:
    # en/em dashes inside inline math ($...$) are invalid in math mode under
    # pdflatex (inputenc -> \textendash/\textemdash, silently dropped); these are
    # numeric ranges, so render them as text dashes. All math here is inline and
    # balanced per line (no display/equation envs), so a $...$ scan is safe.
    def fix(m):
        inner = m.group(1).replace("\u2014", r"\mbox{---}").replace("\u2013", r"\mbox{--}")
        return "$" + inner + "$"
    return re.sub(r"\$([^$]*)\$", fix, seg)


def unicode_safe(tex: str) -> str:
    parts = re.split(r"(\\begin\{verbatim\}.*?\\end\{verbatim\})", tex, flags=re.S)
    out = []
    for seg in parts:
        if seg.startswith(r"\begin{verbatim}"):
            for k, v in _UNI_ASCII.items():
                seg = seg.replace(k, v)
        else:
            for k, v in _UNI_MATH.items():
                seg = seg.replace(k, v)
            seg = _math_dash_safe(seg)
        out.append(seg)
    return "".join(out)


def build():
    md = open(SRC, encoding="utf-8").read()
    title = md2tex.inline(md2tex.title_of(md))

    abstract_md = slice_between(md, r"##\s+Abstract\s*\n", r"\n##\s+").strip()
    abstract_md = re.sub(r"\n-{3,}\s*$", "", abstract_md).strip()
    body_md = slice_between(md, r"\n##\s+1\.\s*Introduction",
                            r"\n##\s+Data and code availability")
    body_md = "## 1. Introduction" + body_md
    dataavail_md = slice_between(md, r"##\s+Data and code availability\s*\n",
                                 r"\n##\s+References").strip()
    refs_md = slice_between(md, r"##\s+References\s*\n", r"\Z").strip()

    abstract_tex = md2tex.convert(abstract_md).strip()
    body_tex = postprocess_body(md2tex.convert(body_md))
    # Block conversion preserves multi-paragraph statements and bulleted data
    # citations (with DOIs); \codedataavailability is \long, so itemize is fine.
    dataavail_tex = postprocess_body(md2tex.convert(dataavail_md))

    refs = [r.strip() for r in re.split(r"\n\s*\n", refs_md) if r.strip()]
    bib = ["\\begin{thebibliography}{}"]
    for k, r in enumerate(refs, 1):
        one = " ".join(line.strip() for line in r.split("\n"))
        bib.append("\\bibitem{ref%d}" % k)
        bib.append(md2tex.inline(one))
        bib.append("")
    bib.append("\\end{thebibliography}")
    bib_tex = "\n".join(bib)

    doc = rf"""\documentclass[tc, manuscript]{{copernicus}}

\graphicspath{{{{../}}}}

\begin{{document}}

\title{{{title}}}

% Corresponding-author email: author's AIET institutional account.
\Author[1]{{Abdelrahman}}{{Saifelden}}

\affil[1]{{{AFFIL}}}

\runningtitle{{{RUNNING_TITLE}}}
\runningauthor{{{RUNNING_AUTHOR}}}
\correspondence{{Abdelrahman Saifelden ({CORR_EMAIL})}}

\received{{}}
\pubdiscuss{{}}
\revised{{}}
\accepted{{}}
\published{{}}

\firstpage{{1}}
\maketitle

\begin{{abstract}}
{abstract_tex}
\end{{abstract}}

{body_tex}

\codedataavailability{{{dataavail_tex}}}

\authorcontribution{{A. Saifelden conceived the study, performed all computations and analysis, and wrote the manuscript.}}

\competinginterests{{The author declares that there are no competing interests.}}

{bib_tex}

\end{{document}}
"""
    open(DST, "w", encoding="utf-8").write(unicode_safe(doc))
    print("wrote", DST, "(stem:", STEM + ")")
    print("sections:", len(re.findall(r"\\section\{", body_tex)) + 1,
          "| refs:", len(refs))


if __name__ == "__main__":
    build()
