#!/usr/bin/env python3
"""Build a manuscript in the Physical Review Fluids (REVTeX 4.2) class.

Usage: python3 build_prfluids.py [paper_stem]   (default: paper1_closure_theory)

Reuses the repo's md2tex.py math/markup conversion, then wraps the converted body
in `revtex4-2` with the `prfluids` option (pdflatex). md2tex already emits
`\\section*{N. Title}` with manual numbers, which we keep verbatim so the paper's
own cross-references (e.g. "section 5b") stay correct. We provide the few macros
the shared (xelatex-only) preamble.sty would have supplied (\\code, fancyvrb,
booktabs) and apply the same pdflatex unicode/inline-math-dash safety as the
Copernicus build.

Each source .md uses the layout: `## Abstract`, `## 1. Introduction`, ... numbered
sections ..., `## Data and code availability`, `## References`.
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
AFFIL = "Alexandria Higher Institute of Engineering and Technology (AIET), Alexandria, Egypt"

PAPERS = {
    "paper1_closure_theory": True,
}

STEM = sys.argv[1] if len(sys.argv) > 1 else "paper1_closure_theory"
if STEM not in PAPERS:
    raise SystemExit(f"unknown paper stem {STEM!r}; choose from {sorted(PAPERS)}")
SRC = os.path.join(os.path.dirname(TEXDIR), f"{STEM}.md")
DST = os.path.join(HERE, f"{STEM}.tex")

# Unicode that pdflatex+inputenc(utf8) does not define out of the box; map to LaTeX
# outside verbatim and ASCII inside (where macro expansion would print literally).
_UNI_MATH = {
    "\U0001D4A6": r"\ensuremath{\mathcal{K}}", "\u2093": r"\ensuremath{{}_{x}}",
    "\u03c4": r"\ensuremath{\tau}", "\u03c6": r"\ensuremath{\varphi}",
    "\u0398": r"\ensuremath{\Theta}", "\u03bd": r"\ensuremath{\nu}",
    "\u2207": r"\ensuremath{\nabla}", "\u0393": r"\ensuremath{\Gamma}",
    "\u03b3": r"\ensuremath{\gamma}",
}
_UNI_ASCII = {
    "\U0001D4A6": "K", "\u2093": "x", "\u03c4": "tau", "\u03c6": "phi",
    "\u0398": "Theta", "\u03bd": "nu", "\u2207": "grad", "\u0393": "Gamma",
    "\u03b3": "gamma",
}


def _math_dash_safe(seg: str) -> str:
    def fix(m):
        inner = m.group(1).replace("\u2014", r"\mbox{---}").replace("\u2013", r"\mbox{--}")
        return "$" + inner + "$"
    return re.sub(r"\$([^$]*)\$", fix, seg)


def unicode_safe(tex: str) -> str:
    parts = re.split(r"(\\begin\{verbatim\}.*?\\end\{verbatim\}|\\begin\{Verbatim\}.*?\\end\{Verbatim\})",
                     tex, flags=re.S)
    out = []
    for seg in parts:
        if seg.startswith(r"\begin{verbatim}") or seg.startswith(r"\begin{Verbatim}"):
            for k, v in _UNI_ASCII.items():
                seg = seg.replace(k, v)
        else:
            for k, v in _UNI_MATH.items():
                seg = seg.replace(k, v)
            seg = _math_dash_safe(seg)
        out.append(seg)
    return "".join(out)


def slice_between(md, start_pat, end_pat):
    s = re.search(start_pat, md)
    e = re.search(end_pat, md[s.end():]) if s else None
    if not s:
        return ""
    return md[s.end(): s.end() + e.start()] if e else md[s.end():]


def postprocess_body(tex: str) -> str:
    tex = re.sub(r"^\\medskip\\hrule\\medskip\s*$", "", tex, flags=re.M)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


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

    abstract_tex = postprocess_body(md2tex.convert(abstract_md))
    body_tex = postprocess_body(md2tex.convert(body_md))
    dataavail_tex = postprocess_body(md2tex.convert(dataavail_md))

    refs = [r.strip() for r in re.split(r"\n\s*\n", refs_md) if r.strip()]
    bib = ["\\begin{thebibliography}{99}"]
    for k, r in enumerate(refs, 1):
        one = " ".join(line.strip() for line in r.split("\n"))
        bib.append("\\bibitem{ref%d}" % k)
        bib.append(md2tex.inline(one))
        bib.append("")
    bib.append("\\end{thebibliography}")
    bib_tex = "\n".join(bib)

    doc = rf"""\documentclass[aps,prfluids,preprint,superscriptaddress,nofootinbib]{{revtex4-2}}

\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{fvextra}}
\fvset{{breaklines=true,fontsize=\footnotesize}}
\graphicspath{{{{../}}}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}

\begin{{document}}

\title{{{title}}}

\author{{Abdelrahman Saifelden}}
\email[Correspondence: ]{{{CORR_EMAIL}}}
\affiliation{{{AFFIL}}}

\date{{\today}}

\begin{{abstract}}
{abstract_tex}
\end{{abstract}}

\maketitle

{body_tex}

\section*{{Data and code availability}}
{dataavail_tex}

{bib_tex}

\end{{document}}
"""
    open(DST, "w", encoding="utf-8").write(unicode_safe(doc))
    print("wrote", DST, "(stem:", STEM + ")")
    print("sections:", len(re.findall(r"\\section\*\{", body_tex)),
          "| refs:", len(refs))


if __name__ == "__main__":
    build()
