#!/usr/bin/env python3
"""Tailored Markdown -> LaTeX converter for the K manuscript drafts.

Targets the specific patterns these papers use: claim tags ([VERIFIED] ...),
[cite: ...] slots, pipe tables, `>` blockquotes, inline `code`/math spans,
fenced code, and heavy inline Unicode mathematics. Output compiles with
xelatex + preamble.sty.

Design notes:
  * Soft-wrapped source lines are JOINED per logical block (paragraph / list
    item / blockquote) before inline parsing, so bold/italic/code spans that
    span a source line break are handled correctly.
  * Inline `code` spans are classified as verbatim code (paths, commands,
    snake_case/CamelCase identifiers) -> \\texttt, or as mathematics -> $...$
    with a full Unicode->LaTeX translation (Greek, operators, relations,
    arrows, sub/superscripts, hats/bars/dots).
  * Stray Unicode math characters in prose are wrapped in \\ensuremath.
"""
from __future__ import annotations
import re
import sys

# ----------------------------------------------------------------------------
# Claim tags / citation slots
# ----------------------------------------------------------------------------
TAGS = ("VERIFIED", "DERIVED", "FALSIFIED", "MEASURED", "HYP", "LIT",
        "NOT EARNED", "REAL DATA", "FUTURE")
TAGMACRO = {"VERIFIED": r"\ctV", "DERIVED": r"\ctD", "HYP": r"\ctH",
            "FALSIFIED": r"\ctF", "LIT": r"\ctL", "MEASURED": r"\ctM"}

# ----------------------------------------------------------------------------
# Unicode -> LaTeX (mathematics)
# ----------------------------------------------------------------------------
GREEK = {
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ",
    "ε": r"\varepsilon ", "η": r"\eta ", "θ": r"\theta ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "ν": r"\nu ", "π": r"\pi ",
    "ρ": r"\rho ", "σ": r"\sigma ", "τ": r"\tau ", "φ": r"\phi ",
    "χ": r"\chi ", "ψ": r"\psi ", "ω": r"\omega ",
    "Γ": r"\Gamma ", "Δ": r"\Delta ", "Θ": r"\Theta ", "Λ": r"\Lambda ",
    "Π": r"\Pi ", "Σ": r"\Sigma ", "Φ": r"\Phi ", "Ω": r"\Omega ",
}
OPS = {
    "×": r"\times ", "·": r"\cdot ", "−": "-", "∇": r"\nabla ",
    "∂": r"\partial ", "∝": r"\propto ", "∫": r"\int ", "∞": r"\infty ",
    "±": r"\pm ", "⊥": r"\perp ", "≈": r"\approx ", "≤": r"\le ",
    "≥": r"\ge ", "≪": r"\ll ", "≫": r"\gg ", "≲": r"\lesssim ",
    "≳": r"\gtrsim ", "≡": r"\equiv ", "≠": r"\ne ", "∈": r"\in ",
    "→": r"\to ", "↔": r"\leftrightarrow ", "⇒": r"\Rightarrow ",
    "⟶": r"\longrightarrow ", "⟺": r"\iff ", "⟨": r"\langle ",
    "⇄": r"\rightleftarrows ", "⇆": r"\leftrightarrows ", "↦": r"\mapsto ",
    "⇔": r"\Leftrightarrow ", "⟷": r"\longleftrightarrow ",
    "⟩": r"\rangle ", "ℓ": r"\ell ", "ℙ": r"\mathbb{P}", "∗": "*",
    "‾": r"\,", "°": r"^{\circ}", "½": r"\tfrac{1}{2}",
    "…": r"\ldots ", "∼": r"\sim ", "≃": r"\simeq ",
}
# precomposed accented math letters
PRECOMP = {
    "â": r"\hat{a}", "û": r"\hat{u}", "ŝ": r"\hat{s}", "x̂": r"\hat{x}",
    "ū": r"\bar{u}", "ȳ": r"\bar{y}", "x̄": r"\bar{x}",
    "ŷ": r"\hat{y}", "n̂": r"\hat{n}", "m̂": r"\hat{m}",
    "ṡ": r"\dot{s}", "ẋ": r"\dot{x}", "ÿ": r"\ddot{y}",
}
SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
       "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-",
       "ⁿ": "n", "ᵀ": "T", "ᵗ": "t", "ᵃ": "a", "ᵏ": "k", "ⁱ": "i"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
       "₆": "6", "₇": "7", "₈": "8", "₉": "9", "₊": "+", "₋": "-",
       "ₜ": "t", "ₛ": "s", "ₖ": "k", "ᵢ": "i", "ⱼ": "j", "ₙ": "n"}


def _group_scripts(s: str):
    """Collapse runs of Unicode sub/superscript chars into ^{..}/_{..}.
    Returns a string with LaTeX script syntax (no surrounding math)."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in SUP:
            j = i
            buf = ""
            while j < n and s[j] in SUP:
                buf += SUP[s[j]]
                j += 1
            out.append("^{" + buf + "}")
            i = j
        elif c in SUB:
            j = i
            buf = ""
            while j < n and s[j] in SUB:
                buf += SUB[s[j]]
                j += 1
            out.append("_{" + buf + "}")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _combining(s: str) -> str:
    s = re.sub(r"(.)\u0302", r"\\hat{\1}", s)      # combining circumflex
    s = re.sub(r"(.)\u0304", r"\\bar{\1}", s)       # combining macron
    s = re.sub(r"(.)\u0307", r"\\dot{\1}", s)       # combining dot above
    s = re.sub(r"(.)\u0303", r"\\tilde{\1}", s)     # combining tilde
    for a, b in PRECOMP.items():
        s = s.replace(a, b)
    return s


def math_translate(s: str) -> str:
    """Translate a math span (inner text, no $) to LaTeX."""
    s = _combining(s)
    s = s.replace("overline{", r"\overline{")
    # protect multi-underscore identifiers so they can't form double subscripts
    s = re.sub(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,}",
               lambda m: r"\mathrm{" + m.group(0).replace("_", r"\_") + "}", s)
    # sqrt with parenthesised argument
    s = re.sub(r"√\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = s.replace("√", r"\surd ")
    # Unicode sub/superscripts -> ^{..}/_{..}
    s = _group_scripts(s)
    # ASCII multi-char subscripts/superscripts -> braced groups
    s = re.sub(r"_(?!\{)([A-Z]{2,}|[A-Za-z][a-z0-9]*|\d+)", r"_{\1}", s)
    s = re.sub(r"\^(?!\{)([A-Za-z][a-z0-9]*|\d+)", r"^{\1}", s)
    # Greek + operators
    for a, b in GREEK.items():
        s = s.replace(a, b)
    for a, b in OPS.items():
        s = s.replace(a, b)
    # "scales as / approximately" tilde -> \sim (math space otherwise)
    s = s.replace("~", r"\sim ")
    # function names as operators (only function calls, not subscript labels)
    s = re.sub(r"\b(sin|cos|tan|ln|log|exp)\s*\(", r"\\\1(", s)
    # math-hostile characters
    s = s.replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
    return s


# Unicode chars that, in prose, must be rendered as math
PROSE_MATH = {}
PROSE_MATH.update(GREEK)
PROSE_MATH.update({k: v for k, v in OPS.items()
                   if k not in ("−", "…", "½", "°", "±")})


def prose_unicode(s: str) -> str:
    """Wrap stray Unicode math in \\ensuremath; leave text Unicode literal."""
    s = s.replace("−", "-").replace("\u2011", "-")
    # combining / precomposed math letters
    s = re.sub(r"(.)\u0302", r"\\ensuremath{\\hat{\1}}", s)
    s = re.sub(r"(.)\u0304", r"\\ensuremath{\\bar{\1}}", s)
    s = re.sub(r"(.)\u0307", r"\\ensuremath{\\dot{\1}}", s)
    for a, b in PRECOMP.items():
        s = s.replace(a, r"\ensuremath{" + b + "}")
    # grouped sub/superscript runs
    def sgrp(m):
        return r"\ensuremath{" + _group_scripts(m.group(0)) + "}"
    s = re.sub("[" + "".join(SUP) + "]+", sgrp, s)
    s = re.sub("[" + "".join(SUB) + "]+", sgrp, s)
    # single math symbols
    for a, b in PROSE_MATH.items():
        s = s.replace(a, r"\ensuremath{" + b.strip() + "}")
    return s


# ----------------------------------------------------------------------------
# Text escaping
# ----------------------------------------------------------------------------
def esc_text(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("$", r"\$")):
        s = s.replace(a, b)
    s = s.replace("^", r"\textasciicircum{}").replace("~", r"\textasciitilde{}")
    s = s.replace("…", "...")
    # straight double quotes -> LaTeX curly quotes (alternating)
    if '"' in s:
        buf = []
        openq = True
        for ch in s:
            if ch == '"':
                buf.append("``" if openq else "''")
                openq = not openq
            else:
                buf.append(ch)
        s = "".join(buf)
    return s


def esc_code(s: str) -> str:
    return s  # content goes inside \texttt via \detokenize-free path; see code()


# ----------------------------------------------------------------------------
# Inline code-span classification
# ----------------------------------------------------------------------------
PATH_EXT = re.compile(r"\.(py|md|json|nc|h5|csv|txt|sty|tex|cfg|ya?ml|sh|toml|ipynb|npy|png|pdf)\b")
CMD = re.compile(r"^(python3?|pytest|pip|git|bash|cd|make|sudo|ls|cat|grep|awk|sed|export|source|mkdir|rm|cp|mv|chmod|curl|wget)\b")
MATH_SIGNAL = re.compile(r"[=^√∫∇∂×·−≈≤≥≪≫≲≳≡≠∈→↔⇒⟨⟩ℓℙ±∝∞°"
                         r"αβγδεηθκλμνπρστφχψωΓΔΘΛΠΣΦΩ"
                         r"⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵀᵗₜₛ₀₁₂₃₄₅₆₇₈₉]")
SNAKE2 = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+){2,}$")
CAMEL = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def classify(span: str) -> str:
    t = span.strip()
    if not t:
        return "code"
    if PATH_EXT.search(t):
        return "code"
    if CMD.match(t):
        return "code"
    base = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*(\([^)]*\))?$", t)
    if base and base.group(1).count("_") >= 2:
        return "code"
    if "()" in t:
        return "code"
    if re.search(r"[A-Za-z]\.[A-Za-z_]", t) and "=" not in t and not MATH_SIGNAL.search(t):
        return "code"
    if "/" in t and re.match(r"^[\w./\-]+$", t) and not MATH_SIGNAL.search(t):
        return "code"
    if SNAKE2.match(t):
        return "code"
    if CAMEL.match(t) and re.search(r"[a-z][A-Z]", t):
        return "code"
    return "math"


def code(span: str) -> str:
    """Render a verbatim code span; keep Unicode literal (DejaVu mono)."""
    s = span
    for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                 ("$", r"\$"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("^", r"\textasciicircum{}"),
                 ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    # allow line breaks at path/identifier separators for long spans
    for sep in ("/", r"\_", ".", "-"):
        s = s.replace(sep, sep + r"\allowbreak{}")
    return r"\code{" + s + "}"


# ----------------------------------------------------------------------------
# Inline processing (operates on a fully-joined logical line)
# ----------------------------------------------------------------------------
def inline(s: str) -> str:
    holds = []

    def hold(latex: str) -> str:
        holds.append(latex)
        return "\x00%d\x00" % (len(holds) - 1)

    # claim tags (optionally wrapped in backticks)
    def claim(m):
        full = (m.group(1) + m.group(2))
        for t in sorted(TAGS, key=len, reverse=True):
            if full.upper().startswith(t):
                rest = full[len(t):]
                if t in TAGMACRO:
                    return hold(TAGMACRO[t] + "{" + esc_text(rest) + "}")
                return hold(r"\ctX{" + esc_text(full) + "}")
        return hold(r"\ctX{" + esc_text(full) + "}")
    s = re.sub(r"`?\[(" + "|".join(TAGS) + r")([^\]]*)\]`?", claim, s)
    # citation slots
    s = re.sub(r"`?\[cite:\s*([^\]]*)\]`?",
               lambda m: hold(r"\cmark{" + esc_text(m.group(1).strip()) + "}"), s)
    s = re.sub(r"`?\[cite\]`?", lambda m: hold(r"\cmarke{}"), s)
    # code / math spans
    def span(m):
        raw = m.group(1)
        if classify(raw) == "code":
            return hold(code(raw))
        return hold("$" + math_translate(raw) + "$")
    s = re.sub(r"`([^`]+)`", span, s)
    # links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    # escape prose
    s = esc_text(s)
    # italic first (consume inner * so nested **bold *italic* ** still matches), then bold
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    # stray Unicode math in prose
    s = prose_unicode(s)
    # restore holds
    s = re.sub("\x00(\\d+)\x00", lambda m: holds[int(m.group(1))], s)
    return s


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def _split_row(line: str):
    """Split a markdown table row on | but not inside backtick spans."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, cur, incode = [], "", False
    for ch in line:
        if ch == "`":
            incode = not incode
            cur += ch
        elif ch == "|" and not incode:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


def table(rows):
    cells = [_split_row(r) for r in rows]
    header, body = cells[0], cells[2:]
    ncol = len(header)
    maxw = max((sum(len(c) for c in r) for r in [header] + body), default=0)
    if maxw > 68:
        colspec = (r">{\raggedright\arraybackslash}X" * ncol)
        out = [r"\begin{center}\small", r"\begin{tabularx}{\linewidth}{" + colspec + "}", r"\toprule"]
        closer = [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    else:
        out = [r"\begin{center}\small", r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule"]
        closer = [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    out.append(" & ".join(inline(c) for c in header) + r" \\")
    out.append(r"\midrule")
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += closer
    return out


# ----------------------------------------------------------------------------
# Block-level conversion
# ----------------------------------------------------------------------------
def _is_block_start(ln: str) -> bool:
    s = ln.strip()
    if s == "":
        return True
    if s.startswith("```"):
        return True
    if s.startswith("!["):
        return True
    if re.match(r"^#{1,6}\s", ln):
        return True
    if re.match(r"^>(\s|$)", ln):
        return True
    if re.match(r"^(\s*)[-*]\s+", ln):
        return True
    if re.match(r"^(\s*)\d+\.\s+", ln):
        return True
    if re.match(r"^---+\s*$", ln):
        return True
    if ln.strip().startswith("|"):
        return True
    return False


def convert(md: str) -> str:
    lines = md.split("\n")
    out = []
    in_abstract = False
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        # fenced code
        if ln.strip().startswith("```"):
            out.append(r"\begin{Verbatim}")
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                out.append(lines[i])
                i += 1
            out.append(r"\end{Verbatim}")
            i += 1
            continue
        # standalone image: ![caption](path)
        mimg = re.match(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$", ln)
        if mimg:
            cap, path = mimg.group(1), mimg.group(2).strip()
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=0.82\linewidth]{" + path + "}")
            if cap.strip():
                out.append(r"\caption{" + inline(cap) + "}")
            out.append(r"\end{figure}")
            out.append("")
            i += 1
            continue
        # table
        if ln.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            tb = [ln]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                tb.append(lines[i])
                i += 1
            out += table(tb)
            out.append("")
            continue
        # headers
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                i += 1
                continue  # document title handled by template
            # "Abstract" heading -> standard abstract environment (arXiv parses this)
            if level == 2 and title.lower() == "abstract":
                while out and out[-1].strip() == "":
                    out.pop()
                if out and out[-1].strip() == r"\medskip\hrule\medskip":
                    out.pop()  # drop the decorative rule preceding the abstract
                out.append(r"\begin{abstract}")
                out.append("")
                in_abstract = True
                i += 1
                continue
            if in_abstract:  # a later heading closes the abstract block
                out.append(r"\end{abstract}")
                out.append("")
                in_abstract = False
            cmd = {2: r"\section*", 3: r"\subsection*", 4: r"\subsubsection*",
                   5: r"\paragraph*", 6: r"\subparagraph*"}.get(level, r"\subsubsection*")
            out.append(cmd + "{" + inline(title) + "}")
            out.append("")
            i += 1
            continue
        # horizontal rule
        if re.match(r"^---+\s*$", ln):
            if in_abstract:  # the rule after the abstract closes it (drop the rule)
                out.append(r"\end{abstract}")
                out.append("")
                in_abstract = False
                i += 1
                continue
            out.append(r"\medskip\hrule\medskip")
            out.append("")
            i += 1
            continue
        # blockquote
        if re.match(r"^>(\s|$)", ln):
            paras, cur = [], []
            while i < n and re.match(r"^>(\s|$)", lines[i]):
                content = re.sub(r"^>\s?", "", lines[i])
                if content.strip() == "":
                    if cur:
                        paras.append(" ".join(cur))
                        cur = []
                else:
                    cur.append(content)
                i += 1
            if cur:
                paras.append(" ".join(cur))
            out.append(r"\begin{quote}")
            for k, p in enumerate(paras):
                out.append(inline(p))
                if k != len(paras) - 1:
                    out.append("")
            out.append(r"\end{quote}")
            out.append("")
            continue
        # lists (with soft-wrapped item continuations)
        mb = re.match(r"^(\s*)([-*])\s+(.*)$", ln)
        me = re.match(r"^(\s*)\d+\.\s+(.*)$", ln)
        if mb or me:
            want = "itemize" if mb else "enumerate"
            out.append(r"\begin{%s}" % want)
            while i < n:
                mb = re.match(r"^(\s*)([-*])\s+(.*)$", lines[i])
                me = re.match(r"^(\s*)\d+\.\s+(.*)$", lines[i])
                if not (mb or me):
                    break
                body = (mb.group(3) if mb else me.group(2))
                i += 1
                # gather continuation lines for this item
                while i < n and lines[i].strip() != "" and not _is_block_start(lines[i]):
                    body += " " + lines[i].strip()
                    i += 1
                out.append(r"\item " + inline(body))
            out.append(r"\end{%s}" % want)
            out.append("")
            continue
        # blank
        if ln.strip() == "":
            out.append("")
            i += 1
            continue
        # paragraph (join soft-wrapped lines)
        para = [ln.rstrip()]
        i += 1
        while i < n and lines[i].strip() != "" and not _is_block_start(lines[i]):
            para.append(lines[i].rstrip())
            i += 1
        out.append(inline(" ".join(para)))
        out.append("")
    if in_abstract:
        out.append(r"\end{abstract}")
    return "\n".join(out)


def title_of(md: str) -> str:
    for ln in md.split("\n"):
        m = re.match(r"^#\s+(.*)$", ln)
        if m:
            return m.group(1).strip()
    return "Manuscript"


TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage{preamble}
\title{%s}
\author{Abdelrahman Saifelden\\ Alexandria Higher Institute of Engineering and Technology (AIET), Alexandria, Egypt\\ \texttt{Abdulrahman.Saifelden22011411@aiet.edu.eg}}
\date{\today}
\begin{document}
\maketitle
%s
\end{document}
"""


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        md = fh.read()
    title = inline(title_of(md))
    body = convert(md)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE % (title, body))
    print("wrote", dst)


if __name__ == "__main__":
    main()
