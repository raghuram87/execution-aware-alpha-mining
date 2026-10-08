"""Build the public (non-anonymized) preprint from the anonymized FRO
submission file: adds the author block, the repository URL and the
competing-interest statement, and drops line numbers.

Usage: python scripts/make_preprint.py   (then compile paper/fro/preprint/preprint.tex)
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "paper" / "fro" / "submission"
OUT = ROOT / "paper" / "fro" / "preprint"
REPO = "https://github.com/raghuram87/execution-aware-alpha-mining"

REPLACEMENTS = [
    ("\\linenumbers\n", ""),
    ("\\journal{Finance Research Open}\n",
     "\\makeatletter\n\\def\\ps@pprintTitle{\\let\\@oddhead\\@empty\\let\\@evenhead\\@empty"
     "\\def\\@oddfoot{\\footnotesize\\itshape Working paper. This version: \\today\\hfill}"
     "\\let\\@evenfoot\\@oddfoot}\n\\makeatother\n"),
    ("\\title{Execution-Aware Alpha Mining: Teaching LLM Factor Agents to Account for Trading Costs}\n",
     "\\title{Execution-Aware Alpha Mining: Teaching LLM Factor Agents to Account for Trading Costs}\n\n"
     "\\author{Raghuram Nagireddy\\fnref{orcid}}\n"
     "\\ead{rrn2111@caa.columbia.edu}\n"
     "\\fntext[orcid]{ORCID: 0009-0002-2203-3367.}\n"
     "\\address{Independent Researcher; Columbia University (Alumni)}\n"),
    ("are available in a public code repository; its URL is given on the title page and withheld from this version "
     "to preserve anonymity during review.",
     f"are available at \\url{{{REPO}}}."),
    ("\\section*{Funding}",
     "\\section*{Declaration of competing interest}\n\n"
     "The author declares no known competing financial interests or personal relationships that could have "
     "appeared to influence the work reported in this paper.\n\n\\section*{Funding}"),
]

if __name__ == "__main__":
    tex = (SUB / "manuscript.tex").read_text()
    for old, new in REPLACEMENTS:
        if tex.count(old) != 1:
            raise SystemExit(f"expected exactly one match for: {old[:70]!r}")
        tex = tex.replace(old, new)
    OUT.mkdir(exist_ok=True)
    (OUT / "preprint.tex").write_text(tex)
    for fig in SUB.glob("Figure_*.pdf"):
        shutil.copy(fig, OUT / fig.name)
    print(f"wrote {OUT / 'preprint.tex'}")
