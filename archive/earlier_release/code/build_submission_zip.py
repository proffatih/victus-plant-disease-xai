"""Build the flat submission ZIP for Frontiers in Plant Science.

Frontiers portal accepts a manuscript PDF plus optional source (LaTeX) archive.
We produce one flat ZIP that includes:
  - manuscript.pdf, manuscript.tex, manuscript.bbl
  - references.bib
  - FrontiersinHarvard.cls, Frontiers-Harvard.bst, frontiers_suppmat.cls
  - logo1.pdf, logo2.pdf, logos.pdf
  - all figX_*.pdf files
Paths inside the archive are flat (no folders).
"""
from __future__ import annotations
import os, zipfile
from pathlib import Path

ROOT = Path("/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science")
OUT = ROOT / "submission" / "Victus_Pardus_0011_Plant_Disease_XAI_submission.zip"

want = [
    ROOT / "manuscript.pdf",
    ROOT / "manuscript.tex",
    ROOT / "manuscript.bbl",
    ROOT / "references.bib",
    ROOT / "FrontiersinHarvard.cls",
    ROOT / "Frontiers-Harvard.bst",
    ROOT / "frontiers_suppmat.cls",
    ROOT / "logo1.pdf",
    ROOT / "logo2.pdf",
    ROOT / "logos.pdf",
]
want += sorted((ROOT / "figures").glob("fig*.pdf"))
# also root-level fig*.pdf (they've been copied there for the LaTeX compile)
want += sorted(ROOT.glob("fig*.pdf"))

# dedup by basename
seen = set(); dedup = []
for p in want:
    if p.name not in seen and p.exists():
        seen.add(p.name); dedup.append(p)

OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in dedup:
        z.write(p, arcname=p.name)
        print(f"[zip] {p.name}")

print(f"[done] {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
