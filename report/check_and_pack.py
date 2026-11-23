"""Vérifications structurelles de rapport.tex (sans compilateur) et archive prête pour Overleaf.

Usage (depuis report/) : python check_and_pack.py
"""

import collections
import pathlib
import re
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
s = (HERE / "rapport.tex").read_text(encoding="utf-8")
body = re.sub(r"(?<!\\)%.*", "", s)

b = collections.Counter(re.findall(r"\\begin\{([^}]+)\}", body))
e = collections.Counter(re.findall(r"\\end\{([^}]+)\}", body))
print("environnements non appariés :", {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]} or "aucun")
clean = body.replace("\\{", "").replace("\\}", "")
print("accolades équilibrées :", clean.count("{") == clean.count("}"))
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
missing = [f for f in figs if not (HERE / "figures" / f"{f}.pdf").exists()]
print("figures référencées :", len(figs), "; manquantes :", missing or "aucune")
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
print("références sans label :", (refs - labels) or "aucune")
cites = {c.strip() for g in re.findall(r"\\cite\{([^}]+)\}", body) for c in g.split(",")}
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
print("citations sans entrée :", (cites - bib) or "aucune", "| entrées non citées :", (bib - cites) or "aucune")

with zipfile.ZipFile(HERE / "rapport_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(HERE / "rapport.tex", "rapport.tex")
    for f in sorted((HERE / "figures").glob("*.pdf")):
        z.write(f, f"figures/{f.name}")
print("archive Overleaf :", (HERE / "rapport_overleaf.zip").stat().st_size // 1024, "Ko")
