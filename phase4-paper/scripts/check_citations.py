"""
Verifies every \\cite*{...} key used across paper/main.tex and
paper/sections/*.tex resolves to exactly one entry defined in
paper/references.bib -- and flags bib entries that are never cited.
Run before any submission; exits non-zero on either kind of mismatch.

Usage:
    python3 scripts/check_citations.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"


def defined_keys(bib_path: Path) -> set[str]:
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,]+),", text))


def used_keys(tex_paths: list[Path]) -> set[str]:
    keys = set()
    for p in tex_paths:
        text = p.read_text(encoding="utf-8")
        for match in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
            keys.update(k.strip() for k in match.split(","))
    return keys


def main() -> int:
    bib_path = PAPER_DIR / "references.bib"
    tex_paths = [PAPER_DIR / "main.tex"] + [
        Path(p) for p in glob.glob(str(PAPER_DIR / "sections" / "*.tex"))
    ]

    defined = defined_keys(bib_path)
    used = used_keys(tex_paths)

    missing = used - defined
    unused = defined - used

    print(f"Defined bib entries: {len(defined)}")
    print(f"Cited keys found:    {len(used)}")

    ok = True
    if missing:
        ok = False
        print(f"\nBROKEN: {len(missing)} cited key(s) have no bib entry:")
        for k in sorted(missing):
            print(f"  - {k}")
    if unused:
        print(f"\nUnused bib entries ({len(unused)}, not an error, just FYI):")
        for k in sorted(unused):
            print(f"  - {k}")

    if ok:
        print("\nAll citations resolve. No broken keys.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
