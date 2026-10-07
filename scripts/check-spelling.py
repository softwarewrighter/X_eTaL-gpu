#!/usr/bin/env python3
"""American spellings only: scan this repository's docs and sources for
British spellings and fail if any are found (the gate runs it).

  scripts/check-spelling.py              # check every tracked text file we own
  scripts/check-spelling.py --self-test  # check the checker on known samples

Scanned: every file git tracks, except vendor/ (X_eTaL's own code),
pages/ (generated), the agentrail saga records (append-only), this
script (its samples are British on purpose) and binary or lock files. Each hit prints file:line, the word and the American form.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = ("vendor/", "pages/", ".agentrail/", ".agentrail-archive/")
SKIP_FILES = ("scripts/check-spelling.py",)  # its patterns and samples are British on purpose
SKIP_SUFFIXES = (".png", ".ico", ".jpg", ".jpeg", ".gif", ".wasm", ".lock", ".tdb")

# British form (a regex for the whole word, any case) and its American form.
WORDS = [
    (r"(col|flav|fav|hon|hum|lab|neighb|behavi|harb|rum|arm|vig|od|sav|endeav|parl|splend)our(s|ed|ing|ful|ite|ites|ise|ize|hood|hoods|ly|able|ably)?",
     "-or (color, flavor, neighbor, behavior...)"),
    (r"(cent|met|lit|theat|fib|sab|somb|spect|meag|calib|lust)re(s|d)?", "-er (center, meter, theater, fiber...)"),
    (r"manoeuvre(s|d)?", "maneuver"),
    (r"grey(s|ed|ing|ish|er|est|ness)?", "gray"),
    (r"aluminium", "aluminum"),
    (r"(catalog|dialog|analog)ue(s|d)?", "catalog, dialog, analog"),
    (r"programme(s|d)?", "program"),
    (r"(defen|offen|preten)ce(s)?", "defense, offense, pretense"),
    (r"licence(s|d)?", "license"),
    (r"(trav|mod|lab|canc|lev|sign|fu|tot|marsh|quarr|chann|tunn|duel|jew|counsel|dial|equ|riv|shriv|swiv|ten)ell(ed|ing|er|ers|or|ors)", "a single l (traveled, modeled, labeled, canceled...)"),
    (r"jewellery", "jewelry"),
    (r"(fulfil|enrol|instil|distil)(s|ment|ments)?", "fulfill, enroll, instill, distill"),
    (r"whilst|amongst|learnt|spelt|dreamt", "while, among, learned, spelled, dreamed"),
    (r"artefact(s)?", "artifact"),
    (r"ageing", "aging"),
    (r"sceptic(s|al|ism)?", "skeptic"),
    (r"(tyre|kerb|cheque|storey|plough|draught|aeroplane)(s)?", "tire, curb, check, story, plow, draft, airplane"),
    (r"mould(s|ed|ing|y)?", "mold"),
    (r"(organi|reali|recogni|normali|initiali|optimi|utili|summari|visuali|minimi|maximi|emphasi|apologi|authori|categori|characteri|critici|customi|finali|generali|harmoni|locali|memori|mobili|prioriti|seriali|speciali|standardi|symboli|synchroni|randomi|tokeni|paralleli|digiti|saniti|stabili|materiali|equali|centrali|vectori|parameteri|deseriali|capitali|colouri|initiali|penali|publici|sympathi|utili|visuali)s(e|es|ed|ing|ation|ations|er|ers)",
     "-ize, -ization (organize, normalize, initialization...)"),
    (r"analys(e|es|ed|ing|er|ers)", "analyze"),
    (r"(paraly|cataly)s(e|es|ed|ing)", "paralyze, catalyze"),
]
PATTERNS = [(re.compile(r"\b(" + w + r")\b", re.IGNORECASE), a) for w, a in WORDS]


def hits(text):
    for n, line in enumerate(text.splitlines(), 1):
        for pat, american in PATTERNS:
            for m in pat.finditer(line):
                yield n, m.group(1), american


def files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for f in out.splitlines():
        if f.startswith(SKIP_DIRS) or f in SKIP_FILES or f.endswith(SKIP_SUFFIXES):
            continue
        p = ROOT / f
        try:
            yield f, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue


def self_test():
    british = "colour neighbours centred grey modelled whilst normalise initialisation analysed catalogue licence travelled"
    american = "color neighbors centered gray modeled while normalize initialization analyzed catalog license traveled emphasis parameter exercise precise advertise promise analysis rise wise premise"
    found = [w for _, w, _ in hits(british)]
    assert len(found) == len(british.split()), f"missed: {set(british.split()) - set(found)}"
    wrong = [w for _, w, _ in hits(american)]
    assert not wrong, f"false positives: {wrong}"
    print("check-spelling: self-test ok")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    bad = 0
    for f, text in files():
        for n, word, american in hits(text):
            print(f"{f}:{n}: {word} (American: {american})")
            bad += 1
    if bad:
        print(f"check-spelling: {bad} British spelling(s); this repository uses American spellings only", file=sys.stderr)
        sys.exit(1)
    print("check-spelling: ok (American spellings only)")


if __name__ == "__main__":
    main()
