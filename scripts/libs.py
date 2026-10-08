#!/usr/bin/env python3
"""The libraries: each its own directory, libs/<Name>/.

  libs/<Name>/README.md        a short overview
  libs/<Name>/src/<Name>.xtl   the library (and/or <Name>.xtlm, macros)
  libs/<Name>/tests/           reg-rs: *.xtl programs, each with its
                               baseline NAME.rgt (+ .out, .err);
                               types.rgt pins the exports' types;
                               demo-D.rgt runs demos/D.xtl
  libs/<Name>/docs/README.md   the reference page
  libs/<Name>/demos/*.xtl      programs that show it in use

  scripts/libs.py list     # library names, one per line
  scripts/libs.py table    # name, alias, summary
  scripts/libs.py check    # every library is complete and well-formed
  scripts/libs.py path     # the src/ directories, joined with ':'

A library's header: line 1 is "## Name: summary" (a doc comment, so
xetal doc shows it; "# Name: summary" is accepted too), and a comment line
holds its import line, '"alias:" u_se< "Name"' (the recommended alias).
XETAL_LIBS_ROOT overrides the repository root (scripts/selftest-libs.sh).
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("XETAL_LIBS_ROOT", REPO))
STANDARD = {p.stem for p in (REPO / "work/xetal/lib").glob("*.xtl")}


def names():
    return sorted(d.name for d in (ROOT / "libs").glob("*/") if d.is_dir())


def sources(name):
    src = ROOT / "libs" / name / "src"
    return [p for p in (src / f"{name}.xtl", src / f"{name}.xtlm") if p.exists()]


def header(name):
    s = sources(name)
    if not s:
        return None, None
    text = s[0].read_text()
    lines = text.splitlines()
    m = re.match(rf"##? {re.escape(name)}: (.+)", lines[0] if lines else "")
    a = re.search(rf'"([a-z]+:)" u_se< "{re.escape(name)}"', text)
    return (m.group(1) if m else None), (a.group(1) if a else None)


def exports(name):
    types = ROOT / "libs" / name / "tests" / "types.out"
    if not types.exists():
        return []
    # l: exports of a library, m: macros of a macro library (name<).
    return [m.group(1) for m in re.finditer(r"^[lm]:(\S+) :", types.read_text(), re.M)]


def check():
    errors = []
    for name in names():
        d = ROOT / "libs" / name
        where = f"libs/{name}"
        if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
            errors.append(f"{where}: a library name is UpperCamel")
        if name in STANDARD:
            errors.append(f"{where}: named like the standard library {name}")
        if not sources(name):
            errors.append(f"{where}/src/: no {name}.xtl or {name}.xtlm")
            continue
        extra = sorted(p.name for p in (d / "src").iterdir() if p not in sources(name))
        if extra:
            errors.append(f"{where}/src/: only {name}.xtl and {name}.xtlm belong here, not {', '.join(extra)}")
        summary, alias = header(name)
        if not summary:
            errors.append(f"{where}/src: line 1 must be '## {name}: what it is'")
        if not alias:
            errors.append(f"{where}/src: no import line '\"x:\" u_se< \"{name}\"' in its header")
        if not (d / "README.md").exists():
            errors.append(f"{where}/README.md: missing")
        if not list((d / "tests").glob("*.xtl")):
            errors.append(f"{where}/tests/: no test programs")
        if not list((d / "demos").glob("*.xtl")):
            errors.append(f"{where}/demos/: no demo programs")
        page = d / "docs" / "README.md"
        if not page.exists():
            errors.append(f"{where}/docs/README.md: missing")
        else:
            text = page.read_text()
            for e in exports(name):
                if f"{alias or ''}{e}" not in text:
                    errors.append(f"{where}/docs/README.md: export {alias or ''}{e} not documented")
    for e in errors:
        print(f"libs: {e}", file=sys.stderr)
    return 1 if errors else 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for n in names():
            print(n)
    elif cmd == "table":
        for n in names():
            summary, alias = header(n)
            print(f"{n}\t{alias or '?'}\t{summary or '?'}")
    elif cmd == "check":
        sys.exit(check())
    elif cmd == "path":
        print(":".join(str(ROOT / "libs" / n / "src") for n in names()))
    else:
        sys.exit(f"libs.py: unknown command {cmd!r} (list, table, check, path)")


if __name__ == "__main__":
    main()
