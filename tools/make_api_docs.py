"""Generate ``docs/API.md`` from the docstrings: ``python tools/make_api_docs.py``."""
from __future__ import annotations

import importlib
import inspect
import pathlib
import re

import matplotlib

matplotlib.use("Agg")

MODULES = [
    "spectrapy.spectrum", "spectrapy.observers", "spectrapy.illuminants", "spectrapy.colorimetry",
    "spectrapy.colorspaces", "spectrapy.uniform", "spectrapy.appearance", "spectrapy.adaptation",
    "spectrapy.temperature",
    "spectrapy.quality", "spectrapy.lines", "spectrapy.thinfilm", "spectrapy.atmosphere", "spectrapy.optics",
    "spectrapy.render", "spectrapy.io", "spectrapy.icc", "spectrapy.plot", "spectrapy.style", "spectrapy.cmap",
    "spectrapy.interactive",
]
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "API.md"


def _public(mod):
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n, v in vars(mod).items() if not n.startswith("_")
                 and (inspect.isfunction(v) or inspect.isclass(v)) and getattr(v, "__module__", "") == mod.__name__]
    out = []
    for n in names:
        v = getattr(mod, n, None)
        if inspect.isfunction(v) or inspect.isclass(v):
            out.append((n, v))
    return out


def _sig(obj) -> str:
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return s.replace("'", '"')


def _md(text: str) -> str:
    """reST-flavoured docstring -> Markdown (roles, double backticks, underlined section titles)."""
    text = re.sub(r":(?:class|func|mod|data|meth|attr):`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r"``(.+?)``", r"`\1`", text)
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if lines[i].strip() and re.fullmatch(r"\s*[-=~]{3,}\s*", nxt) and len(nxt.strip()) >= len(lines[i].strip()):
            out.append(f"**{lines[i].strip()}**")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _doc(obj) -> str:
    return _md((inspect.getdoc(obj) or "").strip())


def main() -> None:
    lines = ["# spectrapy API reference", "",
             "Generated from the docstrings by `tools/make_api_docs.py`. Everything listed is importable from",
             "its module; the most used names are also available directly as `spectrapy.<name>`.", ""]
    toc = []
    body = []
    for mname in MODULES:
        mod = importlib.import_module(mname)
        short = mname.split(".", 1)[1]
        toc.append(f"- [`{mname}`](#{mname.replace('.', '')})")
        body += [f"## `{mname}`", ""]
        mdoc = _md((inspect.getdoc(mod) or "").strip())
        if mdoc:
            body += [mdoc, ""]
        for name, obj in _public(mod):
            kind = "class" if inspect.isclass(obj) else "function"
            body += [f"### `{short}.{name}`", "", f"```python\n{kind} {name}{_sig(obj)}\n```", ""]
            d = _doc(obj)
            if d:
                body += [d, ""]
            if inspect.isclass(obj):
                for mn, meth in inspect.getmembers(obj):
                    if mn.startswith("_") or not (inspect.isfunction(meth) or isinstance(meth, property)):
                        continue
                    if isinstance(meth, property):
                        md = _md((inspect.getdoc(meth) or "").strip())
                        body += [f"- **`{mn}`** (property){': ' + md.splitlines()[0] if md else ''}"]
                    else:
                        md = _md((inspect.getdoc(meth) or "").strip())
                        first = md.split("\n\n")[0].replace("\n", " ") if md else ""
                        body += [f"- **`{mn}{_sig(meth)}`**{' — ' + first if first else ''}"]
                body += [""]
    lines += ["## Modules", ""] + toc + [""] + body
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines) + len(body)} lines)")


if __name__ == "__main__":
    main()
