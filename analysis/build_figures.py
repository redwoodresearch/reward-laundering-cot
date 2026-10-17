"""Render the post's figures from their frozen data: analysis/figures/page.template.html with each figure's data
(analysis/figures/data/<figure>.json) inlined, written as one self-contained HTML page. Open it in a browser: every
figure is drawn as an SVG and turned into a PNG (right-click to copy or save).

    uv run python analysis/build_figures.py [--data analysis/figures/data] [--out figures.html]

The data files are the aggregates the published figures were drawn from; analysis/compute/ recomputes them from the
raw run outputs (not published; see analysis/README.md)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PLACEHOLDER = re.compile(r"\{\{data:(?P<fig>[0-9a-z]+):(?P<style>compact|spaced)\}\}")
SEPARATORS = {"compact": (",", ":"), "spaced": (", ", ": ")}  # the serialisation each published figure used


def build(data_dir: Path) -> str:
    template = (HERE / "figures" / "page.template.html").read_text()
    figures = {m["fig"] for m in PLACEHOLDER.finditer(template)}
    missing = sorted(f for f in figures if not (data_dir / f"{f}.json").exists())
    if missing:
        raise SystemExit(f"no data for {missing} in {data_dir}")
    return PLACEHOLDER.sub(lambda m: json.dumps(json.loads((data_dir / f"{m['fig']}.json").read_text()),
                                                separators=SEPARATORS[m["style"]]), template)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "figures" / "data")
    ap.add_argument("--out", type=Path, default=Path("figures.html"))
    a = ap.parse_args()
    a.out.write_text(build(a.data))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
