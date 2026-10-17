"""Stage 2 for every figure: analysis/figures/data/<figure>.json from the stage-1 intermediates in
analysis/data/intermediate/. It only selects, renames, averages and rounds, so it needs no raw results.

    uv run python -m analysis.compute.figures [--data analysis/data/intermediate] [--out analysis/figures/data] [--check]

--check compares with the files in --out instead of writing them, and fails on any difference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from analysis.compute import ablations, cotcontrol, curriculum, exposure, fig1, length, transfer

ANALYSIS = Path(__file__).resolve().parents[1]
INTERMEDIATE, FIGURES = ANALYSIS / "data" / "intermediate", ANALYSIS / "figures" / "data"

# figure -> its data from a loader of intermediates by name
BUILDERS: dict[str, Callable[[Callable[[str], object]], object]] = {
    "fig1": lambda i: fig1.figure1(i("grid"), i("cond_leakage"), i("maths_t1")),
    "fig2": lambda i: exposure.fig2(i("exposure")),
    "fig2b": lambda i: exposure.fig2b(i("exposure")),
    "fig3": lambda i: ablations.figure(i("cond_leakage"), i("ablations")),
    "fig4": lambda i: transfer.figure(i("transfer")),
    "fig4b": lambda i: cotcontrol.figure(i("cotcontrol")),
    "fig5": lambda i: curriculum.fig5(i("curriculum"), i("cond_leakage")),
    "fig6": lambda i: length.fig6(i("cond_leakage"), i("length")),
}


def build(data: Path = INTERMEDIATE) -> dict[str, str]:
    """figure -> its data file's text, serialised as the published files are."""
    def intermediate(name: str):
        return json.loads((data / f"{name}.json").read_text())

    return {fig: json.dumps(make(intermediate), indent=1) + "\n" for fig, make in BUILDERS.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=INTERMEDIATE)
    ap.add_argument("--out", type=Path, default=FIGURES)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    differ = []
    for fig, text in build(a.data).items():
        path = a.out / f"{fig}.json"
        if a.check:
            differ += [fig] if not path.exists() or path.read_text() != text else []
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if differ:
        raise SystemExit(f"differ from {a.out}: {', '.join(differ)}")
    if a.check:
        print(f"all {len(BUILDERS)} figure data files match")


if __name__ == "__main__":
    main()
