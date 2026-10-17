"""Build data/sources/side_mmlu_pro_mini.jsonl from CoTControl's MMLU-Pro mini (MIT, YuehHanChen/CoTControl @ 5d78aef).

    uv run python -m scripts.data.build_side_questions csv_path=<CoTControl>/CoT-Control-QA/datasets/mmlu_pro_mini_w_keyword.csv

Upstream stores the answer as option text; it is mapped to a letter, refusing rows where that is not unique.
"""

import csv
import json
import string
from pathlib import Path

import chz

OUT = Path(__file__).resolve().parents[2] / "data" / "sources" / "side_mmlu_pro_mini.jsonl"


def main(csv_path: str) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; committed fixtures are never rewritten")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    lines = []
    for i, row in enumerate(rows):
        options = json.loads(row["options"])
        hits = [k for k, option in enumerate(options) if option.strip() == row["answer"].strip()]
        if len(hits) != 1:
            raise SystemExit(f"row {i}: answer matches {len(hits)} options")
        lines.append(json.dumps({"id": f"mmlu_pro_mini-{i:03d}", "domain": row["domain"], "question": row["question"],
                                 "options": options, "answer": string.ascii_uppercase[hits[0]]}))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} questions to {OUT}")


if __name__ == "__main__":
    chz.entrypoint(main)
