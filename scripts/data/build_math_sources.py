"""Build data/sources/math_gsm8k.jsonl, the GSM8K source behind every gate pool, from the pinned Hugging Face files
(openai/gsm8k, MIT):

    uv run python -m scripts.data.build_math_sources gsm8k_dir=<dir with main/{train,test}-00000-of-00001.parquet>

Integer gold answers only. The official test split is kept as test; val is a hash-chosen 10% of the official train
split; train copies of test questions are dropped. Reference solutions are not stored: grading needs only the gold
answer. The committed data/curation/math_sources_report.json was written by the research version of this script, which
also built a DeepScaleR source the post does not use; its GSM8K pins and counts are the ones this script reproduces.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

import chz
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[2] / "data"
PINS = {
    "gsm8k": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
              "sha256": {"main/train-00000-of-00001.parquet": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
                         "main/test-00000-of-00001.parquet": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"}},
}
INTEGER = re.compile(r"-?\d+")

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    """Grouping key (versioned with the build): whitespace collapsed, case and symbols kept."""
    return " ".join(text.split())


def _bucket(text: str) -> int:
    return int(hashlib.sha1(norm(text).encode()).hexdigest(), 16) % 100


def build_gsm8k(directory: Path) -> tuple[list[dict], dict]:
    out, counts = [], collections.Counter()
    test_texts = set()
    for source_split in ("test", "train"):  # test first, so train copies of test questions can be dropped
        for i, r in enumerate(pq.read_table(directory / f"main/{source_split}-00000-of-00001.parquet").to_pylist()):
            gold = r["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
            if not INTEGER.fullmatch(gold):
                counts["non_integer_answer"] += 1
                continue
            if source_split == "train" and norm(r["question"]) in test_texts:
                counts["train_copy_of_test"] += 1
                continue
            if source_split == "test":
                test_texts.add(norm(r["question"]))
            split = "test" if source_split == "test" else ("val" if _bucket(r["question"]) < 10 else "train")
            out.append({"id": f"gsm8k-{source_split}-{i:04d}", "split": split, "problem": r["question"],
                        "gold": int(gold), "source_split": source_split, "source_row": i})
    counts["kept"] = len(out)
    return out, dict(counts)


def main(gsm8k_dir: str) -> None:
    out = DATA / "sources" / "math_gsm8k.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; committed fixtures are never rewritten")
    for f, sha in PINS["gsm8k"]["sha256"].items():
        if _sha256(Path(gsm8k_dir) / f) != sha:
            raise SystemExit(f"{Path(gsm8k_dir) / f} is not the pinned GSM8K file")
    rows, counts = build_gsm8k(Path(gsm8k_dir))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({**counts, "splits": dict(collections.Counter(r["split"] for r in rows))}, indent=2))


if __name__ == "__main__":
    chz.entrypoint(main)
