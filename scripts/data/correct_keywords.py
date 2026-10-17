"""Write a corrected version of committed word-suppression keyword files (data/keywords/) without touching them:
adjudicated form additions/removals and question exclusions from a JSON corrections file, each recorded on its row.

    uv run python -m scripts.data.correct_keywords sources=<file>[,<file>] corrections=data/curation/<file>.json out=<name>.jsonl

corrections = {"reason": ..., "add_forms": {id: [forms]}, "remove_forms": {id: [forms]}, "exclude": {id: reason}}."""

import json
from pathlib import Path

import chz

KEYWORDS = Path(__file__).resolve().parents[2] / "data" / "keywords"


def main(sources: str, corrections: str, out: str) -> None:
    out_path = KEYWORDS / out
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; committed data files are never rewritten")
    fix = json.loads(Path(corrections).read_text())
    rows = [{**json.loads(line), "source_file": name} for name in sources.split(",")
            for line in (KEYWORDS / name).read_text().splitlines()]
    ids = [r["id"] for r in rows]
    named = set(fix.get("add_forms", {})) | set(fix.get("remove_forms", {})) | set(fix.get("exclude", {}))
    if len(ids) != len(set(ids)) or named - set(ids):
        raise SystemExit(f"duplicate source rows, or corrections name absent questions {sorted(named - set(ids))}")
    kept = []
    for r in rows:
        if r["id"] in fix.get("exclude", {}):
            continue
        added, removed = fix.get("add_forms", {}).get(r["id"], []), fix.get("remove_forms", {}).get(r["id"], [])
        if missing := [f for f in removed if f not in r["forms"]]:
            raise SystemExit(f"{r['id']}: cannot remove absent forms {missing}")
        if added or removed:
            r = {**r, "forms": [f for f in r["forms"] if f not in removed] + [f for f in added if f not in r["forms"]],
                 "corrections_v2": {"added": added, "removed": removed, "reason": fix["reason"], "file": corrections}}
        kept.append(r)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{out_path}: {len(kept)} keywords from {len(rows)} ({len(rows) - len(kept)} excluded); "
          f"{sum('corrections_v2' in r for r in kept)} rows corrected")


if __name__ == "__main__":
    chz.entrypoint(main)
