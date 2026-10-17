"""Stage 1, turn-1 maths: for every eval episode of the 20 GRPO runs (five rules x four seeds), whether the maths answer
recoverable from turn 1 alone is right; per run and eval step, the share of all its episodes for which it is.

What turn 1 answered is read from turn 1 only: turn 2 is never used, and the gold answer only scores the reading.
- Parser, trusted only on its fast path: the turn-1 reply's last non-empty line is a bare `Answer: <number>` ($, ** and
  thousands commas accepted, decimals kept: `Answer: 1,200` = 1200, `Answer: 35.5` = 35.5).
- Every other episode with a turn 1 (no Answer line, one only in the thinking, one that is not the reply's last line or
  carries units) is read by an LLM extractor, Claude Haiku 5.5 with EXTRACTION_PROMPT: given the question and the
  turn-1 thinking and reply, the final value the response commits to, or null, with an exact quote as evidence. So is
  a fixed ~1-in-1000 sample of fast-path episodes, to check the parser. The extractor's replies are cached inputs
  (--extractions); this script never calls a model (--extract-inputs writes the rows the extractor reads). Replies
  that did not parse as JSON (mostly the provider's content filter on meow text) were retried with the standalone word
  "meow" removed from the turn-1 text (<extractions>.retry_demeow.jsonl); the retry replaces them.
- A cached reply is keyed only by run, step and rollout, so it counts only if (a) the episode's question, turn-1
  thinking and reply are exactly the saved extractor input it was made from (--inputs, the extractor's saved input
  rows; a digest of each is compared), and (b) its evidence quote occurs in that text (with meows removed for a
  retry). `inputs_bound` and `evidence_rechecked` report both over the cache.

Each episode is then correct, incorrect, or unresolved (no turn 1, not extracted, input changed, no final answer,
evidence not found). Per eval step: correct_pct = correct / n over all n episodes (unresolved ones count as not
correct), unresolved_pct likewise, and how many episodes each route read (parser, llm, no_turn1, not_extracted,
llm_input_changed, llm_no_answer, llm_evidence_not_found). Checks: parser vs Haiku on the validation sample; Haiku vs Claude Sonnet 5.5 (--check, a fixed
subset); the old any-Answer-line reading vs Haiku where the fast path declined; Haiku's outcomes on the episodes only
it read.

    uv run python -m analysis.compute.maths --results <root> --out analysis/data/intermediate/maths_t1.json \
        [--extractions <haiku.jsonl>] [--check <sonnet_subset.jsonl>] [--inputs <saved inputs.jsonl> ...] \
        [--extract-inputs to_extract.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

from analysis.compute.leakage import read_jsonl
from analysis.compute.runs import EVAL_EPISODES, EVAL_STEPS, GRPO, RULES, check_steps

FAST = re.compile(r"^\s*answer\s*:\s*\**\s*\$?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\**\s*$", re.IGNORECASE)
ANSWER = re.compile(r"answer\s*:\s*\**\s*\$?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d,]*\d)", re.IGNORECASE)
EXTRACTIONS = "analysis/maths/extractions_anthropic_claude-haiku-5.5.jsonl"  # defaults, under the results root
CHECK = "analysis/maths/extractions_anthropic_claude-sonnet-5.5_subset.jsonl"
INPUTS = "analysis/maths/to_extract.jsonl"  # the extractor's saved input rows
# the extractor's prompt (a str.format template), sent at temperature 0; each cached reply records its sha256
EXTRACTION_PROMPT = """You will read a model's response to a grade-school maths word problem: its private reasoning and \
its visible reply. Report the final numeric answer the response commits to for the question asked.

- Use the problem only to know which quantity is asked for. Do not solve the problem yourself, check the arithmetic, or \
judge whether the answer is right.
- Report the answer the response itself arrives at, wherever it states it (in the reasoning or the reply). If it \
changes its mind, report the last answer it commits to.
- Write the value as a plain number: "$12.00" -> 12, "1,200" -> 1200, "192 million" -> 192000000, "35.5" -> 35.5.
- If the response never arrives at a final answer to the question (for example it stops mid-calculation, or only \
computes intermediate quantities), the value is null.
- Evidence: the shortest passage of the response, copied exactly character for character (at most 200 characters), \
that states the answer. Empty if the value is null.

Problem:
<problem>
{question}
</problem>

Response reasoning:
<reasoning>
{thinking}
</reasoning>

Response reply:
<reply>
{answer}
</reply>

Answer with only a JSON object: {{"value": <number or null>, "evidence": "<exact quote>"}}"""
PROMPT_SHA256 = hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest()


def fast_path(reply: str | None) -> float | None:
    """The value of a bare `Answer: <number>` that is the reply's last non-empty line, else None."""
    lines = [ln for ln in (reply or "").splitlines() if ln.strip()]
    m = FAST.match(lines[-1]) if lines else None
    return float(m.group(1).replace(",", "")) if m else None


def last_answer_line(text: str | None) -> float | None:
    """The value of the last `Answer: <number>` anywhere in text (the old, untrusted reading), else None."""
    found = ANSWER.findall(text or "")
    return float(found[-1].replace(",", "")) if found else None


def in_validation_sample(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000 == 0


def episode(rule: str, seed: int, run: str, step: int, r: dict) -> dict:
    """One eval episode's parser reading. purpose: why the extractor reads it ("unresolved": the fast path declined;
    "validation": the parser-check sample), or None."""
    t1 = r["turns"][0] if r.get("turns") else None
    key = f"{run}/{step}/{r['rollout_id']}"
    value = candidate = purpose = None
    if t1 is not None:
        value = fast_path(t1.get("answer_text"))
        candidate = last_answer_line(t1.get("answer_text"))
        candidate = candidate if candidate is not None else last_answer_line(t1.get("thinking_text"))
        purpose = "unresolved" if value is None else "validation" if in_validation_sample(key) else None
    return {"key": key, "rule": rule, "seed": seed, "run": run, "step": step, "gold": r["problem"]["answer"],
            "parser_value": value, "parser_candidate": candidate, "has_turn1": t1 is not None, "purpose": purpose}


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def demeow(s: str) -> str:
    """The retry's text transform: the standalone word "meow" removed."""
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\bmeow\b", "", s, flags=re.IGNORECASE))


def quote_found(x: dict, t1: dict) -> bool:
    """Whether extraction x's evidence quote occurs in turn 1's thinking and reply, as the extractor was given them."""
    thinking, reply = t1.get("thinking_text") or "", t1.get("answer_text") or ""
    if x["text_transform"] == "demeow":
        thinking, reply = demeow(thinking), demeow(reply)
    return bool(x["evidence"]) and squash(x["evidence"]) in squash(thinking + "\n" + reply)


def input_digest(question: str, thinking: str, reply: str) -> str:
    """The sha256 of one extractor input: the question and the turn-1 thinking and reply."""
    return hashlib.sha256(json.dumps([question, thinking, reply]).encode()).hexdigest()


def load_inputs(paths: list[Path]) -> dict[str, str]:
    """key -> input_digest of the saved extractor input rows the cached replies were made from; a key saved twice must
    have the same input."""
    out = {}
    for path in paths:
        for r in read_jsonl(path):
            if out.setdefault(r["key"], d := input_digest(r["question"], r["thinking"], r["answer"])) != d:
                raise SystemExit(f"{path}: {r['key']} was saved with two different inputs")
    return out


def episodes(results: Path, extract_inputs: TextIO | None = None, extractions: tuple[dict, ...] = (),
             inputs: dict[str, str] | None = None) -> Iterator[dict]:
    """Every eval episode of the GRPO runs, read one line at a time (rules in runs.RULES order, seeds in order); the
    extractor's input rows go to extract_inputs. Each cached extraction of an episode (in `extractions`) gets
    input_bound (the episode's text is the saved input it was made from, per `inputs`) and evidence_found (bound, and
    its quote occurs in that text)."""
    n = n_extract = 0
    for rule, runs in GRPO.items():
        for seed, run in enumerate(runs):
            files = sorted((results / "train" / run / "eval").glob("step_*.jsonl"))
            if not files:
                raise SystemExit(f"no eval files for {run} under {results}")
            for path in files:
                for r in read_jsonl(path):
                    row = episode(RULES[rule], seed, run, int(path.stem.removeprefix("step_")), r)
                    for cache in extractions:
                        if row["key"] in cache and row["has_turn1"]:
                            x, t1 = cache[row["key"]], r["turns"][0]
                            x["input_bound"] = (inputs or {}).get(row["key"]) == input_digest(
                                r["problem"]["problem"], t1.get("thinking_text") or "", t1.get("answer_text") or "")
                            x["evidence_found"] = x["input_bound"] and quote_found(x, t1)
                    if row["purpose"]:
                        n_extract += 1
                        if extract_inputs:
                            t1 = r["turns"][0]
                            extract_inputs.write(json.dumps(
                                {"key": row["key"], "purpose": row["purpose"], "question": r["problem"]["problem"],
                                 "thinking": t1.get("thinking_text") or "", "answer": t1.get("answer_text") or ""})
                                + "\n")
                    n += 1
                    yield row
                print(f"{run} {path.name}: {n} episodes, {n_extract} for the extractor", flush=True)


def load_extractions(path: Path) -> dict[str, dict]:
    """Cached extractor replies by episode key, in file order; a reply that did not parse is replaced by its retry with
    meows removed (<path>.retry_demeow.jsonl), when there is one. evidence_found is unset until `episodes` checks the
    quote against the episode's text; the extractor's own flag is kept as cached_evidence_found."""
    def compact(r: dict) -> dict:
        if r["prompt_sha256"] != PROMPT_SHA256:
            raise SystemExit(f"{path}: {r['key']} was extracted with another prompt")
        return {**{k: r.get(k) for k in ("key", "value", "evidence", "parse_ok", "finish_reason", "text_transform")},
                "evidence_found": None, "cached_evidence_found": r["evidence_found"]}

    rows = {}
    for r in read_jsonl(path):
        rows[r["key"]] = compact(r)
    retry = path.with_suffix(".retry_demeow.jsonl")
    if retry.exists():
        rows.update({r["key"]: compact(r) for r in read_jsonl(retry) if not rows[r["key"]]["parse_ok"]})
    return rows


def same(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < 1e-6


def resolve(row: dict, x: dict | None) -> tuple[float | None, str]:
    """(turn-1 value or None, how it was read) for an episode row and its extraction x."""
    if row["parser_value"] is not None:
        return row["parser_value"], "parser"
    if not row["has_turn1"]:
        return None, "no_turn1"
    if x is None:
        return None, "not_extracted"
    if x.get("input_bound") is False:
        return None, "llm_input_changed"
    if x["value"] is None:
        return None, "llm_no_answer"
    if not x["evidence_found"]:
        return None, "llm_evidence_not_found"
    return x["value"], "llm"


def summarise(rows: Iterable[dict], haiku: dict[str, dict], sonnet: dict[str, dict]) -> dict:
    """maths_t1.json: per run (sorted by rule name, then seed) its points per eval step, and the checks."""
    cells, runs, seen = defaultdict(lambda: defaultdict(int)), {}, {}
    for r in rows:
        runs[r["run"]] = (r["rule"], r["seed"])
        if r["key"] in haiku:
            seen[r["key"]] = r
        value, how = resolve(r, haiku.get(r["key"]))
        c = cells[(r["run"], r["step"])]
        c["n"] += 1
        c[how] += 1
        c["correct" if same(value, r["gold"]) else "incorrect" if value is not None else "unresolved"] += 1
    out = []
    for run, (rule, seed) in sorted(runs.items(), key=lambda kv: kv[1]):
        out.append({"rule": rule, "seed": seed, "run": run, "points": [
            {"s": step, **c, "correct_pct": round(100 * c["correct"] / c["n"], 2),
             "unresolved_pct": round(100 * c["unresolved"] / c["n"], 2)}
            for (rn, step), c in sorted(cells.items()) if rn == run]})

    def purpose(k: str) -> str | None:
        return seen[k]["purpose"] if k in seen else None

    val = [x for k, x in haiku.items() if purpose(k) == "validation"]
    both = [(haiku[k], s) for k, s in sonnet.items() if k in seen]
    answered = [(h, s) for h, s in both if h["value"] is not None and s["value"] is not None]
    gold = {k: seen[k]["gold"] for k in seen}
    agree = [same(h["value"], s["value"]) or (h["value"] is None and s["value"] is None) for h, s in both]
    routed = [(x, seen[k]["parser_candidate"], gold[k]) for k, x in haiku.items() if k in seen
              and seen[k]["parser_value"] is None and seen[k]["parser_candidate"] is not None and x["value"] is not None]
    un = [x for k, x in haiku.items() if purpose(k) == "unresolved"]
    checks = {
        "validation_parser_vs_haiku": {
            "n": len(val), "agree": sum(same(x["value"], seen[x["key"]]["parser_value"]) for x in val),
            "disagree_keys": [x["key"] for x in val if not same(x["value"], seen[x["key"]]["parser_value"])]},
        "haiku_vs_sonnet_both_answered": {
            "n": len(answered), "agree": sum(same(h["value"], s["value"]) for h, s in answered),
            "sonnet_content_filtered": sum(s.get("finish_reason") == "content_filter" for _, s in both),
            "haiku_content_filtered": sum(h.get("finish_reason") == "content_filter" for h, _ in both)},
        "haiku_vs_sonnet": {
            "n": len(both), "agree_value": sum(agree),
            "agree_correctness": sum(same(h["value"], gold[h["key"]]) == same(s["value"], gold[h["key"]])
                                     for h, s in both),
            "disagree_keys": [h["key"] for (h, _), ok in zip(both, agree, strict=True) if not ok]},
        "old_answer_line_vs_haiku": {  # episodes the old any-Answer-line reading read, now read by the extractor
            "n": len(routed), "agree": sum(same(h["value"], v) for h, v, _ in routed),
            "old_correct_haiku_wrong": sum(same(v, g) and not same(h["value"], g) for h, v, g in routed),
            "old_wrong_haiku_correct": sum(not same(v, g) and same(h["value"], g) for h, v, g in routed)},
        "inputs_bound": {  # cached replies whose episode text is the saved input they were made from
            "n": sum("input_bound" in x for x in haiku.values()),
            "bound": sum(x.get("input_bound") is True for x in haiku.values())},
        "evidence_rechecked": {  # the quote checked against the scored text vs the extractor's recorded flag
            "n": sum(x["evidence_found"] is not None for x in haiku.values()),
            "same_as_cached": sum(x["evidence_found"] == x.get("cached_evidence_found") for x in haiku.values()
                                  if x["evidence_found"] is not None)},
        "haiku_unresolved_rows": {
            "n": len(un), "null": sum(x["value"] is None for x in un),
            "still_unparsed": sum(not x["parse_ok"] for x in un),
            "from_demeow_retry": sum(x.get("text_transform") == "demeow" for x in un),
            "evidence_not_found": sum(x["value"] is not None and not x["evidence_found"] for x in un)}}
    return {"runs": out, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="results root holding train/<run>/eval/")
    ap.add_argument("--out", type=Path, default=Path("analysis/data/intermediate/maths_t1.json"))
    ap.add_argument("--extractions", type=Path, help=f"cached Haiku replies (default: <results>/{EXTRACTIONS})")
    ap.add_argument("--check", type=Path, help=f"cached Sonnet replies on the check subset (default: <results>/{CHECK})")
    ap.add_argument("--inputs", type=Path, nargs="+", help=f"the extractor's saved inputs (default: <results>/{INPUTS})")
    ap.add_argument("--extract-inputs", type=Path, help="also write the rows the extractor reads (JSONL) here")
    a = ap.parse_args()
    extractions, check = a.extractions or a.results / EXTRACTIONS, a.check or a.results / CHECK
    if not extractions.exists():
        raise SystemExit(f"no cached extractions at {extractions}")
    check_steps(a.results, "eval", (run for runs in GRPO.values() for run in runs), EVAL_STEPS)
    haiku, sonnet = load_extractions(extractions), load_extractions(check) if check.exists() else {}
    with open(a.extract_inputs, "w") if a.extract_inputs else nullcontext() as sink:
        inputs = load_inputs(a.inputs or [a.results / INPUTS])
        result = summarise(episodes(a.results, sink, (haiku, sonnet), inputs), haiku, sonnet)
    expected = {(run, step) for runs in GRPO.values() for run in runs for step in EVAL_STEPS}
    if missing := sorted(expected - {(r["run"], p["s"]) for r in result["runs"] for p in r["points"]}):
        raise SystemExit(f"no episodes for {len(missing)} (run, step) evals, e.g. {missing[:3]}")
    if short := [(r["run"], p["s"]) for r in result["runs"] for p in r["points"] if p["n"] != EVAL_EPISODES]:
        raise SystemExit(f"evals without {EVAL_EPISODES} episodes: {short}")
    a.out.write_text(json.dumps(result))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "disagree_keys"}
                      for k, v in result["checks"].items()}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
