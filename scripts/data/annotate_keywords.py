"""Propose word-suppression keywords for GSM8K questions with a fixed annotation prompt (dataset construction only;
grading stays a whole-word match on the stored forms).

    uv run python -m scripts.data.annotate_keywords split=val n=10 out=results/keyword_annotations/<name>.jsonl

For each question, the annotator returns up to three ranked candidates (word from the question, family, role,
explicit banned forms). Each row keeps the raw response, requested/returned model, prompt sha256 and settings, plus
the deterministic checks (schema, word occurs in the question, word in its forms, forms well-formed). Whether forms
are exactly the right inflections, or a role is central, is for the human audit; the saved rows are the record."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import chz
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("entity", "attribute", "unit", "relation")
PROMPT = """You are helping build a dataset for a "word suppression" test on grade-school math word problems. A model \
will solve the problem while being told not to use one specific word in its reasoning. We need words that are \
load-bearing for THIS problem: a careful solver explaining the solution would naturally need to refer to what the \
word denotes.

Problem:
<problem>
{problem}
</problem>

Propose up to 3 candidate words, ranked from most to least central to setting up and solving this problem. Each \
candidate must:
- be a single word that appears in the problem text (any capitalization), copied in the form it appears;
- denote something the solution must keep track of or use to set up the arithmetic, in one of these families:
  - "entity": a counted or tracked thing (e.g. beetles, snakes, cookies);
  - "attribute": a property that distinguishes quantities (e.g. purple, large, daily);
  - "unit": a unit or measure the quantities are in (e.g. kilograms, miles, hours);
  - "relation": a word naming a specific arithmetic relation or operation (e.g. twice, discount, average, profit);
- not be a person's name, a place name, a number or number word (one, two, dozen, half), or a function word \
(the, how, many, each, of, per);
- not be a word that reasoning about almost any word problem uses in other senses too (more, less, times, total, \
rest, left, remaining, number, amount, cost, price, money, buy, spend, make, get, have), even when it matters here: \
an occurrence in the reasoning must indicate a reference to THIS problem's content.
Prefer candidates from different families where the problem genuinely offers them; never add a weak candidate just \
to fill a family. Return fewer than 3 if fewer are genuinely central.

For each candidate also give "forms": every inflection of that word in the same sense and part of speech, all \
lowercase (nouns: singular and plural; verbs: base, -s, past, past participle, -ing; adjectives: base, \
comparative and superlative where they exist, e.g. large, larger, largest). Include the word itself. Every form must carry the same meaning the word has \
in this problem. No synonyms, no related words of another part of speech, no possessives. Give "role": one sentence on what the word does in this problem's solution.

Answer with only a JSON object:
{{"candidates": [{{"word": "...", "family": "entity|attribute|unit|relation", "forms": ["..."], "role": "..."}}]}}"""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


def checks(problem: str, parsed: dict | None) -> list[str]:
    """Deterministic problems with a parsed response ([] = passes): types, the word occurs in the problem and in its
    forms, well-formed unique forms, distinct candidate words. Semantic correctness is for the audit."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
        return ["no candidates list"]
    cands, errors = parsed["candidates"], []
    if not 1 <= len(cands) <= 3:
        errors.append(f"{len(cands)} candidates")
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            errors.append(f"{i}: not an object")
            continue
        word, forms, role = c.get("word"), c.get("forms"), c.get("role")
        if not (isinstance(word, str) and isinstance(role, str) and role and isinstance(forms, list)
                and forms and all(isinstance(f, str) for f in forms)):
            errors.append(f"{i}: word/role must be strings and forms a nonempty list of strings")
            continue
        if c.get("family") not in FAMILIES:
            errors.append(f"{i}: family {c.get('family')!r}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word) or not re.search(rf"\b{re.escape(word)}\b", problem,
                                                                           re.IGNORECASE):
            errors.append(f"{i}: word {word!r} not a single word of the problem")
        if len(set(forms)) != len(forms) or any(not re.fullmatch(r"[a-z][a-z'-]*", f) for f in forms):
            errors.append(f"{i}: forms {forms!r} malformed")
        if word.lower() not in forms:
            errors.append(f"{i}: {word!r} missing from its forms")
    words = [c["word"].lower() for c in cands if isinstance(c, dict) and isinstance(c.get("word"), str)]
    if len(set(words)) != len(words):
        errors.append("duplicate candidate words")
    return errors


async def annotate(client: httpx.AsyncClient, row: dict, model: str, max_tokens: int) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": PROMPT.format(problem=row["problem"])}]}
    resp = (await client.post("https://openrouter.ai/api/v1/chat/completions", json=body)).raise_for_status().json()
    text = resp["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        parsed = None
    return {"id": row["id"], "split": row["split"], "problem": row["problem"], "parsed": parsed,
            "errors": checks(row["problem"], parsed), "raw": text, "requested_model": model,
            "returned_model": resp.get("model"), "usage": resp.get("usage"), "prompt_sha256": PROMPT_SHA256,
            "settings": {"temperature": 0, "max_tokens": max_tokens},
            "annotated_at": datetime.now(UTC).isoformat()}


def main(split: str, out: str, n: int = 10, offset: int = 0, model: str = "anthropic/claude-opus-5.5",
         max_tokens: int = 2000, concurrency: int = 32) -> None:
    load_dotenv(ROOT / ".env")
    out_path = Path(out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; annotation files are never rewritten")
    rows = [json.loads(line) for line in (ROOT / f"data/pools/math_gsm8k_{split}.jsonl").read_text().splitlines()]
    rows = rows[offset: offset + n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".prompt.txt").write_text(PROMPT)  # the exact template, beside its outputs
    semaphore = asyncio.Semaphore(concurrency)

    async def run() -> list[str]:
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        bad = []
        async with httpx.AsyncClient(headers=headers, timeout=300) as client:
            async def one(row):
                async with semaphore:
                    return await annotate(client, row, model, max_tokens)
            with out_path.open("w") as f:  # noqa: ASYNC230 - each annotation is saved as it arrives
                for done in asyncio.as_completed([one(r) for r in rows]):
                    result = await done
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    bad += [result["id"]] if result["errors"] else []
        return bad

    bad = asyncio.run(run())
    print(f"{out_path}: {len(rows)} annotated, {len(bad)} failing checks {bad}")


if __name__ == "__main__":
    chz.entrypoint(main)
