"""Fixed problem pools, committed as JSONL under data/pools/ (sources in data/sources/, curation records in
data/curation/, keyword lists in data/keywords/). Each pool pairs a task with a recipe for its problems; `rl make-pools`
builds a pool's files once, and every later load re-checks each stored answer against the task."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from src.tasks import Problem, task_from_spec
from src.tasks.math_gates import CotControlMathGate, CotControlWarmup, RepeatSentencesMathGate, WordSuppressionMathGate
from src.tasks.maths import MathQA
from src.tasks.repeat_sentences import RepeatSentencesDirectCorrect
from src.utils.files import write_jsonl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SOURCES, CURATION, KEYWORDS, POOLS_DIR = (DATA_DIR / d for d in ("sources", "curation", "keywords", "pools"))
SIDE_QUESTIONS = SOURCES / "side_mmlu_pro_mini.jsonl"
SIDE_SOURCE = "YuehHanChen/CoTControl@5d78aef CoT-Control-QA/datasets/mmlu_pro_mini_w_keyword.csv (MIT)"


@cache
def side_questions() -> list[dict]:
    return [json.loads(line) for line in SIDE_QUESTIONS.read_text().splitlines()]


def _provenance(path: Path) -> dict:
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _side_inputs(side: dict) -> dict:
    return {"side_id": side["id"], "side_question": side["question"], "side_options": side["options"],
            "side_answer": side["answer"]}


def split_side_questions(seed: int) -> dict[str, list[dict]]:
    """dev = the first 20 side questions (used while developing prompts); the other 280 are shuffled into
    train (180) and eval (100), keeping questions with identical text in the same split."""
    sides = side_questions()
    groups: dict[str, list[dict]] = {}
    for side in sides[20:]:
        groups.setdefault(" ".join(side["question"].lower().split()), []).append(side)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train: list[dict] = []
    eval_: list[dict] = []
    for key in keys:
        (train if len(train) < 180 else eval_).extend(groups[key])
    return {"dev": sides[:20], "train": train, "eval": eval_}


@dataclass(frozen=True)
class SideQuestionsOnly:
    """Each split's side questions as problems on their own (answer = the correct option letter)."""

    side_split_seed: int
    splits: tuple[str, ...] = ("train", "eval")

    def build(self, name: str, task) -> dict[str, list[Problem]]:
        split = split_side_questions(self.side_split_seed)
        return {s: [Problem(f"{name}-{s}-{i:04d}", s, _side_inputs(side), side["answer"])
                    for i, side in enumerate(split[s])] for s in self.splits}

    def sources(self) -> dict:
        return {"side_questions": {**_provenance(SIDE_QUESTIONS), "upstream": SIDE_SOURCE},
                "side_split_seed": self.side_split_seed}


@dataclass(frozen=True)
class FromSource:
    """Rows of a built source file (scripts/data/build_math_sources.py), which assigns each row its split, minus the
    audited exclusions in data/curation/math_exclusions.jsonl."""

    file: str  # under data/sources/
    splits: tuple[str, ...] = ("train", "val", "test")

    def build(self, name: str, task) -> dict[str, list[Problem]]:
        excluded = {json.loads(line)["id"] for line in (CURATION / "math_exclusions.jsonl").read_text().splitlines()}
        rows = [r for r in map(json.loads, (SOURCES / self.file).read_text().splitlines()) if r["id"] not in excluded]
        # Problem ids are the source ids, so they stay stable when the curation changes.
        return {s: [Problem(r["id"], s, {"problem": r["problem"], "gold": r["gold"], "source_id": r["id"]}, r["gold"])
                    for r in rows if r["split"] == s] for s in self.splits}

    def sources(self) -> dict:
        return {"source": _provenance(SOURCES / self.file),
                "curation": {"math_exclusions.jsonl": _provenance(CURATION / "math_exclusions.jsonl")},
                "report": _provenance(CURATION / "math_sources_report.json")}


@dataclass(frozen=True)
class WithKeywords:
    """FromSource rows that have a selected word-suppression keyword (scripts/data/select_keywords.py), in source order,
    with the keyword, its banned forms and its family added to the inputs."""

    keywords: str  # under data/keywords/; several files comma-separated (e.g. a train and a val file)
    source: FromSource = FromSource("math_gsm8k.jsonl")

    def build(self, name: str, task) -> dict[str, list[Problem]]:
        chosen = {r["id"]: r for f in self.keywords.split(",") for r in map(json.loads, (KEYWORDS / f).read_text().splitlines())}
        built = {s: [Problem(p.id, s, {**p.inputs, "keyword": chosen[p.id]["keyword"], "forms": chosen[p.id]["forms"],
                                       "keyword_family": chosen[p.id]["family"]}, p.answer)
                     for p in problems if p.id in chosen] for s, problems in self.source.build(name, task).items()}
        return {s: problems for s, problems in built.items() if problems}

    def sources(self) -> dict:
        return {**self.source.sources(), "keywords": {f: _provenance(KEYWORDS / f) for f in self.keywords.split(",")}}


@dataclass(frozen=True)
class Without:
    """Another recipe's problems minus the ids listed in data/curation/<exclusions> (e.g. warm-up sources and
    calibration questions kept out of RL training)."""

    recipe: FromSource | WithKeywords
    exclusions: str

    def build(self, name: str, task) -> dict[str, list[Problem]]:
        ids = set(json.loads((CURATION / self.exclusions).read_text())["ids"])
        return {s: [p for p in problems if p.id not in ids] for s, problems in self.recipe.build(name, task).items()}

    def sources(self) -> dict:
        return {**self.recipe.sources(), "exclusions": _provenance(CURATION / self.exclusions)}


@dataclass(frozen=True)
class PoolSpec:
    task: MathQA | RepeatSentencesDirectCorrect | RepeatSentencesMathGate | CotControlMathGate | CotControlWarmup \
        | WordSuppressionMathGate
    recipe: SideQuestionsOnly | FromSource | WithKeywords | Without


GSM8K = FromSource("math_gsm8k.jsonl")
WS_KEYWORDS_V1 = WithKeywords("word_suppression_keywords_train_0_1000.jsonl")

POOLS: dict[str, PoolSpec] = {
    # ---- the post's training and evaluation pools ----
    # the gates on curated GSM8K (splits train/val/test); gsm8k_gate's files are symlinks to math_gsm8k's (same rows)
    "gsm8k_gate": PoolSpec(RepeatSentencesMathGate(), GSM8K),
    # the other rules: the gate minus each rule's warm-up source and calibration questions
    # (data/curation/rl_exclusions_<rule>.json); warm-ups are added at train time via task_options
    **{f"gsm8k_{c}_rl_gate": PoolSpec(CotControlMathGate(c), Without(GSM8K, f"rl_exclusions_{c}.json"))
       for c in ("lowercase_thinking", "ok_start_sentence", "meow_between_words")},
    # word suppression: keywords v2 for train[0:1000] (training) and val[0:130] (evaluation)
    "gsm8k_word_suppression_rl_gate": PoolSpec(WordSuppressionMathGate(), Without(
        WithKeywords("word_suppression_keywords_train_0_1000.v2.jsonl,word_suppression_keywords_val_0_130.v2.jsonl"),
        "rl_exclusions_word_suppression.json")),
    # the direct-reward control: CoT-Control's MMLU-Pro questions on their own (train 180, eval 100)
    "repeat_direct_correct": PoolSpec(RepeatSentencesDirectCorrect(), SideQuestionsOnly(side_split_seed=20261011)),
    # ---- pools the data builders sample (scripts/data/) ----
    # plain GSM8K, for keyword selection traces
    "math_gsm8k": PoolSpec(MathQA(), GSM8K),
    # the gates' warm-up message alone, to sample warm-up pool traces (the rule set by task_options)
    "gsm8k_cot_warmup": PoolSpec(CotControlWarmup(), GSM8K),
    "gsm8k_word_suppression_train_warmup": PoolSpec(CotControlWarmup("word_suppression"), WS_KEYWORDS_V1),
}

def build_pool(name: str) -> dict[str, list[Problem]]:
    return POOLS[name].recipe.build(name, POOLS[name].task)


def pool_sources(name: str) -> dict:
    """Provenance of a derived pool's inputs (file hashes), recorded in each run's manifest."""
    return POOLS[name].recipe.sources()


def pool_path(name: str, split: str) -> Path:
    return POOLS_DIR / f"{name}_{split}.jsonl"


def write_pool(name: str) -> list[Path]:
    built = build_pool(name)
    if any(pool_path(name, split).exists() for split in built):
        raise FileExistsError(f"{name} pool files already exist; committed pools are never rewritten")
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    for split, problems in built.items():
        write_jsonl(pool_path(name, split), [p.to_row() for p in problems])
    return [pool_path(name, split) for split in built]


def load_pool(name: str, split: str) -> list[Problem]:
    """The committed problems, with every stored answer re-checked against the task."""
    task = POOLS[name].task
    problems = [Problem.from_row(json.loads(line)) for line in pool_path(name, split).read_text().splitlines()]
    for p in problems:
        if task.solve(p.inputs) != p.answer:
            raise ValueError(f"{p.id}: stored answer {p.answer} != {task.name} of {p.inputs}")
    return problems


def pool_task(name: str, task_options: str = ""):
    """The pool's task, with `task_options` (JSON) merged into its spec: how rollouts and training get e.g. warm-ups."""
    task = POOLS[name].task
    return task_from_spec({**task.spec(), **json.loads(task_options)}) if task_options else task
