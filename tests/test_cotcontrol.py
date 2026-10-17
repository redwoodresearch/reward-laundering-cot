"""The CoT-Control-QA checks (scripts/experiments/cotcontrol_checks.py) as grading and the Docent export use them, on
small offline runs with a stand-in for the CoT-Control repo's loader, prompt builder and grader, so they run without
the external clone; nothing is graded or uploaded unless the run's scope, prompts and receipts all check out."""

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.experiments import cotcontrol_checks as checks

UPSTREAM = {
    "run_cceval.py": '''
        def load_evaluation_data(path, test_mode, mode):
            return [{"id": i, "question": f"q{i} of {path.stem}"} for i in range(2)]

        def create_prompt_for_sample(sample, mode, model_name, *flags):
            return "", f"{sample['question']} [{mode}]"
    ''',
    "grade_compliance_csv.py": '''
        import pandas as pd
        CALLS = []
        DROP_LAST = False  # a faulty grader that loses a row

        def grade_csv(input_path, output_path, num_workers=1, skip_meta_discussion=True):
            CALLS.append(input_path)
            df = pd.read_csv(input_path, keep_default_na=False)
            df["compliance"] = ["" if r == "UNGRADED" else int(r == r.lower()) for r in df["reasoning"]]
            df.iloc[: len(df) - DROP_LAST].to_csv(output_path, index=False)
    ''',
}


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    qa = tmp_path / "upstream"
    qa.mkdir()
    for name, code in UPSTREAM.items():
        (qa / name).write_text(textwrap.dedent(code))
    monkeypatch.setattr(checks, "check_clone", lambda qa_dir: checks.PINNED_COMMIT)
    monkeypatch.syspath_prepend(str(qa))
    import grade_compliance_csv

    grade_compliance_csv.CALLS.clear()
    grade_compliance_csv.DROP_LAST = False
    yield qa


def make_run(tmp_path, qa, *, status="complete", commit=checks.PINNED_COMMIT, modes=("lowercase_thinking",),
             rows=None, datasets=("mmlu_pro_mini_w_keyword",), reasoning=("all lower", "Not Lower"), name="run"):
    """A finished run of the stub benchmark: two questions per (dataset, mode), with the given reasoning."""
    run = tmp_path / name
    (run / "csv").mkdir(parents=True)
    config = {"label": "m", "datasets": ",".join(datasets), "modes": ",".join(modes), "n": None,
              "cotcontrol_dir": str(qa)}
    (run / "manifest.json").write_text(json.dumps({"status": status, "config": config, "cotcontrol_commit": commit}))
    for d in datasets:
        for mode in modes:
            default = [(mode, f"q{i} of {d} [{mode}]", r) for i, r in enumerate(reasoning)]
            pd.DataFrame([{"dataset": f"{d}.csv", "mode": m, "question": q, "reasoning": r, "response": "ANSWER: A",
                           "extracted_answer": "A", "correct_answer": "A", "correctness": True, "control_value": ""}
                          for m, q, r in (rows or default)]).to_csv(run / "csv" / f"r_{d}_{mode}.csv", index=False)
    return run


def grade(run, tmp_path, **kw):
    from scripts.experiments.cotcontrol_grade import Config, main

    main(Config(out_dir=str(run), cost_file=str(tmp_path / "costs.jsonl"), **kw))
    return json.loads((run / "summary.json").read_text())


def grader_calls():
    import grade_compliance_csv

    return grade_compliance_csv.CALLS


def test_a_run_is_scored_over_its_own_scope_and_marked_a_subset(tmp_path, upstream):
    summary = grade(make_run(tmp_path, upstream), tmp_path)
    assert (summary["controllability_score"], summary["pairs"], summary["full_benchmark"]) == (50.0, 1, False)


@pytest.mark.parametrize("change", ["unfinished", "unpinned_commit", "mixed_pairs", "wrong_prompts", "missing_row"])
def test_a_bad_run_is_refused_before_anything_is_graded(tmp_path, upstream, change):
    mode, q = "lowercase_thinking", "q{} of mmlu_pro_mini_w_keyword [lowercase_thinking]"
    rows = {"mixed_pairs": [(mode, q.format(0), "a"), ("uppercase_thinking", q.format(1), "b")],
            "wrong_prompts": [(mode, q.format(0), "a"), (mode, q.format(0), "b")],  # right count, wrong questions
            "missing_row": [(mode, q.format(0), "a")]}.get(change)
    run = make_run(tmp_path, upstream, status="running" if change == "unfinished" else "complete",
                   commit="c" if change == "unpinned_commit" else checks.PINNED_COMMIT, rows=rows)
    with pytest.raises(SystemExit):
        grade(run, tmp_path)
    assert grader_calls() == [] and not (run / "summary.json").exists()


def test_a_graded_file_is_reused_only_with_its_receipt_intact(tmp_path, upstream):
    run = make_run(tmp_path, upstream)
    grade(run, tmp_path)
    grade(run, tmp_path)  # unchanged: reused, not regraded
    receipt = next((run / "graded").glob("*.receipt.json"))
    receipt.write_text(json.dumps({**json.loads(receipt.read_text()), "backfilled": "written after grading"}))
    grade(run, tmp_path)  # a backfilled receipt's note is not compared
    assert len(grader_calls()) == 1
    graded = next((run / "graded").glob("*_graded.csv"))
    pd.read_csv(graded).assign(compliance=1).to_csv(graded, index=False)  # an edited verdict: 50% would read 100%
    with pytest.raises(SystemExit, match="receipt"):
        grade(run, tmp_path)
    graded.unlink()
    grade(run, tmp_path)
    with pytest.raises(SystemExit, match="receipt"):
        grade(run, tmp_path, meta=True)  # other grading options


def test_a_graded_file_missing_a_row_is_neither_sealed_nor_scored(tmp_path, upstream):
    import grade_compliance_csv

    run = make_run(tmp_path, upstream)
    grade_compliance_csv.DROP_LAST = True  # the one row left is compliant: it would score 100%
    with pytest.raises(SystemExit, match="rows"):
        grade(run, tmp_path)
    assert not list((run / "graded").glob("*.receipt.json")) and not (run / "summary.json").exists()


def exportable(run, monkeypatch, reasoning=None):
    """A graded run with the saved samples and manifest fields the export reads, and no tokenizer download."""
    rows = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted((run / "csv").glob("*.csv"))])
    samples = [{"messages": [{"role": "user", "content": q}], "reasoning": r, "response": "ANSWER: A", "raw_text": "x",
                "prompt_text": "", "is_clean": True, "stop_reason": "stop", "n_sample_tokens": 1}
               for q, r in zip(rows["question"], reasoning or rows["reasoning"], strict=True)]
    (run / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["renderer"], manifest["config"]["model"], manifest["config"]["model_path"] = "r", "m", None
    (run / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("src.backends.tinker.clients.get_tokenizer_and_renderer", lambda model, name: (None, None))
    monkeypatch.setattr("src.backends.tinker.clients.normalized_messages", lambda renderer, messages: messages)
    return run


def test_the_export_preflight_refuses_bad_labels_and_unmatched_runs(tmp_path, upstream, monkeypatch):
    from scripts.docent.export import cotcontrol_preflight

    run = make_run(tmp_path, upstream)
    with pytest.raises(SystemExit, match="not graded"):
        cotcontrol_preflight([run], ["m"])
    grade(run, tmp_path)
    with pytest.raises(SystemExit, match="samples"):
        cotcontrol_preflight([run], ["m"])  # no saved samples
    exportable(run, monkeypatch)
    assert cotcontrol_preflight([run], ["m"])["m"]["rows"] == 2
    for labels in (["a/b"], [""], ["m", "m"], []):
        with pytest.raises(SystemExit):
            cotcontrol_preflight([run] * len(labels), labels)
    summary = json.loads((run / "summary.json").read_text())
    (run / "summary.json").write_text(json.dumps({**summary, "controllability_score": 99}))
    with pytest.raises(SystemExit, match="recomputed: 50.0"):
        cotcontrol_preflight([run], ["m"])


def test_the_export_refuses_a_graded_row_that_is_not_its_saved_sample(tmp_path, upstream, monkeypatch):
    from scripts.docent import export

    run = make_run(tmp_path, upstream)
    grade(run, tmp_path)
    exportable(run, monkeypatch, reasoning=["all lower", "SOMETHING ELSE"])
    graded = export.cotcontrol_preflight([run], ["m"])["m"]["graded"]
    with pytest.raises(SystemExit, match="not the saved sample"):
        list(export.cotcontrol_agent_runs(run, "m", graded))


def test_the_export_receipt_keeps_the_collection_id_in_a_new_directory(tmp_path, upstream, monkeypatch):
    from scripts.docent import export

    class FakeDocent:
        frontend_url = "https://docent"

        def create_collection(self, name, metadata, description):
            return "collection-1"

        def add_agent_runs(self, collection_id, runs):
            pass

    run = make_run(tmp_path, upstream)
    grade(run, tmp_path)
    exportable(run, monkeypatch)
    monkeypatch.setattr(export, "Docent", FakeDocent)
    monkeypatch.setattr(export, "verify_cotcontrol", lambda client, cid, names: {"identities_match": True})
    out = tmp_path / "fresh" / "receipts"
    export.cotcontrol(run_dirs=str(run), models="m", name="n", out=str(out))
    receipt = json.loads((out / "docent.json").read_text())
    assert (receipt["collection_id"], receipt["status"], receipt["source_rows"]) == ("collection-1", "verified", 2)


def test_the_figure_computation_reads_only_whole_validated_runs(tmp_path, upstream, monkeypatch):
    from analysis.compute import cotcontrol, runs

    root = tmp_path / "results"
    (root / "cotcontrol").mkdir(parents=True)
    run = make_run(root / "cotcontrol", upstream, datasets=checks.DATASETS, modes=checks.MODES, name="full",
                   reasoning=("all lower", "UNGRADED"))  # one compliant question, one the grader left ungraded
    grade(run, tmp_path)
    monkeypatch.setattr(runs, "COTCONTROL", {"m": "full"})
    with pytest.raises(SystemExit, match="samples"):
        cotcontrol.compute(root)  # the checks the export makes come first
    exportable(run, monkeypatch)
    modes = cotcontrol.compute(root)["m"]["modes"]
    assert {m: (v["compliance"], v["n"]) for m, v in modes.items()} == {**{m: (50.0, 6) for m in checks.MODES},
                                                                         "score": (50.0, 54)}  # ungraded: not compliant
    # each dataset's questions are drawn once per resample for all nine modes, which here comply identically
    assert all(modes[m]["ci"] == modes["score"]["ci"] for m in checks.MODES)
    subset = make_run(root / "cotcontrol", upstream, name="subset")
    grade(subset, tmp_path)
    exportable(subset, monkeypatch)
    monkeypatch.setattr(runs, "COTCONTROL", {"m": "subset"})
    with pytest.raises(SystemExit, match="full benchmark"):
        cotcontrol.compute(root)


def test_full_benchmark_needs_every_dataset_mode_and_question():
    config = {"datasets": ",".join(checks.DATASETS), "modes": ",".join(checks.MODES), "n": None}
    assert checks.is_full({"config": config})
    assert not checks.is_full({"config": {**config, "n": 5}})
    assert not checks.is_full({"config": {**config, "modes": "lowercase_thinking"}})


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "external/CoTControl/CoT-Control-QA").exists(),
                    reason="needs the CoT-Control clone in external/CoTControl")
def test_the_pinned_clone_passes_the_clone_check():
    assert checks.check_clone(Path(__file__).resolve().parents[1] / "external/CoTControl/CoT-Control-QA") == checks.PINNED_COMMIT
