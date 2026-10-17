"""Recomputes tests/golden/golden.json's values from its frozen inputs with the code it runs under, as a CANDIDATE for
review:

    uv run python -m tests.golden.make_golden            # writes tests/golden/golden.candidate.json
    uv run python -m tests.golden.make_golden --accept   # overwrites golden.json (only when a behaviour change is intended)

golden.json was first computed by the code the post's runs used (its `baseline` revision). Its grading cases are saved
eval episodes of those runs, copied into the file (per setup, one per distinct grade and thinking kind: empty, bare
meow, text); they stay fixed, and only the values computed from them are recomputed here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.golden import freeze

GOLDEN = Path(__file__).with_name("golden.json")

if __name__ == "__main__":
    golden = json.loads(GOLDEN.read_text())
    rates = freeze.learning_rates()
    # the stored multipliers are the schedule's own; lr / 1e-4 can differ in the last bit, so keep them while they
    # still give exactly these learning rates (the test's comparison)
    same = [1e-4 * m for m in golden["lr_multipliers"]] == rates
    golden.update(grades=freeze.grades(golden["cases"]), prompts=freeze.prompts(), orders=freeze.orders(),
                  curriculum_definition=freeze.curriculum_definition(), shuffles=freeze.shuffles(),
                  lr_multipliers=golden["lr_multipliers"] if same else [lr / 1e-4 for lr in rates])
    path = GOLDEN if "--accept" in sys.argv else GOLDEN.with_name("golden.candidate.json")
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"{path}: {len(golden['cases'])} grading cases")
