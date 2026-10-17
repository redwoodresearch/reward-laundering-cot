"""Stage 2, Figure 1's data (analysis/figures/data/fig1.json) from three stage-1 intermediates. One entry per rule (the
figure's rows, in runs.RULES order) with its four GRPO seeds (runs.GRPO order):

- seeds[i].eval, per eval step s: rule-following c (%), leakage lk (% of rule-breakers with right turn-1 maths that
  still got the reward; None when there are none) and its denominator lm, from cond_leakage.json (leakage.eval_rows);
- seeds[i].win, per 3-step window s with at least 3 groups holding both kinds of sample: the maths part m and gate part
  k of the training reward gap (points) and the number of such groups q, from grid.json;
- maths[i]: per eval step, [s, % of the eval's episodes whose turn-1 maths answer is right], from maths_t1.json.

The page draws each seed and the plain mean over seeds. Written by analysis.compute.figures.
"""

from __future__ import annotations

from analysis.compute.leakage import eval_rows
from analysis.compute.runs import GRPO, RULES


def figure1(grid: list[dict], leakage: dict[str, dict], maths: dict) -> list[dict]:
    win = {r["run"]: r["win"] for r in grid}
    points = {r["run"]: r["points"] for r in maths["runs"]}
    return [{"name": name,
             "seeds": [{"eval": [{k: e[k] for k in ("s", "c", "lk", "lm")} for e in eval_rows(leakage, run)],
                        "win": [{k: w[k] for k in ("s", "m", "k", "q")} for w in win[run]]} for run in GRPO[rule]],
             "maths": [[[p["s"], p["correct_pct"]] for p in points[run]] for run in GRPO[rule]]}
            for rule, name in RULES.items()]
