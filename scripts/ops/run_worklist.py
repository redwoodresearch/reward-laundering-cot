r"""Run a worklist of heavy jobs (evals, trainers, analyses), at most --parallel at a time, each inside its own
memory-capped systemd scope. Optional: single commands run fine without it; it is for running many jobs on a small
machine without exhausting its memory (many eval processes at once can take a 15 GB machine down).

It needs Linux with systemd user services and cgroup v2 (memory delegated to user services, the default on recent
distributions), and a capped slice for the jobs, created once:

    mkdir -p ~/.config/systemd/user
    printf '[Slice]\nMemoryHigh=6G\nMemoryMax=8G\nMemorySwapMax=0\n' > ~/.config/systemd/user/research.slice
    systemctl --user daemon-reload && systemctl --user start research.slice

(size the limits to the machine; these suit 15 GB). The runner:
- holds a lock for the whole batch, and refuses to start while research.slice is uncapped or still runs a job of ours
  (`rl-*` scope) left by a launcher that died, so batches cannot overlap;
- starts a job only with at least --min-available-gib of MemAvailable (with nothing running and too little memory, it
  stops);
- runs each job as `systemd-run --user --scope` in research.slice with its own MemoryHigh/MemoryMax and no swap, so a
  runaway job is killed inside its scope;
- logs each job's peak memory and memory events, read inside the scope after the job exits (`<log>.live.json` holds
  the latest in-flight observations, in case the runner itself dies);
- on SIGTERM/SIGINT, stops the running jobs' scopes before releasing the lock;
- after the first failure (no automatic retries of paid work), or once `<log>.stop` exists (removed when honoured),
  starts no new job and stops when the running ones finish.

Worklist: JSONL, one job per newline-terminated line: {"name": unique job name, "cmd": [argv...], "cwd": directory, "stdout": file,
"priority": number (default 0; higher starts first, then file order)}. Lines appended while the batch runs are picked
up. Jobs whose "done_if_exists" path exists are skipped. Environment: the caller's, plus KEY=VALUE lines from
--env-file (values never logged).

    python scripts/ops/run_worklist.py worklist.jsonl --log results/ops/<batch>.log.jsonl --env-file .env --parallel 2
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

LOCK = Path.home() / ".local/state/reward-laundering/batch.lock"
SLICE = "research.slice"
PREFIX = "rl-"
CGROUP = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/{SLICE}")
# runs inside the scope: the job, then the scope's final peak and events (the scope is collected once this shell exits)
RECORD = 'f=$1; shift; "$@"; rc=$?; cg=/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup); ' \
         '{ cat "$cg/memory.peak"; cat "$cg/memory.events"; } > "$f"; exit $rc'


class Interrupted(Exception):
    pass


class WorklistError(ValueError):
    pass


@dataclass
class Running:
    index: int
    job: dict
    unit: str
    proc: subprocess.Popen
    out: IO
    final: Path
    start: float = field(default_factory=time.time)
    observed: int = 0


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError):
        return None


def load_env(path: str | None) -> dict[str, str]:
    env = dict(os.environ)
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return env


def read_worklist(path: str) -> list[dict]:
    # only newline-terminated lines: a line still being appended is picked up on a later read
    jobs = [json.loads(line) for line in Path(path).read_text().split("\n")[:-1] if line.strip()]
    if len({j["name"] for j in jobs}) != len(jobs):
        raise WorklistError("duplicate job names in the worklist")
    return jobs


def preflight() -> None:
    if not CGROUP.exists():
        sys.exit(f"{SLICE} is not active ({CGROUP} missing): create and start it (see this script's docstring)")
    limits = {f: (CGROUP / f).read_text().strip() for f in ("memory.high", "memory.max", "memory.swap.max")}
    if "max" in limits.values():
        sys.exit(f"{SLICE} is not capped ({limits}); refusing to run paid work")
    live = [p.name for p in CGROUP.glob(f"{PREFIX}*.scope") if (p / "cgroup.procs").read_text().strip()]
    if live:
        sys.exit(f"jobs from an earlier batch are still running in {SLICE}: {live}; stop them or wait before starting")


def log(fh, **record) -> None:
    fh.write(json.dumps({"time": datetime.now(UTC).isoformat(timespec="seconds"), **record}) + "\n")
    fh.flush()


def start(index: int, job: dict, a, env: dict) -> Running:
    unit = f"{PREFIX}{job['name']}"[:200]
    final = Path(f"{a.log}.{unit}.final")
    cmd = ["systemd-run", "--user", "--scope", "--collect", "--expand-environment=no", f"--slice={SLICE}", f"--unit={unit}",
           "-p", f"MemoryHigh={a.memory_high}", "-p", f"MemoryMax={a.memory_max}", "-p", "MemorySwapMax=0",
           "--", "/bin/sh", "-c", RECORD, "sh", str(final), *job["cmd"]]
    out = open(job.get("stdout", os.devnull), "a")  # noqa: SIM115 - closed in finish()
    proc = subprocess.Popen(cmd, cwd=job.get("cwd"), env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    return Running(index, job, unit, proc, out, final)


def finish(r: Running) -> dict:
    r.out.close()
    stats = {"returncode": r.proc.returncode, "minutes": round((time.time() - r.start) / 60, 1),
             "observed_peak_mib": round(r.observed / 2**20)}
    if r.final.exists():
        lines = r.final.read_text().split("\n")
        stats["peak_mib"] = round(int(lines[0]) / 2**20)
        stats["memory_events"] = {k: int(v) for k, v in (ln.split() for ln in lines[1:] if ln) if int(v)}
        r.final.unlink()
    else:
        stats["final_measurement"] = "missing (the scope's shell was killed)"
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worklist")
    ap.add_argument("--log", required=True)
    ap.add_argument("--env-file")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--memory-high", default="3G")
    ap.add_argument("--memory-max", default="4G")
    ap.add_argument("--min-available-gib", type=float, default=4.0)
    ap.add_argument("--sample-seconds", type=float, default=5.0)
    a = ap.parse_args()
    a.log = str(Path(a.log).resolve())  # jobs run in their own cwd and write <log>.*.final there
    if a.parallel < 1 or a.sample_seconds <= 0 or a.min_available_gib < 0:
        ap.error("--parallel must be >= 1, --sample-seconds > 0 and --min-available-gib >= 0")
    try:
        read_worklist(a.worklist)
    except (WorklistError, json.JSONDecodeError) as e:
        ap.error(f"{a.worklist}: {e}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another batch holds {LOCK}; refusing to start a second one")
    preflight()
    env = load_env(a.env_file)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    stop, live = Path(a.log + ".stop"), Path(a.log + ".live.json")

    def interrupt(signum, _frame):
        raise Interrupted(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    running: list[Running] = []
    seen: set[str] = set()
    halt = None  # why no new job may start
    with open(a.log, "a") as fh:
        log(fh, event="batch_start", worklist=a.worklist, parallel=a.parallel, memory_high=a.memory_high,
            memory_max=a.memory_max)
        try:
            while True:
                for r in [r for r in running if r.proc.poll() is not None]:
                    running.remove(r)
                    stats = finish(r)
                    log(fh, event="job_end", index=r.index, name=r.job["name"], **stats,
                        mem_available_gib=round(mem_available_gib(), 2))
                    if stats["returncode"] != 0 and not halt:
                        halt = ("job_failed", 1)
                        log(fh, event="halting", reason="job_failed", name=r.job["name"], running=[x.job["name"] for x in running])
                if stop.exists() and not halt:  # graceful stop: `touch <log>.stop`
                    stop.unlink()
                    halt = ("stop_requested", 3)
                    log(fh, event="halting", reason="stop_requested", running=[x.job["name"] for x in running])
                jobs = read_worklist(a.worklist)
                pending = sorted((j for j in jobs if j["name"] not in seen), key=lambda j: -j.get("priority", 0))
                while not halt and pending and len(running) < a.parallel:
                    job = pending.pop(0)
                    if job.get("done_if_exists") and Path(job["done_if_exists"]).exists():
                        seen.add(job["name"])
                        log(fh, event="skip_done", name=job["name"])
                        continue
                    avail = mem_available_gib()
                    if avail < a.min_available_gib:
                        if not running:
                            halt = ("low_memory", 2)
                            log(fh, event="halting", reason="low_memory", mem_available_gib=round(avail, 2))
                        break  # otherwise wait for a running job to finish
                    seen.add(job["name"])
                    index = next(i for i, j in enumerate(jobs) if j["name"] == job["name"])
                    log(fh, event="job_start", index=index, name=job["name"], mem_available_gib=round(avail, 2))
                    running.append(start(index, job, a, env))
                if not running:
                    if halt:
                        log(fh, event="batch_stop", reason=halt[0])
                        return halt[1]
                    if not pending:
                        log(fh, event="batch_done")
                        return 0
                for r in running:
                    r.observed = max(r.observed, read_int(CGROUP / f"{r.unit}.scope" / "memory.peak") or 0)
                live.write_text(json.dumps({"time": datetime.now(UTC).isoformat(timespec="seconds"),
                                            "observed_peak_mib": {r.job["name"]: round(r.observed / 2**20) for r in running}}))
                time.sleep(a.sample_seconds)
        except BaseException as e:  # whatever ends the loop (signal, bad worklist, bug) must not leave jobs running
            for r in running:
                subprocess.run(["systemctl", "--user", "stop", f"{r.unit}.scope"], check=False)
                r.proc.wait()
            log(fh, event="batch_stop", reason=f"{'interrupted' if isinstance(e, Interrupted) else 'runner error'} "
                f"({type(e).__name__}: {e}); job scopes stopped", running=[r.job["name"] for r in running])
            if isinstance(e, Interrupted):
                return 130
            raise


if __name__ == "__main__":
    sys.exit(main())
