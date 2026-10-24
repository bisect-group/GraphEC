import subprocess
import time
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import queue_pipeline  # noqa: E402


def test_run_direct_assigns_one_gpu_per_run(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda argv, check: commands.append(argv[-1]))
    jobs = [
        {
            "split_group": "s",
            "train_argv": ["train", str(seed)],
            "eval_argv": ["eval", str(seed)],
        }
        for seed in (1, 2)
    ]
    queue_pipeline.run_direct(jobs, ["2", "3"], 1)
    runs = [c for c in commands if " train " in c or " eval " in c]
    assert len(runs) == 4
    assert {c.split("CUDA_VISIBLE_DEVICES=")[1][0] for c in runs} <= {"2", "3"}


def stub_run(monkeypatch, log: list, fail: str | None = None, eval_seconds: float = 0.0) -> None:
    def run(argv, check):
        name = argv[-1].split()[-1]
        log.append(("start", name, time.monotonic()))
        if name.startswith("eval"):
            time.sleep(eval_seconds)
        log.append(("end", name, time.monotonic()))
        if name == fail:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", run)


def jobs_for(*seeds: int) -> list[dict]:
    return [{"train_argv": [f"train{seed}"], "eval_argv": [f"eval{seed}"]} for seed in seeds]


def test_next_training_starts_while_the_previous_evaluation_runs(monkeypatch) -> None:
    log: list = []
    stub_run(monkeypatch, log, eval_seconds=0.3)
    queue_pipeline.run_direct(jobs_for(1, 2), ["0"], 1)
    times = {(kind, name): t for kind, name, t in log}
    assert times["start", "train2"] < times["end", "eval1"]
    assert ("end", "eval2") in times


def test_background_evaluations_are_capped(monkeypatch) -> None:
    log: list = []
    stub_run(monkeypatch, log, eval_seconds=0.2)
    queue_pipeline.run_direct(jobs_for(1, 2, 3, 4), ["0"], 1, max_evals=1)
    running = peak = 0
    for kind, name, _ in sorted(log, key=lambda e: e[2]):
        if name.startswith("eval"):
            running += 1 if kind == "start" else -1
            peak = max(peak, running)
    assert peak == 1


def test_failed_evaluation_fails_the_pipeline_after_all_runs(monkeypatch) -> None:
    log: list = []
    stub_run(monkeypatch, log, fail="eval1")
    with pytest.raises(RuntimeError, match="1 training or evaluation"):
        queue_pipeline.run_direct(jobs_for(1, 2), ["0"], 1)
    assert ("end", "eval2") in {(kind, name) for kind, name, _ in log}


def test_failed_training_starts_no_evaluation(monkeypatch) -> None:
    log: list = []
    stub_run(monkeypatch, log, fail="train1")
    with pytest.raises(RuntimeError):
        queue_pipeline.run_direct(jobs_for(1, 2), ["0"], 1)
    assert "eval1" not in {name for _, name, _ in log}
