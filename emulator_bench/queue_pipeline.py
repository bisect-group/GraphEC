from __future__ import annotations

import argparse
import queue
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor

from .dataset_adapter import select_split_groups
from .train import DEFAULT_MAX_RESIDUES
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    conda_python,
    find_spooler,
    runs_root,
    seed_run_root_for_split,
    shell_join,
    split_group_slug,
    submit_ts_job,
    wait_for_ts_jobs,
    write_json,
)

MAX_BACKGROUND_EVALS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache features, then train and evaluate GraphEC for every split and seed"
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", help="default: every split folder")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--env-name", default="mldb")
    parser.add_argument("--execution-mode", choices=["direct", "ts"], default="direct")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--prottrans-model-path", default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="each run gets one of these GPUs through CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--runs-per-gpu", "--runs_per_gpu", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast in train and evaluate")
    parser.add_argument("--no-tf32", action="store_true", help="fp32 matmuls without TF32")
    parser.add_argument("--compile", action="store_true", help="torch.compile in train only")
    parser.add_argument("--active-site-procs-per-gpu", type=int, default=1)
    parser.add_argument("--max-residues", type=int, default=DEFAULT_MAX_RESIDUES)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--skip-feature-generation", action="store_true")
    return parser.parse_args()


def with_repo_prefix(command: list[str], gpu: str) -> str:
    return (
        f"cd {shell_join([BASELINE_ROOT])} && env CUDA_VISIBLE_DEVICES={gpu} {shell_join(command)}"
    )


def build_commands(args: argparse.Namespace, split_group: str, seed: int) -> dict[str, list[str]]:
    python = conda_python(args.env_name)
    flags = ["--amp"] if args.amp else []
    flags += ["--no-tf32"] if args.no_tf32 else []
    flags += ["--max-residues", str(args.max_residues)]
    common = ["--split-group", split_group, "--out-root", args.out_root, "--seed", str(seed)]
    return {
        "train": [
            *python,
            "-m",
            "emulator_bench.train",
            *common,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            *flags,
            *(["--compile"] if args.compile else []),
        ],
        "eval": [
            *python,
            "-m",
            "emulator_bench.evaluate",
            *common,
            "--eval-split",
            args.eval_split,
            "--batch-size",
            str(args.batch_size),
            *flags,
        ],
    }


def cache_command(args: argparse.Namespace, split_group: str) -> list[str]:
    command = [
        *conda_python(args.env_name),
        "-m",
        "emulator_bench.cache_features",
        "--dataset-root",
        args.dataset_root,
        "--split-group",
        split_group,
        "--out-root",
        args.out_root,
    ]
    if args.execution_mode == "direct":
        command += ["--num-gpus", str(len(args.gpus))]
    command += ["--active-site-procs-per-gpu", str(args.active_site_procs_per_gpu)]
    if args.limit_per_split is not None:
        command += ["--limit-per-split", str(args.limit_per_split)]
    if args.prottrans_model_path:
        command += ["--prottrans-model-path", args.prottrans_model_path]
    if args.skip_feature_generation:
        command.append("--skip-feature-generation")
    return command


def run_direct(
    jobs: list[dict], gpus: list[str], runs_per_gpu: int, max_evals: int = MAX_BACKGROUND_EVALS
) -> None:
    """Train each job on a GPU slot. A finished training frees its slot at once; its evaluation
    runs in the background on the same GPU, at most `max_evals` at a time. Raises after all runs
    end if any training or evaluation failed."""
    slots: queue.Queue[str] = queue.Queue()
    for _ in range(runs_per_gpu):
        for gpu in gpus:
            slots.put(gpu)

    def run_command(argv: list[str], gpu: str) -> None:
        subprocess.run(["bash", "-lc", with_repo_prefix(argv, gpu)], check=True)

    def train(job: dict, evaluations: ThreadPoolExecutor) -> Future:
        gpu = slots.get()
        try:
            run_command(job["train_argv"], gpu)
        finally:
            slots.put(gpu)
        return evaluations.submit(run_command, job["eval_argv"], gpu)

    with ThreadPoolExecutor(max_workers=max_evals) as evaluations:
        with ThreadPoolExecutor(max_workers=len(gpus) * runs_per_gpu) as trainings:
            training_futures = [trainings.submit(train, job, evaluations) for job in jobs]
    evaluation_futures = [f.result() for f in training_futures if f.exception() is None]
    failures = [
        future.exception()
        for future in training_futures + evaluation_futures
        if future.exception() is not None
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} training or evaluation runs failed: {failures}")


def main() -> None:
    args = parse_args()
    seeds = args.seed or list(DEFAULT_SEEDS)
    runs = runs_root(args.out_root)
    split_groups = [g.name for g in select_split_groups(args.dataset_root, args.split_group)]
    if args.execution_mode == "ts":
        find_spooler(args.spooler_bin)
    jobs = []
    cache_jobs = {}
    for split_group in split_groups:
        cache_argv = cache_command(args, split_group)
        if args.execution_mode == "direct":
            all_gpus = ",".join(args.gpus)
            subprocess.run(["bash", "-lc", with_repo_prefix(cache_argv, all_gpus)], check=True)
        else:
            cache_jobs[split_group] = submit_ts_job(
                with_repo_prefix(cache_argv, args.gpus[0]),
                label=f"graphec-cache-{split_group_slug(split_group)}",
                log_name=f"graphec-cache-{split_group_slug(split_group)}.log",
                gpus=1,
                spooler_bin=args.spooler_bin,
            )
        for seed in seeds:
            commands = build_commands(args, split_group, seed)
            jobs.append(
                {
                    "split_group": split_group,
                    "seed": seed,
                    "train_argv": commands["train"],
                    "eval_argv": commands["eval"],
                    "seed_run_root": str(seed_run_root_for_split(split_group, seed, runs)),
                }
            )
    write_json(runs / "queued_jobs.json", jobs)
    if args.execution_mode == "direct":
        run_direct(jobs, args.gpus, args.runs_per_gpu)
        return
    # ts assigns GPUs round-robin; set `ts -S` to len(gpus) * runs_per_gpu.
    ts_ids = []
    for index, job in enumerate(jobs):
        gpu = args.gpus[index % len(args.gpus)]
        slug = f"{split_group_slug(job['split_group'])}_seed{job['seed']}"
        train_id = submit_ts_job(
            with_repo_prefix(job["train_argv"], gpu),
            label=f"graphec-train-{slug}",
            log_name=f"graphec-train-{slug}.log",
            depends_on=[cache_jobs[job["split_group"]]],
            gpus=1,
            spooler_bin=args.spooler_bin,
        )
        ts_ids.append(
            submit_ts_job(
                with_repo_prefix(job["eval_argv"], gpu),
                label=f"graphec-eval-{slug}",
                log_name=f"graphec-eval-{slug}.log",
                depends_on=[train_id],
                gpus=1,
                spooler_bin=args.spooler_bin,
            )
        )
    wait_for_ts_jobs(ts_ids, spooler_bin=args.spooler_bin)


if __name__ == "__main__":
    main()
