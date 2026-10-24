"""Throughput of the default and the `--compile` model, in fp32 and bf16, and a loss check.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -m emulator_bench.probes.compile_probe --out result.json \
  --work-dir <folder> --split-group random_splits_grouped_sequence

Batches are collated once on the CPU and reused, so the numbers are training-step throughput
without the loader. Each configuration warms up first; compile time is in `warmup_seconds`.
The loss check uses the eval-mode loss of the fresh model on the first batches. Dropout is off
there: the compiled dropout draws other random numbers, so training losses differ step by step.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .. import train
from ..utils import set_seed
from .common import ModelContext, add_common_args, open_split, write_result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eval_mode_loss(model, micro_lists, device, amp):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total, count = 0.0, 0
    with torch.no_grad():
        for micros in micro_lists:
            for micro in micros:
                logits = train.forward_batch(model, micro.graphs, device, amp)
                total += float(criterion(logits, micro.target.to(device)))
                count += micro.graphs.num_graphs
    return total / count


def run_config(ctx, micro_lists, args, device, amp, compiled):
    split = ctx.split
    original_geo = ctx.model_mod.get_geo_feat
    torch._dynamo.reset()
    try:
        set_seed(args.seed)
        model = train.build_model(ctx.model_mod, split.num_labels, device)
        if compiled:
            train.compile_model(model, ctx.model_mod)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        eval_loss = eval_mode_loss(model, micro_lists[: args.check_batches], device, amp)
        model.train()
        losses, seconds, graphs = [], 0.0, 0
        warmup_seconds = 0.0
        for step, micros in enumerate(micro_lists):
            sync(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = train.run_batch(model, micros, device, criterion, amp)
            optimizer.step()
            losses.append(float(loss))
            sync(device)
            elapsed = time.perf_counter() - start
            if step < args.warmup:
                warmup_seconds += elapsed
            else:
                seconds += elapsed
                graphs += sum(m.graphs.num_graphs for m in micros)
        return {
            "samples_per_second": graphs / seconds,
            "warmup_seconds": warmup_seconds,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None,
            "eval_mode_loss": eval_loss,
            "losses": losses,
        }
    finally:
        ctx.model_mod.get_geo_feat = original_geo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check-batches", type=int, default=3)
    parser.add_argument(
        "--tolerance", type=float, default=1e-3, help="fp32 relative loss difference"
    )
    parser.add_argument("--bf16-tolerance", type=float, default=2e-2)
    args = parser.parse_args()
    device = torch.device(args.device)
    split = open_split(args)
    with ModelContext(split) as ctx:
        count = (args.warmup + args.steps) * args.batch_size
        dataset = ctx.dataset(
            {n: split.train_dataset[n] for n in list(split.train_dataset)[:count]}
        )
        graphs = [dataset[i] for i in range(len(dataset))]
        collate = train.MicroCollate(args.max_residues, split.train_labels, split.num_labels)
        micro_lists = [
            collate(graphs[i : i + args.batch_size])
            for i in range(0, len(graphs), args.batch_size)
        ]
        results = {}
        for compiled in (False, True):
            for amp in (False, True):
                key = f"{'compile' if compiled else 'default'}_{'bf16' if amp else 'fp32'}"
                results[key] = run_config(ctx, micro_lists, args, device, amp, compiled)
                print(
                    f"[compile_probe] {key} {results[key]['samples_per_second']:.2f}/s", flush=True
                )
    summary = {}
    for precision, tolerance in (("fp32", args.tolerance), ("bf16", args.bf16_tolerance)):
        base, comp = results[f"default_{precision}"], results[f"compile_{precision}"]
        relative = abs(base["eval_mode_loss"] - comp["eval_mode_loss"]) / base["eval_mode_loss"]
        summary[precision] = {
            "default_samples_per_second": base["samples_per_second"],
            "compile_samples_per_second": comp["samples_per_second"],
            "speedup": comp["samples_per_second"] / base["samples_per_second"],
            "eval_mode_loss_relative_difference": relative,
            "loss_match": relative <= tolerance,
            "compile_warmup_seconds": comp["warmup_seconds"],
        }
    write_result(args.out, {"configs": results, "summary": summary})


if __name__ == "__main__":
    main()
