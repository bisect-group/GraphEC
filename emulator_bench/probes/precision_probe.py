"""Compare bf16 with fp32 training: fp32 seed A, bf16 seed A and fp32 seed B on one split.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -m emulator_bench.probes.precision_probe --out result.json \
  --work-dir <folder> --split-group random_splits_grouped_sequence --train-fraction 0.1 --epochs 2

The verdict says, for each recorded number, whether bf16 differs from fp32 (same seed) by no more
than fp32 seed B differs from fp32 seed A, or than `--tolerance`, whichever is larger.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from .. import train
from ..evaluate import predict_scores
from ..results import average_accuracy, compute_graphec_metrics, get_accuracy_level
from ..utils import set_seed
from .common import ModelContext, add_common_args, label_names, open_split, write_result


def val_metrics(scores, names, split, labels):
    """Metrics of `evaluate.py` without label diffusion, plus top-1 accuracy per EC level."""
    y_true = np.zeros(scores.shape, dtype=np.float32)
    for row, name in enumerate(names):
        y_true[row, split.val_labels[name]] = 1.0
    top1 = [labels[i] for i in scores.argmax(axis=1)]
    levels = [
        get_accuracy_level([top], [labels[i] for i in split.val_labels[name]])
        for top, name in zip(top1, names)
    ]
    metrics = compute_graphec_metrics(y_true, scores)
    for level in (1, 2, 3, 4):
        metrics[f"top1_level_{level}_accuracy"] = float(
            np.mean([average_accuracy(row, level) for row in levels]) * 100.0
        )
    return {k: v for k, v in metrics.items() if v is not None}


def train_run(ctx, args, device, seed, amp):
    split, labels = ctx.split, label_names(ctx.split)
    set_seed(seed)
    skip: set[str] = set()
    train_loader = train.make_loader(
        ctx.dataset(split.train_dataset),
        args.batch_size,
        True,
        args.num_workers,
        train.MicroCollate(args.max_residues, split.train_labels, split.num_labels, skip),
        device,
    )
    val_loader = train.make_loader(
        ctx.dataset(split.val_dataset),
        args.batch_size,
        False,
        args.num_workers,
        train.MicroCollate(args.max_residues),
        device,
    )
    model = train.build_model(ctx.model_mod, split.num_labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train_epoch(
            model, train_loader, device, optimizer, criterion, amp, skip
        )
        val_loss, _ = train.eval_epoch(
            model, val_loader, split.val_labels, split.num_labels, device, criterion, amp
        )
        names, scores, _ = predict_scores(
            dataset=split.val_dataset,
            active_sites=split.active_sites,
            data_mod=ctx.data_mod,
            model=model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=amp,
            max_residues=args.max_residues,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics(scores, names, split, labels),
            }
        )
        print(f"[precision_probe] seed={seed} amp={amp} {history[-1]}", flush=True)
    return model, history


def layer_errors(model, ctx, args, device):
    """Relative error of every module output, bf16 autocast against fp32, on one fixed batch."""
    split = ctx.split
    dataset = ctx.dataset(split.val_dataset)
    graphs = [dataset[i] for i in range(min(args.probe_graphs, len(dataset)))]
    batch = train.MicroCollate(10**9)(graphs)[0].graphs
    outputs = {False: {}, True: {}}
    model.eval()
    for amp in (False, True):
        hooks = []
        for name, module in model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(collect(outputs[amp], name)))
        with torch.no_grad():
            train.forward_batch(model, batch, device, amp)
        for hook in hooks:
            hook.remove()
    return {
        key: float((outputs[True][key] - ref).norm() / (ref.norm() + 1e-12))
        for key, ref in outputs[False].items()
    }


def collect(store, name):
    def hook(module, inputs, output):
        for i, tensor in enumerate(output if isinstance(output, tuple) else (output,)):
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                store[f"{name}#{i}" if isinstance(output, tuple) else name] = (
                    tensor.detach().float().cpu()
                )

    return hook


def verdict(fp32_a, bf16_a, fp32_b, tolerance):
    rows = {}
    for epoch, (a, bf, b) in enumerate(zip(fp32_a, bf16_a, fp32_b), 1):
        for key in a:
            if key == "epoch" or key not in bf or key not in b:
                continue
            bf16_diff, seed_diff = abs(bf[key] - a[key]), abs(b[key] - a[key])
            rows[f"epoch{epoch}.{key}"] = {
                "fp32_a": a[key],
                "bf16_a": bf[key],
                "fp32_b": b[key],
                "bf16_vs_fp32": bf16_diff,
                "seed_vs_seed": seed_diff,
                "within_noise": bf16_diff <= max(seed_diff, tolerance),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed-b", type=int, default=777)
    parser.add_argument("--probe-graphs", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    device = torch.device(args.device)
    split = open_split(args, args.train_fraction)
    with ModelContext(split) as ctx:
        model_a, fp32_a = train_run(ctx, args, device, args.seed, False)
        _, bf16_a = train_run(ctx, args, device, args.seed, True)
        _, fp32_b = train_run(ctx, args, device, args.seed_b, False)
        errors = layer_errors(model_a, ctx, args, device)
    rows = verdict(fp32_a, bf16_a, fp32_b, args.tolerance)
    outside = [key for key, row in rows.items() if not row["within_noise"]]
    result = {
        "train_samples": len(split.train_dataset),
        "val_samples": len(split.val_dataset),
        "epochs": args.epochs,
        "history": {"fp32_seed_a": fp32_a, "bf16_seed_a": bf16_a, "fp32_seed_b": fp32_b},
        "metrics": rows,
        "layer_relative_error": errors,
        "summary": {
            "verdict": "bf16 within fp32 seed noise" if not outside else "bf16 differs more",
            "outside_noise": outside,
            "layer_error_max": max(errors.values()),
            "layer_error_median": float(np.median(list(errors.values()))),
            "final_epoch": {
                key.split(".", 1)[1]: row
                for key, row in rows.items()
                if key.startswith(f"epoch{args.epochs}.")
            },
        },
    }
    write_result(args.out, result)


if __name__ == "__main__":
    main()
