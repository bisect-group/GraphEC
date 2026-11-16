"""Compare K training steps of the original code path with the current one in fp32.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -m emulator_bench.probes.exactness_probe --out result.json \
  --work-dir <folder> [--split-group ...] [--steps 20]

The original path pads on the CPU, copies the node features through the CPU inside the forward,
reads the loss after each micro-batch and builds each target row by row. It runs the upstream
`GraphEC.forward`. Two original runs give the noise floor of the device (atomic scatter adds).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .. import train
from ..utils import set_seed
from .common import ModelContext, add_common_args, open_split, write_result


def cpu_padding(x, batch_id, feature_dim, activate_site=None):
    counts = torch.bincount(batch_id)
    mask = torch.arange(int(counts.max()), device=x.device)[None] < counts[:, None]
    padded = x.new_zeros(len(counts), mask.shape[1], feature_dim)
    padded[mask] = x
    if activate_site is None:
        return padded, mask.float()
    site = activate_site.new_zeros(len(counts), mask.shape[1], 1)
    site[mask] = activate_site
    return padded, mask.float(), site


def original_forward_batch(model, batch, device):
    padded, mask, site = cpu_padding(
        batch.node_feat, batch.batch, batch.node_feat.shape[1], batch.activate_site
    )
    batch = batch.to(device)
    logits = model(
        batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch,
        padded.to(device), mask.to(device), site.to(device),
    )  # fmt: skip
    return logits.float()


def original_split(batch, max_residues):
    if batch.num_nodes <= max_residues:
        return [batch]
    groups, current, residues = [], [], 0
    for graph in batch.to_data_list():
        if current and residues + graph.num_nodes > max_residues:
            groups.append(current)
            current, residues = [], 0
        current.append(graph)
        residues += graph.num_nodes
    groups.append(current)
    return [Batch.from_data_list(group) for group in groups]


def original_targets(names, labels, num_labels, device):
    target = torch.zeros((len(names), num_labels), dtype=torch.float32, device=device)
    for row, name in enumerate(names):
        target[row, labels[str(name)]] = 1.0
    return target


def original_step(model, graphs, labels, num_labels, device, criterion, max_residues):
    total = 0.0
    for micro in original_split(Batch.from_data_list(graphs), max_residues):
        logits = original_forward_batch(model, micro, device)
        loss = criterion(logits, original_targets(list(micro.name), labels, num_labels, device))
        loss = loss * micro.num_graphs
        loss.backward()
        total += float(loss.detach().cpu())
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(len(graphs))
    return total / len(graphs)


def new_step(model, graphs, labels, num_labels, device, criterion, max_residues):
    micros = train.MicroCollate(max_residues, labels, num_labels)(graphs)
    loss, _ = train.run_batch(model, micros, device, criterion, False)
    return float(loss)


def run_path(ctx, batches, args, device, original: bool):
    split = ctx.split
    set_seed(args.seed)
    model = train.build_model(ctx.model_mod, split.num_labels, device)
    if original:
        ctx.model_mod.GraphEC.forward = ctx.upstream_forward
        ctx.model_mod.GraphEC.padding_ver1 = lambda self, x, b, d: cpu_padding(x, b, d)[:2]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    step = original_step if original else new_step
    model.train()
    losses = []
    for graphs in batches:
        optimizer.zero_grad(set_to_none=True)
        losses.append(
            step(
                model,
                graphs,
                split.train_labels,
                split.num_labels,
                device,
                criterion,
                args.max_residues,
            )
        )
        optimizer.step()
    return losses, {k: v.detach().cpu() for k, v in model.state_dict().items()}


def max_diff(a, b):
    return max(
        (torch.as_tensor(x).double() - torch.as_tensor(y).double()).abs().max().item()
        for x, y in zip(a, b)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    split = open_split(args)
    with ModelContext(split) as ctx:
        names = list(split.train_dataset)[: args.steps * args.batch_size]
        dataset = ctx.dataset({n: split.train_dataset[n] for n in names})
        graphs = [dataset[i] for i in range(len(dataset))]
        batches = [graphs[i : i + args.batch_size] for i in range(0, len(graphs), args.batch_size)]
        runs = {
            key: run_path(ctx, batches, args, device, original)
            for key, original in (("original_a", True), ("original_b", True), ("new", False))
        }
    (loss_a, par_a), (loss_b, par_b), (loss_n, par_n) = runs.values()
    floor = {"loss": max_diff(loss_a, loss_b), "params": max_diff(par_a.values(), par_b.values())}
    diff = {"loss": max_diff(loss_a, loss_n), "params": max_diff(par_a.values(), par_n.values())}
    result = {
        "steps": len(batches),
        "device": args.device,
        "original_vs_new_max_abs_diff": diff,
        "original_vs_original_max_abs_diff": floor,
        "bit_identical": diff == {"loss": 0.0, "params": 0.0},
        "losses": {key: value[0] for key, value in runs.items()},
    }
    result["summary"] = {k: v for k, v in result.items() if k != "losses"}
    write_result(args.out, result)


if __name__ == "__main__":
    main()
