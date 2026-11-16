from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from .utils import (
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    baseline_cwd,
    cache_dirs,
    cache_root,
    ensure_dir,
    ensure_graphec_data_links,
    run_data_root,
    feature_keys_from_manifest,
    import_task_module,
    load_feature_failures,
    load_run_metadata,
    raise_open_file_limit,
    read_json,
    runs_root,
    seed_run_root_for_split,
    seed_train_metadata_path_for_split,
    set_seed,
    set_tf32,
    write_json,
)


DEFAULT_MAX_RESIDUES = 5000
LOG_EVERY = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphEC on an EMULaToR split")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    parser.add_argument(
        "--no-tf32", dest="tf32", action="store_false", help="fp32 matmuls without TF32"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile (dynamic shapes) of the geometric features and the GNN layers",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=DEFAULT_MAX_RESIDUES,
        help="residues per micro-batch; a batch above it is split and gradients are accumulated",
    )
    return parser.parse_args()


def load_records(
    path: str | Path,
    label_to_idx: dict[str, int],
    no_structure: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict, dict[str, list[int]], list[str]]:
    """Samples and labels per sample name, plus the names skipped for lack of a structure."""
    records = pd.read_csv(path)
    dataset = {}
    labels_by_key = {}
    skipped = []
    for _, row in records.iterrows():
        key = str(row["cache_key"])
        labels = [
            label_to_idx[label]
            for label in str(row["EC number"]).split(";")
            if label in label_to_idx
        ]
        if not labels:
            continue
        if row["sequence_sha256"] in no_structure:
            skipped.append(key)
            continue
        dataset[key] = [str(row["Sequence"])]
        labels_by_key[key] = labels
    if not dataset:
        raise ValueError(f"No usable rows in {path}")
    return dataset, labels_by_key, skipped


class ActiveSites:
    """Active-site scores per sample name. Each per-sequence cache file is read when a sample is
    first used, in the DataLoader worker, so a run starts without reading every file."""

    def __init__(self, feature_keys: dict[str, str], directory: Path) -> None:
        self.feature_keys, self.directory = feature_keys, directory

    def __getitem__(self, name: str) -> list[torch.Tensor]:
        return torch.load(self.directory / f"{self.feature_keys[name]}.pt", map_location="cpu")


def load_active_sites(feature_keys: dict[str, str], cache_root: str | Path) -> ActiveSites:
    """Active-site scores per sample name; fails at once if a cache file is missing."""
    active_dir = cache_dirs(cache_root)["active_sites"]
    available = set(os.listdir(active_dir))
    for key in set(feature_keys.values()):
        if f"{key}.pt" not in available:
            raise FileNotFoundError(f"Missing active-site cache file: {active_dir / f'{key}.pt'}")
    return ActiveSites(feature_keys, active_dir)


def targets_for_names(
    names: list[str], labels_by_key: dict[str, list[int]], num_labels: int, device
) -> torch.Tensor:
    rows = [row for row, name in enumerate(names) for _ in labels_by_key[str(name)]]
    columns = [label for name in names for label in labels_by_key[str(name)]]
    target = torch.zeros((len(names), num_labels), dtype=torch.float32, device=device)
    target[rows, columns] = 1.0
    return target


def padded_positions(batch_id, counts, max_len: int) -> torch.Tensor:
    """Row of each residue in the flattened [graphs * max_len] padded layout."""
    starts = counts.cumsum(0) - counts
    return (
        batch_id * max_len + torch.arange(len(batch_id), device=batch_id.device) - starts[batch_id]
    )


def pad_rows(x, positions, num_graphs: int, max_len: int) -> torch.Tensor:
    """Pad per-residue rows into [graphs, max_len, features]. It copies values and never syncs."""
    padded = x.new_zeros(num_graphs * max_len, *x.shape[1:])
    padded[positions] = x
    return padded.view(num_graphs, max_len, *x.shape[1:])


def make_forward(model_mod):
    """`GraphEC.forward` that pads the node features on the device. It equals the original, which
    copies them to the CPU and back."""

    def forward(
        self, X, h_V, edge_index, seq, batch_id, batch_data, mask_data, batch_activate_site
    ):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

        h_V_baseline = self.hidden_block(self.input_block(batch_data))

        h_V_geo, h_E = model_mod.get_geo_feat(X, edge_index)
        if h_V.shape[0] != h_V_geo.shape[0]:
            print(h_V.size(), h_V_geo.size(), seq, flush=True)
            h_V_geo = torch.ones([h_V.shape[0], 184], device=h_V.device, dtype=h_V_geo.dtype)
        h_V = torch.cat([h_V, h_V_geo], dim=-1)
        h_V = self.Graph_encoder(h_V, edge_index, h_E, seq, batch_id)
        graphs, max_len = mask_data.shape
        positions = padded_positions(batch_id, mask_data.sum(1).long(), max_len)
        h_V_stru = pad_rows(h_V, positions, graphs, max_len)

        h_V_baseline = self.weight * h_V_baseline + (1 - self.weight) * h_V_stru
        att = self.ATFC(h_V_baseline)
        att = att.masked_fill(mask_data[:, :, None] == 0, -1e9)
        att = F.softmax(att, dim=1)
        active_pool = batch_activate_site.transpose(1, 2) @ h_V_baseline
        h_V_baseline = att.transpose(1, 2) @ h_V_baseline
        h_V_baseline = torch.cat((h_V_baseline, active_pool), 1)
        return self.output_block(torch.flatten(h_V_baseline, start_dim=1)).float()

    return forward


def resize_output_layer(model: nn.Module, num_labels: int) -> None:
    old_layer = model.output_block[-1]
    model.output_block[-1] = nn.Linear(old_layer.in_features, num_labels)
    model.FC_2 = nn.Linear(model.hidden_dim, num_labels)


def build_model(model_mod, num_labels: int, device: torch.device) -> nn.Module:
    model_mod.GraphEC.forward = make_forward(model_mod)
    model = model_mod.GraphEC(
        node_input_dim=1024 + 9 + 184,
        edge_input_dim=450,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        augment_eps=0,
        device=device,
    )
    resize_output_layer(model, num_labels)
    return model.to(device)


def compile_model(model: nn.Module, model_mod) -> None:
    """Compile the geometric features and the GNN layers for dynamic shapes. State dict keys stay
    the same."""
    model_mod.get_geo_feat = torch.compile(model_mod.get_geo_feat, dynamic=True)
    for layer in model.Graph_encoder.layers:
        layer.compile(dynamic=True)


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Micro(NamedTuple):
    """A micro-batch of graphs and its dense label matrix (None when labels are not needed)."""

    graphs: Batch
    target: torch.Tensor | None


def group_graphs(graphs: list, max_residues: int) -> list[list]:
    """Group graphs in order into micro-batches of at most `max_residues`; a longer graph runs
    alone."""
    groups, current, residues = [], [], 0
    for graph in graphs:
        if current and residues + graph.num_nodes > max_residues:
            groups.append(current)
            current, residues = [], 0
        current.append(graph)
        residues += graph.num_nodes
    if current:
        groups.append(current)
    return groups


class MicroCollate:
    """DataLoader `collate_fn` that splits a batch into micro-batches in the worker. Graphs in
    `skip` are left out. With `labels_by_key` each micro-batch also gets its label matrix. The
    workers start with the epoch, so they see the `skip` of that epoch."""

    def __init__(
        self,
        max_residues: int,
        labels_by_key: dict[str, list[int]] | None = None,
        num_labels: int = 0,
        skip: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.max_residues, self.labels_by_key = max_residues, labels_by_key
        self.num_labels, self.skip = num_labels, skip

    def __call__(self, graphs: list) -> list[Micro]:
        kept = [graph for graph in graphs if str(graph.name) not in self.skip]
        return [
            Micro(
                Batch.from_data_list(group),
                None
                if self.labels_by_key is None
                else targets_for_names(
                    [graph.name for graph in group], self.labels_by_key, self.num_labels, "cpu"
                ),
            )
            for group in group_graphs(kept, self.max_residues)
        ]


def make_loader(dataset, batch_size, shuffle, num_workers, collate, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )


def forward_batch(model, batch: Batch, device: torch.device, amp: bool) -> torch.Tensor:
    """Logits of a batch. The batch is copied once and padded on the device."""
    max_len = int(batch.ptr.diff().max())
    batch = batch.to(device, non_blocking=True)
    counts = batch.ptr.diff()
    positions = padded_positions(batch.batch, counts, max_len)
    mask = (torch.arange(max_len, device=device)[None] < counts[:, None]).float()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        logits = model(
            batch.X,
            batch.node_feat,
            batch.edge_index,
            batch.seq,
            batch.batch,
            pad_rows(batch.node_feat, positions, batch.num_graphs, max_len),
            mask,
            pad_rows(batch.activate_site, positions, batch.num_graphs, max_len),
        )
    return logits.float()


def split_in_half(batch: Batch) -> list[Batch]:
    graphs = batch.to_data_list()
    middle = len(graphs) // 2
    return [Batch.from_data_list(graphs[:middle]), Batch.from_data_list(graphs[middle:])]


def backward_or_split(
    model, micro: Micro, device, criterion, amp, dropped: list[str]
) -> torch.Tensor:
    """Backward pass of a micro-batch; returns its summed per-sample loss as a float64 tensor on
    the device. After a GPU OOM the gradients of the failed attempt are undone and the two halves
    run on their own. A graph that runs out of memory alone is appended to `dropped` and adds
    nothing."""
    saved = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    try:
        logits = forward_batch(model, micro.graphs, device, amp)
        target = micro.target.to(device, non_blocking=True)
        loss = criterion(logits, target) * micro.graphs.num_graphs
        loss.backward()
        return loss.detach().double()
    except torch.cuda.OutOfMemoryError:
        pass
    for p, grad in zip(model.parameters(), saved):
        p.grad = grad
    torch.cuda.empty_cache()
    if micro.graphs.num_graphs == 1:
        dropped.append(str(micro.graphs.name[0]))
        return torch.zeros((), dtype=torch.float64, device=device)
    middle = micro.graphs.num_graphs // 2
    parts = zip(split_in_half(micro.graphs), (micro.target[:middle], micro.target[middle:]))
    return sum(
        backward_or_split(model, Micro(*part), device, criterion, amp, dropped) for part in parts
    )


def run_batch(
    model, micros: list[Micro], device, criterion, amp
) -> tuple[torch.Tensor, list[str]]:
    """Backward pass of the micro-batches of one batch. The gradient is the mean over the samples
    that ran, so it equals the full-batch gradient when none is dropped. Returns the mean loss as
    a float64 tensor on the device, and the names that ran out of GPU memory alone."""
    dropped, total = [], torch.zeros((), dtype=torch.float64, device=device)
    for micro in micros:
        total += backward_or_split(model, micro, device, criterion, amp, dropped)
    used = sum(micro.graphs.num_graphs for micro in micros) - len(dropped)
    if used == 0:
        return total, dropped
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(used)
    return total / used, dropped


def forward_or_split(model, batch: Batch, device: torch.device, amp: bool) -> list:
    """Logits without gradients, as (names, logits) pairs. After a GPU OOM the two halves of the
    batch run on their own; a graph that runs out of memory alone gives (names, None)."""
    try:
        with torch.no_grad():
            logits = forward_batch(model, batch, device, amp).cpu()
        return [([str(name) for name in batch.name], logits)]
    except torch.cuda.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()
    if batch.num_graphs == 1:
        return [([str(batch.name[0])], None)]
    return [
        pair
        for part in split_in_half(batch)
        for pair in forward_or_split(model, part, device, amp)
    ]


def train_epoch(model, dataloader, device, optimizer, criterion, amp, skip: set[str]) -> float:
    """One epoch. Names that run out of GPU memory alone are added to `skip` and never retried.
    The loss stays on the device until the end of the epoch."""
    model.train()
    total, count = torch.zeros((), dtype=torch.float64, device=device), 0
    pbar = progress(dataloader, desc="train", unit="batch", leave=False)
    for micros in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, dropped = run_batch(model, micros, device, criterion, amp)
        skip.update(dropped)
        optimizer.step()
        total += loss
        count += 1
        if count % LOG_EVERY == 1:
            pbar.set_postfix(loss=f"{float(loss):.4f}")
    return float(total) / max(1, count)


def eval_epoch(
    model, dataloader, labels_by_key, num_labels, device, criterion, amp
) -> tuple[float, list[str]]:
    """Mean loss per sample, and the names left out because they ran out of GPU memory alone."""
    model.eval()
    total, count, out_of_memory = 0.0, 0, []
    for micros in progress(dataloader, desc="val", unit="batch", leave=False):
        for micro in micros:
            for names, logits in forward_or_split(model, micro.graphs, device, amp):
                if logits is None:
                    out_of_memory.extend(names)
                    continue
                target = targets_for_names(names, labels_by_key, num_labels, "cpu")
                total += float(criterion(logits, target)) * len(names)
                count += len(names)
    return total / max(1, count), out_of_memory


def train_model(
    *,
    train_dataset,
    val_dataset,
    train_labels,
    val_labels,
    active_sites,
    data_mod,
    model_mod,
    num_labels: int,
    args,
    run_root: Path,
) -> dict:
    """Train one model on the train split and stop early on the val split. Resumes from
    `checkpoint_last.pt`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skip: set[str] = set()

    def loader(dataset, shuffle, collate):
        return make_loader(
            data_mod.ProteinGraphDataset(
                dataset, range(len(dataset)), args=None, active_sites=active_sites
            ),
            args.batch_size,
            shuffle,
            args.num_workers,
            collate,
            device,
        )

    train_loader = loader(
        train_dataset, True, MicroCollate(args.max_residues, train_labels, num_labels, skip)
    )
    val_loader = loader(val_dataset, False, MicroCollate(args.max_residues))
    set_seed(args.seed)
    model = build_model(model_mod, num_labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_path, last_path, resume_path = (
        run_root / "bestmodel.pth",
        run_root / "lastmodel.pth",
        run_root / "checkpoint_last.pt",
    )
    state = {
        "epoch": 0,
        "best_val": float("inf"),
        "best_epoch": 0,
        "stale": 0,
        "history": [],
        "val_out_of_memory": [],
        "train_out_of_memory": [],
    }
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state = saved["state"]
        print(f"[emulator_bench] resumed after epoch {state['epoch']}", flush=True)
    if args.compile:
        compile_model(model, model_mod)

    skip.update(state["train_out_of_memory"])
    while state["epoch"] < args.epochs and state["stale"] < args.patience:
        epoch = state["epoch"] + 1
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion, args.amp, skip)
        state["train_out_of_memory"] = sorted(skip)
        val_loss, state["val_out_of_memory"] = eval_epoch(
            model, val_loader, val_labels, num_labels, device, criterion, args.amp
        )
        state["history"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[emulator_bench] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < state["best_val"]:
            state.update(best_val=val_loss, best_epoch=epoch, stale=0)
            atomic_save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)
        else:
            state["stale"] += 1
        state["epoch"] = epoch
        atomic_save({"model_state_dict": model.state_dict(), "epoch": epoch}, last_path)
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "state": state,
            },
            resume_path,
        )
    return {
        "checkpoint": str(best_path),
        "best_epoch": state["best_epoch"],
        "best_val_loss": state["best_val"],
        "train_out_of_memory": len(state["train_out_of_memory"]),
        "train_out_of_memory_names": state["train_out_of_memory"],
        "val_out_of_memory": len(state["val_out_of_memory"]),
        "val_out_of_memory_names": state["val_out_of_memory"],
        "history": state["history"],
    }


@dataclass
class Split:
    """Records, labels and feature links of one split group, as `train_model` needs them."""

    metadata: dict
    num_labels: int
    train_dataset: dict
    train_labels: dict
    val_dataset: dict
    val_labels: dict
    active_sites: ActiveSites
    data_root: Path
    no_features: list[str]
    val_no_features: list[str]


def load_split(split_group: str, out_root: str | Path, data_dir: Path) -> Split:
    """Read the split, link its cached features under `data_dir`, and open the active sites."""
    metadata = load_run_metadata(split_group, runs_root(out_root))
    cache = cache_root(out_root)
    vocab = read_json(metadata["vocab_path"])
    label_to_idx = {label: int(index) for label, index in vocab["label_to_idx"].items()}
    failures = set(load_feature_failures(cache))
    train_dataset, train_labels, no_features = load_records(
        metadata["manifests"]["train"], label_to_idx, failures
    )
    val_dataset, val_labels, val_no_features = load_records(
        metadata["manifests"]["val"], label_to_idx, failures
    )
    feature_keys = {}
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        keys = feature_keys_from_manifest(metadata["manifests"][split])
        feature_keys.update({name: keys[name] for name in dataset})
    data_root = run_data_root(data_dir)
    ensure_graphec_data_links(feature_keys, cache, data_root)
    return Split(
        metadata,
        int(vocab["num_labels"]),
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        load_active_sites(feature_keys, cache),
        data_root,
        no_features,
        val_no_features,
    )


def main() -> None:
    args = parse_args()
    raise_open_file_limit()
    set_tf32(args.tf32)
    runs = runs_root(args.out_root)
    run_root = ensure_dir(seed_run_root_for_split(args.split_group, args.seed, runs))
    if seed_train_metadata_path_for_split(args.split_group, args.seed, runs).exists():
        print(f"[emulator_bench] training already finished: {run_root}", flush=True)
        return
    split = load_split(args.split_group, args.out_root, run_root / "data")
    skipped = {
        "max_residues": args.max_residues,
        "no_features": len(split.no_features),
        "no_features_names": split.no_features,
        "val_no_features": len(split.val_no_features),
        "val_no_features_names": split.val_no_features,
    }
    write_json(run_root / "skipped_train.json", skipped)
    print(
        f"[emulator_bench] skipped train rows: {skipped['no_features']} without features; "
        f"val rows left out: {skipped['val_no_features']} without features",
        flush=True,
    )

    with baseline_cwd(split.data_root):
        model_mod = import_task_module("EC_number", "model")
        data_mod = sys.modules["data"]
        result = train_model(
            train_dataset=split.train_dataset,
            val_dataset=split.val_dataset,
            train_labels=split.train_labels,
            val_labels=split.val_labels,
            active_sites=split.active_sites,
            data_mod=data_mod,
            model_mod=model_mod,
            num_labels=split.num_labels,
            args=args,
            run_root=run_root,
        )
    write_json(
        run_root / "skipped_train.json",
        {
            **skipped,
            "out_of_memory": result["train_out_of_memory"],
            "out_of_memory_names": result["train_out_of_memory_names"],
            "val_out_of_memory": result["val_out_of_memory"],
            "val_out_of_memory_names": result["val_out_of_memory_names"],
        },
    )
    print(
        f"[emulator_bench] train rows out of memory: {result['train_out_of_memory']}; "
        f"val rows left out: {result['val_out_of_memory']} out of memory",
        flush=True,
    )
    train_metadata = {
        "split_group": args.split_group,
        "seed": args.seed,
        "run_root": str(run_root),
        "num_labels": split.num_labels,
        "vocab_path": split.metadata["vocab_path"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "amp": args.amp,
        "tf32": args.tf32,
        **result,
    }
    train_json = write_json(
        seed_train_metadata_path_for_split(args.split_group, args.seed, runs), train_metadata
    )
    print(f"[emulator_bench] train metadata: {train_json}", flush=True)


if __name__ == "__main__":
    main()
