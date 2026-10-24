"""Shared setup for the GPU probes: data, model import and JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ..train import Split, load_split
from ..utils import DEFAULT_OUT_ROOT, DEFAULT_SEEDS, baseline_cwd, import_task_module, read_json

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="path of the JSON result")
    parser.add_argument("--device", default="cuda", help="one device; use CUDA_VISIBLE_DEVICES")
    parser.add_argument("--split-group", default="random_splits_grouped_sequence")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--work-dir", required=True, help="folder for the feature links")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-residues", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument(
        "--synthetic", type=int, default=0, help="use this many random proteins instead of a split"
    )


def synthetic_split(data_dir: Path, count: int, num_labels: int = 6, seed: int = 0) -> Split:
    """Random proteins with the feature files the GraphEC data loader reads."""
    from ..utils import run_data_root

    rng = np.random.default_rng(seed)
    data_root = run_data_root(data_dir)
    labels = [f"{i % 3 + 1}.{i % 2 + 1}.1.{i + 1}" for i in range(num_labels)]
    parts = {"t": {}, "v": {}}
    active_sites = {}
    for prefix, number in (("t", count), ("v", max(2, count // 4))):
        for i in range(number):
            name, length = f"{prefix}{i}", int(rng.integers(8, 30))
            ca = np.cumsum(rng.normal(size=(length, 3)), axis=0) * 3.8 / 1.7
            atoms = ca[:, None] + rng.normal(scale=0.7, size=(length, 5, 3))
            atoms[:, 1] = ca
            data = data_root / "Data"
            torch.save(
                torch.tensor(atoms, dtype=torch.float32), data / f"Structures/{name}.tensor"
            )
            torch.save(torch.randn(length, 1024), data / f"ProtTrans/{name}.tensor")
            torch.save(torch.randn(length, 9), data / f"DSSP/{name}.tensor")
            active_sites[name] = [torch.rand(length, 1)]
            sequence = "".join(rng.choice(list(AMINO_ACIDS), size=length))
            parts[prefix][name] = ([sequence], [i % num_labels])
    return Split(
        {"labels": labels},
        num_labels,
        {n: v[0] for n, v in parts["t"].items()},
        {n: v[1] for n, v in parts["t"].items()},
        {n: v[0] for n, v in parts["v"].items()},
        {n: v[1] for n, v in parts["v"].items()},
        active_sites,
        data_root,
        [],
        [],
    )


def open_split(args: argparse.Namespace, train_fraction: float = 1.0) -> Split:
    split = (
        synthetic_split(Path(args.work_dir), args.synthetic)
        if args.synthetic
        else load_split(args.split_group, args.out_root, Path(args.work_dir))
    )
    keep = max(1, int(len(split.train_dataset) * train_fraction))
    names = list(split.train_dataset)[:keep]
    split.train_dataset = {n: split.train_dataset[n] for n in names}
    split.train_labels = {n: split.train_labels[n] for n in names}
    return split


def label_names(split: Split) -> list[str]:
    if "labels" in split.metadata:
        return split.metadata["labels"]
    idx_to_label = read_json(split.metadata["vocab_path"])["idx_to_label"]
    return [idx_to_label[str(i)] for i in range(split.num_labels)]


class ModelContext:
    """Imports the GraphEC model code with the run's `Data` folder as working directory and
    keeps the upstream `forward` for comparison."""

    def __init__(self, split: Split) -> None:
        self.split = split
        self.cwd = baseline_cwd(split.data_root)

    def __enter__(self) -> "ModelContext":
        self.cwd.__enter__()
        self.model_mod = import_task_module("EC_number", "model")
        self.data_mod = sys.modules["data"]
        self.upstream_forward = self.model_mod.GraphEC.forward
        self.upstream_padding = self.model_mod.GraphEC.padding_ver1
        return self

    def __exit__(self, *exc) -> None:
        self.cwd.__exit__(*exc)

    def dataset(self, names: dict, active_sites=None):
        return self.data_mod.ProteinGraphDataset(
            names,
            range(len(names)),
            args=None,
            active_sites=active_sites or self.split.active_sites,
        )


def write_result(path: str, result: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result.get("summary", result), indent=2, default=float), flush=True)
