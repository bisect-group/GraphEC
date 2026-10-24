import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as ssp
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402
from emulator_bench.evaluate import block_cg  # noqa: E402
from emulator_bench.utils import (  # noqa: E402
    BASELINE_ROOT,
    ensure_graphec_data_links,
    merge_json,
    read_json,
    write_json,
)

LABELS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(lengths):
    generator = torch.Generator().manual_seed(0)
    graphs = [
        Data(x=torch.randn(n, 4, generator=generator), name=f"s{i}", num_nodes=n)
        for i, n in enumerate(lengths)
    ]
    return Batch.from_data_list(graphs), {f"s{i}": [i % LABELS] for i in range(len(lengths))}


def stub_forward(monkeypatch, fail_over=None, fail_names=()):
    """Linear stub that raises a GPU OOM for a micro-batch above `fail_over` graphs or with a
    name in `fail_names`."""

    def forward(model, batch, device, amp):
        if (fail_over and batch.num_graphs > fail_over) or set(batch.name) & set(fail_names):
            raise torch.cuda.OutOfMemoryError
        return model(global_add_pool(batch.x, batch.batch).to(device))

    monkeypatch.setattr(train, "forward_batch", forward)


def gradient(model, batch, labels, skip=frozenset(), max_residues=10**9):
    model.zero_grad()
    micros = train.MicroCollate(max_residues, labels, LABELS, skip)(batch.to_data_list())
    loss, dropped = train.run_batch(model, micros, DEVICE, nn.BCEWithLogitsLoss(), False)
    return float(loss), dropped, [p.grad.clone() for p in model.parameters()]


def test_oom_splits_the_micro_batch_and_keeps_the_gradient(monkeypatch) -> None:
    torch.manual_seed(1)
    model = nn.Linear(4, LABELS).to(DEVICE)
    batch, labels = make_batch([3, 9, 2, 7, 4])
    stub_forward(monkeypatch)
    full_loss, _, full = gradient(model, batch, labels)
    stub_forward(monkeypatch, fail_over=1)
    loss, dropped, split = gradient(model, batch, labels)
    assert not dropped
    assert abs(loss - full_loss) < 1e-6
    assert max((a - b).abs().max().item() for a, b in zip(full, split)) < 1e-6


def test_lone_oom_graph_is_dropped_and_the_mean_uses_the_rest(monkeypatch) -> None:
    torch.manual_seed(1)
    model = nn.Linear(4, LABELS).to(DEVICE)
    batch, labels = make_batch([3, 9, 2, 7, 4])
    stub_forward(monkeypatch)
    rest = Batch.from_data_list([g for g in batch.to_data_list() if g.name != "s1"])
    expected_loss, _, expected = gradient(model, rest, labels)
    stub_forward(monkeypatch, fail_names=("s1",))
    loss, dropped, got = gradient(model, batch, labels)
    assert dropped == ["s1"]
    assert abs(loss - expected_loss) < 1e-6
    assert max((a - b).abs().max().item() for a, b in zip(expected, got)) < 1e-6


def test_graph_over_the_budget_is_not_dropped(monkeypatch) -> None:
    model = nn.Linear(4, LABELS).to(DEVICE)
    batch, labels = make_batch([3, 50, 2])
    stub_forward(monkeypatch)
    _, dropped, _ = gradient(model, batch, labels, max_residues=10)
    assert not dropped


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, index, args, active_sites):
        self.names = list(dataset)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return Data(x=torch.randn(4, 4, generator=generator), name=self.names[index], num_nodes=4)


def train_args(epochs):
    return SimpleNamespace(
        batch_size=4, num_workers=0, seed=1, lr=1e-2, amp=False,
        max_residues=100, epochs=epochs, patience=9, compile=False,
    )  # fmt: skip


def run_training(monkeypatch, tmp_path, epochs, fail_names):
    names = [f"t{i}" for i in range(8)]
    monkeypatch.setattr(
        train, "build_model", lambda mod, labels, device: nn.Linear(4, LABELS).to(device)
    )
    monkeypatch.setattr(train, "eval_epoch", lambda *a, **k: (1.0, []))
    calls = []

    def forward(model, batch, device, amp):
        calls.append(list(batch.name))
        if set(batch.name) & set(fail_names):
            raise torch.cuda.OutOfMemoryError
        return model(global_add_pool(batch.x, batch.batch).to(device))

    monkeypatch.setattr(train, "forward_batch", forward)
    result = train.train_model(
        train_dataset={n: ["A"] for n in names},
        val_dataset={"v": ["A"]},
        train_labels={n: [0] for n in names},
        val_labels={"v": [0]},
        active_sites={},
        data_mod=SimpleNamespace(ProteinGraphDataset=Dataset),
        model_mod=None,
        num_labels=LABELS,
        args=train_args(epochs),
        run_root=tmp_path,
    )
    return result, calls


def test_oom_skip_set_persists_in_the_checkpoint(tmp_path: Path, monkeypatch) -> None:
    result, _ = run_training(monkeypatch, tmp_path, 1, ["t3"])
    assert result["train_out_of_memory_names"] == ["t3"]
    saved = torch.load(tmp_path / "checkpoint_last.pt", weights_only=False)
    assert saved["state"]["train_out_of_memory"] == ["t3"]
    assert ("cuda_rng" in saved) and (saved["cuda_rng"] is not None) == torch.cuda.is_available()
    result, calls = run_training(monkeypatch, tmp_path, 2, ["t3"])
    assert result["train_out_of_memory"] == 1
    assert not any(names == ["t3"] for names in calls)


def test_block_cg_raises_when_not_converged() -> None:
    A = ssp.diags(np.linspace(1, 1000, 50)).tocsr()
    B = np.random.default_rng(0).random((50, 3))
    with pytest.raises(RuntimeError, match="did not reach"):
        block_cg(A, B, max_iter=2)
    assert np.abs(A @ block_cg(A, B) - B).max() < 1e-8


def test_runs_do_not_share_data_links(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    from emulator_bench.utils import cache_dirs

    dirs = cache_dirs(cache)
    for key in ("k1", "k2"):
        torch.save(torch.zeros(1), dirs["prottrans"] / f"{key}.tensor")
    before = sorted((BASELINE_ROOT / "Data/ProtTrans").iterdir())
    ensure_graphec_data_links({"a": "k1"}, cache, tmp_path / "run1")
    ensure_graphec_data_links({"a": "k2"}, cache, tmp_path / "run2")
    one = (tmp_path / "run1/Data/ProtTrans/a.tensor").resolve()
    two = (tmp_path / "run2/Data/ProtTrans/a.tensor").resolve()
    assert one != two
    assert sorted((BASELINE_ROOT / "Data/ProtTrans").iterdir()) == before


def test_json_writes_are_atomic_and_merge(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json(path, {"a": 1})
    merge_json(path, {"b": 2})
    merge_json(path, {"c": 3})
    assert read_json(path) == {"a": 1, "b": 2, "c": 3}
    assert not list(tmp_path.glob("*.tmp"))
