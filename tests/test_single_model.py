import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402

LABELS = 5


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, index, args, active_sites):
        self.names = list(dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        generator = torch.Generator().manual_seed(index)
        n = len(self.dataset[name][0])
        return Data(x=torch.randn(n, 4, generator=generator), name=name, num_nodes=n)


def records(prefix: str, count: int):
    dataset = {f"{prefix}{i}": ["A" * (3 + i)] for i in range(count)}
    return dataset, {name: [i % LABELS] for i, name in enumerate(dataset)}


def setup(monkeypatch, val_losses):
    monkeypatch.setattr(
        train, "build_model", lambda mod, labels, device: nn.Linear(4, LABELS).to(device)
    )
    monkeypatch.setattr(
        train,
        "forward_batch",
        lambda model, batch, device, amp: model(global_add_pool(batch.x, batch.batch).to(device)),
    )
    calls = iter(val_losses)
    monkeypatch.setattr(train, "eval_epoch", lambda *a, **k: (next(calls), []))
    train_set, train_labels = records("t", 12)
    val_set, val_labels = records("v", 6)
    return dict(
        train_dataset=train_set,
        val_dataset=val_set,
        train_labels=train_labels,
        val_labels=val_labels,
        active_sites={},
        data_mod=SimpleNamespace(ProteinGraphDataset=FakeDataset),
        model_mod=None,
        num_labels=LABELS,
    )


def make_args(epochs: int, patience: int):
    return SimpleNamespace(
        batch_size=4,
        num_workers=0,
        seed=1,
        lr=1e-2,
        amp=False,
        max_residues=10,
        epochs=epochs,
        patience=patience,
        compile=False,
    )


def test_one_model_per_run_in_the_seed_folder(tmp_path: Path, monkeypatch) -> None:
    kwargs = setup(monkeypatch, [1.0, 0.9])
    result = train.train_model(**kwargs, args=make_args(2, 4), run_root=tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "bestmodel.pth",
        "lastmodel.pth",
        "checkpoint_last.pt",
    }
    assert result["checkpoint"] == str(tmp_path / "bestmodel.pth")
    assert result["best_epoch"] == 2


def test_early_stopping_follows_the_val_loss(tmp_path: Path, monkeypatch) -> None:
    kwargs = setup(monkeypatch, [1.0, 0.5, 0.6, 0.7, 0.8, 0.9])
    result = train.train_model(**kwargs, args=make_args(35, 2), run_root=tmp_path)
    assert result["best_epoch"] == 2
    assert result["best_val_loss"] == 0.5
    assert len(result["history"]) == 4


def test_resume_continues_after_the_last_epoch(tmp_path: Path, monkeypatch) -> None:
    kwargs = setup(monkeypatch, [1.0, 0.9, 0.8, 0.7])
    train.train_model(**kwargs, args=make_args(2, 4), run_root=tmp_path)
    kwargs = setup(monkeypatch, [0.8, 0.7])
    result = train.train_model(**kwargs, args=make_args(4, 4), run_root=tmp_path)
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3, 4]
    assert result["best_epoch"] == 4


def test_val_graph_out_of_memory_is_left_out_of_the_loss(monkeypatch) -> None:
    def forward(model, batch, device, amp):
        if "v1" in list(batch.name):
            raise torch.cuda.OutOfMemoryError
        return model(global_add_pool(batch.x, batch.batch))

    monkeypatch.setattr(train, "forward_batch", forward)
    dataset, labels = records("v", 4)
    loader = torch.utils.data.DataLoader(
        FakeDataset(dataset, None, None, None),
        batch_size=4,
        collate_fn=train.MicroCollate(10),
    )
    loss, out_of_memory = train.eval_epoch(
        nn.Linear(4, LABELS),
        loader,
        labels,
        LABELS,
        torch.device("cpu"),
        nn.BCEWithLogitsLoss(),
        False,
    )
    assert out_of_memory == ["v1"]
    assert loss > 0
