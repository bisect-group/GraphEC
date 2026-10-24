import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402
from emulator_bench.results import compute_care_metrics, write_ranked_outputs  # noqa: E402

LABELS = 5
BENCH = Path(__file__).resolve().parents[1] / "emulator_bench"


def make_batch(lengths: list[int]) -> tuple[Batch, dict[str, list[int]]]:
    generator = torch.Generator().manual_seed(0)
    graphs = [
        Data(x=torch.randn(n, 4, generator=generator), name=f"s{i}", num_nodes=n)
        for i, n in enumerate(lengths)
    ]
    labels = {f"s{i}": [i % LABELS] for i in range(len(lengths))}
    return Batch.from_data_list(graphs), labels


def test_accumulated_gradient_equals_full_batch(monkeypatch) -> None:
    torch.manual_seed(1)
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, LABELS)).eval()
    monkeypatch.setattr(
        train,
        "forward_batch",
        lambda model, batch, device, amp: model(global_add_pool(batch.x, batch.batch)),
    )
    batch, labels = make_batch([3, 9, 2, 7, 4, 12, 5])
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    def gradient(max_residues: int):
        model.zero_grad()
        micros = train.MicroCollate(max_residues, labels, LABELS)(batch.to_data_list())
        loss, dropped = train.run_batch(model, micros, device, criterion, False)
        assert not dropped
        return float(loss), [p.grad.clone() for p in model.parameters()], len(micros)

    full_loss, full, _ = gradient(10**9)
    micro_loss, micro, micros = gradient(10)
    assert micros > 1
    assert abs(full_loss - micro_loss) < 1e-6
    assert max((a - b).abs().max().item() for a, b in zip(full, micro)) < 1e-6


def test_graph_longer_than_budget_runs_alone() -> None:
    batch, _ = make_batch([3, 20, 2])
    micros = train.MicroCollate(10)(batch.to_data_list())
    assert [micro.graphs.num_nodes for micro in micros] == [3, 20, 2]


def test_load_records_skips_failed_sequences(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "cache_key": ["a", "b"],
            "EC number": ["1.1.1.1", "1.1.1.1"],
            "Sequence": ["AAA", "CCC"],
            "sequence_sha256": ["ha", "hb"],
        }
    ).to_csv(path, index=False)
    dataset, _, skipped = train.load_records(path, {"1.1.1.1": 0}, {"hb"})
    assert list(dataset) == ["a"]
    assert skipped == ["b"]


def test_missing_rows_count_in_the_denominator(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    pd.DataFrame(
        {"Entry": ["a", "b"], "EC number": ["1.1.1.1", "1.1.1.1"], "Sequence": ["A", "C"]}
    ).to_csv(path, index=False)
    scores = np.array([[0.9, 0.1], [0.0, 0.0]])
    care = write_ranked_outputs(
        path, scores, ["1.1.1.1", "2.2.2.2"], tmp_path / "out.csv", missing=np.array([False, True])
    )
    assert care.loc[1, ["0", "1"]].isna().all()
    metrics = compute_care_metrics(care, k_values=(1,))
    assert metrics["k=1"]["level_4_accuracy"] == 50.0
    assert metrics["k=1"]["level_4_support"] == 2


def test_padding_on_the_device_equals_original_loop() -> None:
    lengths = [5, 2, 7]
    graphs = [
        Data(
            X=torch.randn(n, 5, 3),
            node_feat=torch.randn(n, 4),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            seq=torch.zeros(n, dtype=torch.long),
            activate_site=torch.rand(n, 1),
            name=f"s{i}",
            num_nodes=n,
        )
        for i, n in enumerate(lengths)
    ]
    seen = []
    model = lambda *args: seen.append(args) or torch.zeros(3, LABELS)  # noqa: E731
    batch = Batch.from_data_list(graphs)
    train.forward_batch(model, batch, torch.device("cpu"), False)
    data, mask, site = seen[0][5:]
    expected_data, expected_mask = torch.zeros(3, 7, 4), torch.zeros(3, 7)
    expected_site = torch.zeros(3, 7, 1)
    start = 0
    for i, n in enumerate(lengths):
        expected_data[i, :n] = batch.node_feat[start : start + n]
        expected_site[i, :n] = batch.activate_site[start : start + n]
        expected_mask[i, :n] = 1
        start += n
    assert torch.equal(data, expected_data)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(site, expected_site)


def test_worker_side_split_gives_the_original_groups() -> None:
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

    batch, labels = make_batch([3, 9, 2, 7, 4, 12, 5, 30, 1])
    skip = {"s4"}
    kept = Batch.from_data_list([g for g in batch.to_data_list() if g.name not in skip])
    for budget in (10, 12, 40, 10**9):
        expected = original_split(kept, budget)
        got = train.MicroCollate(budget, labels, LABELS, skip)(batch.to_data_list())
        assert [list(m.graphs.name) for m in got] == [list(e.name) for e in expected]
        for micro, ref in zip(got, expected):
            assert torch.equal(micro.graphs.x, ref.x)
            names = list(ref.name)
            assert torch.equal(micro.target, train.targets_for_names(names, labels, LABELS, "cpu"))


def test_targets_set_every_label() -> None:
    target = train.targets_for_names(["a", "b"], {"a": [0, 3], "b": [1]}, 5, "cpu")
    assert target.tolist() == [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]


def test_epoch_loss_accumulated_on_the_device_equals_the_python_sum(monkeypatch) -> None:
    values = [0.1234567, 1.5e-3, 2.718281828, 0.3333333, 9.87654321]
    losses = iter(values)
    monkeypatch.setattr(
        train,
        "run_batch",
        lambda *a: (torch.tensor(next(losses), dtype=torch.float32).double(), []),
    )

    class Bar(list):
        def set_postfix(self, **kwargs):
            pass

    monkeypatch.setattr(train, "progress", lambda loader, **k: Bar(loader))
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    loss = train.train_epoch(
        nn.Linear(1, 1), [[]] * 5, torch.device("cpu"), optimizer, None, False, set()
    )
    singles = [float(torch.tensor(v, dtype=torch.float32)) for v in values]
    assert loss == float(sum(singles) / len(singles))


def test_compile_is_off_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--split-group", "g"])
    assert train.parse_args().compile is False
    monkeypatch.setattr(sys, "argv", ["train", "--split-group", "g", "--compile"])
    assert train.parse_args().compile is True
    assert json.loads((BENCH / "default_hparams.json").read_text())["compile"] is False
