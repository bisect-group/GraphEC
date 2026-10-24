import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402
from emulator_bench.probes.common import ModelContext, synthetic_split  # noqa: E402
from emulator_bench.probes.exactness_probe import cpu_padding  # noqa: E402
from emulator_bench.utils import set_seed  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    split = synthetic_split(tmp_path_factory.mktemp("synthetic"), 10)
    with ModelContext(split) as ctx:
        yield ctx


def test_device_padding_forward_equals_the_upstream_forward(context) -> None:
    split, mod = context.split, context.model_mod
    dataset = context.dataset(split.train_dataset)
    graphs = [dataset[i] for i in range(6)]
    batch = train.MicroCollate(10**9)(graphs)[0].graphs
    set_seed(1)
    model = train.build_model(mod, split.num_labels, CPU).eval()
    with torch.no_grad():
        new = train.forward_batch(model, batch, CPU, False)
        mod.GraphEC.forward = context.upstream_forward
        mod.GraphEC.padding_ver1 = lambda self, x, b, d: cpu_padding(x, b, d)[:2]
        padded, mask, site = cpu_padding(
            batch.node_feat, batch.batch, batch.node_feat.shape[1], batch.activate_site
        )
        old = model(
            batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch, padded, mask, site
        ).float()
    assert torch.equal(new, old)


def test_training_loop_runs_on_the_real_model(context, tmp_path: Path) -> None:
    split = context.split
    args = SimpleNamespace(
        batch_size=4, num_workers=2, seed=1, lr=1e-3, amp=False, compile=False,
        max_residues=60, epochs=2, patience=4,
    )  # fmt: skip
    result = train.train_model(
        train_dataset=split.train_dataset,
        val_dataset=split.val_dataset,
        train_labels=split.train_labels,
        val_labels=split.val_labels,
        active_sites=split.active_sites,
        data_mod=context.data_mod,
        model_mod=context.model_mod,
        num_labels=split.num_labels,
        args=args,
        run_root=tmp_path,
    )
    losses = [h["train_loss"] for h in result["history"]]
    assert len(losses) == 2 and all(0 < loss < 10 for loss in losses)
    assert (tmp_path / "bestmodel.pth").exists()


def test_mismatched_geometric_features_take_the_upstream_fallback(context, capsys) -> None:
    split, mod = context.split, context.model_mod
    dataset = context.dataset(split.train_dataset)
    batch = train.MicroCollate(10**9)([dataset[i] for i in range(3)])[0].graphs
    set_seed(1)
    model = train.build_model(mod, split.num_labels, CPU).eval()
    real_geo = mod.get_geo_feat

    def short_geo(X, edge_index):
        node, edge = real_geo(X, edge_index)
        return node[:-1], edge

    def ones_geo(X, edge_index):
        node, edge = real_geo(X, edge_index)
        return torch.ones_like(node), edge

    try:
        with torch.no_grad():
            mod.get_geo_feat = short_geo
            fallback = train.forward_batch(model, batch, CPU, False)
            assert "torch.Size" in capsys.readouterr().out
            mod.get_geo_feat = ones_geo
            expected = train.forward_batch(model, batch, CPU, False)
    finally:
        mod.get_geo_feat = real_geo
    assert torch.equal(fallback, expected)
