"""Wrapper vs original GraphEC code on tiny inputs, on the CPU.

Run: CUDA_VISIBLE_DEVICES= /home/adhil/.conda/envs/mldb/bin/python -m pytest \
  tests/test_original_equivalence.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402
from emulator_bench.probes import exactness_probe as probe  # noqa: E402
from emulator_bench.probes.common import ModelContext, synthetic_split  # noqa: E402
from emulator_bench.utils import baseline_cwd, import_task_module, set_seed  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    split = synthetic_split(tmp_path_factory.mktemp("synthetic"), 12)
    with ModelContext(split) as ctx:
        yield ctx


def test_wrapper_step_gradients_equal_the_original_path(context) -> None:
    # Same code as the exactness probe: original = upstream forward, CPU padding, per-row targets;
    # new = wrapper forward, micro-batching (60 residues, so every batch splits), vectorised
    # targets. One backward pass in fp32 and train mode (the dropout seed is shared). Loss and
    # gradients differ only by summation order: loss <= 1e-6, gradients <= 1e-5 of the largest
    # gradient. Parameters after Adam are not compared: Adam turns 1e-7 gradient noise on
    # near-zero gradients into steps of the learning rate.
    mod, split = context.model_mod, context.split
    dataset = context.dataset(split.train_dataset)
    graphs = [dataset[i] for i in range(6)]
    criterion = nn.BCEWithLogitsLoss()
    old_forward, old_padding = mod.GraphEC.forward, mod.GraphEC.padding_ver1
    args = (graphs, split.train_labels, split.num_labels, CPU, criterion, 60)
    try:
        set_seed(1)
        model = train.build_model(mod, split.num_labels, CPU).train()
        mod.GraphEC.forward = context.upstream_forward
        mod.GraphEC.padding_ver1 = lambda self, x, b, d: probe.cpu_padding(x, b, d)[:2]
        old_loss = probe.original_step(model, *args)
        old_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        set_seed(1)
        model = train.build_model(mod, split.num_labels, CPU).train()
        new_loss = probe.new_step(model, *args)
        new_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    finally:
        mod.GraphEC.forward, mod.GraphEC.padding_ver1 = old_forward, old_padding
    assert abs(old_loss - new_loss) <= 1e-6
    scale = max(g.abs().max().item() for g in old_grads)
    assert probe.max_diff(old_grads, new_grads) <= 1e-5 * scale


def test_wrapper_split_matches_the_original_split_and_targets(context) -> None:
    dataset = context.dataset(context.split.train_dataset)
    graphs = [dataset[i] for i in range(8)]
    labels, num = context.split.train_labels, context.split.num_labels
    micros = train.MicroCollate(40, labels, num)(graphs)
    original = probe.original_split(Batch.from_data_list(graphs), 40)
    assert [list(m.graphs.name) for m in micros] == [list(b.name) for b in original]
    for micro, batch in zip(micros, original):
        expected = probe.original_targets(list(batch.name), labels, num, CPU)
        assert torch.equal(micro.target, expected)


@pytest.fixture(scope="module")
def active_sites_model():
    with baseline_cwd():
        yield import_task_module("Active_sites", "model")


def test_last_nodes_only_forward_equals_full_forward_at_last_nodes(active_sites_model) -> None:
    # The full forward also updates the edges of the last layer; nothing reads them, so the node
    # output is identical. Eval mode, fp64, same ops in the same order: tolerance 1e-12.
    torch.manual_seed(0)
    nodes, hidden = 30, 32
    encoder = active_sites_model.Graph_encoder(
        node_in_dim=12, edge_in_dim=10, hidden_dim=hidden, num_layers=3, drop_rate=0.1
    ).double()
    for module in encoder.modules():  # non-trivial running statistics
        if isinstance(module, nn.BatchNorm1d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
    encoder.eval()
    edge_index = torch.randint(0, nodes, (2, 90))
    batch_id = torch.arange(nodes) // 10
    h_V = torch.randn(nodes, 12, dtype=torch.float64)
    h_E = torch.randn(90, 10, dtype=torch.float64)
    with torch.no_grad():
        full = encoder(h_V, edge_index, h_E, None, batch_id)
        last_only = encoder.forward_last_nodes_only(h_V, edge_index, h_E, None, batch_id)
    assert torch.allclose(full, last_only, rtol=0, atol=1e-12)


def test_forward_geo_equals_head_on_the_full_encoder(active_sites_model) -> None:
    # `forward_geo` = last-nodes-only encoder + the FC head. Compare with the head on the full
    # encoder output. Eval mode, fp64, tolerance 1e-12.
    torch.manual_seed(1)
    nodes, hidden, task = 24, 32, "AS"
    model = active_sites_model.GraphEC_AS(12, 10, hidden, 3, 0.1, 0, task).double().eval()
    edge_index = torch.randint(0, nodes, (2, 70))
    batch_id = torch.arange(nodes) // 8
    h_V = torch.randn(nodes, 12, dtype=torch.float64)
    h_E = torch.randn(70, 10, dtype=torch.float64)
    with torch.no_grad():
        new = model.forward_geo(h_V, h_E, edge_index, None, batch_id)
        encoded = model.Graph_encoder(h_V, edge_index, h_E, None, batch_id)
        old = model._modules[f"FC_{task}2"](F.elu(model._modules[f"FC_{task}1"](encoded)))
    assert torch.allclose(new, old, rtol=0, atol=1e-12)
