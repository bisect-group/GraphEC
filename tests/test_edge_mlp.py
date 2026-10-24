import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.utils import baseline_cwd, import_task_module  # noqa: E402


def concat_reference(edge_mlp, h_V, edge_index, h_E):
    h_EV = torch.cat([h_V[edge_index[0]], h_E, h_V[edge_index[1]]], dim=-1)
    h_message = edge_mlp.W12(edge_mlp.act(edge_mlp.W11(h_EV)))
    return edge_mlp.norm(h_E + edge_mlp.dropout(h_message))


def test_factorized_edge_mlp_equals_the_concat_form() -> None:
    with baseline_cwd():
        model = import_task_module("EC_number", "model")
    torch.manual_seed(0)
    edge_mlp = model.EdgeMLP(256, 0.2)
    assert list(edge_mlp.state_dict()) == [
        "norm.weight", "norm.bias", "norm.running_mean", "norm.running_var",
        "norm.num_batches_tracked", "W11.weight", "W11.bias", "W12.weight", "W12.bias",
    ]  # fmt: skip
    assert edge_mlp.W11.weight.shape == (256, 768)
    h_V, h_E = torch.randn(50, 256), torch.randn(400, 256)
    edge_index = torch.randint(0, 50, (2, 400))
    edge_mlp.train()
    torch.manual_seed(1)
    new = edge_mlp(h_V, edge_index, h_E)
    torch.manual_seed(1)
    old = concat_reference(edge_mlp, h_V, edge_index, h_E)
    assert (new - old).abs().max() < 1e-5
    h_V.requires_grad_()
    edge_mlp(h_V, edge_index, h_E).sum().backward()
    assert edge_mlp.W11.weight.grad.shape == (256, 768)
