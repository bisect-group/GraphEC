import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

ACTIVE_SITES = str(Path(__file__).resolve().parents[1] / "Active_sites")
SIZES = (7, 12, 5)


@pytest.fixture(scope="module")
def mods():
    saved = {k: sys.modules.pop(k, None) for k in ("data", "model")}
    sys.path.insert(0, ACTIVE_SITES)
    import data
    import model

    yield data, model
    sys.path.remove(ACTIVE_SITES)
    for k, v in saved.items():
        sys.modules.pop(k, None)
        if v is not None:
            sys.modules[k] = v


def make_graphs() -> list[Data]:
    torch.manual_seed(0)
    graphs = []
    for n in SIZES:
        X = torch.cumsum(torch.randn(n, 5, 3), 0)
        edge_index = radius_graph(X[:, 1], r=8.0, loop=True)
        graphs.append(Data(X=X, node_feat=torch.randn(n, 1024 + 9), edge_index=edge_index))
    return graphs


def build_model(model_mod):
    torch.manual_seed(1)
    return model_mod.GraphEC_AS(1024 + 9 + 184, 450, 16, 2, 0.2, 0.1, task="ActiveSite").eval()


def test_batched_geometry_with_ptr_equals_per_graph(mods) -> None:
    data_mod, _ = mods
    graphs = make_graphs()
    batch = Batch.from_data_list(graphs)
    node, edge = data_mod.get_geo_feat(batch.X, batch.edge_index, batch.ptr)
    singles = [data_mod.get_geo_feat(g.X, g.edge_index) for g in graphs]
    torch.testing.assert_close(node, torch.cat([s[0] for s in singles]), atol=1e-6, rtol=0)
    torch.testing.assert_close(edge, torch.cat([s[1] for s in singles]), atol=1e-6, rtol=0)


def test_batched_geometry_without_ptr_differs_at_boundaries(mods) -> None:
    data_mod, _ = mods
    graphs = make_graphs()
    batch = Batch.from_data_list(graphs)
    node, _ = data_mod.get_geo_feat(batch.X, batch.edge_index)
    singles = torch.cat([data_mod.get_geo_feat(g.X, g.edge_index)[0] for g in graphs])
    assert (node - singles).abs().max() > 1e-3


def test_forward_geo_equals_forward(mods) -> None:
    data_mod, model_mod = mods
    model = build_model(model_mod)
    graphs = make_graphs()
    batch = Batch.from_data_list(graphs)

    def ids(g):
        return torch.zeros(g.X.shape[0], dtype=torch.long)

    def forward_geo(b, batch_id, ptr):
        h_V_geo, h_E = data_mod.get_geo_feat(b.X, b.edge_index, ptr)
        h_V = torch.cat([b.node_feat, h_V_geo], dim=-1)
        return model.forward_geo(h_V, h_E, b.edge_index, None, batch_id)

    with torch.no_grad():
        single = graphs[0]
        expected = model(single.X, single.node_feat, single.edge_index, None, ids(single))
        torch.testing.assert_close(
            forward_geo(single, ids(single), None), expected, atol=1e-6, rtol=0
        )

        expected = torch.cat([model(g.X, g.node_feat, g.edge_index, None, ids(g)) for g in graphs])
        torch.testing.assert_close(
            forward_geo(batch, batch.batch, batch.ptr), expected, atol=1e-6, rtol=0
        )
