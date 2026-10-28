"""Training precision of the wrapper: TF32 (the production setting, no --amp) against fp32 on a synthetic split.

Run on a CUDA device: python -m pytest tests/test_precision_gpu.py -q
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import train  # noqa: E402
from emulator_bench.probes.common import ModelContext, synthetic_split  # noqa: E402
from emulator_bench.utils import set_seed, set_tf32  # noqa: E402

STEPS, BATCH = 20, 4
K = 10  # the production gap may be 10 times the fp32 noise (the TF32 run with initial weights perturbed by one TF32 rounding) before it counts as a defect


U = 2**-11  # one rounding of the weights at the production precision (TF32 keeps 10 mantissa bits)


def perturb(module, rel=U):
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(rel * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    split = synthetic_split(tmp_path_factory.mktemp("synthetic"), 16)
    with ModelContext(split) as ctx:
        yield ctx


def train_and_predict(ctx, tf32, amp, noise=False, scale=1.0):
    """Per-step losses and val probabilities after STEPS Adam steps (lr 1e-4) from seed 1."""
    device, split = torch.device("cuda"), ctx.split
    set_tf32(tf32)
    graphs = list(ctx.dataset(split.train_dataset))
    set_seed(1)
    model = train.build_model(ctx.model_mod, split.num_labels, device).train()
    if noise:
        perturb(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion, losses = nn.BCEWithLogitsLoss(), []
    collate = train.MicroCollate(2500, split.train_labels, split.num_labels)
    for step in range(STEPS):
        start = step * BATCH % len(graphs)
        optimizer.zero_grad(set_to_none=True)
        micros = collate(graphs[start : start + BATCH])
        loss, _ = train.run_batch(model, micros, device, criterion, amp)
        optimizer.step()
        losses.append(float(loss))
    val = Batch.from_data_list(list(ctx.dataset(split.val_dataset)))
    ((_, logits),) = train.forward_or_split(model.eval(), val, device, amp)
    return torch.tensor(losses), torch.sigmoid(logits.float())


@pytest.mark.gpu
def test_production_precision_tracks_fp32(context):
    """Same seed, weights and batches; 20 steps on 16 synthetic proteins (random features at the
    feature-cache boundary).

    The reference is fp32 with TF32 off; the production path is TF32; the noise control is the TF32 run with initial
    weights perturbed by 2**-11 relative. Bounds: max per-step loss gap and final mean absolute
    val probability gap each within K times their noise floor; the step-0 loss gap within K times its floor.
    """
    scale = 1.05 if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1" else 1.0  # scales only the gap run's initial weights
    try:
        runs = [
            train_and_predict(context, False, False),
            train_and_predict(context, True, False, noise=True),
            train_and_predict(context, True, False),
        ]
        if scale != 1.0:  # the gap uses this run; the floor keeps the unscaled production run
            runs.append(train_and_predict(context, True, False, scale=scale))
    finally:
        set_tf32(True)
    (ref_loss, ref_prob), (noise_loss, noise_prob), (clean_loss, clean_prob), (got_loss, got_prob) = runs[0], runs[1], runs[2], runs[-1]
    ref, noise, clean, prod = (x.double() for x in (ref_loss, noise_loss, clean_loss, got_loss))
    floor, gap = (noise - clean).abs().max().item(), (prod - ref).abs().max().item()
    tiny, tiny0 = 1e-7 * ref.abs().max().item(), 1e-7 * ref[0].abs().item()
    gap0, floor0 = (prod[0] - ref[0]).abs().item(), (noise[0] - clean[0]).abs().item()
    final_gap, final_floor = (got_prob - ref_prob).abs().mean().item(), (noise_prob - clean_prob).abs().mean().item()
    print(
        f"graphec tf32: max gap={gap:.3e}, max floor={floor:.3e}, ratio={gap / max(floor, tiny):.2f}, K={K}, "
        f"step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}; final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap <= K * max(floor, tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * ref_prob.abs().mean().item())
