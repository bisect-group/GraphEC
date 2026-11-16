import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_features import manifest_items  # noqa: E402
from emulator_bench.dataset_adapter import normalize_sequence, sequence_hash  # noqa: E402
from emulator_bench.evaluate import compute_laplacian, diffuse  # noqa: E402
from emulator_bench.utils import (  # noqa: E402
    feature_keys_from_manifest,
    seed_run_root_for_split,
)


def test_features_are_keyed_by_sequence(tmp_path: Path) -> None:
    long_sequence = "ACDE" * 600
    rows = [("P1", "AAAA"), ("P2", "AAAA"), ("P3", long_sequence)]
    manifest = pd.DataFrame(
        {
            "cache_key": [r[0] for r in rows],
            "Original Entry": [r[0] for r in rows],
            "Sequence": [normalize_sequence(r[1]) for r in rows],
            "sequence_sha256": [sequence_hash(r[1]) for r in rows],
        }
    )
    path = tmp_path / "train.csv"
    manifest.to_csv(path, index=False)
    items = manifest_items({"manifests": {"train": str(path)}})
    assert len(items) == 2
    shared = next(item for item in items if item["sequence"] == "AAAA")
    assert shared["entries"] == ["P1", "P2"]
    assert any(len(item["sequence"]) == len(long_sequence) for item in items)
    keys = feature_keys_from_manifest(path)
    assert keys["P1"] == keys["P2"] != keys["P3"]


def test_diffusion_matches_dense_inverse() -> None:
    rng = np.random.default_rng(0)
    n = 60
    W = ssp.random(n, n, density=0.15, random_state=1, format="csr")
    W = W.maximum(W.T)
    W = (W + ssp.identity(n)).tocsr()
    laplacian = compute_laplacian(W)
    initial = ssp.csr_matrix(rng.random((n, 700)) * (rng.random((n, 700)) < 0.3))
    skip = 20
    dense = np.linalg.inv((ssp.identity(n) + laplacian.multiply(0.1)).toarray())
    expected = dense[skip:] @ initial.toarray()
    result = diffuse(laplacian, 0.1, initial, skip, block=256)
    assert result.shape == expected.shape
    assert np.abs(result - expected).max() < 1e-8


def test_run_layout(tmp_path: Path) -> None:
    flat = seed_run_root_for_split("random_splits_grouped_sequence", 666, tmp_path)
    assert flat == tmp_path / "random_splits_grouped_sequence/all/seed_666"
    threshold = seed_run_root_for_split("enzyme_sequence_splits/threshold_0.39", 7, tmp_path)
    assert threshold == tmp_path / "enzyme_sequence_splits/threshold_0.39/seed_7"


def test_active_site_oom_is_recorded_per_sequence() -> None:
    from torch_geometric.data import Batch, Data

    from emulator_bench.cache_features import predict_or_fail

    graphs = [Data(x=torch.zeros(n, 1), name=f"s{n}") for n in (2, 3, 9)]
    done = []

    def predict(batch) -> None:
        if batch.num_nodes > 5:
            raise torch.cuda.OutOfMemoryError
        done.extend(batch.name)

    assert predict_or_fail(predict, Batch.from_data_list(graphs)) == ["s9"]
    assert done == ["s2", "s3"]


def test_dssp_failure_is_returned_not_raised(tmp_path: Path) -> None:
    from emulator_bench.cache_features import generate_dssp_features

    dirs = {name: tmp_path / name for name in ("structures", "dssp", "tmp")}
    for path in dirs.values():
        path.mkdir()
    (dirs["structures"] / "k.pdb").write_text("not a structure\n")
    assert generate_dssp_features([{"cache_key": "k", "sequence": "AA"}], dirs, workers=1) == ["k"]
    assert not list(dirs["dssp"].iterdir())


def test_experimental_chain_with_long_id(tmp_path: Path) -> None:
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.info import residue

    from emulator_bench.cache_features import experimental_pdb_lines

    atoms = residue("ALA") + residue("GLY")
    atoms = atoms[atoms.element != "H"]
    atoms.res_id = np.where(np.arange(atoms.array_length()) < 5, 1, 2)
    atoms.chain_id[:] = "AA"
    path = tmp_path / "long.cif"
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms, data_block="x")
    cif.write(path)
    lines, sequence = experimental_pdb_lines(path, "AA")
    assert sequence == "AG"
    assert sum(line.startswith("ATOM") for line in lines) == np.sum(atoms.atom_name != "OXT")


def test_experimental_long_residue_name_becomes_unk(tmp_path: Path) -> None:
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.info import residue

    from emulator_bench.cache_features import experimental_pdb_lines

    atoms = residue("ALA")
    atoms = atoms[atoms.element != "H"]
    atoms.res_name = np.array(["A1A2I"] * atoms.array_length(), dtype="U5")
    atoms.chain_id[:] = "A"
    path = tmp_path / "long_name.cif"
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms, data_block="x")
    cif.write(path)
    lines, sequence = experimental_pdb_lines(path, "A")
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    assert len(sequence) == 1
    assert len(atom_lines) == np.sum(atoms.atom_name != "OXT")
    assert {line[17:20] for line in atom_lines} == {"UNK"}


def test_lookup_skips_chain_the_pdb_format_cannot_hold(tmp_path: Path, monkeypatch) -> None:
    from biotite.structure import BadStructureError

    from emulator_bench import cache_features

    def refuse(cif: Path, chain: str) -> None:
        raise BadStructureError("too large")

    monkeypatch.setattr(cache_features, "experimental_pdb_lines", refuse)
    dirs = {"structures": tmp_path}
    item = {"cache_key": "k", "sequence": "AG", "entries": ["P1"]}
    unresolved, sources = cache_features.link_chunk(
        [item], dirs, tmp_path, {"P1": [("1ABC", "A")]}
    )
    assert unresolved == [item] and sources == {}
