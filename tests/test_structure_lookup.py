import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import cache_features  # noqa: E402
from emulator_bench.utils import cache_dirs, read_json, safe_id  # noqa: E402


def pdb_text(residues: int) -> str:
    lines = []
    for index in range(1, residues + 1):
        for atom in ("N", "CA", "C", "O"):
            lines.append(
                f"ATOM  {len(lines) + 1:5d}  {atom:<3s} ALA A{index:4d}    "
                f"{index:8.3f}{1.0:8.3f}{2.0:8.3f}  1.00  0.00           {atom[0]}"
            )
    return "\n".join(lines) + "\nTER\n"


def test_priority_and_residue_check(tmp_path: Path) -> None:
    root = tmp_path / "structures"
    for folder, name, residues in (
        ("esmfold2_pdb", "ESMFold2-A.pdb", 3),
        ("esm", "ESM3-open-small-B.pdb", 3),
        ("alphafold", "AF-B-F1-model_v6.pdb", 3),
        ("esmfold2_pdb", "ESMFold2-C.pdb", 5),
        ("alphafold", "AF-D-F1-model_v6.pdb", 3),
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / name).write_text(pdb_text(residues))
    items = [{"cache_key": entry, "entries": [entry], "sequence": "AAA"} for entry in "ABCDE"]
    dirs = cache_dirs(tmp_path / "cache")
    unresolved = cache_features.link_structures(items, dirs, root, {})
    assert [item["cache_key"] for item in unresolved] == ["C", "E"]
    sources = read_json(tmp_path / "cache" / "structure_sources.json")
    assert sources == {"A": "esmfold2", "B": "esm3", "D": "alphafold"}
    assert (dirs["structures"] / "A.tensor").exists()


def test_safe_id_has_no_shell_pipe() -> None:
    assert "|" not in safe_id("P1|A2V")


def test_structure_with_missing_backbone_atom_is_not_used() -> None:
    import Features.features as features

    full = pdb_text(2).splitlines(keepends=True)
    broken = [line for line in full if not line.startswith("ATOM      8")]
    assert cache_features.has_backbone(features, full, 2)
    assert not cache_features.has_backbone(features, full, 3)
    assert not cache_features.has_backbone(features, broken, 2)


def test_dssp_failure_tries_the_next_structure_source(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    root = tmp_path / "structures"
    for folder, name in (
        ("esmfold2_pdb", "ESMFold2-{}.pdb"),
        ("alphafold", "AF-{}-F1-model_v6.pdb"),
    ):
        (root / folder).mkdir(parents=True)
        for entry in "AB":
            (root / folder / name.format(entry).replace("{}", entry)).write_text(pdb_text(3))
    items = [{"cache_key": entry, "entries": [entry], "sequence": "AAA"} for entry in "ABC"]
    dirs = cache_dirs(tmp_path / "cache")
    cache_features.link_structures(items, dirs, root, {})
    assert read_json(tmp_path / "cache" / "structure_sources.json") == {
        "A": "esmfold2",
        "B": "esmfold2",
    }

    # mkdssp rejects every esmfold2 file; C has no structure at all and is not passed in.
    def fake_dssp(pending, dirs, workers):
        sources = read_json(dirs["structures"].parent / "structure_sources.json")
        return [i["cache_key"] for i in pending if sources[i["cache_key"]] == "esmfold2"]

    monkeypatch.setattr(cache_features, "generate_dssp_features", fake_dssp)
    args = SimpleNamespace(dssp_workers=1, num_gpus=1, lookup_workers=1, gpu_memory_gb=None)
    no_source, esmfold_oom = cache_features.dssp_with_fallback(items[:2], dirs, args, root, {})
    assert (no_source, esmfold_oom) == ([], [])
    assert read_json(tmp_path / "cache" / "structure_sources.json")["A"] == "alphafold"
    assert (dirs["structures"] / "B.tensor").exists()

    # With no other source left, the sequence is reported as having none.
    (root / "alphafold" / "AF-B-F1-model_v6.pdb").unlink()
    cache_features.link_structures(items[1:2], dirs, root, {}, {"B": {"alphafold"}})
    monkeypatch.setattr(cache_features, "ESMFOLD_SOURCE", "esmfold2")
    no_source, _ = cache_features.dssp_with_fallback(items[1:2], dirs, args, root, {})
    assert no_source == ["B"]


def test_predicted_structure_needs_the_same_residues(tmp_path: Path) -> None:
    root = tmp_path / "structures"
    (root / "esmfold2_pdb").mkdir(parents=True)
    (root / "esmfold2_pdb" / "ESMFold2-A.pdb").write_text(pdb_text(3))
    items = [{"cache_key": "A", "entries": ["A"], "sequence": "AGA"}]
    unresolved = cache_features.link_structures(items, cache_dirs(tmp_path / "cache"), root, {})
    assert unresolved == items


def test_predicted_structure_loses_oxt(tmp_path: Path) -> None:
    path = tmp_path / "p.pdb"
    oxt = "ATOM      9  OXT ALA A   3       1.000   1.000   2.000  1.00  0.00           O\n"
    path.write_text(pdb_text(3).replace("TER\n", oxt + "TER\n"))
    lines = cache_features.predicted_pdb_lines(path)
    assert not any(" OXT " in line for line in lines)
    assert sum(line.startswith("ATOM") for line in lines) == 12


def test_dssp_failure_of_experimental_falls_to_esmfold(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    lines = pdb_text(3).splitlines(keepends=True)
    monkeypatch.setattr(cache_features, "experimental_pdb_lines", lambda cif, chain: (lines, "AAA"))
    items = [{"cache_key": "A", "entries": ["A"], "sequence": "AAA"}]
    dirs = cache_dirs(tmp_path / "cache")
    experimental = {"A": [("1ABC", "A")]}
    assert cache_features.link_structures(items, dirs, tmp_path, experimental) == []
    sources_path = tmp_path / "cache" / "structure_sources.json"
    assert read_json(sources_path)["A"] == "experimental:1ABC:A"

    def fake_dssp(pending, dirs, workers):
        sources = read_json(sources_path)
        return [i["cache_key"] for i in pending if sources[i["cache_key"]].startswith("exp")]

    def fake_fold(items, dirs, **kwargs):
        for item in items:
            (dirs["structures"] / f"{item['cache_key']}.pdb").write_text(pdb_text(3))
            cache_features.save_structure(item["cache_key"], lines, dirs)
        cache_features.record_sources(
            dirs, {item["cache_key"]: cache_features.ESMFOLD_SOURCE for item in items}
        )
        return []

    monkeypatch.setattr(cache_features, "generate_dssp_features", fake_dssp)
    monkeypatch.setattr(
        cache_features, "run_sharded", lambda f, items, n, **kw: fake_fold(items, **kw)
    )
    args = SimpleNamespace(dssp_workers=1, num_gpus=1, lookup_workers=1, gpu_memory_gb=None)
    assert cache_features.dssp_with_fallback(items, dirs, args, tmp_path, experimental) == ([], [])
    assert read_json(sources_path)["A"] == cache_features.ESMFOLD_SOURCE


def test_parallel_lookup_equals_serial(tmp_path: Path) -> None:
    root = tmp_path / "structures"
    for folder, name, residues in (
        ("esmfold2_pdb", "ESMFold2-{}.pdb", 3),
        ("alphafold", "AF-{}-F1-model_v6.pdb", 3),
    ):
        (root / folder).mkdir(parents=True)
        for entry in "ACE":
            (root / folder / name.format(entry)).write_text(pdb_text(residues))
    items = [{"cache_key": e, "entries": [e], "sequence": "AAA"} for e in "ABCDE"]
    results = []
    for workers in (1, 2):
        dirs = cache_dirs(tmp_path / f"cache{workers}")
        unresolved = cache_features.link_structures(items, dirs, root, {}, workers=workers)
        sources = read_json(tmp_path / f"cache{workers}" / "structure_sources.json")
        results.append((unresolved, sources, sorted(p.name for p in dirs["structures"].iterdir())))
    assert results[0] == results[1]
    assert [item["cache_key"] for item in results[0][0]] == ["B", "D"]


def backbone_pdb(residues: int, drop: tuple[int, str] | None = None) -> str:
    """Backbone-only PDB with 3.8 A spacing; `drop` removes one (residue, atom)."""
    coords = {
        "N": (0.0, 1.4, 0.0),
        "CA": (1.4, 1.4, 0.0),
        "C": (2.0, 0.0, 0.0),
        "O": (3.2, 0.0, 0.5),
    }
    lines = []
    for index in range(1, residues + 1):
        for atom, (x, y, z) in coords.items():
            if drop != (index, atom):
                lines.append(
                    f"ATOM  {len(lines) + 1:5d} {atom:^4s} ALA A{index:4d}    "
                    f"{x + 4 * index:8.3f}{y:8.3f}{z:8.3f}  1.00 12.34           {atom[0]}"
                )
    return "\n".join(lines) + "\nTER\n"


def link_one(tmp_path: Path, text: str):
    folder = tmp_path / "root" / "esm"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ESM3-open-small-A.pdb").write_text(text.removesuffix("TER\n").rstrip("\n"))
    dirs = cache_dirs(tmp_path / "cache")
    item = {"cache_key": "k", "entries": ["A"], "sequence": "AAA"}
    unresolved, sources = cache_features.link_chunk([item], dirs, tmp_path / "root", {})
    return dirs, unresolved, sources


def test_missing_last_oxygen_is_added(tmp_path: Path) -> None:
    import numpy as np
    import torch

    dirs, unresolved, sources = link_one(tmp_path, backbone_pdb(3, drop=(3, "O")))
    assert not unresolved
    assert sources == {"k": "esm3+terminal_O"}
    path = dirs["structures"] / "k.pdb"
    assert path.is_file() and not path.is_symlink()
    lines = path.read_text().splitlines()
    atoms = [line for line in lines if line.startswith("ATOM")]
    assert lines.index(atoms[-1]) == lines.index(atoms[-2]) + 1
    assert len(atoms) == 12 and atoms[-1][12:16] == " O  " and atoms[-2][12:16] == " C  "
    assert atoms[-1][6:11] == "   12" and atoms[-1][17:27] == atoms[-2][17:27]
    assert atoms[-1][60:66] == "12.34".rjust(6) and atoms[-1][76:78] == " O"

    def xyz(line: str):
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    assert abs(np.linalg.norm(xyz(atoms[-1]) - xyz(atoms[-2])) - 1.231) < 1e-2
    assert len(torch.load(dirs["structures"] / "k.tensor")) == 3


def test_other_missing_backbone_atoms_are_rejected(tmp_path: Path) -> None:
    for drop in ((2, "O"), (3, "CA"), (1, "CA")):
        _, unresolved, sources = link_one(tmp_path / str(drop), backbone_pdb(3, drop=drop))
        assert len(unresolved) == 1 and not sources


def test_sources_are_saved_during_a_long_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache_features, "SOURCE_FLUSH", 2)
    saved = []
    monkeypatch.setattr(cache_features, "record_sources", lambda d, new: saved.append(dict(new)))
    folder = tmp_path / "root" / "esm"
    folder.mkdir(parents=True)
    items = [{"cache_key": key, "entries": [key], "sequence": "AAA"} for key in "ABC"]
    for key in "ABC":
        (folder / f"ESM3-open-small-{key}.pdb").write_text(backbone_pdb(3))
    cache_features.link_chunk(items, cache_dirs(tmp_path / "cache"), tmp_path / "root", {})
    assert saved == [{"A": "esm3"}]
