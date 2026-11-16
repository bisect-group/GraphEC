import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as ssp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import evaluate  # noqa: E402


def fake_diamond(monkeypatch, calls: list, version: str = "2.0.14") -> None:
    def run(argv, check, capture_output=False, text=False):
        calls.append(argv[1])
        if argv[1] == "version":
            return subprocess.CompletedProcess(argv, 0, stdout=f"diamond version {version}\n")
        if argv[1] == "blastp":
            Path(argv[argv.index("-o") + 1]).write_text(f"hits for {argv[argv.index('-q') + 1]}\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run)
    evaluate.diamond_version.cache_clear()


def search(tmp_path: Path, name: str, train: str, query: str) -> Path:
    work = tmp_path / name
    work.mkdir(exist_ok=True)
    (work / "train.fasta").write_text(train)
    (work / "eval.fasta").write_text(query)
    out = work / "out.tsv"
    evaluate.diamond_search(
        Path("diamond"), work / "eval.fasta", work / "train.fasta", out, tmp_path / "cache", work
    )
    return out


def test_runs_with_equal_fasta_share_one_search(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    fake_diamond(monkeypatch, calls)
    first = search(tmp_path, "seed1", ">a\nAAA\n", ">b\nCCC\n")
    second = search(tmp_path, "seed2", ">a\nAAA\n", ">b\nCCC\n")
    assert calls.count("blastp") == 1 and calls.count("makedb") == 1
    assert first.read_text().split("\n")[0].startswith("hits for ")
    assert first.read_bytes() == second.read_bytes()
    assert len(list((tmp_path / "cache").glob("*.tsv"))) == 1
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_changed_fasta_or_version_starts_a_new_search(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    fake_diamond(monkeypatch, calls)
    search(tmp_path, "a", ">a\nAAA\n", ">b\nCCC\n")
    search(tmp_path, "b", ">a\nAAA\n", ">b\nCCD\n")
    assert calls.count("blastp") == 2
    fake_diamond(monkeypatch, calls, version="9")
    search(tmp_path, "c", ">a\nAAA\n", ">b\nCCC\n")
    assert calls.count("blastp") == 3


def test_parallel_diffusion_equals_the_serial_solve() -> None:
    rng = np.random.default_rng(0)
    W = ssp.random(40, 40, density=0.1, random_state=1, format="csr")
    laplacian = evaluate.compute_laplacian((W + W.T + ssp.identity(40)).tocsr())
    initial = ssp.csr_matrix(rng.random((40, 9)))
    parallel = evaluate.diffuse(laplacian, 0.1, initial, 5, block=2, workers=3)
    A = (ssp.identity(40) + laplacian.multiply(0.1)).tocsr()
    dense = initial.toarray()
    serial = np.concatenate(
        [evaluate.block_cg(A, dense[:, i : i + 2])[5:] for i in range(0, 9, 2)], axis=1
    )
    assert np.array_equal(parallel, serial)
