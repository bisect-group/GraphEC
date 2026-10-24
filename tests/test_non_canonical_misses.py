import contextlib
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import evaluate  # noqa: E402
from emulator_bench.dataset_adapter import load_split_records  # noqa: E402


def test_non_canonical_test_rows_count_as_misses(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source" / "grp"
    source.mkdir(parents=True)
    rows = {
        "uniprot_id": ["P1", "P2", "P3", "P4"],
        "sequence": ["AAA", "CXC", "DDD", "EEE"],
        "ec_number": ["1.1.1.1", "2.2.2.2;3.-.-.-", "1.1.1.1", "3.-.-.-"],
    }
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_parquet(source / f"{split}.parquet")
    manifests = {}
    for split in ("train", "val", "test"):
        records, _ = load_split_records(source / f"{split}.parquet", split_name=split)
        records["Entry"] = records["cache_key"] = records["Original Entry"]
        manifests[split] = str(tmp_path / f"{split}.csv")
        records.to_csv(manifests[split], index=False)
    assert list(pd.read_csv(manifests["test"])["Entry"]) == ["P1", "P3"]
    metadata = {
        "dataset_root": str(tmp_path / "source"),
        "split_group": "grp",
        "limit_per_split": None,
        "manifests": manifests,
        "stats": {"test": {"rows_with_no_full_label": 1}},
    }
    labels = ["1.1.1.1", "2.2.2.2"]
    label_to_idx = {label: i for i, label in enumerate(labels)}

    def predict(dataset, **kwargs):
        assert sorted(dataset) == ["P1", "P3"]
        return ["P1", "P3"], np.array([[0.9, 0.1], [0.8, 0.2]]), []

    monkeypatch.setattr(evaluate, "predict_scores", predict)
    monkeypatch.setattr(evaluate, "load_feature_failures", lambda cache: [])
    monkeypatch.setattr(evaluate, "ensure_graphec_data_links", lambda *a: None)
    monkeypatch.setattr(evaluate, "load_active_sites", lambda *a: {})
    monkeypatch.setattr(evaluate, "baseline_cwd", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(evaluate, "import_task_module", lambda *a: None)
    monkeypatch.setattr(evaluate, "load_model", lambda *a: None)
    monkeypatch.setitem(sys.modules, "data", types.ModuleType("data"))
    args = Namespace(
        out_root=str(tmp_path / "out"),
        split_group="grp",
        seed=0,
        label_diffusion=False,
        batch_size=1,
        num_workers=0,
        amp=False,
        max_residues=100,
        rank_top_k=2,
    )
    evaluate.evaluate_split(args, metadata, {}, "test", labels, label_to_idx)

    result = next((tmp_path / "out").rglob("skipped_test.json"))
    skipped = json.loads(result.read_text())
    assert skipped["non_canonical"] == 1
    assert skipped["non_canonical_names"] == ["P2"]
    assert skipped["rows"] == 3 and skipped["misses"] == 1 and skipped["no_features"] == 0
    metrics = json.loads((result.parent / "test_metrics.json").read_text())
    assert metrics["rows"] == 3 and metrics["misses"] == 1
    scores = np.load(result.parent / "test_scores.npz")
    assert list(scores["names"]) == ["P1", "P3", "P2"]
    assert scores["final_scores"][2].sum() == 0
    care = pd.read_csv(result.parent / "test_results_df.csv")
    assert care.loc[2, ["0", "1"]].isna().all()
    assert care.loc[0, "0"] == "1.1.1.1"
