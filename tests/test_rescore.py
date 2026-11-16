import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import rescore  # noqa: E402

LABELS = ["1.1.1.1", "2.2.2.2"]
COLUMNS = ["Entry", "cache_key", "Original Entry", "Sequence", "EC number", "sequence_sha256"]


def row(entry: str, sequence: str, ec: str) -> list[str]:
    return [entry, entry, entry, sequence, ec, f"sha_{entry}"]


def test_x_mapping_and_items() -> None:
    assert rescore.x_mapped("AUZOBC") == "AXXXXC"
    assert rescore.known_letters("AUXJC") == "AAAAC"
    rows = pd.DataFrame(
        [row("P1", "AUC", "1.1.1.1"), row("P2", "BBC", "2.2.2.2")], columns=COLUMNS
    )
    assert rescore.feature_items(rows)[0] == {
        "cache_key": "sha_P1",
        "raw": "AUC",
        "sequence": "AXC",
        "entries": ["P1"],
    }


def test_rescored_rows_leave_the_misses(tmp_path: Path, monkeypatch) -> None:
    result = tmp_path / "group" / "all" / "seed_1"
    result.mkdir(parents=True)
    train = tmp_path / "train.csv"
    pd.DataFrame([row("T1", "AAA", "1.1.1.1")], columns=COLUMNS).to_csv(train, index=False)
    test = pd.DataFrame(
        [row("P1", "AAA", "1.1.1.1"), row("P3", "DDD", "1.1.1.1")], columns=COLUMNS
    )
    extra = pd.DataFrame(
        [row("P2", "CUC", "2.2.2.2"), row("P4", "EUE", "1.1.1.1")], columns=COLUMNS
    )
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.concat([test, extra]).to_csv(result / "test_eval_manifest.csv", index=False)
    names = ["P1", "P3", "P2", "P4"]
    saved = np.zeros((4, 2))
    saved[0] = [0.9, 0.1]
    np.savez(result / "test_scores.npz", names=names, raw_scores=saved, final_scores=saved)
    (result / "skipped_test.json").write_text(
        json.dumps(
            {
                "no_features_names": ["P3"],
                "non_canonical_names": ["P2", "P4"],
                "out_of_memory_names": [],
            }
        )
    )
    (result / "test_metrics.json").write_text(json.dumps({"rows": 4, "misses": 3, "seed": 1}))
    (result / "test_results_df.csv").write_text("old")
    metadata = {"manifests": {"test": str(tmp_path / "test.csv"), "train": str(train)}}

    def diffuse(*, scores, eval_df, **kwargs):
        assert list(eval_df["cache_key"]) == ["P1", "P2"]
        return scores

    monkeypatch.setattr(rescore, "run_label_diffusion", diffuse)
    new = {"names": np.array(["sha_P2"]), "raw_scores": np.array([[0.2, 0.8]])}
    rescore.rescore_seed(result, metadata, LABELS, extra, new)

    scores = np.load(result / "test_scores_rescored.npz")
    assert scores["predicted"].tolist() == [True, False, True, False]
    assert scores["final_scores"][2].tolist() == [0.2, 0.8]
    assert scores["final_scores"][3].sum() == 0
    metrics = json.loads((result / "test_metrics.json").read_text())
    assert (metrics["rows"], metrics["misses"], metrics["rescored_rows"]) == (4, 2, 1)
    assert metrics["care_task1"]["k=1"]["level_4_accuracy"] == 50.0
    assert json.loads((result / "test_metrics_pre_rescore.json").read_text())["misses"] == 3
    assert (result / "test_results_df_pre_rescore.csv").read_text() == "old"
    care = pd.read_csv(result / "test_results_df.csv")
    assert care.loc[2, "0"] == "2.2.2.2" and care.loc[3, ["0", "1"]].isna().all()
