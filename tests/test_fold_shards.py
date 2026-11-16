import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench import cache_features  # noqa: E402
from emulator_bench.utils import cache_dirs, cache_root, read_json  # noqa: E402

KEYS = [f"{value:08x}" + "0" * 56 for value in range(0, 40, 3)]


def make_items(keys: list[str]) -> list[dict]:
    return [{"cache_key": key, "entries": [key], "sequence": "AAA"} for key in keys]


def make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = dict(
        dataset_root=str(tmp_path),
        split_group=None,
        out_root=str(tmp_path / "out"),
        structure_root=str(tmp_path),
        limit_per_split=None,
        skip_feature_generation=False,
        lookup_workers=1,
        num_gpus=1,
        gpu_memory_gb=None,
        prottrans_model_path="p",
        fold_shards=None,
        num_fold_shards=None,
        fold_from=None,
        esmfold_model="local-folder",
    )
    return argparse.Namespace(**{**values, **overrides})


def test_shards_partition_keys() -> None:
    items = make_items(KEYS)
    parts = [cache_features.select_fold_shards(items, [i], 4) for i in range(4)]
    assert sorted(item["cache_key"] for part in parts for item in part) == sorted(KEYS)
    for index, part in enumerate(parts):
        assert all(int(item["cache_key"][:8], 16) % 4 == index for item in part)
    two = cache_features.select_fold_shards(items, [0, 3], 4)
    assert two == [i for i in items if i in parts[0] + parts[3]]
    assert cache_features.select_fold_shards(items, None, None) == items


def test_sharded_run_folds_selected_shards_and_stops(tmp_path: Path, monkeypatch) -> None:
    items = make_items(KEYS)
    folded = []

    def fake_run_sharded(function, run_items, num_gpus, *args, **kwargs):
        if function is cache_features.fold_with_esmfold:
            folded.extend(item["cache_key"] for item in run_items)
            assert kwargs["model_name"] == "local-folder"
        return []

    def no_dssp(*args, **kwargs):
        raise AssertionError("DSSP must not run")

    monkeypatch.setattr(cache_features, "prepare_dataset", lambda **kw: [{"metadata_path": "x"}])
    monkeypatch.setattr(cache_features, "manifest_items", lambda metadata: items)
    monkeypatch.setattr(cache_features, "load_experimental_chains", lambda path: {})
    monkeypatch.setattr(cache_features, "link_structures", lambda items, *a, **kw: items)
    monkeypatch.setattr(cache_features, "run_sharded", fake_run_sharded)
    monkeypatch.setattr(cache_features, "dssp_with_fallback", no_dssp)
    args = make_args(tmp_path, fold_shards=[1, 2], num_fold_shards=4)
    cache_features.prepare_and_cache(args)
    expected = [key for key in KEYS if cache_features.fold_shard(key, 4) in (1, 2)]
    assert folded == expected
    todo = read_json(cache_root(args.out_root) / "esmfold_todo.json")
    assert [row["cache_key"] for row in todo] == KEYS
    assert todo[0] == {"cache_key": KEYS[0], "sequence": "AAA"}
    assert not (cache_root(args.out_root) / "feature_failures.json").exists()


def test_fold_from_skips_done_and_records_oom(tmp_path: Path, monkeypatch) -> None:
    from emulator_bench.utils import write_json

    args = make_args(tmp_path, fold_shards=[0], num_fold_shards=2)
    dirs = cache_dirs(cache_root(args.out_root))
    todo = [{"cache_key": key, "sequence": "AAA"} for key in KEYS]
    write_json(tmp_path / "todo.json", todo)
    args.fold_from = str(tmp_path / "todo.json")
    shard0 = [key for key in KEYS if cache_features.fold_shard(key, 2) == 0]
    (dirs["structures"] / f"{shard0[0]}.tensor").touch()
    seen = []

    def fake_run_sharded(function, run_items, num_gpus, **kwargs):
        seen.extend(item["cache_key"] for item in run_items)
        assert function is cache_features.fold_with_esmfold
        return [shard0[1]]

    monkeypatch.setattr(cache_features, "run_sharded", fake_run_sharded)
    cache_features.fold_from_list(args)
    assert seen == shard0[1:]
    failures = read_json(cache_root(args.out_root) / "feature_failures.json")
    assert failures == {shard0[1]: {"length": 3, "reason": "esmfold_oom"}}
