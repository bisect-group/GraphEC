import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.dataset_adapter import discover_split_groups, select_split_groups  # noqa: E402


def make_split(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (folder / f"{name}.parquet").touch()


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    for name in (
        "random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.39",
        "enzyme_structure_splits/threshold_0.85",
        "enzyme_structure_splits/backup_2026/threshold_0.7",
        "enzyme_structure_splits/backup_2026",
        "backup_other",
        "unrelated_splits",
    ):
        make_split(tmp_path / name)
    return tmp_path


def test_discovery_skips_backup_and_unknown_folders(dataset_root: Path) -> None:
    names = [group.name for group in discover_split_groups(dataset_root)]
    assert names == [
        "random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.39",
        "enzyme_structure_splits/threshold_0.85",
    ]


def test_select_unknown_group_fails(dataset_root: Path) -> None:
    with pytest.raises(ValueError):
        select_split_groups(dataset_root, ["backup_other"])


def test_empty_root_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_split_groups(tmp_path)
