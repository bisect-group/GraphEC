from __future__ import annotations

import fcntl
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    BASELINE_ROOT.parents[1] / "data/processed/datasets/enzyme_classification_dataset"
)
DEFAULT_OUT_ROOT = BASELINE_ROOT.parents[1] / "outputs/baselines/GraphEC"
TASK = "enzyme_classification"
DEFAULT_RUNS_ROOT = DEFAULT_OUT_ROOT / TASK
DEFAULT_CACHE_ROOT = DEFAULT_OUT_ROOT / "embeddings"
DEFAULT_STRUCTURE_ROOT = BASELINE_ROOT.parents[1] / "data/intermediate"
DEFAULT_SEEDS = (666, 777, 888)
SPLIT_NAMES = ("train", "val", "test")


def runs_root(out_root: str | Path) -> Path:
    return Path(out_root) / TASK


def cache_root(out_root: str | Path) -> Path:
    return Path(out_root) / "embeddings"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, *, base: str | Path = BASELINE_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def read_json(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: str | Path, data: object) -> Path:
    """Write through a temporary file and an atomic rename, so readers never see half a file."""
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def merge_json(path: str | Path, new: dict) -> None:
    """Merge `new` into the JSON object at `path` under a file lock, so parallel jobs add up."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.with_name(f"{path.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(path, {**(read_json(path) if path.exists() else {}), **new})


def split_group_slug(split_group: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", split_group.strip("/")).replace("/", "_")


def safe_id(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text[:180] or "missing_id"


def split_run_root(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """`<group>/<threshold_*>` for threshold folders, `<group>/all` for flat split folders."""
    group = split_group.strip("/")
    return Path(runs_root) / (group if "/" in group else f"{group}/all")


def metadata_path_for_split(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return split_run_root(split_group, runs_root) / "metadata.json"


def load_run_metadata(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> dict:
    path = metadata_path_for_split(split_group, runs_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return read_json(path)


def seed_run_root_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return split_run_root(split_group, runs_root) / f"seed_{seed}"


def seed_train_metadata_path_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root) / "train.json"


def seed_results_root_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root)


def conda_python(env_name: str) -> list[str]:
    if env_name in {"", "current", "none"}:
        return [sys.executable]
    executable = Path(sys.executable).resolve()
    env_root = executable.parent.parent
    if env_root.name == env_name and env_root.parent.name == "envs":
        return [str(executable)]
    return ["conda", "run", "-n", env_name, "python"]


def shell_join(parts: Sequence[str | Path]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in parts)


def find_spooler(explicit: str | None = None) -> str:
    candidates = [explicit] if explicit else ["ts", "tsp"]
    for candidate in candidates:
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found:
            return found
        candidate_path = resolve_path(candidate)
        if candidate_path.exists() and os.access(candidate_path, os.X_OK):
            return str(candidate_path)
    raise FileNotFoundError("Could not find task-spooler. Install it or pass --spooler-bin.")


def submit_ts_job(
    command: str,
    *,
    label: str,
    log_name: str,
    depends_on: Iterable[str] | None = None,
    gpus: int | None = None,
    spooler_bin: str | None = None,
) -> str:
    spooler = find_spooler(spooler_bin)
    args = [spooler, "-L", label, "-O", log_name]
    if gpus is not None and gpus > 0:
        args.extend(["-G", str(gpus)])
    depends = [str(job_id) for job_id in (depends_on or []) if str(job_id)]
    if depends:
        args.extend(["-W", ",".join(depends)])
    args.extend(["bash", "-lc", command])
    output = subprocess.check_output(args, text=True).strip()
    job_id = output.splitlines()[-1].strip()
    print(f"[emulator_bench] queued {label}: job {job_id}", flush=True)
    return job_id


def wait_for_ts_jobs(
    job_ids: Sequence[str],
    *,
    spooler_bin: str | None = None,
    poll_seconds: float = 10.0,
) -> None:
    try:
        from src.utils.rich_progress import progress
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from src.utils.rich_progress import progress

    spooler = find_spooler(spooler_bin)
    remaining = {str(job_id) for job_id in job_ids}
    with progress(total=len(remaining), desc="ts jobs", unit="job") as progress:
        while remaining:
            finished = []
            for job_id in sorted(remaining):
                status = subprocess.check_output([spooler, "-s", job_id], text=True).strip()
                lowered = status.lower()
                if "finished" in lowered:
                    finished.append(job_id)
                elif "failed" in lowered or "error" in lowered:
                    raise RuntimeError(f"task-spooler job {job_id} failed: {status}")
            for job_id in finished:
                remaining.remove(job_id)
                progress.update(1)
            if remaining:
                time.sleep(poll_seconds)


@contextmanager
def baseline_cwd(data_root: str | Path | None = None):
    """Work from `data_root` (see `run_data_root`) or the baseline root, with the baseline root on
    `sys.path` so its packages import from either."""
    old_cwd = Path.cwd()
    added = str(BASELINE_ROOT) not in sys.path
    if added:
        sys.path.insert(0, str(BASELINE_ROOT))
    os.chdir(data_root or BASELINE_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if added and str(BASELINE_ROOT) in sys.path:
            sys.path.remove(str(BASELINE_ROOT))


def import_task_module(task_dir: str, module_name: str):
    """Import a GraphEC task module that uses local absolute imports like `from data import *`."""
    task_path = BASELINE_ROOT / task_dir
    if not task_path.exists():
        raise FileNotFoundError(task_path)
    for stale_name in ("data", "model", "utils", "label_diffusion"):
        sys.modules.pop(stale_name, None)
    sys.path.insert(0, str(task_path))
    try:
        __import__(module_name)
        return sys.modules[module_name]
    finally:
        try:
            sys.path.remove(str(task_path))
        except ValueError:
            pass


def link_or_copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    ensure_dir(target.parent)
    if target.is_symlink() and target.resolve() != source.resolve():
        target.unlink()
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(source, target)


def run_data_root(base: str | Path) -> Path:
    """Working directory of one run. The original dataset code reads `./Data/...`, so each run
    gets its own `Data` folder under `base` and never shares links with another run."""
    base = ensure_dir(base)
    for name in ("Structures", "ProtTrans", "DSSP"):
        ensure_dir(base / "Data" / name)
    for name in ("Max_ProtTrans.npy", "Min_ProtTrans.npy"):
        link_or_copy(BASELINE_ROOT / "Data/ProtTrans" / name, base / "Data/ProtTrans" / name)
    return base


FEATURE_CACHE_VERSION = 1


def cache_dirs(cache_root: str | Path) -> dict[str, Path]:
    """Feature folders. The version is in the name, so a bump forces a rebuild."""
    root = Path(cache_root)
    v = FEATURE_CACHE_VERSION
    return {
        "structures": ensure_dir(root / f"structures_v{v}"),
        "prottrans": ensure_dir(root / f"prottrans_v{v}"),
        "dssp": ensure_dir(root / f"dssp_v{v}"),
        "active_sites": ensure_dir(root / f"active_sites_v{v}"),
        "fastas": ensure_dir(root / "fastas"),
        "tmp": ensure_dir(root / "tmp"),
    }


def ensure_graphec_data_links(
    feature_keys: dict[str, str], cache_root: str | Path, data_root: str | Path
) -> None:
    """Link cached features under the sample names the original data loader reads, inside the
    `Data` folder of `data_root` (see `run_data_root`).

    `feature_keys` maps a sample name to the sequence hash that names its cache files.
    """
    dirs = cache_dirs(cache_root)
    data = run_data_root(data_root) / "Data"
    for kind, folder in (
        ("structures", "Structures"),
        ("prottrans", "ProtTrans"),
        ("dssp", "DSSP"),
    ):
        source_dir = dirs[kind].resolve()
        target_dir = ensure_dir(data / folder)
        available = set(os.listdir(source_dir))
        for name, key in feature_keys.items():
            if f"{key}.tensor" not in available:
                continue
            source, target = f"{source_dir}/{key}.tensor", f"{target_dir}/{name}.tensor"
            try:
                os.symlink(source, target)
            except FileExistsError:
                link_or_copy(Path(source), Path(target))
            except OSError:
                shutil.copy2(source, target)


def feature_failures_path(cache_root: str | Path) -> Path:
    return Path(cache_root) / "feature_failures.json"


def load_feature_failures(cache_root: str | Path) -> dict[str, dict]:
    """Sequences that could not be featurized, as {sequence hash: {"length", "reason"}}."""
    path = feature_failures_path(cache_root)
    return read_json(path) if path.exists() else {}


def record_feature_failures(cache_root: str | Path, failures: dict[str, dict]) -> None:
    if failures:
        merge_json(feature_failures_path(cache_root), failures)


def feature_keys_from_manifest(path: str | Path) -> dict[str, str]:
    import pandas as pd

    manifest = pd.read_csv(path, dtype=str, usecols=["cache_key", "sequence_sha256"])
    return dict(zip(manifest["cache_key"], manifest["sequence_sha256"]))


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def raise_open_file_limit() -> None:
    """Raise the soft open-file limit to the hard limit; data loader workers pass tensors through
    file descriptors."""
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def set_tf32(enabled: bool) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
