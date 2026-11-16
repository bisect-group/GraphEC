from __future__ import annotations

import argparse
import gc
import os
import multiprocessing
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from .dataset_adapter import prepare_dataset
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_STRUCTURE_ROOT,
    baseline_cwd,
    cache_dirs,
    cache_root,
    ensure_dir,
    ensure_graphec_data_links,
    load_feature_failures,
    merge_json,
    run_data_root,
    record_feature_failures,
    import_task_module,
    read_json,
    resolve_path,
    runs_root,
    write_json,
)


# Predicted full-length structures, in priority order: (source, folder, file name).
STRUCTURE_SOURCES = (
    ("esmfold2", "esmfold2_pdb", "ESMFold2-{}.pdb"),
    ("esm3", "esm", "ESM3-open-small-{}.pdb"),
    ("alphafold", "alphafold", "AF-{}-F1-model_v6.pdb"),
)
ESMFOLD_SOURCE = "esmfold_v1_hf"
ESMFOLD_MODEL = "facebook/esmfold_v1"
SLICES = (slice(30, 38), slice(38, 46), slice(46, 54))
SOURCE_FLUSH = 5000
TERMINAL_O = "+terminal_O"
PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GraphEC features for EMULaToR splits")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--split-group", action="append", help="default: every split folder on disk"
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--structure-root", default=str(DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--prottrans-model-path",
        default=os.environ.get("PROTTRANS_MODEL_PATH", PROTT5_MODEL),
    )
    parser.add_argument("--active-site-batch-size", type=int, default=1)
    parser.add_argument(
        "--active-site-procs-per-gpu",
        type=int,
        default=1,
        help="active-site shard processes on each GPU; batch size 1 gives the same outputs",
    )
    parser.add_argument(
        "--lookup-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="CPU processes for the structure lookup",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="shard ESMFold, ProtT5 and active-site work over the first N visible GPUs",
    )
    parser.add_argument(
        "--dssp-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="CPU threads",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=None,
        help="cap on GPU memory in GB for each GPU process (default: no cap); "
        "with --active-site-procs-per-gpu N, a GPU can use N times this value",
    )
    parser.add_argument("--skip-feature-generation", action="store_true")
    parser.add_argument(
        "--fold-shards",
        type=int,
        nargs="+",
        help="fold only these ESMFold shards (of --num-fold-shards), then stop before DSSP",
    )
    parser.add_argument(
        "--num-fold-shards",
        type=int,
        help="split the ESMFold sequences by key into N shards",
    )
    parser.add_argument(
        "--fold-from",
        help="fold the sequences of this esmfold_todo.json without the dataset, then exit",
    )
    parser.add_argument(
        "--esmfold-model",
        default=ESMFOLD_MODEL,
        help="ESMFold model id or local folder",
    )
    args = parser.parse_args()
    if (args.fold_shards is None) != (args.num_fold_shards is None):
        parser.error("--fold-shards and --num-fold-shards go together")
    return args


def fold_shard(cache_key: str, num_shards: int) -> int:
    return int(cache_key[:8], 16) % num_shards


def select_fold_shards(
    items: list[dict], shards: list[int] | None, num_shards: int | None
) -> list[dict]:
    """The items in the listed shards, or all items when there is no sharding."""
    if shards is None:
        return items
    return [item for item in items if fold_shard(item["cache_key"], num_shards) in shards]


def fold_from_list(args: argparse.Namespace) -> None:
    """Fold the items of an `esmfold_todo.json` that have no structure tensor yet, and record
    the keys that ran out of memory."""
    root = cache_root(args.out_root)
    dirs = cache_dirs(root)
    items = select_fold_shards(
        missing_items(read_json(args.fold_from), dirs["structures"], ".tensor"),
        args.fold_shards,
        args.num_fold_shards,
    )
    print(
        f"[emulator_bench] folding {len(items)} sequences from {args.fold_from}",
        flush=True,
    )
    failed = run_sharded(
        fold_with_esmfold,
        items,
        args.num_gpus,
        gpu_memory_gb=args.gpu_memory_gb,
        dirs=dirs,
        model_name=args.esmfold_model,
    )
    lengths = {item["cache_key"]: len(item["sequence"]) for item in items}
    record_feature_failures(
        root, {key: {"length": lengths[key], "reason": "esmfold_oom"} for key in failed}
    )
    print(f"[emulator_bench] folded {len(items) - len(failed)}; out of memory: {len(failed)}")


def manifest_items(metadata: dict) -> list[dict]:
    """One item per unique sequence, keyed by its hash, with every entry id that has it."""
    seen = {}
    for manifest_path in metadata["manifests"].values():
        records = pd.read_csv(manifest_path, dtype=str)
        for key, entry, sequence in zip(
            records["sequence_sha256"], records["Original Entry"], records["Sequence"]
        ):
            item = seen.setdefault(key, {"cache_key": key, "entries": set(), "sequence": sequence})
            item["entries"].add(entry)
    return [{**item, "entries": sorted(item["entries"])} for item in seen.values()]


def missing_items(items: list[dict], directory: Path, suffix: str) -> list[dict]:
    return [item for item in items if not (directory / f"{item['cache_key']}{suffix}").exists()]


def load_experimental_chains(parquet_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Map each entry to its experimental (PDB id, chain) pairs that cover the whole sequence."""
    if not parquet_path.exists():
        return {}
    columns = ["uniprot_id", "pdbs", "pdb_type", "pdb_chain", "pdb_coverage"]
    df = pd.read_parquet(parquet_path, columns=columns)
    df = df[(df["pdb_type"] == "experimental") & (df["pdb_coverage"] >= 1.0 - 1e-9)]
    chains: dict[str, list[tuple[str, str]]] = {}
    for entry, pdbs, chain in zip(df["uniprot_id"], df["pdbs"], df["pdb_chain"]):
        chains.setdefault(entry, []).append((pdbs.partition(":")[0].upper(), chain))
    return {entry: sorted(set(pairs)) for entry, pairs in chains.items()}


def experimental_pdb_lines(cif_path: Path, chain: str) -> tuple[list[str], str]:
    """PDB text and residue sequence of one amino-acid chain, without hydrogens or OXT.

    Residues are renumbered from 1 so that `get_pdb_xyz` sees one residue per number.
    """
    import biotite.structure as struc
    import biotite.structure.info as struc_info
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx
    import numpy as np

    atoms = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True)
    atoms = atoms[
        struc.filter_amino_acids(atoms)
        & (atoms.chain_id == chain)
        & (atoms.element != "H")
        & (atoms.atom_name != "OXT")
    ]
    starts = struc.get_residue_starts(atoms)
    atoms.res_id = np.cumsum(np.isin(np.arange(atoms.array_length()), starts))
    atoms.ins_code[:] = ""
    atoms.hetero[:] = False
    sequence = "".join(struc_info.one_letter_code(name) or "X" for name in atoms.res_name[starts])
    if len(chain) > 1:
        atoms.chain_id[:] = "A"  # PDB format holds 1-character chain IDs; one chain is kept
    # PDB format holds 3-character residue names; longer modified-residue codes become UNK
    atoms.res_name[np.char.str_len(atoms.res_name) > 3] = "UNK"
    handle = pdb.PDBFile()
    handle.set_structure(atoms)
    return handle.lines, sequence


def save_structure(key: str, lines: list[str], dirs: dict[str, Path]) -> None:
    """Write the coordinate tensor GraphEC reads."""
    with baseline_cwd():
        import Features.features as features
    coord = features.get_pdb_xyz(lines + ["TER\n"])
    torch.save(torch.tensor(coord, dtype=torch.float32), dirs["structures"] / f"{key}.tensor")


def record_sources(dirs: dict[str, Path], new: dict[str, str]) -> None:
    """Merge structure sources into `structure_sources.json`. Shards can write at the same time."""
    merge_json(dirs["structures"].parent / "structure_sources.json", new)


def predicted_pdb_lines(path: Path) -> list[str]:
    """Lines of a predicted PDB without OXT, as for experimental chains."""
    return [
        line for line in path.read_text().splitlines(keepends=True) if line[13:16].strip() != "OXT"
    ]


def pdb_residues(lines: list[str]) -> str:
    """One-letter residue sequence of the ATOM lines of a PDB file."""
    from Bio.SeqUtils import seq1

    names = {}
    for line in lines:
        if line.startswith("ATOM"):
            names.setdefault((line[21], line[22:27]), line[17:20])
    return "".join(seq1(name) for name in names.values())


def has_backbone(features, lines: list[str], length: int) -> bool:
    """True when every residue has N, CA, C and O atoms and the residue count is `length`."""
    try:
        return len(features.get_pdb_xyz(lines + ["TER\n"])) == length
    except KeyError:
        return False


def add_terminal_oxygen(lines: list[str]) -> list[str] | None:
    """Lines with the O atom of the last residue added, or None when it has all its backbone atoms.
    The O is placed with C=O 1.231 A, CA-C-O 120.5 degrees and N-CA-C-O 180 degrees."""
    last = [i for i, line in enumerate(lines) if line.startswith("ATOM")]
    if not last:
        return None
    residue = lines[last[-1]][21:27]
    rows = {lines[i][12:16].strip(): i for i in last if lines[i][21:27] == residue}
    if "O" in rows or not {"N", "CA", "C"} <= rows.keys():
        return None
    n, ca, c = (np.array([float(lines[rows[a]][s]) for s in SLICES]) for a in ("N", "CA", "C"))
    bc = (c - ca) / np.linalg.norm(c - ca)
    normal = np.cross(ca - n, bc)
    normal /= np.linalg.norm(normal)
    theta = np.radians(120.5)
    o = c + 1.231 * (-np.cos(theta) * bc - np.sin(theta) * np.cross(normal, bc))
    serial = max(int(line[6:11]) for line in lines if line.startswith("ATOM")) + 1
    base = lines[rows["C"]].rstrip().ljust(78)
    new = (
        f"{base[:6]}{serial:5d}  O  {base[16:30]}{o[0]:8.3f}{o[1]:8.3f}{o[2]:8.3f}{base[54:76]} O\n"
    )
    return [*lines[: rows["C"]], base + "\n", new, *lines[rows["C"] + 1 :]]


def link_chunk(
    items: list[dict],
    dirs: dict[str, Path],
    structure_root: Path,
    experimental: dict[str, list[tuple[str, str]]],
    rejected: dict[str, set[str]] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Write a full-length PDB file and coordinate tensor for each item that has one.

    Sources in priority order: an experimental chain whose residues equal the sequence, then
    predicted structures (`STRUCTURE_SOURCES`) whose residues equal the sequence.
    Entry ids that share the sequence are all tried. The chosen source is recorded per sequence
    as the second return value. `rejected` maps a key to the source families (`experimental`, a
    `STRUCTURE_SOURCES` name) not to use for it. Returns the items that still need ESMFold and the
    chosen sources. Sources are also saved every `SOURCE_FLUSH` items, so an interrupted run keeps
    them. A predicted structure that lacks only the O atom of its last residue gets that atom added.
    It is then a real file, and its source ends with `TERMINAL_O`.
    """
    from biotite.structure import BadStructureError

    with baseline_cwd():
        import Features.features as features
    unresolved, sources = [], {}
    for done, item in enumerate(progress(items, desc="Structures", unit="seq"), 1):
        if done % SOURCE_FLUSH == 0:
            record_sources(dirs, sources)
        key, sequence = item["cache_key"], item["sequence"]
        skip = (rejected or {}).get(key, set())
        pdb_path = dirs["structures"] / f"{key}.pdb"
        found = None
        for entry in item["entries"] if "experimental" not in skip else ():
            for pdb_id, chain in experimental.get(entry, []):
                cif = structure_root / "mmcif" / f"{pdb_id}_updated.cif"
                try:
                    lines, residues = experimental_pdb_lines(cif, chain)
                except (FileNotFoundError, KeyError, BadStructureError):
                    continue
                if residues == sequence and has_backbone(features, lines, len(sequence)):
                    pdb_path.unlink(missing_ok=True)
                    pdb_path.write_text("\n".join(lines) + "\n")
                    found = (f"experimental:{pdb_id}:{chain}", lines)
                    break
            if found:
                break
        for source, folder, pattern in STRUCTURE_SOURCES if not found else ():
            if source in skip:
                continue
            for entry in item["entries"]:
                path = structure_root / folder / pattern.format(entry)
                if not path.exists():
                    continue
                lines = predicted_pdb_lines(path)
                if pdb_residues(lines) != sequence:
                    continue
                if has_backbone(features, lines, len(sequence)):
                    pdb_path.unlink(missing_ok=True)
                    pdb_path.symlink_to(path)
                    found = (source, lines)
                    break
                fixed = add_terminal_oxygen(lines)
                if fixed and has_backbone(features, fixed, len(sequence)):
                    pdb_path.unlink(missing_ok=True)
                    pdb_path.write_text("".join(fixed))
                    found = (source + TERMINAL_O, fixed)
                    break
            if found:
                break
        if found:
            save_structure(key, found[1], dirs)
            sources[key] = found[0]
        else:
            unresolved.append(item)
    return unresolved, sources


def link_structures(
    items: list[dict],
    dirs: dict[str, Path],
    structure_root: Path,
    experimental: dict[str, list[tuple[str, str]]],
    rejected: dict[str, set[str]] | None = None,
    workers: int = 1,
) -> list[dict]:
    """`link_chunk` on consecutive chunks of `items` in `workers` processes, with one write of
    `structure_sources.json` at the end. Returns the items that still need ESMFold, in order."""
    workers = min(workers, len(items))
    if workers <= 1:
        unresolved, sources = link_chunk(items, dirs, structure_root, experimental, rejected)
    else:
        size = -(-len(items) // workers)
        chunks = [items[start : start + size] for start in range(0, len(items), size)]
        with ProcessPoolExecutor(
            len(chunks), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    link_chunk,
                    chunk,
                    dirs,
                    structure_root,
                    {
                        entry: experimental[entry]
                        for item in chunk
                        for entry in item["entries"]
                        if entry in experimental
                    },
                    rejected,
                )
                for chunk in chunks
            ]
            results = [future.result() for future in futures]
        unresolved = [item for part, _ in results for item in part]
        sources = {key: source for _, part in results for key, source in part.items()}
    record_sources(dirs, sources)
    return unresolved


def shard_positions(num_gpus: int, procs_per_gpu: int) -> list[int]:
    """Position in the visible-device list for each shard."""
    return list(range(num_gpus)) * procs_per_gpu


def limit_gpu_memory(gb: float | None) -> None:
    """Cap the caching allocator of the current device at `gb` GB. An allocation over the cap
    frees the cached blocks and retries, then raises an out-of-memory error."""
    if gb is not None:
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 1024**3 / total))


def _shard_worker(
    function,
    position: int | None,
    items: list[dict],
    kwargs: dict,
    gpu_memory_gb: float | None = None,
) -> list[str]:
    # The child inherits CUDA_VISIBLE_DEVICES. The current device selects the GPU, so every
    # "cuda" device and `.cuda()` call in `function` uses it.
    if position is not None:
        torch.cuda.set_device(position)
    limit_gpu_memory(gpu_memory_gb)
    return function(items, **kwargs)


def run_sharded(
    function,
    items: list[dict],
    num_gpus: int,
    procs_per_gpu: int = 1,
    gpu_memory_gb: float | None = None,
    **kwargs,
) -> list[str]:
    """Run `function(items, **kwargs)` on `items` split over the first `num_gpus` visible GPUs,
    with `procs_per_gpu` shards on each. Each shard runs in its own process. Returns the failed
    keys from all shards. `gpu_memory_gb` caps the memory of each process."""
    if num_gpus * procs_per_gpu <= 1 or len(items) <= 1:
        limit_gpu_memory(gpu_memory_gb)
        return function(items, **kwargs)
    positions = shard_positions(num_gpus, procs_per_gpu)
    shards = [items[i :: len(positions)] for i in range(len(positions))]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(positions), mp_context=context) as pool:
        futures = [
            pool.submit(_shard_worker, function, position, shard, kwargs, gpu_memory_gb)
            for position, shard in zip(positions, shards)
        ]
        return [key for future in futures for key in future.result()]


def fold_with_esmfold(
    items: list[dict], dirs: dict[str, Path], model_name: str = ESMFOLD_MODEL
) -> list[str]:
    """Fold sequences with the Hugging Face ESMFold v1. Returns the keys that ran out of memory."""
    if not items:
        return []
    from transformers import EsmForProteinFolding

    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True).eval()
    model.esm = model.esm.half()
    model = model.cuda()
    model.trunk.set_chunk_size(128)
    print(f"[emulator_bench] folding {len(items)} sequences with ESMFold v1", flush=True)
    failed, sources = [], {}
    for item in progress(items, desc="ESMFold", unit="seq"):
        try:
            with torch.no_grad():
                pdb_text = model.infer_pdb(item["sequence"])
        except torch.cuda.OutOfMemoryError:
            failed.append(item["cache_key"])
            torch.cuda.empty_cache()
            continue
        (dirs["structures"] / f"{item['cache_key']}.pdb").write_text(pdb_text)
        save_structure(item["cache_key"], pdb_text.splitlines(keepends=True), dirs)
        sources[item["cache_key"]] = ESMFOLD_SOURCE
    record_sources(dirs, sources)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return failed


def dssp_one(item: dict, mkdssp: Path, dirs: dict[str, Path], features) -> str | None:
    """Write the DSSP tensor of one sequence. Returns its key when mkdssp fails on the structure."""
    pdb = dirs["structures"] / f"{item['cache_key']}.pdb"
    raw = dirs["dssp"] / f"{item['cache_key']}.dssp"
    raw.unlink(missing_ok=True)
    subprocess.run([str(mkdssp), "-i", str(pdb), "-o", str(raw)], check=False)
    if not raw.exists():
        return item["cache_key"]
    dssp_seq, dssp_matrix = features.process_dssp(str(raw))
    if dssp_seq != item["sequence"]:
        dssp_matrix = features.match_dssp(dssp_seq, dssp_matrix, item["sequence"])
    torch.save(
        torch.tensor(dssp_matrix, dtype=torch.float32),
        dirs["dssp"] / f"{item['cache_key']}.tensor",
    )
    raw.unlink()
    return None


def generate_dssp_features(items: list[dict], dirs: dict[str, Path], *, workers: int) -> list[str]:
    """Run mkdssp on CPU threads. Each sequence gives the same tensor as `features.get_dssp`.
    Returns the keys that mkdssp could not process (for example a degenerate structure)."""
    if not items:
        return []
    with baseline_cwd():
        import Features.features as features

        print(
            f"[emulator_bench] generating DSSP features for {len(items)} entries",
            flush=True,
        )
        dssp_dir = ensure_dir(dirs["tmp"] / "dssp-2.0.4")
        shutil.copy2(BASELINE_ROOT / "Features/dssp-2.0.4/mkdssp", dssp_dir / "mkdssp")
        (dssp_dir / "mkdssp").chmod(0o755)
        with ThreadPoolExecutor(workers) as pool:
            results = progress(
                pool.map(
                    lambda item: dssp_one(item, dssp_dir / "mkdssp", dirs, features),
                    items,
                ),
                total=len(items),
                desc="DSSP",
                unit="seq",
            )
            return [key for key in results if key]


def generate_prottrans_features(
    items: list[dict], dirs: dict[str, Path], *, model_path: str
) -> list[str]:
    """Embed one sequence at a time. Returns the keys that ran out of GPU memory."""
    if not items:
        return []
    import re

    from transformers import T5EncoderModel, T5Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).eval().to(device)
    print(
        f"[emulator_bench] generating ProtTrans features for {len(items)} sequences on {device}",
        flush=True,
    )
    failed = []
    for item in progress(items, desc="ProtTrans", unit="seq"):
        spaced = " ".join(re.sub(r"[UZOB]", "X", item["sequence"]))
        tokens = tokenizer([spaced], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(tokens["input_ids"], device=device)
        attention_mask = torch.tensor(tokens["attention_mask"], device=device)
        try:
            with torch.no_grad():
                embedding = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state.cpu()
        except torch.cuda.OutOfMemoryError:
            failed.append(item["cache_key"])
            torch.cuda.empty_cache()
            continue
        seq_len = int(attention_mask.sum().item())
        torch.save(
            embedding[0, : seq_len - 1].clone(),
            dirs["prottrans"] / f"{item['cache_key']}.tensor",
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failed


def predict_or_fail(predict, batch) -> list[str]:
    """Run `predict` on a batch. After a GPU OOM, run each graph alone and return the names of
    graphs that still run out of memory."""
    try:
        predict(batch)
        return []
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    failed = []
    for graph in batch.to_data_list():
        try:
            predict(Batch.from_data_list([graph]))
        except torch.cuda.OutOfMemoryError:
            failed.append(graph.name)
            torch.cuda.empty_cache()
    return failed


def generate_active_site_features(
    items: list[dict],
    dirs: dict[str, Path],
    *,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    """Predict active-site scores. Returns the keys that ran out of GPU memory, even alone.
    The process works from its own `Data` links, removed at the end."""
    missing = missing_items(items, dirs["active_sites"], ".pt")
    if not missing:
        return []
    data_root = run_data_root(dirs["tmp"] / f"data_{os.getpid()}")
    ensure_graphec_data_links(
        {item["cache_key"]: item["cache_key"] for item in missing},
        dirs["structures"].parent,
        data_root,
    )
    try:
        return predict_active_sites(missing, dirs, batch_size, num_workers, data_root)
    finally:
        shutil.rmtree(data_root)


def predict_active_sites(
    missing: list[dict],
    dirs: dict[str, Path],
    batch_size: int,
    num_workers: int,
    data_root: Path,
) -> list[str]:
    with baseline_cwd(data_root):
        model_mod = import_task_module("Active_sites", "model")
        data_mod = __import__("data")
        from torch_geometric.loader import DataLoader
        import torch_geometric

        dataset_dict = {item["cache_key"]: [item["sequence"]] for item in missing}
        dataset = data_mod.ProteinGraphDataset(dataset_dict, range(len(dataset_dict)), args=None)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        for fold in range(5):
            state_dict = torch.load(
                BASELINE_ROOT / "Active_sites/model" / f"fold{fold}.ckpt", device
            )
            model = model_mod.GraphEC_AS(
                1024 + 9 + 184,
                450,
                128,
                4,
                0.2,
                0.1,
                task="ActiveSite",
            ).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        print(
            f"[emulator_bench] generating active-site features for {len(missing)} entries",
            flush=True,
        )

        def predict(batch):
            batch = batch.to(device)
            with torch.no_grad():
                h_V_geo, h_E = data_mod.get_geo_feat(batch.X, batch.edge_index, batch.ptr)
                h_V = torch.cat([batch.node_feat, h_V_geo], dim=-1)
                outputs = [
                    model.forward_geo(h_V, h_E, batch.edge_index, batch.seq, batch.batch).sigmoid()
                    for model in models
                ]
                outputs = torch.stack(outputs, 0).mean(0)
            for name, tensor in zip(
                batch.name, torch_geometric.utils.unbatch(outputs, batch.batch)
            ):
                torch.save([tensor.detach().cpu()], dirs["active_sites"] / f"{name}.pt")

        failed = []
        for batch in progress(dataloader, desc="GraphEC-AS", unit="batch"):
            failed += predict_or_fail(predict, batch)
        return failed


def dssp_with_fallback(
    items: list[dict],
    dirs: dict[str, Path],
    args,
    structure_root: Path,
    experimental: dict,
) -> tuple[list[str], list[str]]:
    """Run DSSP. A structure that mkdssp rejects is replaced by the next source (experimental,
    predicted, ESMFold v1) and DSSP runs again. Returns the keys with no usable source and the
    keys whose ESMFold run was out of memory."""
    rejected: dict[str, set[str]] = {}
    no_source, esmfold_oom = [], []
    pending = items
    while pending:
        failed = generate_dssp_features(pending, dirs, workers=args.dssp_workers)
        if not failed:
            break
        sources = read_json(dirs["structures"].parent / "structure_sources.json")
        retry = [item for item in pending if item["cache_key"] in set(failed)]
        for key in failed:
            family = sources[key].split(":")[0].removesuffix(TERMINAL_O)
            rejected.setdefault(key, set()).add(family)
            for suffix in (".pdb", ".tensor"):
                (dirs["structures"] / f"{key}{suffix}").unlink(missing_ok=True)
        unresolved = link_structures(
            retry, dirs, structure_root, experimental, rejected, args.lookup_workers
        )
        fold = [i for i in unresolved if ESMFOLD_SOURCE not in rejected[i["cache_key"]]]
        no_source += [i["cache_key"] for i in unresolved if i not in fold]
        esmfold_oom += run_sharded(
            fold_with_esmfold,
            fold,
            args.num_gpus,
            gpu_memory_gb=args.gpu_memory_gb,
            dirs=dirs,
        )
        dropped = set(no_source) | set(esmfold_oom)
        pending = [item for item in retry if item["cache_key"] not in dropped]
    return no_source, esmfold_oom


def prepare_and_cache(args: argparse.Namespace) -> list[dict]:
    metadata_list = prepare_dataset(
        dataset_root=args.dataset_root,
        split_groups=args.split_group,
        runs_root=runs_root(args.out_root),
        limit_per_split=args.limit_per_split,
    )
    dirs = cache_dirs(cache_root(args.out_root))
    all_items = {}
    for metadata in metadata_list:
        for item in manifest_items(metadata):
            known = all_items.setdefault(item["cache_key"], item)
            known["entries"] = sorted(set(known["entries"]) | set(item["entries"]))
    items = list(all_items.values())
    print(f"[emulator_bench] unique sequences: {len(items)}", flush=True)

    if not args.skip_feature_generation:
        failures = load_feature_failures(cache_root(args.out_root))
        items = [item for item in items if item["cache_key"] not in failures]
        structure_root = Path(args.structure_root)
        experimental = load_experimental_chains(
            resolve_path(args.dataset_root) / "enzyme_classification_dataset_3d.parquet"
        )
        lengths = {item["cache_key"]: len(item["sequence"]) for item in items}
        failed_keys: dict[str, str] = {}

        def record(keys: list[str], reason: str) -> None:
            nonlocal items
            keys = [key for key in keys if key not in failed_keys]
            if keys:
                record_feature_failures(
                    cache_root(args.out_root),
                    {key: {"length": lengths[key], "reason": reason} for key in keys},
                )
                failed_keys.update(dict.fromkeys(keys, reason))
                items = [item for item in items if item["cache_key"] not in failed_keys]

        # ProtT5 needs only the sequences, so it runs on the GPUs while the CPUs look up structures.
        with ThreadPoolExecutor(1) as pool:
            prottrans = pool.submit(
                run_sharded,
                generate_prottrans_features,
                missing_items(items, dirs["prottrans"], ".tensor"),
                args.num_gpus,
                gpu_memory_gb=args.gpu_memory_gb,
                dirs=dirs,
                model_path=args.prottrans_model_path,
            )
            unresolved = link_structures(
                missing_items(items, dirs["structures"], ".tensor"),
                dirs,
                structure_root,
                experimental,
                workers=args.lookup_workers,
            )
            prottrans_failed = prottrans.result()
        write_json(
            cache_root(args.out_root) / "esmfold_todo.json",
            [{"cache_key": item["cache_key"], "sequence": item["sequence"]} for item in unresolved],
        )
        record(
            run_sharded(
                fold_with_esmfold,
                select_fold_shards(unresolved, args.fold_shards, args.num_fold_shards),
                args.num_gpus,
                gpu_memory_gb=args.gpu_memory_gb,
                dirs=dirs,
                model_name=args.esmfold_model,
            ),
            "esmfold_oom",
        )
        record(prottrans_failed, "prott5_oom")
        if args.fold_shards is not None:
            print(
                f"[emulator_bench] fold shards {args.fold_shards} of {args.num_fold_shards} "
                "done; rerun without --fold-shards to continue",
                flush=True,
            )
            return metadata_list
        no_source, esmfold_oom = dssp_with_fallback(
            missing_items(items, dirs["dssp"], ".tensor"),
            dirs,
            args,
            structure_root,
            experimental,
        )
        record(no_source, "dssp_failed")
        record(esmfold_oom, "esmfold_oom")
        record(
            run_sharded(
                generate_active_site_features,
                missing_items(items, dirs["active_sites"], ".pt"),
                args.num_gpus,
                args.active_site_procs_per_gpu,
                gpu_memory_gb=args.gpu_memory_gb,
                dirs=dirs,
                batch_size=args.active_site_batch_size,
                num_workers=args.num_workers,
            ),
            "active_site_oom",
        )
        print(f"[emulator_bench] feature failures: {len(failed_keys)}", flush=True)
    else:
        print("[emulator_bench] skipping feature generation by request", flush=True)

    for metadata in metadata_list:
        metadata["cache_root"] = str(cache_root(args.out_root))
        metadata["feature_cache"] = {key: str(path) for key, path in dirs.items()}
        write_json(metadata["metadata_path"], metadata)
    return metadata_list


def main() -> None:
    args = parse_args()
    if args.fold_from:
        fold_from_list(args)
    else:
        prepare_and_cache(args)


if __name__ == "__main__":
    main()
