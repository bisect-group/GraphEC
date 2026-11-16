from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch

try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from .dataset_adapter import CANONICAL_AA, full_ec_labels, normalize_sequence, sequence_hash
from .results import (
    compact_evaluation_summary,
    compute_care_metrics,
    compute_graphec_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    multilabel_targets,
    write_ranked_outputs,
)
from .train import (
    DEFAULT_MAX_RESIDUES,
    MicroCollate,
    build_model,
    forward_or_split,
    load_active_sites,
    make_loader,
)
from .utils import (
    BASELINE_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    baseline_cwd,
    cache_root,
    ensure_dir,
    ensure_graphec_data_links,
    run_data_root,
    safe_id,
    feature_keys_from_manifest,
    import_task_module,
    load_feature_failures,
    load_run_metadata,
    raise_open_file_limit,
    read_json,
    runs_root,
    seed_results_root_for_split,
    seed_train_metadata_path_for_split,
    set_tf32,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GraphEC on an EMULaToR split")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rank-top-k", type=int, default=50)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    parser.add_argument(
        "--no-tf32", dest="tf32", action="store_false", help="fp32 matmuls without TF32"
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=DEFAULT_MAX_RESIDUES,
        help="residues per forward pass; a longer sequence runs alone",
    )
    diffusion = parser.add_mutually_exclusive_group()
    diffusion.add_argument("--label-diffusion", dest="label_diffusion", action="store_true")
    diffusion.add_argument("--no-label-diffusion", dest="label_diffusion", action="store_false")
    parser.set_defaults(label_diffusion=True)
    parser.add_argument("--diffusion-lambda", type=float, default=0.1)
    parser.add_argument("--identity-cutoff", type=float, default=0.1)
    return parser.parse_args()


def load_manifest_dataset(
    path: str | Path,
    label_to_idx: dict[str, int],
    no_features: set[str] = frozenset(),
    non_canonical: set[str] = frozenset(),
) -> tuple[dict, list[str], np.ndarray, list[str], list[str]]:
    """Samples to predict, all row names in order, targets for all rows, the names without
    features and the names with non-canonical residues (both stay in the rows and count as
    misses)."""
    manifest = pd.read_csv(path)
    dataset = {}
    names = []
    targets = []
    missing = []
    skipped = []
    for _, row in manifest.iterrows():
        key = str(row["cache_key"])
        labels = [label for label in str(row["EC number"]).split(";") if label in label_to_idx]
        if not labels:
            continue
        if key in non_canonical:
            skipped.append(key)
        elif row["sequence_sha256"] in no_features:
            missing.append(key)
        else:
            dataset[key] = [str(row["Sequence"])]
        names.append(key)
        targets.append(multilabel_targets(labels, label_to_idx))
    if not names:
        raise ValueError(f"No usable rows in {path}")
    return dataset, names, np.stack(targets, axis=0), missing, skipped


def non_canonical_rows(
    metadata: dict, eval_split: str, label_to_idx: dict[str, int]
) -> pd.DataFrame:
    """Rows of the source split file that the adapter dropped only for non-canonical residues,
    with the adapter's full-EC labels, one row per entry and sequence, and cache keys that stay
    unique next to the manifest rows."""
    path = Path(metadata["dataset_root"]) / metadata["split_group"] / f"{eval_split}.parquet"
    raw = pd.read_parquet(path, columns=["uniprot_id", "sequence", "ec_number"])
    df = raw.head(metadata["limit_per_split"]).dropna()
    df = df.assign(
        **{
            "Original Entry": df["uniprot_id"].astype(str),
            "Sequence": df["sequence"].map(normalize_sequence),
            "labels": df["ec_number"].map(full_ec_labels),
        }
    )
    df = df[df["Sequence"].map(lambda sequence: not set(sequence) <= CANONICAL_AA)]
    rows = (
        df.explode("labels")
        .dropna(subset=["labels"])
        .groupby(["Original Entry", "Sequence"], as_index=False)["labels"]
        .agg(lambda labels: ";".join(sorted(set(labels))))
        .rename(columns={"labels": "EC number"})
    )
    rows = rows[rows["EC number"].map(lambda v: any(x in label_to_idx for x in v.split(";")))]
    rows = rows.reset_index(drop=True)
    sequences: dict[str, set[str]] = {}
    for manifest in metadata["manifests"].values():
        known = pd.read_csv(manifest, usecols=["Original Entry", "Sequence"]).drop_duplicates()
        for entry, sequence in zip(known["Original Entry"].astype(str), known["Sequence"]):
            sequences.setdefault(entry, set()).add(sequence)
    for entry, sequence in zip(rows["Original Entry"], rows["Sequence"]):
        sequences.setdefault(entry, set()).add(sequence)
    rows["sequence_sha256"] = rows["Sequence"].map(sequence_hash)
    rows["Entry"] = [
        safe_id(entry) if len(sequences[entry]) == 1 else f"{safe_id(entry)}__{digest[:12]}"
        for entry, digest in zip(rows["Original Entry"], rows["sequence_sha256"])
    ]
    rows["cache_key"] = rows["Entry"]
    return rows


def load_model(
    model_mod, train_metadata: dict, num_labels: int, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(train_metadata["checkpoint"], map_location=device)
    model = build_model(model_mod, num_labels, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def predict_scores(
    *,
    dataset: dict,
    active_sites: dict,
    data_mod,
    model: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    amp: bool = False,
    max_residues: int = DEFAULT_MAX_RESIDUES,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Sigmoid of the logits. Sequences that run out of GPU memory even alone are returned as the
    third value and get no score."""
    device = next(model.parameters()).device
    dataloader = make_loader(
        data_mod.ProteinGraphDataset(
            dataset, range(len(dataset)), args=None, active_sites=active_sites
        ),
        batch_size,
        False,
        num_workers,
        MicroCollate(max_residues),
        device,
    )
    names, scores, out_of_memory = [], [], []
    for micros in progress(dataloader, desc="GraphEC evaluate", unit="batch"):
        for micro in micros:
            for micro_names, logits in forward_or_split(model, micro.graphs, device, amp):
                if logits is None:
                    out_of_memory.extend(micro_names)
                else:
                    names.extend(micro_names)
                    scores.append(logits.sigmoid().numpy())
    return names, np.concatenate(scores, axis=0) if scores else np.zeros((0, 0)), out_of_memory


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def jaccard(W):
    co = W.dot(W.T)
    clen = W.sum(axis=1)
    nonzero_mask = co.astype("bool")
    denominator = nonzero_mask.multiply(clen) + nonzero_mask.multiply(clen.T) - co + nonzero_mask
    return sparse_divide_nonzero(co, denominator)


def compute_laplacian(W):
    jw = jaccard(W).multiply(W)
    degree = 1.0 / jw.sum(axis=1)
    w_s2f = 0.5 * (jw.multiply(degree) + jw.multiply(degree.T))
    degree = w_s2f.sum(axis=1)
    d_s2f = ssp.spdiags(degree.T, 0, W.shape[0], W.shape[0])
    return d_s2f - w_s2f


def read_fasta_ids(fasta: Path) -> list[str]:
    ids = []
    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip())
    return ids


DIAMOND_FLAGS = ["--very-sensitive", "--quiet", "-p", "8"]


@functools.cache
def diamond_version(diamond: Path) -> str:
    return subprocess.run(
        [str(diamond), "version"], check=True, capture_output=True, text=True
    ).stdout.strip()


def diamond_search(
    diamond: Path,
    query: Path,
    subject: Path,
    out_tsv: Path,
    cache_dir: Path,
    work_dir: Path,
) -> None:
    """Write the DIAMOND hits of `query` against `subject` to `out_tsv`. The hits are cached in
    `cache_dir` by the fasta contents, DIAMOND version and flags, so runs that share the inputs
    search once."""
    digest = hashlib.sha256()
    for part in (diamond_version(diamond), *DIAMOND_FLAGS):
        digest.update(part.encode() + b"\0")
    for fasta in (subject, query):
        digest.update(fasta.read_bytes() + b"\0")
    ensure_dir(cache_dir)
    cached = cache_dir / f"{digest.hexdigest()}.tsv"
    with cached.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cached.exists():
            db_prefix = work_dir / f"{subject.stem}_db"
            subprocess.run(
                [
                    str(diamond),
                    "makedb",
                    "--in",
                    str(subject),
                    "-d",
                    str(db_prefix),
                    "--quiet",
                ],
                check=True,
            )
            partial = cached.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run(
                [
                    str(diamond),
                    "blastp",
                    "-d",
                    str(db_prefix) + ".dmnd",
                    "-q",
                    str(query),
                    "-o",
                    str(partial),
                    *DIAMOND_FLAGS,
                ],
                check=True,
            )
            partial.replace(cached)
    shutil.copyfile(cached, out_tsv)


def homology_matrix(
    fasta: Path, diamond: Path, work_dir: Path, cache_dir: Path, cutoff: float = 0.1
):
    ensure_dir(work_dir)
    ids = read_fasta_ids(fasta)
    id_to_idx = dict(zip(ids, range(len(ids))))
    out_tsv = work_dir / "homology.tsv"
    diamond_search(diamond, fasta, fasta, out_tsv, cache_dir, work_dir)
    homology = {}
    with out_tsv.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            query_idx = id_to_idx[fields[0]]
            subject_idx = id_to_idx[fields[1]]
            identity = float(fields[2]) / 100.0
            homology.setdefault(query_idx, {})[subject_idx] = max(
                homology.get(query_idx, {}).get(subject_idx, 0.0), identity
            )
            homology.setdefault(subject_idx, {})[query_idx] = max(
                homology.get(subject_idx, {}).get(query_idx, 0.0), identity
            )
    rows, cols, data = [], [], []
    for row, values in homology.items():
        for col, identity in values.items():
            value = 1.0 if row == col else identity
            if value >= cutoff:
                rows.append(row)
                cols.append(col)
                data.append(value)
    return ssp.csr_matrix((np.array(data), (rows, cols)), shape=(len(ids), len(ids)))


def block_cg(A, B: np.ndarray, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Solve A X = B for symmetric positive definite sparse A, one CG run per column of B.
    Raises if a column ends with a relative residual above `tol`."""
    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rs = (R * R).sum(0)
    limit = (tol**2) * rs
    for _ in range(max_iter):
        active = rs > limit
        if not active.any():
            break
        AP = A @ P
        denom = (P * AP).sum(0)
        alpha = np.where(active, rs / np.where(denom > 0, denom, 1.0), 0.0)
        X += alpha * P
        R -= alpha * AP
        rs_new = (R * R).sum(0)
        beta = np.where(active, rs_new / np.where(rs > 0, rs, 1.0), 0.0)
        P = R + beta * P
        rs = rs_new
    residual = np.linalg.norm(B - A @ X, axis=0)
    scale = np.linalg.norm(B, axis=0)
    if (residual > tol * np.where(scale > 0, scale, 1.0)).any():
        raise RuntimeError(
            f"label diffusion CG did not reach relative residual {tol} in {max_iter} iterations"
        )
    return X


_DIFFUSION_SYSTEM: tuple | None = None


def solve_block(start: int) -> np.ndarray:
    A, Y, block = _DIFFUSION_SYSTEM
    return block_cg(A, Y[:, start : start + block].toarray())


def diffuse(
    laplacian, lamda: float, initial_pred, skip: int, block: int = 256, workers: int = 8
) -> np.ndarray:
    """Rows `skip:` of (I + lamda L)^-1 @ initial_pred. Blocks of label columns are independent;
    forked workers solve them."""
    global _DIFFUSION_SYSTEM
    A = (ssp.identity(laplacian.shape[0]) + laplacian.multiply(lamda)).tocsr()
    Y = initial_pred.tocsc()
    _DIFFUSION_SYSTEM = (A, Y, block)
    starts = range(0, Y.shape[1], block)
    with ProcessPoolExecutor(
        min(workers, len(starts)), mp_context=multiprocessing.get_context("fork")
    ) as pool:
        return np.concatenate([x[skip:] for x in pool.map(solve_block, starts)], axis=1)


def run_label_diffusion(
    *,
    scores: np.ndarray,
    train_manifest: str | Path,
    eval_df: pd.DataFrame,
    labels: list[str],
    work_dir: Path,
    cache_dir: Path,
    lamda: float,
    identity_cutoff: float,
) -> np.ndarray:
    diamond = Path(
        os.environ.get("DIAMOND_BIN", BASELINE_ROOT.parents[1] / "external/diamond-2.0.14/diamond")
    )
    if not diamond.is_file():
        raise FileNotFoundError(
            f"Missing DIAMOND executable: {diamond}. Run: bash scripts/install_external_tools.sh"
        )
    ensure_dir(work_dir)
    train_df = pd.read_csv(train_manifest)
    label_to_idx = {label: index for index, label in enumerate(labels)}
    train_fasta = work_dir / "train.fasta"
    eval_fasta = work_dir / "eval.fasta"
    train_seed_fasta = work_dir / "train_seed.fasta"
    combined_fasta = work_dir / "train_seed_and_eval.fasta"
    blast_tsv = work_dir / "eval_vs_train.tsv"
    with train_fasta.open("w") as handle:
        for _, row in train_df.loc[:, ["Entry", "Sequence"]].drop_duplicates().iterrows():
            handle.write(f">{row['Entry']}\n{row['Sequence']}\n")
    with eval_fasta.open("w") as handle:
        for _, row in eval_df.loc[:, ["Entry", "Sequence"]].drop_duplicates().iterrows():
            handle.write(f">{row['Entry']}\n{row['Sequence']}\n")
    diamond_search(diamond, eval_fasta, train_fasta, blast_tsv, cache_dir, work_dir)
    train_seed_ids = set()
    with blast_tsv.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) >= 3 and float(fields[2]) / 100.0 > identity_cutoff:
                train_seed_ids.add(fields[1])
    if not train_seed_ids:
        print("[emulator_bench] label diffusion skipped: no homologous train seeds", flush=True)
        return scores
    seed_df = train_df[train_df["Entry"].isin(train_seed_ids)].drop_duplicates("Entry")
    with train_seed_fasta.open("w") as handle:
        for row in seed_df.itertuples(index=False):
            handle.write(f">{row.Entry}\n{row.Sequence}\n")
    combined_fasta.write_text(train_seed_fasta.read_text() + eval_fasta.read_text())
    W = homology_matrix(
        combined_fasta, diamond, work_dir / "homology", cache_dir, cutoff=identity_cutoff
    )
    row_indices, col_indices = [], []
    for row_index, (_, row) in enumerate(seed_df.iterrows()):
        for label in str(row["EC number"]).split(";"):
            if label in label_to_idx:
                row_indices.append(row_index)
                col_indices.append(label_to_idx[label])
    train_seed_label = ssp.csr_matrix(
        ([1] * len(row_indices), (row_indices, col_indices)),
        shape=(len(seed_df), len(labels)),
    )
    initial_pred = ssp.vstack([train_seed_label, ssp.csr_matrix(scores)])
    laplacian = compute_laplacian(W)
    return diffuse(laplacian, lamda, initial_pred, len(seed_df))


def evaluate_split(
    args,
    metadata,
    train_metadata,
    eval_split: str,
    labels: list[str],
    label_to_idx: dict[str, int],
):
    cache = cache_root(args.out_root)
    manifest = metadata["manifests"][eval_split]
    extra = non_canonical_rows(metadata, eval_split, label_to_idx)
    result_root = ensure_dir(
        seed_results_root_for_split(args.split_group, args.seed, runs_root(args.out_root))
    )
    eval_manifest = result_root / f"{eval_split}_eval_manifest.csv"
    pd.concat([pd.read_csv(manifest), extra], ignore_index=True).to_csv(eval_manifest, index=False)
    dataset, ordered_names, y_true, no_features, non_canonical = load_manifest_dataset(
        eval_manifest, label_to_idx, set(load_feature_failures(cache)), set(extra["cache_key"])
    )
    feature_keys = feature_keys_from_manifest(manifest)
    feature_keys = {name: feature_keys[name] for name in dataset}
    data_root = run_data_root(result_root / f"data_{eval_split}")
    ensure_graphec_data_links(feature_keys, cache, data_root)
    active_sites = load_active_sites(feature_keys, cache) if dataset else {}
    pred_names, pred_scores, out_of_memory = [], np.zeros((0, len(labels))), []
    if dataset:
        with baseline_cwd(data_root):
            model_mod = import_task_module("EC_number", "model")
            data_mod = sys.modules["data"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(model_mod, train_metadata, len(labels), device)
            pred_names, pred_scores, out_of_memory = predict_scores(
                dataset=dataset,
                active_sites=active_sites,
                data_mod=data_mod,
                model=model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp=args.amp,
                max_residues=args.max_residues,
            )
    pred_scores = pred_scores.reshape(len(pred_names), len(labels))
    position = {name: index for index, name in enumerate(ordered_names)}
    predicted = np.zeros(len(ordered_names), dtype=bool)
    predicted[[position[name] for name in pred_names]] = True
    raw_scores = np.zeros((len(ordered_names), len(labels)))
    raw_scores[[position[name] for name in pred_names]] = pred_scores
    final_scores = raw_scores
    if args.label_diffusion and predicted.any():
        eval_df = pd.read_csv(manifest)
        final_scores = np.zeros_like(raw_scores)
        final_scores[predicted] = run_label_diffusion(
            scores=raw_scores[predicted],
            train_manifest=metadata["manifests"]["train"],
            eval_df=eval_df[eval_df["cache_key"].astype(str).isin(set(pred_names))],
            labels=labels,
            work_dir=result_root / f"{eval_split}_label_diffusion",
            cache_dir=result_root.parent / "diamond_cache" / eval_split,
            lamda=args.diffusion_lambda,
            identity_cutoff=args.identity_cutoff,
        )
    misses = {
        "rows": len(ordered_names),
        "misses": int((~predicted).sum()),
        "no_features": len(no_features),
        "non_canonical": len(non_canonical),
        "out_of_memory": len(out_of_memory),
        "no_features_names": no_features,
        "non_canonical_names": non_canonical,
        "out_of_memory_names": out_of_memory,
    }
    write_json(result_root / f"skipped_{eval_split}.json", misses)
    print(
        f"[emulator_bench] {eval_split} misses: {misses['no_features']} without features, "
        f"{misses['non_canonical']} non-canonical, "
        f"{misses['out_of_memory']} out of memory, of {misses['rows']} rows",
        flush=True,
    )
    np.savez_compressed(
        result_root / f"{eval_split}_scores.npz",
        names=np.array(ordered_names),
        raw_scores=raw_scores,
        final_scores=final_scores,
        labels=np.array(labels),
    )
    care_csv = result_root / f"{eval_split}_results_df.csv"
    care_df = write_ranked_outputs(
        eval_manifest,
        final_scores,
        labels,
        care_csv,
        top_k=args.rank_top_k,
        missing=~predicted,
    )
    metrics = {
        "split_group": args.split_group,
        "seed": args.seed,
        "eval_split": eval_split,
        "label_diffusion": args.label_diffusion,
        "rows": misses["rows"],
        "misses": misses["misses"],
        "rows_dropped_no_full_ec": metadata["stats"][eval_split]["rows_with_no_full_label"],
        "graphec": compute_graphec_metrics(y_true, final_scores),
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(
                y_true,
                (final_scores >= 0.5).astype(int),
            ),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
        "artifacts": {
            "care_ranked_csv": str(care_csv),
            "scores_npz": str(result_root / f"{eval_split}_scores.npz"),
        },
    }
    metrics_path = write_json(result_root / f"{eval_split}_metrics.json", metrics)
    print(f"[emulator_bench] {eval_split} metrics: {metrics_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    raise_open_file_limit()
    set_tf32(args.tf32)
    runs = runs_root(args.out_root)
    metadata = load_run_metadata(args.split_group, runs)
    train_metadata = read_json(
        seed_train_metadata_path_for_split(args.split_group, args.seed, runs)
    )
    vocab = read_json(metadata["vocab_path"])
    idx_to_label = {int(index): label for index, label in vocab["idx_to_label"].items()}
    labels = [idx_to_label[index] for index in sorted(idx_to_label)]
    label_to_idx = {label: index for index, label in enumerate(labels)}
    eval_splits = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    result_root = ensure_dir(seed_results_root_for_split(args.split_group, args.seed, runs))
    all_metrics = []
    for eval_split in eval_splits:
        metrics_path = result_root / f"{eval_split}_metrics.json"
        if metrics_path.exists():
            print(f"[emulator_bench] {eval_split} already evaluated: {metrics_path}", flush=True)
            all_metrics.append(read_json(metrics_path))
        else:
            all_metrics.append(
                evaluate_split(args, metadata, train_metadata, eval_split, labels, label_to_idx)
            )
    write_json(
        result_root / "evaluation_summary.json",
        compact_evaluation_summary(all_metrics, train_metadata=train_metadata),
    )


if __name__ == "__main__":
    main()
