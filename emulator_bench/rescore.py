"""Score the test rows with non-canonical residues instead of counting them as misses.

`evaluate` leaves these rows without a prediction. Here each one gets features from its sequence
with U, Z, O and B written as X (cache `<out_root>/embeddings_xrescore`, keyed by the hash of the
raw sequence), a score from every seed's best model, and a place in the label diffusion. Rows whose
features fail stay misses. Per seed, `test_metrics.json` and `test_results_df.csv` are replaced by
the rescored ones and the first versions are kept as `*_pre_rescore.*`. The rescored scores go to
`test_scores_rescored.npz`. A seed with that file is skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from . import cache_features as cf
from .dataset_adapter import select_split_groups
from .evaluate import load_manifest_dataset, load_model, non_canonical_rows, predict_scores
from .evaluate import run_label_diffusion
from .results import (
    compute_care_metrics,
    compute_graphec_metrics,
    compute_supplemental_classification_metrics,
    compute_supplemental_ranking_metrics,
    write_ranked_outputs,
)
from .train import DEFAULT_MAX_RESIDUES, load_active_sites
from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_STRUCTURE_ROOT,
    baseline_cwd,
    cache_dirs,
    ensure_dir,
    ensure_graphec_data_links,
    import_task_module,
    load_run_metadata,
    read_json,
    run_data_root,
    runs_root,
    seed_results_root_for_split,
    seed_train_metadata_path_for_split,
    set_tf32,
    split_group_slug,
    write_json,
)

DIFFUSION_LAMBDA = IDENTITY_CUTOFF = 0.1  # as in evaluate
TOP_K = 50


def x_mapped(sequence: str) -> str:
    return re.sub(r"[UZOB]", "X", sequence)


def known_letters(sequence: str) -> str:
    """The EC and active-site models never read the residue letter; the loader needs a known one."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "A", sequence)


def feature_items(rows: pd.DataFrame) -> list[dict]:
    """One item per unique raw sequence: its hash, X-mapped sequence and entry ids."""
    return [
        {
            "cache_key": key,
            "raw": group["Sequence"].iloc[0],
            "sequence": x_mapped(group["Sequence"].iloc[0]),
            "entries": sorted(set(group["Original Entry"])),
        }
        for key, group in rows.groupby("sequence_sha256", sort=False)
    ]


def build_features(items: list[dict], args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    """Missing features of `items`. A sequence whose step fails gets no active-site file."""
    experimental = cf.load_experimental_chains(
        Path(args.dataset_root) / "enzyme_classification_dataset_3d.parquet"
    )
    structure_root = Path(args.structure_root)
    # A chain residue such as selenocysteine (U) is compared with the X-mapped sequence.
    pdb_residues, experimental_lines = cf.pdb_residues, cf.experimental_pdb_lines

    def chain_lines(cif, chain):
        lines, residues = experimental_lines(cif, chain)
        return lines, x_mapped(residues)

    with (
        mock.patch.object(cf, "pdb_residues", lambda lines: x_mapped(pdb_residues(lines))),
        mock.patch.object(cf, "experimental_pdb_lines", chain_lines),
    ):
        unresolved = cf.link_structures(
            cf.missing_items(items, dirs["structures"], ".tensor"),
            dirs,
            structure_root,
            experimental,
        )
        failed = set(cf.fold_with_esmfold(unresolved, dirs))
        for item in unresolved:  # ESMFold gives no coordinates for an X residue
            key, tensor = item["cache_key"], dirs["structures"] / f"{item['cache_key']}.tensor"
            if key not in failed and len(torch.load(tensor)) != len(item["sequence"]):
                failed.add(key)
                tensor.unlink()
                (dirs["structures"] / f"{key}.pdb").unlink()
        items = [item for item in items if item["cache_key"] not in failed]
        failed = set(
            cf.generate_prottrans_features(
                cf.missing_items(items, dirs["prottrans"], ".tensor"),
                dirs,
                model_path=args.prottrans_model_path,
            )
        )
        items = [item for item in items if item["cache_key"] not in failed]
        no_source, oom = cf.dssp_with_fallback(
            cf.missing_items(items, dirs["dssp"], ".tensor"),
            dirs,
            Namespace(
                dssp_workers=args.dssp_workers, lookup_workers=1, num_gpus=1, gpu_memory_gb=None
            ),
            structure_root,
            experimental,
        )
    items = [item for item in items if item["cache_key"] not in {*no_source, *oom}]
    cf.generate_active_site_features(
        cf.missing_items(
            [{**item, "sequence": known_letters(item["raw"])} for item in items],
            dirs["active_sites"],
            ".pt",
        ),
        dirs,
        batch_size=1,
        num_workers=args.num_workers,
    )


def predict_rows(
    items: list[dict], split_group: str, seeds, labels: list[str], args, dirs: dict[str, Path]
) -> dict[int, dict]:
    """Scores of the rows with features, per seed: {"names": hashes, "raw_scores": ...}."""
    ready = [i for i in items if (dirs["active_sites"] / f"{i['cache_key']}.pt").exists()]
    keys = {item["cache_key"]: item["cache_key"] for item in ready}
    dataset = {item["cache_key"]: [known_letters(item["raw"])] for item in ready}
    data_root = run_data_root(dirs["tmp"] / f"data_{split_group_slug(split_group)}")
    ensure_graphec_data_links(keys, dirs["structures"].parent, data_root)
    set_tf32(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    with baseline_cwd(data_root):
        model_mod = import_task_module("EC_number", "model")
        data_mod = sys.modules["data"]
        active_sites = load_active_sites(keys, dirs["structures"].parent)
        for seed in seeds:
            train_metadata = read_json(
                seed_train_metadata_path_for_split(split_group, seed, runs_root(args.out_root))
            )
            model = load_model(model_mod, train_metadata, len(labels), device)
            names, raw_scores, _ = predict_scores(
                dataset=dataset,
                active_sites=active_sites,
                data_mod=data_mod,
                model=model,
                batch_size=32,
                num_workers=args.num_workers,
                max_residues=DEFAULT_MAX_RESIDUES,
            )
            scores[seed] = {"names": np.array(names), "raw_scores": raw_scores}
    return scores


def metrics_of(eval_manifest, scores, predicted, y_true, labels, care_csv) -> dict:
    care_df = write_ranked_outputs(
        eval_manifest, scores, labels, care_csv, top_k=TOP_K, missing=~predicted
    )
    return {
        "graphec": compute_graphec_metrics(y_true, scores),
        "care_task1": compute_care_metrics(care_df),
        "supplemental": {
            "classification": compute_supplemental_classification_metrics(
                y_true, (scores >= 0.5).astype(int)
            ),
            "ranking": compute_supplemental_ranking_metrics(care_df),
        },
    }


def rescore_seed(
    result_root: Path, metadata: dict, labels: list[str], rows: pd.DataFrame, new: dict
) -> None:
    """Add the scores `new` of the non-canonical rows to the saved test scores, rerun the label
    diffusion and replace the test metrics."""
    saved = np.load(result_root / "test_scores.npz")
    names, raw_scores = list(saved["names"]), saved["raw_scores"]
    skipped = read_json(result_root / "skipped_test.json")
    missed = {
        n for k in ("no_features", "non_canonical", "out_of_memory") for n in skipped[f"{k}_names"]
    }
    predicted = np.array([name not in missed for name in names])
    eval_manifest = result_root / "test_eval_manifest.csv"
    label_to_idx = {label: i for i, label in enumerate(labels)}
    _, ordered, y_true, _, _ = load_manifest_dataset(
        eval_manifest,
        label_to_idx,
        set(skipped["no_features_names"]),
        set(skipped["non_canonical_names"]),
    )
    assert ordered == names, "row order differs from saved scores"

    position = {name: i for i, name in enumerate(names)}
    manifest = pd.read_csv(metadata["manifests"]["test"])
    eval_df = manifest[manifest["cache_key"].astype(str).isin(set(np.array(names)[predicted]))]
    score_of = dict(zip(new["names"], new["raw_scores"]))
    added = rows[rows["sequence_sha256"].isin(score_of)]
    for row in added.itertuples():
        raw_scores[position[row.cache_key]] = score_of[row.sequence_sha256]
        predicted[position[row.cache_key]] = True
    eval_df = pd.concat([eval_df, added], ignore_index=True)
    order = [position[key] for key in eval_df["cache_key"].astype(str)]
    assert order == sorted(order), "eval rows differ in order from the score rows"
    final_scores = np.zeros_like(raw_scores)
    final_scores[predicted] = run_label_diffusion(
        scores=raw_scores[predicted],
        train_manifest=metadata["manifests"]["train"],
        eval_df=eval_df,
        labels=labels,
        work_dir=result_root / "test_label_diffusion_rescore",
        cache_dir=result_root.parent / "diamond_cache" / "test",
        lamda=DIFFUSION_LAMBDA,
        identity_cutoff=IDENTITY_CUTOFF,
    )

    for stem, suffix in (("test_metrics", "json"), ("test_results_df", "csv")):
        (result_root / f"{stem}.{suffix}").replace(result_root / f"{stem}_pre_rescore.{suffix}")
    care_csv = result_root / "test_results_df.csv"
    metrics = metrics_of(eval_manifest, final_scores, predicted, y_true, labels, care_csv)
    old = read_json(result_root / "test_metrics_pre_rescore.json")
    write_json(
        result_root / "test_metrics.json",
        {
            **old,
            **metrics,
            "misses": int((~predicted).sum()),
            "remaining_misses": int((~predicted).sum()),
            "rescored_rows": len(added),
            "artifacts": {
                "care_ranked_csv": str(care_csv),
                "scores_npz": str(result_root / "test_scores_rescored.npz"),
            },
        },
    )
    np.savez_compressed(
        result_root / "test_scores_rescored.npz",
        names=np.array(names),
        raw_scores=raw_scores,
        final_scores=final_scores,
        predicted=predicted,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the non-canonical test rows of GraphEC runs"
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", help="default: every split folder")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--structure-root", default=str(DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--prottrans-model-path", default=cf.PROTT5_MODEL)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--scores-dir",
        help="<slug>_seed<seed>.npz scores of the non-canonical rows (names = sequence hashes); "
        "missing files are computed and saved here (default: <out-root>/xrescore_scores)",
    )
    parser.add_argument("--dssp-workers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = runs_root(args.out_root)
    scores_dir = ensure_dir(args.scores_dir or Path(args.out_root) / "xrescore_scores")
    seeds = args.seed or DEFAULT_SEEDS
    for group in select_split_groups(args.dataset_root, args.split_group):
        metadata = load_run_metadata(group.name, runs)
        vocab = read_json(metadata["vocab_path"])
        labels = [
            name for _, name in sorted((int(i), name) for i, name in vocab["idx_to_label"].items())
        ]
        rows = non_canonical_rows(metadata, "test", {label: i for i, label in enumerate(labels)})
        slug = split_group_slug(group.name)
        todo = [
            seed
            for seed in seeds
            if not (
                seed_results_root_for_split(group.name, seed, runs) / "test_scores_rescored.npz"
            ).exists()
        ]
        need = [seed for seed in todo if not (scores_dir / f"{slug}_seed{seed}.npz").exists()]
        if need:
            dirs = cache_dirs(Path(args.out_root) / "embeddings_xrescore")
            items = feature_items(rows)
            build_features(items, args, dirs)
            for seed, new in predict_rows(items, group.name, need, labels, args, dirs).items():
                np.savez_compressed(scores_dir / f"{slug}_seed{seed}.npz", **new)
        for seed in todo:
            result_root = seed_results_root_for_split(group.name, seed, runs)
            rescore_seed(
                result_root, metadata, labels, rows, np.load(scores_dir / f"{slug}_seed{seed}.npz")
            )
            print(f"[emulator_bench] rescored {slug} seed {seed}: {result_root}", flush=True)


if __name__ == "__main__":
    main()
