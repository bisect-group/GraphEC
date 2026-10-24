# GraphEC EMULaToR wrapper

Multi-seed evaluation of GraphEC (EC classification) with its published settings.
GraphEC ships no training loop, so `train.py` adds one: one model trained on the whole train
split, Adam (lr 1e-4), BCE loss, at most 35 epochs, early stopping on the val split of the
benchmark (patience 4, val loss), dropout 0.1, batch size 32, 3 GNN layers, hidden size 256.
The paper averages 5 cross-validation models. The learning rate is not published; the wrapper
uses 1e-4. Evaluation applies a sigmoid to the logits of the best model, then runs
label diffusion (lambda 0.1, DIAMOND `--very-sensitive`, identity cutoff 0.1) against the train
split. The diffusion solves `(I + lambda L) X = Y` with blocked conjugate gradients
(tolerance 1e-10) and returns the eval rows. This equals the dense inverse.

## Data

Split folders are found on disk under
`data/processed/datasets/enzyme_classification_dataset`: `random_splits_grouped_sequence`,
`uniprot_time_splits`, `enzyme_sequence_splits/threshold_*` and
`enzyme_structure_splits/threshold_*`. `backup_*` folders are never read.
Columns: `uniprot_id`, `sequence`, `ec_number`.
Partial EC labels (`3.2.2.-`) are dropped. Rows with non-canonical residues are skipped in
train and count as misses in val and test.
Sequences have no length cap. The label vocabulary is every full EC label seen in train, val
and test of that split.

## Requirements

- Env: `/home/adhil/.conda/envs/mldb/bin/python`. It needs no `torch_cluster`: `radius_graph`
  runs on pyg-lib.
- ProtT5-XL-U50: `--prottrans-model-path` or `$PROTTRANS_MODEL_PATH`. The default is the
  Hugging Face id `Rostlab/prot_t5_xl_uniref50`. Each sequence runs alone on the GPU. There is no CPU
  fallback: a sequence that runs out of GPU memory goes to `feature_failures.json`.
- Structures, one per unique sequence, in this order:
  1. an experimental mmCIF chain whose residues equal the sequence
     (`data/intermediate/mmcif`, chains from the 3d parquet);
  2. a predicted structure from `esmfold2_pdb`, `esm` or `alphafold` under `--structure-root`,
     when its residue count equals the sequence length;
  3. Hugging Face `facebook/esmfold_v1` (ESM language model in fp16, chunk size 128).
  `cache/structure_sources.json` records the source of each sequence.
  `embeddings/feature_failures.json` lists sequences whose ProtT5, ESMFold or active-site step
  ran out of GPU memory, or whose structure mkdssp cannot read.
  They are skipped in train and count as misses in val and test.
  Two-machine fold: every run writes `embeddings/esmfold_todo.json`. `--fold-shards I.. --num-fold-shards N`
  folds only those shards and stops before DSSP. On another machine, `--fold-from <todo.json>
  [--esmfold-model <local folder>]` folds a shard without the dataset. Copy its `structures_v*/*.{pdb,tensor}`
  back, merge its `structure_sources.json` and `feature_failures.json`, and rerun without `--fold-shards`.
- DIAMOND 2.0.14 at `<EMULaToR>/external/diamond-2.0.14/diamond` (or `$DIAMOND_BIN`).

## Commands

Everything goes to `--out-root` (default `<EMULaToR>/outputs/baselines/GraphEC`).
Seeds default to 666 777 888. Run from the submodule root.

```bash
cd baselines/GraphEC
python -m emulator_bench.queue_pipeline --gpus 0 1 2 --runs-per-gpu 1 --eval-split both
python -m emulator_bench.aggregate_results
```

The launcher caches features for each split folder, then runs train and eval for every seed.
Each run gets one GPU through `CUDA_VISIBLE_DEVICES`. A finished run frees its slot.
`--amp` turns on bf16 autocast in train and evaluate; the default keeps fp32. Run the
precision probe before you use it (see Probes).
TF32 matmuls are on by default in train and evaluate; `--no-tf32` turns them off.
In direct mode a finished training frees its GPU slot at once; its evaluation runs in the
background on the same GPU (at most 2 at a time). DIAMOND hits are cached per split and eval
split in `diamond_cache/` next to the seed folders and shared by the seeds.
`--compile` (train only, default off) compiles the geometric features and the GNN layers with
`torch.compile(dynamic=True)`. The compiled dropout draws other random numbers than the eager one.
`--execution-mode ts` queues the jobs with task-spooler and assigns GPUs round-robin.
The cache build shards ESMFold, ProtT5 and active-site work over the first `--num-gpus` visible
GPUs (`queue_pipeline` passes its `--gpus`). `--active-site-procs-per-gpu` (default 1) runs more
active-site shards on each GPU. ProtT5 runs on the GPUs while `--lookup-workers` CPU processes look
up structures. `--gpu-memory-gb` (default: no cap) limits the GPU memory of each GPU process; an
allocation over the cap counts as an out-of-memory failure. DSSP runs on `--dssp-workers` CPU threads. Outputs are identical to the serial
build.
`--max-residues` (default 5000) is the residue budget of one micro-batch. A batch of 32 is
split into micro-batches under this budget, with gradient accumulation and one optimizer step per
batch. A sequence longer than the budget runs alone; there is no length cap. On a GPU OOM a
micro-batch is split in halves and retried on the GPU, in train and in evaluation. A train
sequence that runs out of memory alone is dropped, kept in a skip set in `checkpoint_last.pt`, and
the gradient uses the samples that ran. It is listed in `skipped_train.json` (`no_features`,
`out_of_memory`, `val_no_features`, `val_out_of_memory` and their names). Val and test rows
without features or that run out of memory alone get no prediction; they count as misses in every
metric (`skipped_<split>.json`). The metrics JSON has `rows_dropped_no_full_ec`. Each run has its
own `Data` folder for the feature links. See `REPORT.md`.
The DataLoader workers split each batch into micro-batches and build the label matrix. The main
process copies a batch once, pads it on the GPU and reads the loss once per epoch.
`--epochs` is only for smoke tests. Finished runs are skipped on a rerun, and an interrupted
run resumes from `checkpoint_last.pt`.

## Outputs

- `<out_root>/embeddings`: features per unique sequence hash (`structures_v1`, `prottrans_v1`,
  `dssp_v1`, `active_sites_v1`).
- `<out_root>/enzyme_classification/<split_group>/<split_name>/`: `metadata.json`, manifests,
  fastas and vocab. `<split_name>` is `threshold_*` or `all`.
- `.../seed_<seed>/`: `train.json`, `{bestmodel.pth,lastmodel.pth,checkpoint_last.pt}`,
  `skipped_train.json`, `skipped_<split>.json`, and `<split>_metrics.json`, `<split>_scores.npz`, `<split>_results_df.csv`.
- `<out_root>/enzyme_classification/aggregated_seed_metrics_{long,summary}.csv`: mean and std
  over seeds.

The original code reads features from `baselines/GraphEC/Data/{Structures,ProtTrans,DSSP}`.
The wrapper symlinks the cached files there under the sample names (the folders are git-ignored).

## Probes

GPU probes in `emulator_bench/probes`. Each writes JSON to `--out`, runs on one GPU
(`CUDA_VISIBLE_DEVICES=<gpu>`), and needs the feature cache of `--split-group`. `--synthetic N`
uses random proteins instead, for a dry run on the CPU.

- `exactness_probe`: K steps in fp32 with the upstream forward and CPU padding against the current
  path. It reports the largest loss and parameter differences, and the same for two original runs.
- `precision_probe`: fp32 seed A, bf16 seed A and fp32 seed B. It records train and val loss, val
  metrics, top-1 EC accuracy per level, and the per-layer relative error of bf16 on one batch.
- `compile_probe`: samples per second with and without `--compile`, in fp32 and bf16.
