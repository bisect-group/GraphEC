# GraphEC EMULaToR deviations

## Deviations from the paper

- Single model with early stopping on the benchmark val split. The paper averages 5
  cross-validation models. Val rows without features, or that run out of GPU memory alone, are
  left out of the val loss; the counts are in `skipped_train.json` and `train.json`.

- Learning rate: the paper gives none. The wrapper uses 1e-4.
- Structures: one per unique sequence. Order: exact experimental mmCIF chain, then a predicted
  structure (`esmfold2_pdb`, `esm`, `alphafold`) with the same residue count, then Hugging Face
  ESMFold v1. `structure_sources.json` records the source.
- No length cap. The original code has none in the data path, and long sequences are kept.
- Gradient accumulation: a batch of 32 is split into micro-batches of at most `--max-residues`
  residues (default 5000). Each micro loss is scaled by its share of the batch and there is
  one optimizer step per batch. The gradient equals the full-batch gradient (tested to 1e-6
  with BatchNorm in eval mode). In train mode BatchNorm statistics come from micro-batches.
- Label diffusion uses blocked conjugate gradients: an iterative solve of `(I + lambda L) X = B`
  to a relative residual of 1e-10 per column. It equals the dense solve within that tolerance
  (tested) and is not a factorisation. It raises an error if a column does not reach the
  tolerance in 1000 iterations.
- Each run has its own `Data` folder (`<run>/data` for training, `<result>/data_<split>` for
  evaluation) and the code runs with that folder as working directory, because the original code
  reads `./Data`. Runs never share or change links, and the files in the submodule `Data` folder
  are not touched.
- A predicted structure must have the same residues as the sequence, not only the same length.
  OXT atoms are removed from predicted structures, as from experimental ones.
- The unit of data is a unique (entry, sequence) protein after the full-EC filter, not a parquet
  row. Rows with no full EC (4 digits) are dropped. The count is in the metrics JSON as
  `rows_dropped_no_full_ec` (random split: 19,083 train, 1,080 val and 1,043 test rows).
- Padding of node features uses one indexed copy on the GPU instead of the original per-graph CPU
  loop. `GraphEC.forward` is replaced at model build by a copy that pads the node features on the
  GPU instead of copying them through the CPU. The original fallback stays: when the
  geometric node features do not match the node count, they become ones (found by a shape check,
  not by catching an error). Outputs are equal.
- Micro-batches and label matrices are built in the DataLoader workers. Loss sums stay on the GPU
  (float64) and are read once per epoch. Values and order are the same.
- Active-site files are read when a sample is first used, and only the `.tensor` links are made.
- Feature build: ESMFold, ProtT5 and active-site work shard over `--num-gpus` GPUs, with
  `--active-site-procs-per-gpu` shards on each. ProtT5 runs while the structure lookup runs on
  `--lookup-workers` CPU processes. DSSP runs on CPU threads. Outputs equal the serial build (DSSP bit-equal, active site within 1.4e-6).
- A predicted or experimental structure with a missing N, CA, C or O atom is not used. The next
  source is tried. The same holds when mkdssp rejects a structure (a degenerate predicted
  structure gives NaN): experimental, then predicted, then ESMFold v1. Only when every source
  fails does the sequence go to `feature_failures.json` (reason `dssp_failed`).

## Skipped rows and misses

- No CPU retry anywhere. A ProtT5, ESMFold or active-site step that runs out of GPU memory
  writes the sequence to `embeddings/feature_failures.json`. So does a sequence whose structures
  mkdssp rejects from every source.
- Train: rows with no features (no usable structure, or a feature failure) are skipped. There is
  no length cap: `--max-residues` only sets the micro-batch size, and a longer sequence runs
  alone. If a micro-batch runs out of GPU memory, its gradient is undone and its two halves run
  again on the GPU (no CPU retry). A sequence that runs out of memory alone is dropped from the
  step, the gradient is divided by the samples that ran, and the sequence goes to a skip set. The
  set is saved in `checkpoint_last.pt`, so later epochs and resumes skip it without a retry.
  Counts and names (`no_features`, `out_of_memory`) are in `seed_<s>/skipped_train.json`.
  Val rows that run out of memory alone are left out of the val loss (`val_out_of_memory`).
- Val and test: these rows stay. A row with no features, or one that runs out of GPU memory at
  batch size 1, gets no prediction. It counts as a miss: zero score, NaN rank, and it stays in
  every metric denominator. Counts are in `seed_<s>/skipped_<split>.json` and the metrics JSON
  (`rows`, `misses`).
- Any other missing cache still fails with an error.

## Equivalence with the original code (6 real sequences, CPU, fp32, no accumulation split)

- Padding: identical. Loss: identical (0.7356116).
- Gradients: max difference 2.3e-9. Two runs of the original code differ by 2.3e-9 as well, so
  this is noise. Parameters after one Adam step: 1.7e-5 (original against original: 1.2e-5).
- Eval scores, original `predict` path against `predict_scores`: 0.0.
- Label diffusion on a 100-sequence graph: homology matrix identical; diffusion max difference
  1.4e-10 against the original `compute_L` and sparse inverse (lambda 0.1 and 0.5); metrics
  identical. The original code has no metric code.
- Known limit of the original code: `get_geo_feat` computes angles over the concatenated batch,
  so the first and last two residues of a graph depend on the neighbour graph (max feature
  difference 2.0 on those residues; 0 elsewhere). Splitting a batch into micro-batches therefore
  changes the scores a little: 1.4e-3 at most, with every graph alone against one batch. The same
  effect comes from shuffling in the original. The accumulated gradient equals the full-batch
  gradient exactly only for a model without this coupling (tested with a stand-in model).

## Memory and speed (GPU 3; earlier measurements shared it with other jobs, the epoch timing did not)

- Peak memory of a training step on real batches: bf16 (`--amp`) 2500 residues 4.70 GiB, 3500
  6.57, 4500 8.07 (about 1.8 GiB per 1000 residues). fp32: 1500 5.87 GiB, 2200 7.15; fp32 at
  3000 ran out of memory under the 9.3 GiB test cap.
- Estimate at the default of 5000 residues: bf16 about 9 GiB, fp32 about 12 to 13 GiB. Both fit
  a 22 GB A10 with one run per GPU. For two runs per GPU use `--max-residues 3000` with `--amp`.
- Training throughput (400 real sequences, mean length 397, batch 32): bf16 34 samples/s,
  fp32 24 samples/s. Forward only: bf16 66, fp32 52 samples/s. bf16 is 1.4 times faster.
- The loader is not the limit: 0 workers is 6 percent slower than 4; 8 workers with pinned
  memory gives no gain.
- Two runs on one GPU: 17 samples/s each against 30 for one (bf16, 1500 residues). That is
  1.14 times the throughput of one run.
- Epoch time from real sample counts (unique proteins after the full-EC filter, before any
  feature failure). Rates: 37 train and 77 val samples/s (bf16, batch 32, budget 3500; 1,000 real
  sequences, mean length 391). Epoch = train/37 + val/77.

  | Split group | Train | Val | Test | Epoch | Run, 35 ep | Run, 12 ep |
  |---|---|---|---|---|---|---|
  | random_splits_grouped_sequence | 222,073 | 12,406 | 12,374 | 1.71 h | 60 h | 21 h |
  | uniprot_time_splits | 223,131 | 11,395 | 12,327 | 1.72 h | 60 h | 21 h |
  | enzyme_sequence_splits/threshold_0.39 | 224,457 | 12,074 | 10,322 | 1.73 h | 61 h | 21 h |
  | enzyme_structure_splits/threshold_0.85 | 225,113 | 10,992 | 10,748 | 1.73 h | 61 h | 21 h |

  The 12 runs (4 groups, 3 seeds) on 4 A10s with one run per GPU take 3 waves: about 182 h
  (7.6 d) at 35 epochs and 62 h (2.6 d) at 12 epochs. The 12-epoch figure is an assumption;
  early stopping is untested at this scale. Test and val evaluation add a few minutes per run.
  GPU 3 was not shared during this timing. The 5000-residue budget was not timed (memory cap);
  throughput did not change with the budget in earlier tests.
- Feature build per sequence (one shared GPU): ProtT5 0.14 s, active site 0.15 s, DSSP 0.1 s
  serial or 0.005 s on 16 threads, structure lookup 0.03 s (single CPU process). ESMFold takes
  about 4 s per sequence. For 226,400 sequences and 1,775 ESMFold folds this is about 22 h on
  one GPU and 7.5 h on four GPUs (structure lookup, 2 h serial, now runs on CPU processes while ProtT5 runs).
- Under the 9.5 GB test cap ESMFold ran out of memory on sequences longer than 333 residues
  (54 of 102). Expect far fewer on a 22 GB A10.

## bf16 against fp32 (measured with the earlier 2-fold code: 6 epochs, 1,451 train samples, one seed)

- Best val loss per fold: fp32 0.00687 and 0.00629; bf16 0.00684 and 0.00629.
- Test micro AUC 0.6776 and 0.6779; micro AUPR 0.0883 and 0.0881; raw scores differ by at most
  0.008 (correlation 0.9999). The model is far from trained, so this only shows that bf16 does
  not change the training path. Rerun on a longer run before a final choice: `probes/precision_probe.py`.

## Open issues

- The fp32 peak at 5000 residues is extrapolated, not measured. Train and val OOM handling is
  tested with stubs, not on real long sequences.
- Real ProtT5 and ESMFold OOM paths and `--execution-mode ts` are not tested on long sequences.
- `--compile` is untested on a GPU: run `probes/compile_probe.py`.
- Training cost is the main risk: see the ETAs above.
- The 12-epoch figure depends on how early stopping behaves at full scale.
- The full feature build and per-source counts for the full dataset are not run.
