# Business Entity Resolution: pipeline code

Amazon ML Challenge 2026. For every Source 1 (reference) business record, finds all
Source 2 / Source 3 records of the same business. Pipeline: LaBSE bi-encoder fine-tuned
on the training pairs -> exact bidirectional top-k blocking on GPU -> LightGBM pair
classifier on embedding, string and list-context features -> one-parent rule + threshold
tuned for macro F0.5. Written for a Slurm cluster with P100 GPUs (`cse-gpu-all`,
`--gres=gpu:P100-PCIE-16GB:1`) and CPU nodes (`cse-cpu-all`); every step also runs as a
plain `python src/<script>.py` command (see each script's docstring).

Models: LaBSE (`sentence-transformers/LaBSE`, Apache 2.0, ~471M parameters) and LightGBM
(MIT). No external data, APIs or lookup services; everything is learned from the
provided training files. Indic scripts are romanized offline with `indic-transliteration`
(MIT, rule-based).

## Layout

```
student_resource/
├── dataset/{train,test}/            # challenge data (given)
├── utils/validate_submission.py     # given
├── hf_cache/                        # base model weights (setup_env.sh)
├── work/                            # emb/<tag>, cand/<tag>, norm/, feat/<tag>, models/
├── output/                          # matching_results.tsv, candidate_pairs.tsv, history/<tag>_<time>/
└── code/business_entity_resolution/ # this folder
    ├── setup_env.sh                 # one-time conda env (Python 3.11, torch 2.4.1+cu121) + model download
    ├── requirements.txt             # pinned dependencies; requirements.lock.txt = full pip freeze
    ├── run_all.sh                   # submits the whole pipeline as chained Slurm jobs
    ├── make_submission.sh           # builds <team>_submission.zip in the required layout
    ├── sbatch/                      # one job script per step; env.sh is sourced by every job
    └── src/                         # Python source (below)
```

## Reproduce end to end

```bash
cd student_resource/code/business_entity_resolution
bash setup_env.sh      # login node, needs internet once (conda env + LaBSE download)
bash run_all.sh        # submits every step with dependencies; ~12 P100-hours + ~6 CPU-hours
```

Outputs: `student_resource/output/matching_results.tsv` and `candidate_pairs.tsv`
(also copied to `output/history/final_<time>/`), validated with `utils/validate_submission.py`.
Randomness is seeded (folds by crc32 of the entity id, samplers and LightGBM seed 0);
GPU fp16 arithmetic can still move individual borderline pairs slightly between runs.

## Steps

| Step | Script | Job | What it does |
|---|---|---|---|
| 00 | `gpu_check.py` | `00_gpu_check.sbatch` | Checks torch/sm_60/fp16 on the P100, times LaBSE encoding |
| 01 | `data_stats.py` | `01_data_stats.sbatch` | Data profile: matches per entity, one-parent check, unmatched share, scripts, countries |
| 02 | `encode.py` | `02_encode.sbatch` (array 0-27) | LaBSE embeddings (fp16, L2-normalised) of "name \| address", 1M rows per task |
| 03 | `block.py` | `03_block.sbatch` (array 0-9) | Exact top-k search in both directions per (split, country, source); stores forward top-15 and reverse top-5 |
| 04 | `normalize.py` | `04_normalize.sbatch` | Romanize Indic scripts, expand abbreviations / state codes, extract numbers and postal codes |
| 05 | `features.py` | `05_features.sbatch` | Cut-off (forward rank < k_fwd or reverse rank < k_rev, or a shared rare key) = candidate set; embedding-context, string, rarity (idf per country label), soft TF-IDF, character TF-IDF and list-context features; labels + folds for train |
| 06 | `train_match.py` | `06_train.sbatch` | LightGBM; grid search of the decision rule for macro F0.5 on validation fold C |
| 07 | `predict.py` | `07_predict.sbatch` | Scores test candidates, applies the rule, writes both TSVs, runs the validator |
| 08 | `finetune.py` | `08_finetune.sbatch` | Fine-tunes LaBSE on fold-A pairs (previous misses first, hard negatives, MNRL) |
| 09 | `evaluate.py` | `09_evaluate.sbatch` | Writes a validation-fold matching TSV, scores it with a stand-alone macro F0.5, prints sample wrong merges and misses |
| 10 | `diagnose.py` | `10_diagnose.sbatch` | Per-country test statistics and sample records; "blank one country" probe file |
| 11 | `key_block.py` | `11_keyblock.sbatch` | Second candidate channel: pairs sharing a rare exact key (name word, postal + house number, house number + street word), merged into the embedding candidates |
| – | `ensemble.py` | – | Averages the probabilities of runs that share one candidate set; tunes weights + threshold on fold C |
| 12 | `efdecide.py` | `12_efdecide.sbatch` | Per-record expected-F0.5 decision (keeps the top-m pairs that maximise each record's expected F0.5), optionally on averaged runs; tuned on fold C, written only if it beats the threshold rule |

Country handling: country labels are an open set. They only decide which records are
searched against each other (no match crosses labels); they are never a model feature, and
no code branches on a specific label. Normalization uses one set of tables for every record
(English, French and Indian abbreviations side by side; ambiguous short forms such as
St = Street/Saint collapse to one canonical token).

Shared modules: `common.py` (TSV reading without quote handling, text cleaning, folds,
paths), `decision.py` (one-parent rule, threshold, top-1 floor, macro F0.5).

Folds (hash of the S1 entity id): A = 40% (encoder fine-tuning only), B = 40% (LightGBM
training), C = 20% (threshold tuning and local score). LightGBM never trains on pairs the
encoder was fine-tuned on, and fold C is never used for fitting.

## Useful commands

```bash
squeue -u $USER; tail -f logs/<job>_<id>.log; scancel <id>
grep BEST logs/ber_train_*.log                       # validation macro F0.5 per run
TAG=<tag> sbatch sbatch/09_evaluate.sbatch            # independent local score of a run
bash make_submission.sh <team_name> <history folder>  # final zip
```
