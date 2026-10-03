#!/usr/bin/env bash
# End-to-end reproduction of the final submission on the Slurm cluster (all jobs chained).
#
#   cd student_resource/code/business_entity_resolution
#   bash run_all.sh            # after setup_env.sh; submits every job with dependencies
#
# Stages (tags are folder names under student_resource/work/):
#   r0  : off-the-shelf LaBSE, train split only -> candidates used to mine misses for fine-tune 1
#   ft1 : fine-tune LaBSE on fold A (r0 misses first, hard negatives)            -> models/labse_ft
#   r1  : encode + block train with labse_ft -> candidates used to mine misses for fine-tune 2
#   ft2 : continue from labse_ft on r1 misses, half the examples candidate->S1  -> models/labse_ft2
#   final: encode train+test with labse_ft2, block (20% of train S1 left out of the index so train
#          has test-like density of unmatched records), add the exact-key candidate channel,
#          normalize, features (k_fwd 15, k_rev 5, rarity / TF-IDF / soft TF-IDF / list context;
#          LightGBM rows = all of fold B), train LightGBM + tune decision, predict -> output/
# Total: ~12 GPU-hours (P100) + ~6 CPU-hours; wall time ~9 h with 6 GPUs.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
sub() { sbatch --parsable "$@"; }

n=$(sub sbatch/04_normalize.sbatch)

e0=$(TAG=r0 sub --array=0-14 sbatch/02_encode.sbatch)
b0=$(TAG=r0 sub --array=0-3 --dependency=afterok:$e0 sbatch/03_block.sbatch)
f1=$(FROM_TAG=r0 sub --dependency=afterok:$b0 sbatch/08_finetune.sbatch)

e1=$(TAG=r1 MODEL=../../work/models/labse_ft sub --array=0-14 --dependency=afterok:$f1 sbatch/02_encode.sbatch)
b1=$(TAG=r1 sub --array=0-3 --dependency=afterok:$e1 sbatch/03_block.sbatch)
f2=$(BASE_MODEL=../../work/models/labse_ft FROM_TAG=r1 OUT_MODEL=../../work/models/labse_ft2 \
     LR=1e-5 REVERSE_FRAC=0.5 MAX_EXAMPLES=450000 sub --dependency=afterok:$b1 sbatch/08_finetune.sbatch)

e=$(TAG=final_emb MODEL=../../work/models/labse_ft2 sub --dependency=afterok:$f2 sbatch/02_encode.sbatch)
b=$(TAG=final_emb EXCLUDE=0-19 sub --dependency=afterok:$e sbatch/03_block.sbatch)
k=$(TAG=final_emb OUT_TAG=final EXCLUDE=0-19 sub --dependency=afterok:$b:$n sbatch/11_keyblock.sbatch)
f=$(TAG=final K_FWD=15 K_REV=5 TRAIN_FOLDS=B TRAIN_FRAC=1.0 \
    EXTRA="--idf --idf-addr --char-tfidf --soft-tfidf --char-tfidf-addr" \
    sub --dependency=afterok:$k sbatch/05_features.sbatch)
t=$(TAG=final sub --time=05:00:00 --mem=120G --dependency=afterok:$f sbatch/06_train.sbatch)
p=$(TAG=final sub --time=02:00:00 --mem=120G --dependency=afterok:$t sbatch/07_predict.sbatch)

echo "submitted: normalize $n | r0 $e0/$b0 | ft1 $f1 | r1 $e1/$b1 | ft2 $f2 | final $e/$b/$f/$t | predict $p"
echo "outputs: ../../output/matching_results.tsv and ../../output/candidate_pairs.tsv"
