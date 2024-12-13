#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

for split in test testb;do
    echo "crux-"${split}
for tau in 1 2 3 4 5;do
    echo " [Threshold tau ="${tau}"]"
for retrieval in bm25 contriever splade;do
    printf '%-15s|' ' baseline' ${retrieval}
    ir_measures \
        ${DATASET_DIR}/crux/ranking_${tau}/${split}_qrels_oracle_context_pr.txt \
        runs/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec RPrec(rel=2) RPrec(rel=3) MAP@30' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
# printf '%-15s|' ' reranking' ${retrieval}+CE
# ir_measures \
#     ${DATASET_DIR}/crux/ranking_${tau}/${split}_qrels_oracle_context_pr.txt \
#     runs/reranking.${retrieval}+monoT5.race-${split}.passages.run \
#     'RPrec RPrec(rel=2) RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
done
done
done
