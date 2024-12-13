# bm25 evaluate
for split in test testb;do
    echo "RACE-"${split}

for retrieval in bm25 contriever splade;do
    printf '%-15s|' ' baseline' ${retrieval}
    ir_measures \
        ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        runs/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec(rel=1) RPrec(rel=2) RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'

    printf '%-15s|' ' reranking' ${retrieval}+CE
    ir_measures \
        ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        runs/reranking.${retrieval}+monoT5.race-${split}.passages.run \
        'RPrec(rel=1) RPrec(rel=2) RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
done

done

for split in test testb;do
    echo "RACE-"${split}
for retrieval in bm25 contriever splade;do
    printf '%-15s|' ' baseline' ${retrieval}
    ir_measures \
        ${DATASET_DIR}/RACE/ranking_4/${split}_qrels_oracle_context_pr.txt \
        runs/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
    ir_measures \
        ${DATASET_DIR}/RACE/ranking_5/${split}_qrels_oracle_context_pr.txt \
        runs/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'

    printf '%-15s|' ' reranking' ${retrieval}+CE
    ir_measures \
        ${DATASET_DIR}/RACE/ranking_4/${split}_qrels_oracle_context_pr.txt \
        runs/reranking.${retrieval}+monoT5.race-${split}.passages.run \
        'RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
    ir_measures \
        ${DATASET_DIR}/RACE/ranking_5/${split}_qrels_oracle_context_pr.txt \
        runs/reranking.${retrieval}+monoT5.race-${split}.passages.run \
        'RPrec(rel=3)' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
done
done
