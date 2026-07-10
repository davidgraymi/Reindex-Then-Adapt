# Reindex step for Apple-silicon (MPS) machines.
#
# Differences from reindex_step.sh:
#   - One-time preprocessing of the JSONL encodings into compact float16
#     tensors (the original loader needs ~25GB+ RAM for the raw JSONL).
#   - A single run with the paper's best config (rnn, lr 1e-4, decay 0)
#     instead of the 12-run hyperparameter grid.
#   - redditv1.5 is excluded (its 800k train samples/epoch are impractical
#     on a laptop GPU); training is capped at 2 hours and the best
#     checkpoint by valid/Recall@10 is kept.
#
# Run from the repo root with the repo's .venv activated.

python reindex_step/preprocess.py --data_names "[inspired,redial,wikipedia]"

python reindex_step/train.py \
    --aggregator rnn \
    --lr 1e-4 \
    --decay 0 \
    --data_names "[inspired,redial,wikipedia]" \
    --epochs 200 \
    --max_minutes 120

# Then copy the best checkpoint for the adapt step, e.g.:
# cp reindex_step/logs/inspired_redial_wikipedia/rnn/version_0/checkpoints/best.ckpt ckpts/best_aggregator/
