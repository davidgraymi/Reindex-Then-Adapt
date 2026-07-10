# Overnight Reindex-Step Training on an M1 Mac mini

Verified working on Apple silicon (M3, 16GB) with the modified loaders in
this repo: `reindex_step/preprocess.py` converts the JSONL encodings to
compact float16 tensors once, and `train.py`/`test.py` read those instead
of parsing multi-GB JSONL into Python lists (which needs ~25GB+ RAM).

Peak RAM during training is roughly 3–4GB, so an 8GB or 16GB M1 mini is
fine. Expect ~20–25 min/epoch on an M1 (702 steps/epoch at batch 256), so
an 8-hour overnight run covers ~20 epochs with validation every 2 epochs.

## 1. Copy the repo to the mini

Only ~900MB of data is actually needed (not the full 22GB of JSONL):

```shell
# from this machine, after preprocessing has been run
rsync -avR --exclude '.venv' --exclude 'data/**/output-encoding.jsonl' \
    --exclude 'Reindex-Then-Adapt-*' --exclude 'logs' \
    ./Reindex-Then-Adapt user@mac-mini.local:~/git/
```

Required files if copying manually:
- `reindex_step/data/{inspired,redial,wikipedia}/{train,valid,test}/output-encoding.pt`
  (wikipedia only has `train`)
- `data/llm_embedding.pt`
- `data/item_entity_name_to_item_indices.json`
- `data/{inspired,redial,redditv1.5,wikipedia}/id2name.json`
- the source tree (`reindex_step/`, `adapt_step/`, `tools/`, `scripts/`)

If the `.pt` files have not been generated yet, copy the `.jsonl` files too
and run step 3 on the mini.

## 2. Set up the environment (Python 3.11)

```shell
cd ~/git/Reindex-Then-Adapt
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -r requirements-apple-silicon.txt
# or: .venv/bin/pip install -r requirements-apple-silicon.txt

# sanity check — must print True
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Do NOT use the original `requirements.txt`: it pins torch 2.0.1, which has
no usable MPS acceleration. Also avoid torch >= 2.6 — it flips the
`torch.load(weights_only=...)` default and breaks Lightning checkpoint
loading here.

## 3. (Only if needed) preprocess the JSONL encodings

```shell
.venv/bin/python reindex_step/preprocess.py --data_names "[inspired,redial,wikipedia]"
```

## 4. Launch the overnight run

`caffeinate -is` stops the mini from sleeping; `nohup` keeps the run alive
after you disconnect. `max_minutes 480` = 8 hours; Lightning stops
gracefully at the cap and keeps the best checkpoint by `valid/Recall@10`.

```shell
cd ~/git/Reindex-Then-Adapt
nohup caffeinate -is .venv/bin/python reindex_step/train.py \
    --aggregator rnn \
    --lr 1e-4 \
    --decay 0 \
    --data_names "[inspired,redial,wikipedia]" \
    --epochs 200 \
    --max_minutes 480 \
    --print_every 1 \
    > reindex_step/logs/overnight_run.log 2>&1 &
echo "PID: $!"
```

`--print_every 1` validates (and refreshes `best.ckpt`) after every epoch
instead of every other one — worth the ~2 min/epoch overhead overnight.
Two checkpoints are maintained in `.../checkpoints/`:
- `best.ckpt` — highest `valid/Recall@10` so far (kept even on a crash)
- `last-epoch=<N>.ckpt` — the most recent completed epoch, refreshed every
  epoch regardless of validation, so a crash loses at most one epoch

Notes:
- redditv1.5 is intentionally excluded: its 800k train samples/epoch push
  epoch time past 2 hours on a laptop-class GPU.
- The config (rnn aggregator, lr 1e-4, decay 0) is the paper's best from
  `scripts/best_reindex.yaml`; no need to run the 12-run grid in
  `scripts/reindex_step.sh`.

## 5. Check progress / results

```shell
# live progress (epoch counter, it/s)
tail -f reindex_step/logs/overnight_run.log

# validation metrics so far
column -s, -t reindex_step/logs/inspired_redial_wikipedia/rnn/version_*/metrics.csv | less -S

# charts, redrawn after every validation
open reindex_step/logs/inspired_redial_wikipedia/rnn/version_*/charts/
```

When the run finishes (8h cap or early stop), the log ends with a test
metrics table, and the best checkpoint is at:

```
reindex_step/logs/inspired_redial_wikipedia/rnn/version_<N>/checkpoints/best.ckpt
```

## 6. Stage the checkpoint for the adapt step

```shell
cp reindex_step/logs/inspired_redial_wikipedia/rnn/version_0/checkpoints/best.ckpt \
    ckpts/best_aggregator/
```

Then copy `best.ckpt` back to this machine (or continue on the mini) for
the adapt step (`scripts/adapt_step_for_*.sh`).

To share the trained aggregator on the Hugging Face Hub, use
`scripts/publish_aggregator.py` — it strips the frozen Llama-2 embedding
(a licensed derivative, and ~525MB of the checkpoint) so only the weights
you trained are published. Run it without `--push` first for a dry run.

## Reference: single-epoch smoke test

To verify the setup end-to-end in ~3 minutes before committing to the
overnight run:

```shell
.venv/bin/python reindex_step/train.py --aggregator rnn --lr 1e-4 --decay 0 \
    --data_names "[inspired]" --epochs 2 --max_minutes 15 \
    --ckpt_dir reindex_step/logs_smoke
```

Expect test Recall@10 ≈ 0.39 on inspired after 2 epochs.
