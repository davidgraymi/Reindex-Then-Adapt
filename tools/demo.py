"""Reindex-Then-Adapt in action: replay held-out test conversations and show
top-k movie recommendations before vs. after the adapt step.

Pipeline per conversation (no LLM needed at demo time — the LLM's 4096-dim
output encoding for each test turn is precomputed in {split}-for-adapt.jsonl):

    encoding --aggregator--> scores over items --(+a)*m biases--> adapted scores

Examples:
    .venv/bin/python tools/demo.py --dataset inspired
    .venv/bin/python tools/demo.py --dataset redial --num_examples 5 --seed 7
"""

import glob
import json
import os
import random
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)

import torch
from jsonargparse import CLI, Namespace

from reindex_step.aggregators import AGGREGATORS


def load_aggregator(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    args = Namespace(**ckpt["config"])
    model = AGGREGATORS[args.aggregator](args)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def find_adapt_ckpt(dataset):
    pattern = os.path.join(
        ROOT_DIR,
        f"logs/adapt/{dataset}/rnn/add_bias_*/version_*/checkpoints/best.ckpt",
    )
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(
            f"No adapt checkpoint for {dataset}; run"
            " adapt_step/bias_term_adjustment/train.py first"
        )
    return paths[-1]


def load_biases(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return (
        sd["additive_bias.weight"].squeeze(1),
        sd["multiplicative_bias.weight"].squeeze(1),
    )


@torch.no_grad()
def build_item_embeddings(aggregator, tokens, single2llm, llm_embed, bs=2048):
    """Squeeze each candidate item's llm-token embeddings to one vector."""
    out = []
    for i in range(0, len(tokens), bs):
        chunk = tokens[i : i + bs]
        llm_tokens = single2llm[chunk]  # (bs, llm_len)
        mask = llm_tokens != 0
        out.append(
            aggregator(inputs_embeds=llm_embed[llm_tokens], attention_mask=mask)
        )
    return torch.cat(out)


def sample_conversations(jsonl_path, num, seed):
    """Reservoir-sample `num` labeled test turns without loading the file."""
    rng = random.Random(seed)
    picked = []
    n_seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if not sample.get("labels_from_data_as_single_token"):
                continue
            n_seen += 1
            if len(picked) < num:
                picked.append(sample)
            else:
                j = rng.randrange(n_seen)
                if j < num:
                    picked[j] = sample
    return picked


@torch.no_grad()
def pick_improved_conversations(
    dataset, num, k, candidates, item_embed, cand_add, cand_mul
):
    """Score the whole test split (preprocessed .pt) and return the turns
    where adaptation moves a ground-truth item into the top-k, ordered by
    how many ranks it climbed. Falls back to best after-adapt ranks."""
    blob = torch.load(
        os.path.join(ROOT_DIR, f"data/{dataset}/test-for-adapt.pt"),
        map_location="cpu",
    )
    tok2col = {t.item(): c for c, t in enumerate(candidates)}
    scored = []
    for i, labels in enumerate(blob["labels_from_data_as_single_token"]):
        cols = [tok2col[t] for t in labels if t in tok2col]
        if not cols:
            continue
        q = blob["encodings"][i].float()
        base = q @ item_embed.T
        adapted = (base + cand_add) * cand_mul
        # rank of the best-ranked ground-truth item (0-indexed)
        rank_b = min((base > base[c]).sum().item() for c in cols)
        rank_a = min((adapted > adapted[c]).sum().item() for c in cols)
        scored.append((rank_a, rank_a - rank_b, i))
    scored.sort(key=lambda x: (x[0] >= k, x[1], x[0]))
    line_idxs = {i for _, _, i in scored[:num]}

    picked = {}
    with open(
        os.path.join(ROOT_DIR, f"data/{dataset}/test-for-adapt.jsonl"),
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f):
            if line_no in line_idxs:
                picked[line_no] = json.loads(line)
            if len(picked) == len(line_idxs):
                break
    return [picked[i] for _, _, i in scored[:num]]


def show_ranked(title, ranked_tokens, token2name, gt, k):
    print(f"  {title}")
    for r, tok in enumerate(ranked_tokens[:k], 1):
        mark = " <-- ground truth" if tok in gt else ""
        print(f"    {r:2d}. {token2name.get(tok, f'<item {tok}>')}{mark}")


def main(
    dataset: str = "inspired",
    num_examples: int = 3,
    k: int = 10,
    seed: int = 0,
    context_lines: int = 8,
    pick: str = "random",  # "random" or "improved"
    aggregator_path: str = "ckpts/best_aggregator/aggregator.pt",
    adapt_ckpt: str = None,
):
    torch.manual_seed(seed)

    # Models
    aggregator = load_aggregator(os.path.join(ROOT_DIR, aggregator_path))
    adapt_ckpt = adapt_ckpt or find_adapt_ckpt(dataset)
    add_bias, mul_bias = load_biases(adapt_ckpt)
    print(f"aggregator: {aggregator_path}")
    print(f"adapt biases: {os.path.relpath(adapt_ckpt, ROOT_DIR)}")

    # Item vocabulary for this dataset
    mapping = json.load(
        open(os.path.join(ROOT_DIR, "data/item_entity_name_to_item_indices.json"))
    )
    names = json.load(
        open(os.path.join(ROOT_DIR, f"data/{dataset}/id2name.json"))
    ).values()
    candidates = torch.LongTensor(
        sorted({mapping[n]["single_token"] for n in names})
    )
    token2name = {
        m["single_token"]: m["friendly_title"] for m in mapping.values()
    }
    single2llm = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(m["llm_tokens"]) for m in mapping.values()],
        batch_first=True,
        padding_value=0,
    )
    # pad_sequence above is ordered by mapping insertion; reorder by token id
    order = torch.LongTensor([m["single_token"] for m in mapping.values()])
    single2llm = single2llm[order.argsort()]

    llm_embed = torch.load(
        os.path.join(ROOT_DIR, "data/llm_embedding.pt"), map_location="cpu"
    )

    print(f"building embeddings for {len(candidates)} {dataset} items...")
    item_embed = build_item_embeddings(
        aggregator, candidates, single2llm, llm_embed
    )
    cand_add = add_bias[candidates]
    cand_mul = mul_bias[candidates]

    # Conversations
    if pick == "improved":
        samples = pick_improved_conversations(
            dataset, num_examples, k, candidates, item_embed, cand_add, cand_mul
        )
    else:
        samples = sample_conversations(
            os.path.join(ROOT_DIR, f"data/{dataset}/test-for-adapt.jsonl"),
            num_examples,
            seed,
        )

    hits = {"before": [0, 0, 0], "after": [0, 0, 0]}  # @1, @5, @10
    for n, sample in enumerate(samples, 1):
        gt = set(sample["labels_from_data_as_single_token"])
        q = torch.tensor(sample["encoding"], dtype=torch.float32)

        base = q @ item_embed.T
        adapted = (base + cand_add) * cand_mul
        rank_before = candidates[base.argsort(descending=True)].tolist()
        rank_after = candidates[adapted.argsort(descending=True)].tolist()

        print("\n" + "=" * 72)
        print(f"Conversation {n}  (conv {sample['conv_id']}, turn {sample['turn_id']})")
        print("-" * 72)
        for line in sample["context"].strip().split("\n")[-context_lines:]:
            print(f"  {line}")
        print("-" * 72)
        print(
            "  Ground truth:",
            ", ".join(token2name.get(t, str(t)) for t in sorted(gt)),
        )
        print()
        show_ranked("LLM + aggregator (before adapt):", rank_before, token2name, gt, k)
        print()
        show_ranked("+ bias-term adaptation (after adapt):", rank_after, token2name, gt, k)

        for mode, ranked in [("before", rank_before), ("after", rank_after)]:
            for i, kk in enumerate([1, 5, 10]):
                hits[mode][i] += bool(gt & set(ranked[:kk]))

    print("\n" + "=" * 72)
    print(f"Hits over these {len(samples)} conversations (@1 / @5 / @10):")
    print(f"  before adapt: {hits['before'][0]} / {hits['before'][1]} / {hits['before'][2]}")
    print(f"  after  adapt: {hits['after'][0]} / {hits['after'][1]} / {hits['after'][2]}")


if __name__ == "__main__":
    CLI(main)
