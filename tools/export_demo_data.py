"""Export demo conversations + rankings as JS data for demo/index.html.

Reuses the pipeline from tools/demo.py and writes demo/data.js containing
everything the frontend needs (chat turns, before/after top-k with scores,
ground-truth ranks), so the page runs from file:// with no server.

Example:
    .venv/bin/python tools/export_demo_data.py --dataset inspired --num_examples 3
"""

import json
import os
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
sys.path.append(CUR_DIR)

import torch
from jsonargparse import CLI

from demo import (
    build_item_embeddings,
    find_adapt_ckpt,
    load_aggregator,
    load_biases,
    pick_improved_conversations,
    sample_conversations,
)


def parse_turns(context):
    turns = []
    for line in context.strip().split("\n"):
        if ": " in line:
            speaker, text = line.split(": ", 1)
        else:
            speaker, text = "SYSTEM", line
        turns.append({"speaker": speaker.strip(), "text": text.strip()})
    return turns


@torch.no_grad()
def main(
    dataset: str = "inspired",
    num_examples: int = 3,
    k: int = 10,
    seed: int = 0,
    context_lines: int = 0,  # 0 = full conversation
    pick: str = "improved",
    aggregator_path: str = "ckpts/best_aggregator/aggregator.pt",
    adapt_ckpt: str = None,
    out_path: str = "demo/data.js",
):
    aggregator = load_aggregator(os.path.join(ROOT_DIR, aggregator_path))
    adapt_ckpt = adapt_ckpt or find_adapt_ckpt(dataset)
    add_bias, mul_bias = load_biases(adapt_ckpt)

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

    conversations = []
    for sample in samples:
        gt = set(sample["labels_from_data_as_single_token"])
        q = torch.tensor(sample["encoding"], dtype=torch.float32)
        base = q @ item_embed.T
        adapted = (base + cand_add) * cand_mul

        def ranked_entries(scores):
            order = scores.argsort(descending=True)
            entries = []
            for col in order[:k].tolist():
                tok = candidates[col].item()
                entries.append(
                    {
                        "name": token2name.get(tok, f"<item {tok}>"),
                        "token": tok,
                        "score": round(scores[col].item(), 3),
                        "gt": tok in gt,
                    }
                )
            gt_rank = min(
                (scores > scores[c]).sum().item()
                for c, t in enumerate(candidates.tolist())
                if t in gt
            )
            return entries, gt_rank + 1  # 1-indexed

        before, gt_rank_before = ranked_entries(base)
        after, gt_rank_after = ranked_entries(adapted)

        conversations.append(
            {
                "convId": str(sample["conv_id"]),
                "turnId": str(sample["turn_id"]),
                "turns": parse_turns(sample["context"])[
                    -context_lines if context_lines > 0 else None :
                ],
                "groundTruth": [token2name.get(t, str(t)) for t in sorted(gt)],
                "before": before,
                "after": after,
                "gtRankBefore": gt_rank_before,
                "gtRankAfter": gt_rank_after,
            }
        )

    payload = {
        "dataset": dataset,
        "k": k,
        "aggregator": "bi-GRU aggregator (39.9M params, frozen)",
        "adaptCkpt": os.path.relpath(adapt_ckpt, ROOT_DIR),
        "conversations": conversations,
    }
    out = os.path.join(ROOT_DIR, out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("const DEMO_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
    print(f"wrote {out} ({len(conversations)} conversations)")


if __name__ == "__main__":
    CLI(main)
