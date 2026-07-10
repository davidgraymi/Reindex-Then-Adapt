"""Package a trained reindex-step checkpoint for distribution and
(optionally) push it to the Hugging Face Hub.

A Lightning checkpoint from reindex_step/train.py embeds a frozen copy of
Llama-2's token-embedding matrix (~525MB, and a Llama-2 derivative subject
to Meta's license). This script extracts only the weights that were
actually trained (the aggregator) plus the minimal config needed to
rebuild it, producing a small artifact that is safe to publish.

Examples:
    # package only (writes <ckpt_dir>/aggregator.pt + README.md)
    python scripts/publish_aggregator.py \
        --ckpt_path reindex_step/logs/inspired_redial_wikipedia/rnn/version_0/checkpoints/best.ckpt

    # package and push (needs `pip install huggingface_hub` and `hf auth login`)
    python scripts/publish_aggregator.py \
        --ckpt_path .../best.ckpt --repo_id davidgraymi/rta-aggregator --push true

To reload for inference:
    from aggregators import AGGREGATORS
    from jsonargparse import Namespace
    blob = torch.load("aggregator.pt", map_location="cpu")
    model = AGGREGATORS[blob["config"]["aggregator"]](Namespace(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
"""

import os
import sys

import torch
from jsonargparse import CLI

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

README_TEMPLATE = """---
license: mit
tags:
  - conversational-recommendation
  - reindex-then-adapt
---

# Reindex-Then-Adapt {aggregator} aggregator

Trained aggregator from the reindex step of
[Reindex-Then-Adapt (WSDM'25)](https://dl.acm.org/doi/10.1145/3701551.3703573),
trained on: {data_names}.

This is only the trained aggregator ({size_mb}MB). The frozen Llama-2
token-embedding matrix it operates on is NOT included; get
`llm_embedding.pt` from the
[authors' data repo](https://huggingface.co/datasets/ZhankuiHe/Reindex-Then-Adapt-Real-World-Data)
(subject to the Llama 2 license).

## Usage

With the [Reindex-Then-Adapt code](https://github.com/ZhankuiHe/Reindex-Then-Adapt):

```python
import torch
from jsonargparse import Namespace
from reindex_step.aggregators import AGGREGATORS

blob = torch.load("aggregator.pt", map_location="cpu")
model = AGGREGATORS[blob["config"]["aggregator"]](Namespace(**blob["config"]))
model.load_state_dict(blob["state_dict"])
model.eval()
```

`model(inputs_embeds, attention_mask)` maps an item's Llama-2 token
embeddings `(batch, num_tokens, {embed_size})` to one item embedding
`(batch, {embed_size})`.

## Config

```python
{config}
```
"""


def main(
    ckpt_path: str = None,
    repo_id: str = None,
    push: bool = False,
    private: bool = True,
    out_dir: str = None,
):
    """
    Args:
        ckpt_path: path to a best.ckpt produced by reindex_step/train.py
        repo_id: Hugging Face repo id, e.g. "davidgraymi/rta-aggregator"
        push: if True, upload to the Hub (requires huggingface_hub + login)
        private: create the Hub repo as private
        out_dir: where to write aggregator.pt + README.md (default: ckpt dir)
    """
    assert ckpt_path, "--ckpt_path is required"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["hyper_parameters"]["args"]

    # Keep only the trained aggregator weights ("model." prefix); the
    # remaining state_dict entries (llm_embed, single2llm) are frozen
    # copies of distributed data files.
    state_dict = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    assert state_dict, f"no aggregator weights found in {ckpt_path}"

    # The minimal Namespace fields needed by any AGGREGATORS[...] __init__.
    # single2llm is only used for its padded length (WeightedAggregator).
    config = {
        "aggregator": args.aggregator,
        "embed_size": args.embed_size,
        "dropout_prob": args.dropout_prob,
        "max_llm_tokens": args.single2llm.size(1),
        # provenance, not needed to rebuild the model
        "data_names": list(args.data_names),
        "label": args.label,
        "lr": args.lr,
        "decay": args.decay,
        "epoch": ckpt["epoch"],
        "pytorch_lightning_version": ckpt["pytorch-lightning_version"],
    }

    out_dir = out_dir or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aggregator.pt")
    torch.save({"state_dict": state_dict, "config": config}, out_path)
    size_mb = os.path.getsize(out_path) // 1_000_000
    print(f"Saved {out_path} ({size_mb}MB, from {ckpt_path})")

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            README_TEMPLATE.format(
                aggregator=config["aggregator"],
                data_names=", ".join(config["data_names"]),
                size_mb=size_mb,
                embed_size=config["embed_size"],
                config=config,
            )
        )
    print(f"Saved {readme_path}")

    # Sanity check: rebuild the aggregator from the packaged artifact alone.
    # weights_only=True also proves the file loads under torch>=2.6 defaults.
    from reindex_step.aggregators import AGGREGATORS
    from jsonargparse import Namespace

    blob = torch.load(out_path, map_location="cpu", weights_only=True)
    cfg = Namespace(**blob["config"])
    cfg.single2llm = torch.zeros(1, cfg.max_llm_tokens)  # only .size(1) is used
    model = AGGREGATORS[cfg.aggregator](cfg)
    model.load_state_dict(blob["state_dict"])
    print("Sanity check passed: aggregator rebuilt from packaged file.")

    if not push:
        print("\nDry run (no --push). To upload later:")
        print(f"  hf upload {repo_id or '<user>/<repo>'} {out_path} aggregator.pt")
        print(f"  hf upload {repo_id or '<user>/<repo>'} {readme_path} README.md")
        return

    assert repo_id, "--repo_id is required with --push true"
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=out_path, path_in_repo="aggregator.pt", repo_id=repo_id
    )
    api.upload_file(
        path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=repo_id
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    CLI(main)
