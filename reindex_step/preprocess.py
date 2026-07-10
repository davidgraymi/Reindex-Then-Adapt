"""Convert output-encoding.jsonl files into compact .pt tensors.

The original training loader parses multi-GB JSONL into nested Python float
lists, which needs far more RAM than a laptop has (the redditv1.5 train split
alone is ~20GB as Python objects). This script streams each split once and
stores the 4096-dim encodings as a single float16 tensor plus the per-line
label lists, so training can start in seconds instead of minutes.

Example:
    python reindex_step/preprocess.py --data_names [inspired,redial,wikipedia]
"""

import json
import os

import torch
from jsonargparse import CLI
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(jsonl_path: str, pt_path: str):
    encodings = []
    labels_llm = []
    labels_data = []
    conv_ids = []
    turn_ids = []
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=f"converting {jsonl_path}"):
            sample = json.loads(line)
            encodings.append(
                torch.tensor(sample["encoding"], dtype=torch.float16)
            )
            # LLM-generated data (e.g. wikipedia) has no data-derived labels
            labels_llm.append(sample["labels_from_llm_as_single_token"])
            labels_data.append(sample.get("labels_from_data_as_single_token", []))
            conv_ids.append(sample.get("conv_id"))
            turn_ids.append(sample.get("turn_id"))
    torch.save(
        {
            "encodings": torch.stack(encodings),
            "labels_from_llm_as_single_token": labels_llm,
            "labels_from_data_as_single_token": labels_data,
            "conv_ids": conv_ids,
            "turn_ids": turn_ids,
        },
        pt_path,
    )
    print(f"Saved {pt_path}")


def main(
    data_names: list = ["inspired", "redial", "wikipedia"],
    overwrite: bool = False,
):
    for data_name in data_names:
        for split in ["train", "valid", "test"]:
            jsonl_path = os.path.join(
                CUR_DIR, "data", data_name, split, "output-encoding.jsonl"
            )
            if not os.path.exists(jsonl_path):
                print(f"Skip {data_name} {split}: {jsonl_path} does not exist.")
                continue
            pt_path = jsonl_path.replace(".jsonl", ".pt")
            if os.path.exists(pt_path) and not overwrite:
                print(f"Skip {data_name} {split}: {pt_path} already exists.")
                continue
            convert(jsonl_path, pt_path)


if __name__ == "__main__":
    CLI(main)
