"""Evaluate the bare aggregator checkpoint published to HF
(davidgray/rta-aggregator, file ckpts/best_aggregator/aggregator.pt).

Unlike a Lightning best.ckpt, that file holds only the aggregator's own
state_dict (keys `rnn.*` / `proj.*`) plus a `config` dict, so it is loaded
into `Model.model` rather than the full LightningModule.

Example:
    .venv/bin/python tools/eval_hf_aggregator.py
"""

import os
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "reindex_step"))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from jsonargparse import Namespace
from pytorch_lightning.loggers import CSVLogger

import test as test_mod  # reindex_step/test.py


def main():
    ckpt_path = os.path.join(ROOT_DIR, "ckpts/best_aggregator/aggregator.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    print("checkpoint config:", config)

    args = Namespace(
        data_dir=os.path.join(ROOT_DIR, "data"),
        llm_embed_path=os.path.join(ROOT_DIR, "data/llm_embedding.pt"),
        freeze_llm_embed=True,
        aggregator=config["aggregator"],
        dropout_prob=config["dropout_prob"],
        data_names=config["data_names"],
        batch_size=256,
        embed_update_batch_size=2048,
        lr=config["lr"],
        decay=config["decay"],
        negs=1000,
        ks=[1, 5, 10, 50],
        ckpt_dir=os.path.join(ROOT_DIR, "reindex_step/logs/eval_hf"),
    )

    for label in [
        "labels_from_llm_as_single_token",
        "labels_from_data_as_single_token",
    ]:
        args.label = label
        data = test_mod.Data(args=args)
        args.num_items = data.num_items
        args.embed_size = data.embed_size
        args.single2llm = data.single2llm
        args.data2single = data.data2single

        val_loader = DataLoader(
            test_mod.Dataset(data=data, mode="valid", args=args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_mod.Dataset(data=data, mode="test", args=args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = test_mod.Model(args)
        missing, unexpected = model.model.load_state_dict(
            ckpt["state_dict"], strict=True
        )
        print(f"loaded aggregator weights: missing={missing} unexpected={unexpected}")

        trainer = pl.Trainer(
            default_root_dir=args.ckpt_dir,
            accelerator="auto",
            logger=CSVLogger(args.ckpt_dir, name=label),
            enable_checkpointing=False,
        )
        print(f"\n===== label = {label} =====")
        trainer.validate(model=model, dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
