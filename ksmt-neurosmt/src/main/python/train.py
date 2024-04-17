#!/usr/bin/python3
import os
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GraphDataloader import get_dataloader
from LightningModel import LightningModel


def get_args():
    parser = ArgumentParser(description="main training script")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ds", required=True, nargs="+")
    parser.add_argument("--oenc", required=True)
    parser.add_argument("--epochs", required=False, type=int)
    parser.add_argument("--lr", required=False, type=float)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--ckpt", required=False)

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    # enable usage of nvidia's TensorFloat if available
    torch.set_float32_matmul_precision("medium")

    num_threads = int(os.environ["NUM_THREADS"])
    print("NUM_THREADS =", num_threads)

    args = get_args()

    train_dl = get_dataloader(
        args.ds, args.oenc, targets="train",
        cache_path="./cache", batch_size=args.batch_size, num_threads=num_threads,
        shuffle=True, drop_last=True
    )
    val_dl = get_dataloader(
        args.ds, args.oenc, targets="val",
        cache_path="./cache", batch_size=args.batch_size, num_threads=num_threads,
        shuffle=False, drop_last=False
    )

    pl_model = LightningModel(learning_rate=args.lr)
    trainer = Trainer(
        accelerator="auto",
        devices=-1,
        # precision="bf16-mixed",
        logger=TensorBoardLogger("./train-logs", name=args.name),
        callbacks=[ModelCheckpoint(
            filename="epoch_{epoch:03d}_roc-auc_{val/roc-auc:.3f}_avg-prc_{val/avg-precision:.3f}",
            monitor="val/loss",
            verbose=True,
            save_last=True, save_top_k=3, mode="min",
            auto_insert_metric_name=False, save_on_train_epoch_end=False
        )],
        max_epochs=args.epochs,
        log_every_n_steps=1,
        accumulate_grad_batches=max(1, 64 // args.batch_size),
        gradient_clip_val=10,
        enable_checkpointing=True,
        barebones=False,
        default_root_dir="."
    )

    trainer.fit(pl_model, train_dl, val_dl, ckpt_path=args.ckpt)
