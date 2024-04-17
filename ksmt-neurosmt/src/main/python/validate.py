#!/usr/bin/python3

import os
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer

from GraphDataloader import get_dataloader
from LightningModel import LightningModel


def get_args():
    parser = ArgumentParser(description="validation script")
    parser.add_argument("--ds", required=True, nargs="+")
    parser.add_argument("--oenc", required=True)
    parser.add_argument("--batch_size", required=False, type=int, default=16)
    parser.add_argument("--run_full", required=False, action="store_true")
    parser.add_argument("--ckpt", required=True)

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

    if args.run_full:
        test_dl = get_dataloader(
            args.ds, args.oenc, targets=["train", "val", "test"],
            cache_path="./cache", batch_size=args.batch_size, num_threads=num_threads,
            shuffle=False, drop_last=False,
        )

        Trainer(accelerator="auto", devices=1).test(LightningModel(), test_dl, args.ckpt)

    else:
        val_dl = get_dataloader(
            args.ds, args.oenc, targets="val",
            cache_path="./cache", batch_size=args.batch_size, num_threads=num_threads,
            shuffle=False, drop_last=False
        )
        test_dl = get_dataloader(
            args.ds, args.oenc, targets="test",
            cache_path="./cache", batch_size=args.batch_size, num_threads=num_threads,
            shuffle=False, drop_last=False,
        )

        trainer = Trainer(accelerator="auto", devices=1)

        trainer.validate(LightningModel(), val_dl, args.ckpt)
        trainer.test(LightningModel(), test_dl, args.ckpt)
