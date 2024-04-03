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
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--oenc", required=True)
    parser.add_argument("--batch_size", required=False, type=int, default=16)
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

    args = get_args()
    num_threads = int(os.environ["OMP_NUM_THREADS"])

    val_dl = get_dataloader(
        args.ds, args.oenc, target="val",
        metadata_dir=args.metadata, cache_path="./cache", batch_size=args.batch_size,
        num_threads=num_threads
    )
    test_dl = get_dataloader(
        args.ds, args.oenc, target="test",
        metadata_dir=args.metadata, cache_path="./cache", batch_size=args.batch_size,
        num_threads=num_threads
    )

    trainer = Trainer()

    trainer.validate(LightningModel(), val_dl, args.ckpt)
    trainer.test(LightningModel(), test_dl, args.ckpt)
