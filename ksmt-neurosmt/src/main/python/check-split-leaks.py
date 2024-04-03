#!/usr/bin/python3

import os
from argparse import ArgumentParser
from typing import Literal

from tqdm import tqdm


def get_groups_set(paths_to_datasets: list[str], target: Literal["train", "val", "test"], metadata_dir: str) -> set:
    groups = set()

    print(f"loading {target}")
    for path_to_dataset in paths_to_datasets:
        print(f"loading data from '{path_to_dataset}'")

        with open(os.path.join(path_to_dataset, metadata_dir, target), "r") as f:
            for path_to_sample in tqdm(list(f.readlines())):
                path_to_sample = path_to_sample.strip()
                path_to_parent = os.path.dirname(path_to_sample)

                groups.add(str(path_to_parent))

    return groups


def get_args():
    parser = ArgumentParser(description="validation script")
    parser.add_argument("--ds", required=True, nargs="+")
    parser.add_argument("--metadata", required=True)

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    args = get_args()

    train_groups = get_groups_set(args.ds, "train", args.metadata)
    val_groups = get_groups_set(args.ds, "val", args.metadata)
    test_groups = get_groups_set(args.ds, "test", args.metadata)

    assert train_groups.isdisjoint(val_groups)
    assert val_groups.isdisjoint(test_groups)
    assert test_groups.isdisjoint(train_groups)

    print("\nsuccess!")
