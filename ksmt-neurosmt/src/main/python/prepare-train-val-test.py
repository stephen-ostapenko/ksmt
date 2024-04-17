#!/usr/bin/python3

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import trange

from pytorch_lightning import seed_everything

from utils import train_val_test_indices, align_sat_unsat_sizes, select_paths_with_suitable_samples_and_transform_to_paths_from_root


def classic_random_split(
        path_to_dataset,
        val_qty, test_qty,
        align_train_mode, align_val_mode, align_test_mode
):
    path_to_dataset_root, metadata_dir = path_to_dataset.strip().split(":")
    sat_paths, unsat_paths = [], []
    for root, dirs, files in os.walk(path_to_dataset_root, topdown=True):
        if metadata_dir in dirs:
            dirs.remove(metadata_dir)

        for file_name in files:
            cur_path = str(os.path.join(root, file_name))

            if cur_path.endswith("-sat"):
                sat_paths.append(cur_path)
            elif cur_path.endswith("-unsat"):
                unsat_paths.append(cur_path)
            else:
                raise Exception(f"unsupported file path '{cur_path}'")

    sat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, sat_paths)
    unsat_paths = select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root, unsat_paths)

    def split_data_to_train_val_test(data):
        train_ind, val_ind, test_ind = train_val_test_indices(len(data), val_qty=val_qty, test_qty=test_qty)

        return [data[i] for i in train_ind], [data[i] for i in val_ind], [data[i] for i in test_ind]

    sat_train, sat_val, sat_test = split_data_to_train_val_test(sat_paths)
    unsat_train, unsat_val, unsat_test = split_data_to_train_val_test(unsat_paths)

    sat_train, unsat_train = align_sat_unsat_sizes(sat_train, unsat_train, align_train_mode)
    sat_val, unsat_val = align_sat_unsat_sizes(sat_val, unsat_val, align_val_mode)
    sat_test, unsat_test = align_sat_unsat_sizes(sat_test, unsat_test, align_test_mode)

    train_data = sat_train + unsat_train
    val_data = sat_val + unsat_val
    test_data = sat_test + unsat_test

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data


def grouped_random_split(
        path_to_dataset,
        val_qty, test_qty,
        align_train_mode, align_val_mode, align_test_mode
):
    path_to_dataset_root, metadata_dir = path_to_dataset.strip().split(":")

    def get_all_paths(path_to_dataset_root):
        res = []
        for group_name in os.listdir(path_to_dataset_root):
            if group_name.startswith("__"):
                continue

            for sample_path in os.listdir(os.path.join(path_to_dataset_root, group_name)):
                res.append(os.path.join(path_to_dataset_root, group_name, sample_path))

        return res

    paths = get_all_paths(path_to_dataset_root)

    def calc_group_weights(list_of_suitable_samples):
        groups = dict()
        for path_to_sample in list_of_suitable_samples:
            group = path_to_sample.split("/")[0].strip()
            if group not in groups:
                groups[group] = 0

            groups[group] += 1

        return list(groups.items())

    list_of_suitable_samples = select_paths_with_suitable_samples_and_transform_to_paths_from_root(
        path_to_dataset_root, paths
    )
    groups = calc_group_weights(list_of_suitable_samples)

    def pick_best_split(groups):
        attempts = 100_000

        groups_cnt = len(groups)
        samples_cnt = sum(g[1] for g in groups)

        need_val = int(samples_cnt * val_qty)
        need_test = int(samples_cnt * test_qty)
        need_train = samples_cnt - need_val - need_test

        print("picking best split with existing groups")
        print(f"need: {need_train} (train) | {need_val} (val) | {need_test} (test)")
        print(flush=True)

        best = None

        for _ in trange(attempts):
            probs = (np.array([need_train, need_val, need_test]) / samples_cnt + np.array([1, 1, 1]) / 3) / 2
            cur_split = np.random.choice(range(3), size=groups_cnt, p=probs)

            train_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 0)
            val_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 1)
            test_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 2)

            cur_error = (train_size - need_train) ** 2 + (val_size - need_val) ** 2 + (test_size - need_test) ** 2

            if best is None or best[0] > cur_error:
                best = (cur_error, cur_split)

        return best[1]

    split = pick_best_split(groups)

    split_by_group = dict()
    for i, (group, weight) in enumerate(groups):
        split_by_group[group] = split[i]

    train_data, val_data, test_data = [], [], []
    for path_to_sample in list_of_suitable_samples:
        group = path_to_sample.split("/")[0].strip()

        if split_by_group[group] == 0:
            train_data.append(path_to_sample)
        elif split_by_group[group] == 1:
            val_data.append(path_to_sample)
        elif split_by_group[group] == 2:
            test_data.append(path_to_sample)

    def split_data_to_sat_unsat(data):
        sat_data = list(filter(lambda path: path.endswith("-sat"), data))
        unsat_data = list(filter(lambda path: path.endswith("-unsat"), data))

        return sat_data, unsat_data

    def align_data(data, mode):
        sat_data, unsat_data = split_data_to_sat_unsat(data)
        sat_data, unsat_data = align_sat_unsat_sizes(sat_data, unsat_data, mode)

        return sat_data + unsat_data

    if align_train_mode != "none":
        train_data = align_data(train_data, align_train_mode)

    if align_val_mode != "none":
        val_data = align_data(val_data, align_val_mode)

    if align_test_mode != "none":
        test_data = align_data(test_data, align_test_mode)

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data


def create_split(
        path_to_dataset,
        val_qty, test_qty,
        align_train_mode, align_val_mode, align_test_mode,
        grouped
):
    path_to_dataset_root, metadata_dir = path_to_dataset.strip().split(":")

    if grouped:
        train_data, val_data, test_data = grouped_random_split(
            path_to_dataset,
            val_qty, test_qty,
            align_train_mode, align_val_mode, align_test_mode
        )

    else:
        train_data, val_data, test_data = classic_random_split(
            path_to_dataset,
            val_qty, test_qty,
            align_train_mode, align_val_mode, align_test_mode
        )

    print("\nstats:", flush=True)
    print(f"train: {len(train_data)}")
    print(f"val:   {len(val_data)}")
    print(f"test:  {len(test_data)}")
    print(flush=True)

    meta_path = str(os.path.join(path_to_dataset_root, metadata_dir))
    os.makedirs(meta_path, exist_ok=True)

    with open(os.path.join(meta_path, "train"), "w") as f:
        f.write("\n".join(train_data))

        if len(train_data):
            f.write("\n")

    with open(os.path.join(meta_path, "val"), "w") as f:
        f.write("\n".join(val_data))

        if len(val_data):
            f.write("\n")

    with open(os.path.join(meta_path, "test"), "w") as f:
        f.write("\n".join(test_data))

        if len(test_data):
            f.write("\n")


def get_args():
    parser = ArgumentParser(description="train/val/test splitting script")

    parser.add_argument("--ds", required=True)

    parser.add_argument("--val_qty", type=float, default=0.15)
    parser.add_argument("--test_qty", type=float, default=0.1)

    parser.add_argument("--align_train", choices=["none", "upsample", "downsample"], default="none")
    parser.add_argument("--align_val", choices=["none", "upsample", "downsample"], default="none")
    parser.add_argument("--align_test", choices=["none", "upsample", "downsample"], default="none")

    parser.add_argument("--grouped", action="store_true")

    args = parser.parse_args()
    print("args:")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))

    print()

    return args


if __name__ == "__main__":
    seed_everything(24)

    args = get_args()

    create_split(
        args.ds,
        args.val_qty, args.test_qty,
        args.align_train, args.align_val, args.align_test,
        args.grouped
    )
