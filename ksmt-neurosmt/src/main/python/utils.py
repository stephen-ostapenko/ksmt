import os

from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm

from GlobalConstants import MIN_FORMULA_SIZE, MIN_FORMULA_DEPTH, MAX_FORMULA_SIZE, MAX_FORMULA_DEPTH
from GraphReader import read_graph_by_path


def train_val_test_indices(cnt: int, val_qty: float = 0.15, test_qty: float = 0.1)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    perm = np.arange(cnt)
    np.random.shuffle(perm)

    val_cnt = int(cnt * val_qty)
    test_cnt = int(cnt * test_qty)

    return perm[val_cnt + test_cnt:], perm[:val_cnt], perm[val_cnt:val_cnt + test_cnt]


# select paths to suitable samples and transform them to paths from dataset root
def select_paths_with_suitable_samples_and_transform_to_paths_from_root(path_to_dataset_root: str, paths: list[str])\
        -> list[str]:

    print("\nloading paths", flush=True)

    def load_path(path):
        operators, edges, _, _ = read_graph_by_path(
            path,
            min_size=MIN_FORMULA_SIZE, min_depth=MIN_FORMULA_DEPTH,
            max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH
        )

        if operators is None:
            return None

        if len(edges) == 0:
            print(f"w: ignoring formula without edges; file '{path}'")
            return None

        return os.path.relpath(path, path_to_dataset_root)

    num_threads = os.environ["NUM_THREADS"]
    if num_threads is not None:
        num_threads = int(num_threads)
    else:
        num_threads = 16

    parallel = Parallel(n_jobs=num_threads, return_as="generator")
    correct_paths = list(parallel(delayed(load_path)(path) for path in tqdm(paths)))
    correct_paths = list(filter(lambda p: p is not None, correct_paths))

    return correct_paths


def align_sat_unsat_sizes_with_upsamping(sat_data: list[str], unsat_data: list[str]) -> tuple[list[str], list[str]]:
    sat_cnt = len(sat_data)
    unsat_cnt = len(unsat_data)

    sat_indices = list(range(sat_cnt))
    unsat_indices = list(range(unsat_cnt))

    if sat_cnt < unsat_cnt:
        sat_indices += list(np.random.choice(np.array(sat_indices), unsat_cnt - sat_cnt, replace=True))
    elif sat_cnt > unsat_cnt:
        unsat_indices += list(np.random.choice(np.array(unsat_indices), sat_cnt - unsat_cnt, replace=True))

    return (
        list(np.array(sat_data, dtype=object)[sat_indices]),
        list(np.array(unsat_data, dtype=object)[unsat_indices])
    )


def align_sat_unsat_sizes_with_downsamping(sat_data: list[str], unsat_data: list[str]) -> tuple[list[str], list[str]]:
    sat_cnt = len(sat_data)
    unsat_cnt = len(unsat_data)

    sat_indices = list(range(sat_cnt))
    unsat_indices = list(range(unsat_cnt))

    if sat_cnt > unsat_cnt:
        sat_indices = np.random.choice(np.array(sat_indices), unsat_cnt, replace=False)
    elif sat_cnt < unsat_cnt:
        unsat_indices = np.random.choice(np.array(unsat_indices), sat_cnt, replace=False)

    return (
        list(np.array(sat_data, dtype=object)[sat_indices]),
        list(np.array(unsat_data, dtype=object)[unsat_indices])
    )


def align_sat_unsat_sizes(sat_data: list[str], unsat_data: list[str], mode: str) -> tuple[list[str], list[str]]:
    if mode == "none":
        return sat_data, unsat_data
    elif mode == "upsample":
        return align_sat_unsat_sizes_with_upsamping(sat_data, unsat_data)
    elif mode == "downsample":
        return align_sat_unsat_sizes_with_downsamping(sat_data, unsat_data)
    else:
        raise Exception(f"unknown sampling mode {mode}")
