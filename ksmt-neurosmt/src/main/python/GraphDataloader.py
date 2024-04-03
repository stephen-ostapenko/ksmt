import os
import joblib
from joblib import Parallel, delayed
from multiprocessing import Pool
from typing import Literal

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from GlobalConstants import MAX_FORMULA_SIZE, MAX_FORMULA_DEPTH
from GraphReader import read_graph_by_path


joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)


# create a torch_geometric graph
def create_graph(data):
    nodes, edges, label, depths, edge_depths = data

    return Graph(
        x=torch.tensor(nodes, dtype=torch.int32),
        edge_index=torch.tensor(edges, dtype=torch.int64).t(),
        y=torch.tensor([[label]], dtype=torch.int32),
        depth=torch.tensor(depths, dtype=torch.int32),
        edge_depths=torch.tensor(edge_depths, dtype=torch.int32),
    )


# torch Dataset
class GraphDataset(Dataset):
    def __init__(self, graph_data, num_threads=1):
        parallel = Parallel(n_jobs=num_threads, return_as="generator")
        self.graphs = list(parallel(delayed(create_graph)(gd) for gd in graph_data))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


# load one sample
def load_sample(path_to_sample: str):
    operators, edges, depths, edge_depths = read_graph_by_path(
        path_to_sample, max_size=MAX_FORMULA_SIZE, max_depth=MAX_FORMULA_DEPTH
    )

    if operators is None or edges is None or depths is None:
        return None

    if len(edges) == 0:
        print(f"w: ignoring formula without edges; file '{path_to_sample}'")
        return None

    if path_to_sample.endswith("-sat"):
        label = 1
    elif path_to_sample.endswith("-unsat"):
        label = 0
    else:
        raise Exception(f"strange file path '{path_to_sample}'")

    return operators, edges, label, depths, edge_depths


# load all samples from all datasets and return them as a list of tuples
def load_data(
        paths_to_datasets: list[str], target: Literal["train", "val", "test"], num_threads: int
) -> list[tuple[list[str], list[tuple[int, int]], int, list[int], list[int]]]:

    data = []

    print(f"loading {target}")
    for path_to_dataset in paths_to_datasets:
        print(f"loading data from '{path_to_dataset}'")
        path_to_dataset_root, metadata_dir = path_to_dataset.strip().split(":")

        with open(os.path.join(path_to_dataset_root, metadata_dir, target), "r") as f:
            paths_list = list(f.readlines())

            if "SHRINK_DATASET" in os.environ:
                paths_list = paths_list[:int(os.environ["SHRINK_DATASET"])]

            paths_list = map(lambda path: os.path.join(path_to_dataset_root, path.strip()), paths_list)

            with Pool(processes=num_threads) as pool:
                for operators, edges, label, depths, edge_depths in tqdm(pool.imap_unordered(load_sample, paths_list)):
                    data.append((operators, edges, label, depths, edge_depths))

    return data


# load samples from all datasets, transform them and return them in a Dataloader object
def get_dataloader(
        paths_to_datasets: list[str], path_to_ordinal_encoder: str, target: Literal["train", "val", "test"],
        cache_path: str, batch_size: int,
        num_threads: int
) -> DataLoader:

    print(f"creating dataloader for {target}")

    print("loading data")
    data = load_data(paths_to_datasets, target, num_threads)

    print(f"stats: {len(data)} overall; sat fraction is {sum(it[2] for it in data) / len(data)}")

    print("loading encoder")
    encoder = joblib.load(path_to_ordinal_encoder)

    def transform(data_for_one_sample: tuple[list[str], list[tuple[int, int]], int, list[int], list[int]])\
            -> tuple[list[str], list[tuple[int, int]], int, list[int], list[int]]:

        nodes, edges, label, depths, edge_depths = data_for_one_sample
        nodes = encoder.transform(np.array(nodes).reshape(-1, 1))

        return nodes, edges, label, depths, edge_depths

    print("transforming & creating dataset")
    ds_dump_path = f"{target}_{'-'.join(paths_to_datasets).replace('/', '+')}"
    ds_dump_path = f"{ds_dump_path}_{path_to_ordinal_encoder.replace('/', '+')}"
    if "SHRINK_DATASET" in os.environ:
        ds_dump_path = f"{os.environ['SHRINK_DATASET']}_{ds_dump_path}"

    ds_dump_path = os.path.join(cache_path, ds_dump_path)

    if os.path.exists(ds_dump_path):
        print("cache hit!")
        ds = torch.load(ds_dump_path)

    else:
        print("cache miss!")
        parallel = Parallel(n_jobs=num_threads, return_as="generator")
        data = list(parallel(delayed(transform)(data_i) for data_i in data))

        ds = GraphDataset(data, num_threads)

        torch.save(ds, ds_dump_path)

    print("constructing dataloader\n", flush=True)
    return DataLoader(
        ds.graphs,
        batch_size=batch_size, num_workers=num_threads,
        shuffle=(target == "train"), drop_last=(target == "train")
    )
