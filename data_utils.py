import torch
import scanpy as sc
from scipy import sparse
from typing import Tuple
from torch import Tensor
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfTransformer


class TensorDataSetWithIndex(TensorDataset):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor):
        super(TensorDataSetWithIndex, self).__init__(*tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index


def prepare_dataloader(args):
    # Load and Preprocess Source (RNA) Data
    source_adata = sc.read_h5ad(args.data_path + args.source_data)
    if isinstance(source_adata.X, sparse.csr_matrix):
        source_adata.X = source_adata.X.toarray()
    if args.source_preprocess == "Standard":
        sc.pp.normalize_total(source_adata, target_sum=1e4)
        sc.pp.log1p(source_adata)
    elif args.source_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        source_adata.X = tfidf.fit_transform(source_adata.X).toarray()
    else:
        raise NotImplementedError
    sc.pp.scale(source_adata)
    source_adata.obs["Domain"] = args.source_data[:-5]
    source_label = source_adata.obs["CellType"]
    source_label_int = source_label.rank(method="dense", ascending=True).astype(int) - 1
    source_label = source_label.values
    source_label_int = source_label_int.values
    label_map = dict()
    for k in range(source_label_int.max() + 1):
        label_map[k] = source_label[source_label_int == k][0]

    # Load and Preprocess Target (ATAC) Data
    target_adata = sc.read_h5ad(args.data_path + args.target_data)
    if isinstance(target_adata.X, sparse.csr_matrix):
        target_adata.X = target_adata.X.toarray()
    if args.target_preprocess == "Standard":
        sc.pp.normalize_total(target_adata, target_sum=1e4)
        sc.pp.log1p(target_adata)
    elif args.target_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        target_adata.X = tfidf.fit_transform(target_adata.X).toarray()
    else:
        raise NotImplementedError
    sc.pp.scale(target_adata)
    target_adata.obs["Domain"] = args.target_data[:-5]

    # Prepare PyTorch Data
    source_data = torch.from_numpy(source_adata.X).float()
    source_label_int = torch.from_numpy(source_label_int).long()
    target_data = torch.from_numpy(target_adata.X).float()
    target_index = torch.arange(target_data.shape[0]).long()

    # Prepare PyTorch Dataset and DataLoader
    source_dataset = TensorDataset(source_data, source_label_int)
    source_dataloader_train = DataLoader(
        dataset=source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    source_dataloader_eval = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    target_dataset = TensorDataSetWithIndex(target_data, target_index)
    target_dataloader_train = DataLoader(
        dataset=target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_dataloader_eval = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    gene_num = source_data.shape[1]
    type_num = torch.unique(source_label_int).shape[0]

    print("Data Loaded with the Following Configurations:")
    print(
        "Source data:",
        args.source_data[:-5],
        "\tPreprocess:",
        args.source_preprocess,
        "\tShape",
        list(source_data.shape),
    )
    print(
        "Target data:",
        args.target_data[:-5],
        "\tPreprocess:",
        args.target_preprocess,
        "\tShape",
        list(target_data.shape),
    )

    return (
        source_dataset,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataset,
        target_dataloader_train,
        target_dataloader_eval,
        gene_num,
        type_num,
        label_map,
        source_adata,
        target_adata,
    )


def adjacency(X, K=15):
    print("Computing KNN...")
    adj = kneighbors_graph(
        X.cpu().numpy(),
        K,
        mode="connectivity",
        include_self=True,
    ).toarray()
    adj = adj * adj.T
    return adj


def partition_data(
    predictions,
    prob_feature,
    prob_logit,
    source_dataset,
    target_dataset,
    args,
):
    # Partition Reliable/Unreliable Cells
    reliable_index = (prob_feature > args.reliability_threshold) & (
        prob_logit > args.reliability_threshold
    )
    unreliable_index = ~reliable_index

    # Merge Reliable Cells into Source Dataset
    reliable_samples = target_dataset.tensors[0][reliable_index]
    reliable_predictions = predictions[reliable_index]
    source_data = torch.cat((source_dataset.tensors[0], reliable_samples))
    source_type = torch.cat(
        (source_dataset.tensors[1], reliable_predictions)
    )  # Type given as prediction
    source_dataset = TensorDataset(source_data, source_type)

    # Leave Unreliable Cells in Target Dataset
    unreliable_samples = target_dataset.tensors[0][unreliable_index]
    unreliable_index = target_dataset.tensors[1][unreliable_index]
    target_dataset = TensorDataSetWithIndex(unreliable_samples, unreliable_index)

    print(
        "Source dataset size:",
        source_dataset.__len__(),
        "Target dataset size:",
        target_dataset.__len__(),
    )

    # Prepare PyTorch DataLoader
    source_dataloader_train = DataLoader(
        dataset=source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    source_dataloader_eval = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    target_dataloader_train = DataLoader(
        dataset=target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_dataloader_eval = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return (
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        source_dataset,
        target_dataset,
    )
