import os
import torch
import numpy as np
import scanpy as sc
import seaborn as sns
import scanpy.external as sce
from torch import nn
from model_utils import feature_prototype_similarity, gmm
from sklearn.metrics import accuracy_score, f1_score, silhouette_score


def infer_result(net, source_dataloader, target_dataloader, args):
    net.eval()
    feature_vec, type_vec, pred_vec, loss_vec = [], [], [], []
    for (x, y) in source_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
        feature_vec.extend(h.cpu().numpy())
        type_vec.extend(y.numpy())
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    for (x, _), _ in target_dataloader:
        x = x.cuda()
        with torch.no_grad():
            h = net.encoder(x)
            logit = net.classifier(h)
            pred = torch.argmax(logit, dim=-1)
            loss = ce_loss(logit, pred)
        feature_vec.extend(h.cpu().numpy())
        pred_vec.extend(pred.cpu().numpy())
        loss_vec.extend(loss.cpu().numpy())
    feature_vec, type_vec, pred_vec, loss_vec = (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(pred_vec),
        np.array(loss_vec),
    )

    similarity, _ = feature_prototype_similarity(
        feature_vec[: len(source_dataloader.dataset)],
        type_vec,
        feature_vec[len(source_dataloader.dataset) :],
    )
    prob_feature = gmm(1 - similarity)
    prob_logit = gmm(loss_vec)
    reliability_vec = prob_feature * prob_logit

    if args.novel_type:
        prob_gmm = gmm(reliability_vec)
        novel_index = prob_gmm > 0.5
        pred_vec[novel_index] = -1

    return feature_vec, pred_vec, reliability_vec


def save_result(
    feature_vec,
    pred_vec,
    reliability_vec,
    label_map,
    type_num,
    source_adata,
    target_adata,
    args,
):
    adata = sc.AnnData(feature_vec)
    adata.obs["Domain"] = np.concatenate(
        (source_adata.obs["Domain"], target_adata.obs["Domain"]), axis=0
    )
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, "Domain", theta=0.0, verbose=False)
    feature_vec = adata.obsm["X_pca_harmony"]

    source_adata.obsm["Embedding"] = feature_vec[: len(source_adata.obs["Domain"])]
    target_adata.obsm["Embedding"] = feature_vec[len(source_adata.obs["Domain"]) :]
    predictions = np.empty(len(target_adata.obs["Domain"]), dtype=np.dtype("U30"))
    for k in range(type_num):
        predictions[pred_vec == k] = label_map[k]
    if args.novel_type:
        predictions[pred_vec == -1] = "Novel (Most Unreliable)"
    target_adata.obs["Prediction"] = predictions
    target_adata.obs["Reliability"] = reliability_vec

    if args.umap_plot:
        sc.set_figure_params(figsize=(7, 7), dpi=300)
        try:
            target_annotation = target_adata.obs["CellType"]
        except:
            target_annotation = target_adata.obs["Prediction"]
        adata.obs["CellType"] = np.concatenate(
            (source_adata.obs["CellType"], target_annotation), axis=0
        )
        os.makedirs("figures/", exist_ok=True)
        print("Running UMAP visualization...")
        sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        sc.tl.umap(adata)
        sc.pl.umap(
            adata,
            color=["CellType"],
            palette=sns.color_palette(
                "husl", np.unique(adata.obs["CellType"].values).size
            ),
            save="_" + args.data_path[:-1] + "_CellType.png",
            show=False,
        )
        sc.pl.umap(
            adata,
            color=["Domain"],
            palette=sns.color_palette("hls", 2),
            save="_" + args.data_path[:-1] + "_Domain.png",
            show=False,
        )

        source_adata.obsm["X_umap"] = adata.obsm["X_umap"][
            : len(source_adata.obs["Domain"])
        ]
        target_adata.obsm["X_umap"] = adata.obsm["X_umap"][
            len(source_adata.obs["Domain"]) :
        ]

        sc.pl.umap(
            target_adata,
            color=["Reliability"],
            save="_" + args.data_path[:-1] + "_Reliability.png",
            show=False,
        )

    try:
        target_label_int = torch.from_numpy(
            (
                target_adata.obs["CellType"]
                .rank(method="dense", ascending=True)
                .astype(int)
                - 1
            ).values
        )
        evaluation = True
    except:
        print("No Target Cell Type Annotations Provided, Skip Evaluation")
        evaluation = False
    if evaluation and not args.novel_type:
        print("=======Evaluation=======")
        count = torch.unique(target_label_int, return_counts=True, sorted=True)[1]
        f1_weight = 1.0 / count
        f1_weight = f1_weight / f1_weight.sum()
        f1_weight = f1_weight.numpy()
        acc = accuracy_score(target_label_int, pred_vec)
        f1_acc = (f1_score(target_label_int, pred_vec, average=None) * f1_weight).sum()
        print("Label Transfer Accuracy: %.4f, F1-Score: %.4f" % (acc, f1_acc))
        if args.umap_plot:
            sil_type = silhouette_score(
                adata.obsm["X_umap"], adata.obs["CellType"].values
            )
            sil_omic = silhouette_score(
                adata.obsm["X_umap"], adata.obs["Domain"].values
            )
            sil_f1 = (
                2
                * (1 - (sil_omic + 1) / 2)
                * (sil_type + 1)
                / 2
                / (1 - (sil_omic + 1) / 2 + (sil_type + 1) / 2)
            )
            print(
                "Silhouette Score Type: %.4f, Omics: %.4f, Harmonized: %.4f"
                % (sil_type, sil_omic, sil_f1)
            )

    source_save_path = args.data_path + args.source_data[:-5] + "-integrated.h5ad"
    target_save_path = args.data_path + args.target_data[:-5] + "-integrated.h5ad"
    source_adata.write(source_save_path)
    target_adata.write(target_save_path)
    print(
        "Integration Results Saved to %s and %s" % (source_save_path, target_save_path)
    )
