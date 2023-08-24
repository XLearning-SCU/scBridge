import torch
import random
import argparse
import numpy as np
from copy import deepcopy
from model_utils import Net
from eval_utils import infer_result, save_result
from data_utils import prepare_dataloader, partition_data, adjacency


def main(args):
    (
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
    ) = prepare_dataloader(args)

    source_dataloader_eval_all = deepcopy(source_dataloader_eval)
    target_dataloader_eval_all = deepcopy(target_dataloader_eval)
    if args.novel_type:
        target_adj = adjacency(target_dataset.tensors[0])
    else:
        target_adj = None

    source_label = source_dataset.tensors[1]
    count = torch.unique(source_label, return_counts=True, sorted=True)[1]
    ce_weight = 1.0 / count
    ce_weight = ce_weight / ce_weight.sum() * type_num
    ce_weight = ce_weight.cuda()

    print("======= Training Start =======")

    net = Net(gene_num, type_num, ce_weight, args).cuda()
    preds, prob_feat, prob_logit = net.run(
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        target_adj,
        args,
    )

    for iter in range(args.max_iteration):
        (
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            source_dataset,
            target_dataset,
        ) = partition_data(
            preds,
            prob_feat,
            prob_logit,
            source_dataset,
            target_dataset,
            args,
        )

        # Iteration convergence check
        if target_dataset.__len__() <= args.batch_size:
            break
        print("======= Iteration:", iter, "=======")

        source_label = source_dataset.tensors[1]
        count = torch.unique(source_label, return_counts=True, sorted=True)[1]
        ce_weight = 1.0 / count
        ce_weight = ce_weight / ce_weight.sum() * type_num
        ce_weight = ce_weight.cuda()

        net = Net(gene_num, type_num, ce_weight, args).cuda()
        preds, prob_feat, prob_logit = net.run(
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            target_adj,
            args,
        )
    print("======= Training Done =======")

    features, predictions, reliabilities = infer_result(
        net, source_dataloader_eval_all, target_dataloader_eval_all, args
    )
    save_result(
        features,
        predictions,
        reliabilities,
        label_map,
        type_num,
        source_adata,
        target_adata,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--source_data", type=str)
    parser.add_argument("--target_data", type=str)
    parser.add_argument("--source_preprocess", type=str, default="Standard")
    parser.add_argument("--target_preprocess", type=str, default="TFIDF")
    # Model configs
    parser.add_argument("--reliability_threshold", default=0.95, type=float)
    parser.add_argument("--align_loss_epoch", default=1, type=float)
    parser.add_argument("--prototype_momentum", default=0.9, type=float)
    parser.add_argument("--early_stop_acc", default=0.99, type=float)
    parser.add_argument("--max_iteration", default=20, type=int)
    parser.add_argument("--novel_type", action="store_true")
    # Training configs
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--train_epoch", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--random_seed", default=2023, type=int)
    # Evaluation configs
    parser.add_argument("--umap_plot", action="store_true")

    args = parser.parse_args()

    # Randomization
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    main(args)
