import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


class Net(nn.Module):
    def __init__(self, gene_num, type_num, ce_weight, args):
        super(Net, self).__init__()
        self.type_num = type_num
        self.ce_weight = ce_weight
        self.align_loss_epoch = args.align_loss_epoch
        self.encoder = nn.Sequential(
            nn.Linear(gene_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.InstanceNorm1d(64),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, type_num),
        )
        self.adj_decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def run(
        self,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        target_adj,
        args,
    ):
        optim = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        wce_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        align_loss = AlignLoss(type_num=self.type_num, feature_dim=64, args=args)
        epochs = args.train_epoch
        target_iter = iter(target_dataloader_train)
        for epoch in range(epochs):
            wce_loss_epoch = align_loss_epoch = stc_loss_epoch = 0.0
            train_acc = train_tot = 0.0
            self.train()
            for (source_x, source_y) in source_dataloader_train:
                source_x = source_x.cuda()
                source_y = source_y.cuda()
                try:
                    (target_x, adj_index), target_index = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_dataloader_train)
                    (target_x, adj_index), target_index = next(target_iter)
                target_x = target_x.cuda()

                source_h = self.encoder(source_x)
                source_pred = self.classifier(source_h)
                target_h = self.encoder(target_x)

                loss_wce = wce_loss(source_pred, source_y)

                wce_loss_epoch += loss_wce.item()
                train_acc += (
                    torch.argmax(
                        source_pred,
                        dim=-1,
                    )
                    == source_y
                ).sum()
                train_tot += source_x.shape[0]

                loss_epoch = loss_wce

                if epoch >= self.align_loss_epoch:
                    loss_align = align_loss(
                        source_h,
                        source_y,
                        target_h,
                        preds[target_index],
                        prob_feature[target_index] * prob_logit[target_index],
                    )
                    loss_epoch += loss_align
                    align_loss_epoch += loss_align.item()

                if args.novel_type:
                    adj = target_adj[adj_index, :][:, adj_index]
                    cos_sim_x = torch.from_numpy(adj).float().cuda()
                    target_h = F.normalize(self.adj_decoder(target_h), dim=-1)
                    cos_sim_h = F.relu(target_h @ target_h.T)
                    stc_loss = (cos_sim_x - cos_sim_h) * (cos_sim_x - cos_sim_h)
                    stc_loss = torch.clamp(stc_loss - 0.01, min=0).mean()
                    loss_epoch += stc_loss
                    stc_loss_epoch += stc_loss.item()

                optim.zero_grad()
                loss_epoch.backward()
                optim.step()

            train_acc /= train_tot
            wce_loss_epoch /= len(source_dataloader_train)
            align_loss_epoch /= len(source_dataloader_train)
            stc_loss_epoch /= len(source_dataloader_train)

            feature_vec, type_vec, omic_vec, loss_vec = self.inference(
                source_dataloader_eval, target_dataloader_eval
            )
            similarity, preds = feature_prototype_similarity(
                feature_vec[omic_vec == 0],
                type_vec,
                feature_vec[omic_vec == 1],
            )
            if epoch == self.align_loss_epoch - 1:
                align_loss.init_prototypes(
                    feature_vec[omic_vec == 0],
                    type_vec,
                    feature_vec[omic_vec == 1],
                    preds,
                )
            prob_feature = gmm(1 - similarity)
            prob_logit = gmm(loss_vec)

            preds = torch.from_numpy(preds).long().cuda()
            prob_feature = torch.from_numpy(prob_feature).float().cuda()
            prob_logit = torch.from_numpy(prob_logit).float().cuda()

            if args.novel_type:
                print(
                    "Epoch [%d/%d] WCE Loss: %.4f, ALG Loss: %.4f, STC Loss: %.4f, Train ACC: %.4f"
                    % (
                        epoch,
                        epochs,
                        wce_loss_epoch,
                        align_loss_epoch,
                        stc_loss_epoch,
                        train_acc,
                    )
                )
            else:
                print(
                    "Epoch [%d/%d] WCE Loss: %.4f, ALG Loss: %.4f, Train ACC: %.4f"
                    % (epoch, epochs, wce_loss_epoch, align_loss_epoch, train_acc)
                )

            if train_acc > args.early_stop_acc:
                print("Early Stop.")
                break
        return preds.cpu(), prob_feature.cpu(), prob_logit.cpu()

    def inference(self, source_dataloader, target_dataloader):
        self.eval()
        feature_vec, type_vec, omic_vec, loss_vec = [], [], [], []
        for (x, y) in source_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
            feature_vec.extend(h.cpu().numpy())
            type_vec.extend(y.numpy())
            omic_vec.extend(np.zeros(x.shape[0]))
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        for (x, _), _ in target_dataloader:
            x = x.cuda()
            with torch.no_grad():
                h = self.encoder(x)
                logit = self.classifier(h)
                pred = torch.argmax(logit, dim=-1)
                loss = ce_loss(logit, pred)
            feature_vec.extend(h.cpu().numpy())
            omic_vec.extend(np.ones(x.shape[0]))
            loss_vec.extend(loss.cpu().numpy())
        feature_vec, type_vec, omic_vec, loss_vec = (
            np.array(feature_vec),
            np.array(type_vec),
            np.array(omic_vec),
            np.array(loss_vec),
        )
        return feature_vec, type_vec, omic_vec, loss_vec


def gmm(X):
    X = ((X - X.min()) / (X.max() - X.min())).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4).fit(X)
    prob = gmm.predict_proba(X)[:, gmm.means_.argmin()]
    return prob


def feature_prototype_similarity(source_feature, source_label, target_feature):
    type_num = source_label.max() + 1
    source_prototypes = np.zeros((type_num, source_feature.shape[1])).astype(float)
    for k in range(type_num):
        source_prototypes[k] = source_feature[source_label == k].sum(axis=0)
    similarity = cosine_similarity(target_feature, source_prototypes)
    pred = np.argmax(similarity, axis=1)
    similarity = np.max(similarity, axis=1)
    return similarity, pred


class AlignLoss(nn.Module):
    def __init__(self, type_num, feature_dim, args):
        super(AlignLoss, self).__init__()
        self.type_num = type_num
        self.feature_dim = feature_dim
        self.source_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.target_prototypes = torch.zeros(self.type_num, self.feature_dim).cuda()
        self.momentum = args.prototype_momentum
        self.criterion = nn.MSELoss()

    def init_prototypes(
        self, source_feature, source_label, target_feature, target_prediction
    ):
        source_feature = torch.from_numpy(source_feature).cuda()
        source_label = torch.from_numpy(source_label).cuda()
        target_feature = torch.from_numpy(target_feature).cuda()
        target_prediction = torch.from_numpy(target_prediction).cuda()
        for k in range(self.type_num):
            self.source_prototypes[k] = source_feature[source_label == k].mean(dim=0)
            target_index = target_prediction == k
            if target_index.sum() != 0:
                self.target_prototypes[k] = target_feature[target_index].mean(dim=0)

    def forward(
        self,
        source_feature,
        source_label,
        target_feature,
        target_prediction,
        target_reliability,
    ):
        self.source_prototypes.detach_()
        self.target_prototypes.detach_()
        for k in range(self.type_num):
            source_index = source_label == k
            if source_index.sum() != 0:
                self.source_prototypes[k] = self.momentum * self.source_prototypes[
                    k
                ] + (1 - self.momentum) * source_feature[source_label == k].mean(dim=0)
            target_index = target_prediction == k
            if target_index.sum() != 0:
                if torch.abs(self.target_prototypes[k]).sum() > 1e-7:
                    self.target_prototypes[k] = self.momentum * self.target_prototypes[
                        k
                    ] + (1 - self.momentum) * (
                        target_reliability[target_index].unsqueeze(1)
                        * target_feature[target_index]
                    ).mean(
                        dim=0
                    )
                else:  # Not Initialized
                    self.target_prototypes[k] = (
                        target_reliability[target_index].unsqueeze(1)
                        * target_feature[target_index]
                    ).mean(dim=0)
        loss = self.criterion(
            F.normalize(self.source_prototypes, dim=-1),
            F.normalize(self.target_prototypes, dim=-1),
        )
        # In the absence of some prototypes
        if (torch.abs(self.target_prototypes).sum(dim=1) > 1e-7).sum() < self.type_num:
            loss *= 0
        return loss
