import torch
from .grad_rev import RevGrad
from torch import nn
from torch.nn import functional as F


class AdversarialClassifierHead(nn.Module):
    def __init__(self, feat_dim, attr_dim, adv_objective, adv_grad_rev_strength, hidden_layer_num=1):
        super().__init__()
        input_dim = feat_dim

        mlp = []
        for i in range(hidden_layer_num):
            mlp.append(nn.Linear(input_dim, input_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(input_dim, attr_dim))
        self.mlp = nn.Sequential(*mlp)
        self.adv_grad_rev_strength = adv_grad_rev_strength
        self.rev_grad = RevGrad(self.adv_grad_rev_strength)
        self.adv_objective = adv_objective

    def forward(self, hidden, rev_grad):
        if rev_grad and self.adv_objective != 'adv_uni':
            hidden = self.rev_grad(hidden)
        pred = self.mlp(hidden)
        return pred

    def compute_loss(self, *args, **kwargs):
        if self.adv_objective == 'eq_odds_laftr': # LAFTR, eq odds objective
            return self.compute_loss_equal_odds(*args, **kwargs)
        elif self.adv_objective == 'eq_odds_ce':
            return self.compute_loss_ce_equal_odds(*args, **kwargs, weighted=True)
        elif self.adv_objective == 'adv_ce':
            return self.compute_loss_ce_equal_odds(*args, **kwargs, weighted=False)
        elif self.adv_objective == 'adv_uni':
            return self.compute_loss_uni(*args, **kwargs,)
        else:
            raise NotImplementedError(self.adv_objective)

    def get_weights(self, label_stats, attr_gt, label_gt, attr_pred):
        weights = []
        label_weights = 1 / (label_stats + 1e-10)
        normalized_label_weights = label_weights / label_weights.sum()
        attr_gt_list, label_gt_list = attr_gt.cpu().tolist(), label_gt.cpu().tolist()
        for label, attr in zip(label_gt_list, attr_gt_list):
            weights.append(normalized_label_weights[0 if label == 0 else 1, attr])
        weights = torch.FloatTensor(weights).to(attr_pred.device)
        return weights

    def compute_loss_equal_odds(self, attr_pred, attr_gt, label_gt, label_stats):
        assert (attr_pred.size(1) == 2)
        weights = self.get_weights(label_stats, attr_gt, label_gt, attr_pred)
        attr_prob = F.softmax(attr_pred, -1) # [B,C=2]
        loss = torch.abs(attr_gt - attr_prob[:,1]) # [B]
        weighted_loss = loss * weights
        return weighted_loss.mean()

    def compute_loss_ce_equal_odds(self, attr_pred, attr_gt, label_gt, label_stats, weighted):
        assert attr_pred.size(1) == 2
        if weighted:
            weights = self.get_weights(label_stats, attr_gt, label_gt, attr_pred)
            loss = F.cross_entropy(attr_pred, attr_gt, reduction='none', ignore_index=-1)
            weighted_loss = loss * weights
        else:
            weighted_loss = F.cross_entropy(attr_pred, attr_gt, reduction='none', ignore_index=-1)
        return weighted_loss.mean()


