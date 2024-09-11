import torch
import torch.nn as nn


class WeightedCrossEntropy(object):
    def __init__(self, neg_wt, device):
        neg_wt = torch.FloatTensor([neg_wt, 1]).to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(weight=neg_wt, reduction='sum')

    def computer_loss(self, pred, true):
        # get preds with 1 dim, turn to 2 dim
        pred_ = torch.cat([-pred, pred], axis=1).to(self.device)
        loss = self.loss_fn(pred_, true) / len(true)
        return loss
