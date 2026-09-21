import torch
import torch.nn as nn

class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), value)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        smooth_one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        smooth_one_hot += smooth_factor / length
        return smooth_one_hot

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(x.size(0), target.size(0)))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(-x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)