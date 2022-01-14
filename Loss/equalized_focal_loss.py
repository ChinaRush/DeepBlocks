# import from third library

from torch.nn.modules.loss import _Loss


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs
