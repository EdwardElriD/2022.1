import torch
import torch.nn as nn


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def _forward(x, block):
    return block(x)


if __name__ =="__main__":
    Y1 = _forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = _forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    Y = concat_preds([Y1, Y2])
    print(Y1.shape)
    print(Y2.shape)
    print(Y.shape)