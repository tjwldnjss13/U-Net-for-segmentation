import torch


def custom_cross_entropy_loss(output, target):
    _, n, h, w = output.shape
    losses = -1 * (target * torch.log2(output + 1e-12) + (1 - target) * torch.log2(1 - output + 1e-12))
    loss = losses.sum() / (n * h * w)

    return loss