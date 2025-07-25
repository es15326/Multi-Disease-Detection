import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
  loss = (inputs - targets) ** 2
  if weights is not None:
    loss *= weights
  loss = torch.mean(loss)
  return loss

def weighted_BCEWithLogitsLoss(inputs, targets, weights=None):
  criterion = nn.BCEWithLogitsLoss(
      pos_weight=10 * torch.ones(inputs.shape[-1]))
  loss = weights * criterion(inputs, targets)
  loss = torch.mean(loss)
  return loss
