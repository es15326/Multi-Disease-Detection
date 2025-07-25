# base.py

import torch.nn as nn

class BaseNet(nn.Module):
    """
    Abstract base model with common interface for all models.
    """
    def __init__(self, n_classes: int = 29, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = {}

    def log_metric(self, name, value):
        self.metrics[name] = value

    def forward(self, x):
        raise NotImplementedError("Subclasses must override the forward() method.")

