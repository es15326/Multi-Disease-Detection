# nets.py

from typing import Dict, Literal, Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import (AUROC, Accuracy, AveragePrecision, F1Score,
                          Precision, Recall)
from torchvision import models
from transformers import CLIPVisionConfig, CLIPVisionModel


class BaseNet(pl.LightningModule):
    """
    A base LightningModule for multi-label image classification.

    This class provides the core training, validation, and metric-logging
    functionality. Specific model architectures should inherit from this class.
    """

    def __init__(
        self,
        n_classes: int = 29,
        label_weights: List[float] = None,
        freeze_backbone: bool = False,
        lr: float = 7e-6,
        warm_restarts: int = 200,
    ):
        super().__init__()
        # Save hyperparameters like learning rate for logging
        self.save_hyperparameters()

        # Define metrics for validation
        metrics = {
            "accuracy": Accuracy("multilabel", num_labels=n_classes, threshold=0.5),
            "precision": Precision("multilabel", num_labels=n_classes, threshold=0.5),
            "recall": Recall("multilabel", num_labels=n_classes, threshold=0.5),
            "f1": F1Score("multilabel", num_labels=n_classes, threshold=0.5),
        }
        self.val_metrics = nn.ModuleDict(metrics)

        # Metrics for the final S-Score calculation
        self.auc_class_0 = AUROC("binary")
        self.average_precision = AveragePrecision("multilabel", num_labels=n_classes - 1)
        self.mean_auc = AUROC("multilabel", num_labels=n_classes - 1)
        
        # Set position weights for handling class imbalance in the loss function
        if label_weights is None:
            self.pos_weights = torch.ones(n_classes)
        else:
            self.pos_weights = torch.tensor(label_weights, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        
        # Calculate weighted binary cross-entropy loss
        loss = binary_cross_entropy_with_logits(
            y_pred.squeeze(), y.float(), pos_weight=self.pos_weights.to(y.device)
        )
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_pred = self(x)
        
        # Calculate validation loss and return predictions for aggregation
        loss = binary_cross_entropy_with_logits(
            y_pred.squeeze(), y.float(), pos_weight=self.pos_weights.to(y.device)
        )
        
        # Update metrics
        y_pred_prob = torch.sigmoid(y_pred)
        self.val_metrics["accuracy"](y_pred_prob, y)
        self.val_metrics["precision"](y_pred_prob, y)
        self.val_metrics["recall"](y_pred_prob, y)
        self.val_metrics["f1"](y_pred_prob, y)
        self.auc_class_0(y_pred_prob[:, 0], y[:, 0])
        self.average_precision(y_pred_prob[:, 1:], y[:, 1:].int())
        self.mean_auc(y_pred_prob[:, 1:], y[:, 1:].int())

        return {"loss": loss, "preds": y_pred, "targets": y}

    def on_validation_epoch_end(self):
        # Compute and log standard validation metrics
        for name, metric in self.val_metrics.items():
            self.log(f"val_{name}", metric.compute())
            metric.reset() # Reset for the next epoch

        # Compute and log S-Score components
        auc_class_0 = self.auc_class_0.compute()
        avg_precision = self.average_precision.compute()
        mean_auc = self.mean_auc.compute()

        # Calculate the final S-Score
        s_score = 0.5 * auc_class_0 + 0.25 * avg_precision + 0.25 * mean_auc
        
        self.log("val_auc_class_0", auc_class_0)
        self.log("val_mAP", avg_precision)
        self.log("val_mean_auc", mean_auc)
        self.log("val_s_score", s_score, prog_bar=True)

        # Reset S-Score metrics
        self.auc_class_0.reset()
        self.average_precision.reset()
        self.mean_auc.reset()

    @property
    def classification_layer(self) -> nn.Module:
        """Returns the final classification layer of the model."""
        raise NotImplementedError("Subclasses must implement this property.")

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        # If backbone is frozen, only optimize the classification layer
        if self.hparams.freeze_backbone:
            params = self.classification_layer.parameters()
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.hparams.warm_restarts
        )
        return [optimizer], [scheduler]


class ResNext(BaseNet):
    def __init__(self, pretrained: bool = True, n_classes: int = 29, **kwargs):
        super().__init__(n_classes=n_classes, **kwargs)
        self.model = models.resnext101_32x8d(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    @property
    def classification_layer(self) -> nn.Module:
        return self.model.fc


class EfficientNet(BaseNet):
    def __init__(self, pretrained: bool = True, n_classes: int = 29, **kwargs):
        super().__init__(n_classes=n_classes, **kwargs)
        weights = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b7(weights=weights)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, n_classes),
        )
    
    @property
    def classification_layer(self) -> nn.Module:
        return self.model.classifier


class ViTBase(BaseNet):
    def __init__(self, pretrained: bool = True, n_classes: int = 29, **kwargs):
        super().__init__(n_classes=n_classes, **kwargs)
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_32(weights=weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, n_classes)

    @property
    def classification_layer(self) -> nn.Module:
        return self.model.heads.head


# Note: Other ViT variants (Large, Huge) would follow the ViTBase pattern.


class ClipViTLarge(BaseNet):
    def __init__(self, pretrained: bool = True, n_classes: int = 29, **kwargs):
        super().__init__(n_classes=n_classes, **kwargs)
        # Using a pre-trained CLIP vision model as the backbone
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        if self.hparams.freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Define a new classification head
        d_model = self.vision_model.config.hidden_size
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through the vision model and then the classification head
        outputs = self.vision_model(pixel_values=x)
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
    
    @property
    def classification_layer(self) -> nn.Module:
        return self.fc


class DinoBaseNet(BaseNet):
    """A base class for all DINO Vision Transformer models to reduce code duplication."""
    def __init__(self, model_name: str, n_classes: int = 29, **kwargs):
        super().__init__(n_classes=n_classes, **kwargs)
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.fc = nn.Linear(self.model.embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINO models return a dictionary, we take the class token output
        return self.fc(self.model(x))

    @property
    def classification_layer(self) -> nn.Module:
        return self.fc

# DINO model variants inheriting from the base class
class ViTS16(DinoBaseNet):
    def __init__(self, **kwargs):
        super().__init__(model_name="dino_vits16", **kwargs)

class ViTS8(DinoBaseNet):
    def __init__(self, **kwargs):
        super().__init__(model_name="dino_vits8", **kwargs)

class ViTB16(DinoBaseNet):
    def __init__(self, **kwargs):
        super().__init__(model_name="dino_vitb16", **kwargs)

class ViTB8(DinoBaseNet):
    def __init__(self, **kwargs):
        super().__init__(model_name="dino_vitb8", **kwargs)


ARCHS: Dict[str, BaseNet] = {
    "vit-b": ViTBase,
    "resnext": ResNext,
    "clip-l": ClipViTLarge,
    "efficientnet": EfficientNet,
    "vits16": ViTS16,
    "vits8": ViTS8,
    "vitb16": ViTB16,
    "vitb8": ViTB8,
}

def load_model(arch: str, *args, **kwargs) -> BaseNet:
    """Factory function to load a model by its architecture name."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(ARCHS.keys())}")
    return ARCHS[arch](*args, **kwargs)
