# train.py

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

import data  # Assumes data.py is in the same directory
import nets


def get_class_weights(weight_path: str, n_classes: int) -> torch.Tensor:
    """Loads class weights from a CSV file or returns uniform weights."""
    if not os.path.isfile(weight_path):
        print(f"Weight file not found at {weight_path}. Using uniform weights.")
        return torch.ones(n_classes, dtype=torch.float32)
    
    print(f"Loading weights from {weight_path}")
    weights = pd.read_csv(weight_path).to_numpy().squeeze()
    weights = torch.tensor(weights, dtype=torch.float32)

    # Normalize weights for the non-background classes
    if len(weights) > 1:
        fg_weights = weights[1:]
        weights[1:] = (fg_weights / fg_weights.sum()) * (len(fg_weights))
    return weights


def calculate_new_weights(trainer: pl.Trainer, model: pl.LightningModule, valloader: DataLoader) -> np.ndarray:
    """Calculates new class weights based on model performance."""
    print("Evaluating model to calculate new class weights...")
    
    # Use trainer.validate to get predictions
    results = trainer.validate(model, dataloaders=valloader, verbose=False)
    
    # The validate method runs the validation_step, but we need the raw predictions
    # which we returned from validation_step. We can get them from the trainer's state
    # or re-run prediction. For simplicity, we re-run prediction here.
    predictions = trainer.predict(model, dataloaders=valloader)
    
    all_preds = torch.cat([p["preds"] for p in predictions], dim=0)
    all_targets = torch.cat([p["targets"] for p in predictions], dim=0)
    
    all_preds_prob = torch.sigmoid(all_preds).numpy()
    all_targets_np = all_targets.numpy()
    
    new_weights = []
    for i in range(all_targets_np.shape[1]):
        ap = average_precision_score(all_targets_np[:, i], all_preds_prob[:, i])
        new_weights.append(1.0 - ap)
        
    new_weights = np.array(new_weights)
    new_weights = new_weights.clip(0.1, 0.9)
    new_weights[0] = 1.0  # Keep weight for the 'no finding' class at 1.0
    
    return new_weights


def main(args):
    """Main training script for the boosting experts procedure."""
    torch.manual_seed(63)
    os.makedirs(args.savedir, exist_ok=True)
    
    # --- Data Loading ---
    norm_strategy = 'clip' if 'clip' in args.arch else 'imagenet'
    trainset = data.RIADDDataset(
        csv_path=args.train_csv, img_path=args.train_img_path, 
        input_size=args.resolution, normalization=norm_strategy
    )
    valset = data.RIADDDataset(
        csv_path=args.val_csv, img_path=args.val_img_path,
        input_size=args.resolution, normalization=norm_strategy, testing=True
    )
    
    num_workers = min(os.cpu_count(), args.batch_size, 8)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # --- Boosting Loop ---
    for n in range(args.boosting_resume_from, args.boosting_experts):
        print(f"\n--- Training Expert {n+1}/{args.boosting_experts} ---")
        
        # 1. Load weights for the current expert
        weight_path = os.path.join(args.savedir, f"weights_expert_{n:02d}.csv")
        class_weights = get_class_weights(weight_path, n_classes=trainset.N_CLASSES)
        
        # 2. Initialize Model and Trainer
        model = nets.load_model(
            args.arch,
            n_classes=trainset.N_CLASSES,
            label_weights=class_weights.tolist(),
            freeze_backbone=args.freeze_backbone,
            lr=args.lr
        )
        
        project_id = f"boosting-{args.arch}-{args.tag}"
        wandb_logger = WandbLogger(project=project_id, name=f"expert_{n:02d}")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_s_score",
            dirpath=os.path.join(args.savedir, "checkpoints"),
            filename=f"{args.arch}-expert_{n:02d}-{{epoch:03d}}-{{val_s_score:.4f}}",
            save_top_k=1,
            mode="max",
        )
        early_stop_callback = EarlyStopping(monitor="val_s_score", patience=20, mode="max")
        
        trainer = pl.Trainer(
            devices=args.devices,
            accelerator="gpu",
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            max_epochs=args.max_epochs,
            log_every_n_steps=20,
        )
        
        # 3. Train the model
        trainer.fit(model, trainloader, val_dataloaders=valloader)
        
        # 4. Calculate and save weights for the NEXT expert
        print("Finished training. Estimating weights for the next expert...")
        best_model = nets.load_model.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        
        next_weights = calculate_new_weights(trainer, best_model, valloader)
        
        # Save weights for the next iteration
        next_weight_path = os.path.join(args.savedir, f"weights_expert_{n+1:02d}.csv")
        pd.DataFrame(next_weights).to_csv(next_weight_path, index=False)
        print(f"Saved new weights to {next_weight_path}")
        
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence of expert models with dynamic weighting.")
    
    # Paths
    parser.add_argument("--train_img_path", type=str, required=True, help="Path to training images.")
    parser.add_argument("--val_img_path", type=str, required=True, help="Path to validation images.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training labels CSV.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation labels CSV.")
    parser.add_argument("--savedir", type=str, default="./models", help="Directory to save weights and checkpoints.")

    # Training Hyperparameters
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs per expert.")
    parser.add_argument("--lr", type=float, default=7e-6, help="Learning rate.")
    parser.add_argument("--arch", type=str, default="resnext", choices=list(nets.ARCHS.keys()), help="Model architecture.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and only train the classifier.")

    # Boosting Parameters
    parser.add_argument("--boosting_experts", type=int, default=10, help="Total number of experts to train.")
    parser.add_argument("--boosting_resume_from", type=int, default=0, help="Expert index to resume training from.")

    # System
    parser.add_argument("--devices", type=int, nargs='+', default=[0], help="GPU devices to use.")
    parser.add_argument("--tag", type=str, default="v2", help="A tag for this experiment run.")
    
    args = parser.parse_args()
    main(args)
