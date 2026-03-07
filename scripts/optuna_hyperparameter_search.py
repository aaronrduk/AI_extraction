def objective(trial):
"""
Detailed Optuna hyperparameter search for SVAMITVA model training.

This script tunes key hyperparameters (learning rate, batch size, loss weights, etc.)
and connects directly to your project files: config, trainer, model, loss, and dataset.
Results are logged and saved for reproducibility.
"""

import optuna
import argparse
import logging
import os
from pathlib import Path
import torch
from training.config import TrainingConfig
from training.trainer import Trainer
from models.model import SvamitvaModel
from models.losses import MultiTaskLoss
from data.dataset import SVAMITVADataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("optuna_search")

def get_dataloaders(data_dir, image_size, batch_size):
    """Create train and val DataLoaders from SVAMITVADataset."""
    train_dataset = SVAMITVADataset(root_dir=data_dir, image_size=image_size, split="train")
    val_dataset = SVAMITVADataset(root_dir=data_dir, image_size=image_size, split="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def objective(trial):
    """
    Optuna objective function: builds config, model, loss, trainer, and returns best metric.
    All components are imported from project files.
    """
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    roof_weight = trial.suggest_uniform("roof_type_weight", 0.1, 1.0)
    patience = trial.suggest_int("patience", 3, 10)

    # Update config (connected to training/config.py)
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        loss_weights={
            **TrainingConfig().loss_weights,
            "roof_type": roof_weight,
        },
        num_epochs=args.epochs,
        early_stopping=True,
        patience=patience,
        use_wandb=False,
        train_dirs=[str(args.data_dir)],
        val_dir=str(args.data_dir),
        log_dir=args.log_dir,
        experiment_name="optuna_trial",
    )

    # Prepare data/model/loss (connected to data/dataset.py, models/model.py, models/losses.py)
    train_loader, val_loader = get_dataloaders(args.data_dir, config.tile_size, config.batch_size)
    model = SvamitvaModel(config)
    loss_fn = MultiTaskLoss(weights=config.loss_weights, num_roof_classes=config.num_roof_classes)

    # Trainer (connected to training/trainer.py)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
    trainer.fit()

    # Return best validation metric (e.g., avg_iou)
    best_metric = trainer.ckpt_mgr.best_score
    logger.info(f"Trial complete: lr={lr:.2e}, batch={batch_size}, wd={weight_decay:.2e}, roof_w={roof_weight:.2f}, patience={patience}, best_metric={best_metric:.4f}")
    return best_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for SVAMITVA model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Path to dataset root directory")
    parser.add_argument("--log_dir", type=Path, default=Path("logs/optuna"), help="Directory for Optuna logs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger.info(f"Optuna search starting: {args.trials} trials, {args.epochs} epochs per trial.")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    logger.info(f"Best trial: {study.best_trial}")
    # Save study results
    results_path = args.log_dir / "optuna_results.txt"
    with open(results_path, "w") as f:
        f.write(str(study.best_trial))
    logger.info(f"Optuna results saved to {results_path}")
