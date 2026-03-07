import argparse
import logging
from pathlib import Path
import torch

from data.dataset import create_dataloaders
from models.losses import MultiTaskLoss
from models.model import SvamitvaModel
from training.config import TrainingConfig
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("epoch1_train")

def main():
    parser = argparse.ArgumentParser(description="Train for 1 epoch and save checkpoint.")
    parser.add_argument("--val_dir", default=None, help="Validation directory")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", choices=["sam2", "resnet50"], default="sam2")
    parser.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    args = parser.parse_args()

    # Hardcode MAP1 for training
    train_dirs = ["/Users/aaronr/Desktop/DATA/MAP1"]

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    config = TrainingConfig(
        train_dirs=train_dirs,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_epochs=1,
        learning_rate=1e-3,
        tile_size=args.tile_size,
        num_workers=args.num_workers,
        seed=args.seed,
        backbone=args.backbone,
        sam2_checkpoint=args.sam2_checkpoint,
        freeze_encoder=False,
        freeze_backbone_epochs=0,
        fpn_channels=256,
        checkpoint_dir=args.checkpoint_dir,
        force_cpu=False,
        experiment_name="epoch1",
        use_wandb=False,
    )

    if config.backbone == "sam2":
        ckpt_path = Path(config.sam2_checkpoint)
        if not ckpt_path.exists():
            import urllib.request
            SAM2_URL = (
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
            )
            logger.info(f"Downloading SAM2 checkpoint to {ckpt_path}...")
            urllib.request.urlretrieve(SAM2_URL, str(ckpt_path))

    logger.info("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=config.train_dirs,
        val_dir=config.val_dir,
        image_size=config.tile_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
    )

    logger.info("Building model...")
    model = SvamitvaModel(
        backbone=config.backbone,
        sam2_checkpoint=str(config.sam2_checkpoint),
        sam2_model_cfg=config.sam2_model_cfg,
        pretrained=config.pretrained,
        freeze_encoder=False,
        num_roof_classes=config.num_roof_classes,
        fpn_channels=config.fpn_channels,
        dropout=config.dropout,
    ).to(device)

    loss_fn = MultiTaskLoss(weights=config.loss_weights)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)

    logger.info("Starting training for 1 epoch...")
    best_loss = float('inf')
    best_state = None
    mask_keys = [
        "building_mask", "road_mask", "road_centerline_mask", "waterbody_mask",
        "waterbody_line_mask", "waterbody_point_mask", "utility_point_mask",
        "utility_line_mask", "bridge_mask", "railway_mask", "roof_type_mask"
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for batch_idx, batch in enumerate(train_loader):
        model.train()
        inputs = batch["image"].to(device)
        targets = {k: batch[k].to(device) for k in mask_keys if k in batch}
        optimizer.zero_grad()
        loss_tuple = loss_fn(model(inputs), targets)
        loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
        loss.backward()
        optimizer.step()
        logger.info(f"Batch {batch_idx}: loss={loss.item():.4f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()

    # Save best checkpoint after epoch 1
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "epoch1.pt"
    torch.save({
        "model_state_dict": best_state,
        "epoch": 1,
        "best_loss": best_loss,
    }, checkpoint_path)
    logger.info(f"Best checkpoint saved to {checkpoint_path} with loss {best_loss:.4f}")

    # Save best checkpoint after epoch 1
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "epoch1.pt"
    torch.save({
        "model_state_dict": best_state,
        "epoch": 1,
        "best_loss": best_loss,
    }, checkpoint_path)
    logger.info(f"Best checkpoint saved to {checkpoint_path} with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()