"""Main training script for MFVLR.

Usage:
    python train.py --config configs/mfvlr.yaml
    python train.py --config configs/mfvlr.yaml --dry-run
    python train.py --config configs/mfvlr.yaml --resume checkpoints/checkpoint_epoch_5.pt
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mfvlr import MFVLR
from models.losses import MFVLRLoss
from datasets.dummy_dataset import DummyMFVLRDataset
from datasets.manifest_dataset import MFVLRDataset
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.trainer import create_optimizer, create_scheduler, train_step, train_one_epoch
from utils.evaluator import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint, create_checkpoint_state


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFVLR model on face forgery detection and localization.")
    parser.add_argument("--config", type=str, default="configs/mfvlr.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 training step for smoke testing and exit.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision for faster training.")
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Precision mode ('bf16', 'fp16', 'fp32'). Defaults to bf16 if supported, else fp16 when --amp is set.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples for fast experimentation.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints and logs.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(dataset_cfg: Dict[str, Any], split: str = "train"):
    """Build dataset from manifest if present, else fallback to DummyMFVLRDataset."""
    manifest_key = f"{split}_manifest"
    manifest_path = dataset_cfg.get(manifest_key)
    dataset_root = dataset_cfg.get("root")
    real_class_idx = dataset_cfg.get("real_class_index", 0)
    fake_class_idx = dataset_cfg.get("fake_class_index", 1)

    if manifest_path and os.path.exists(manifest_path):
        return MFVLRDataset(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            real_class_index=real_class_idx,
            fake_class_index=fake_class_idx,
        )
    else:
        # Synthetic fallback for dry-run and development without real data
        return DummyMFVLRDataset(
            num_samples=dataset_cfg.get("dummy_samples", 16 if split == "train" else 8),
            seed=42 if split == "train" else 123,
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Config overrides from CLI
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["lr"] = args.lr
    if args.output_dir is not None:
        config.setdefault("training", {})["output_dir"] = args.output_dir

    if args.amp:
        config.setdefault("training", {})["amp"] = True
    if args.precision is not None:
        config.setdefault("training", {})["precision"] = args.precision
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers

    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    # 2. Setup environment, seed, and device
    seed = config.get("seed", 42)
    set_seed(seed)

    device_str = args.device or train_cfg.get("device")
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = train_cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("MFVLR_Train", log_dir=output_dir)
    logger.info(f"Using device: {device}")

    # 3. Build Datasets and DataLoaders
    train_dataset = build_dataset(dataset_cfg, split="train")
    val_dataset = build_dataset(dataset_cfg, split="val")

    if args.max_samples and len(train_dataset) > args.max_samples:
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.max_samples))
        logger.info(f"Subsampled train dataset to {len(train_dataset)} samples for fast training.")

    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # 4. Build Model and Loss Module
    model = MFVLR().to(device)
    loss_cfg = config.get("loss", {})
    cmc_cfg = config.get("cmc", {})
    fd_weights = loss_cfg.get("fd_class_weights")
    if fd_weights is not None and isinstance(fd_weights, list):
        fd_weights = torch.tensor(fd_weights, dtype=torch.float32)

    cmc_l2_norm = bool(cmc_cfg.get("l2_normalize", True))

    loss_fn = MFVLRLoss(
        lambda_fd=float(loss_cfg.get("lambda_fd", 1.0)),
        lambda_lr=float(loss_cfg.get("lambda_lr", 1.0)),
        lambda_cmc=float(loss_cfg.get("lambda_cmc", 1.0)),
        lambda_fl=float(loss_cfg.get("lambda_fl", 1.0)),
        lambda_ar=float(loss_cfg.get("lambda_ar", 1.0)),
        lambda_kl=float(loss_cfg.get("lambda_kl", 1.0)),
        kl_temperature=float(loss_cfg.get("kl_temperature", 0.5)),
        cmc_l2_normalize=cmc_l2_norm,
        fd_class_weights=fd_weights,
    ).to(device)

    # 5. Build Optimizer, Scheduler, and Precision / Scaler configuration
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-3))
    step_size = int(train_cfg.get("lr_step_size", 15))
    gamma = float(train_cfg.get("lr_gamma", 0.1))

    # Determine precision mode
    precision = args.precision or train_cfg.get("precision")
    if precision is None:
        if args.amp or train_cfg.get("amp", False):
            if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                precision = "bf16"
            else:
                precision = "fp16"
        else:
            precision = "fp32"
    precision = precision.lower().strip()

    use_amp = (precision in ("bf16", "fp16")) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0) if (precision == "fp16" and use_amp) else None
    logger.info(f"Precision mode: {precision.upper()} (AMP: {use_amp}, GradScaler: {scaler is not None})")

    optimizer = create_optimizer(model, loss_fn, lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, step_size=step_size, gamma=gamma)

    start_epoch = 0
    global_step = 0

    # 6. Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        ckpt = load_checkpoint(
            filepath=args.resume,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        logger.info(f"Resumed successfully at epoch {start_epoch}, step {global_step}")

    # 7. Dry run mode
    if args.dry_run:
        logger.info("Executing DRY-RUN smoke test (1 training batch)...")
        batch = next(iter(train_loader))
        if isinstance(batch, dict) and "image" in batch:
            dry_bs = min(2, batch["image"].shape[0])
            batch = {k: (v[:dry_bs] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        step_losses = train_step(model, loss_fn, optimizer, batch, device=device, scaler=scaler, use_amp=use_amp, precision=precision)
        logger.info(f"Dry-run training step completed successfully. Losses: {step_losses}")

        logger.info("Executing dry-run image-only evaluation batch...")
        dry_val_loader = [next(iter(val_loader))]
        val_metrics = evaluate(
            model=model,
            dataloader=dry_val_loader,
            device=device,
            positive_label=dataset_cfg.get("fake_class_index", 1),
        )
        logger.info(f"Dry-run evaluation metrics: {val_metrics}")
        logger.info("DRY-RUN test PASSED. Exiting.")
        sys.exit(0)

    # 8. Full training loop
    total_epochs = train_cfg.get("epochs")
    if total_epochs is None:
        raise ValueError(
            "Total epochs is not specified in config ('training.epochs: null'). "
            "Please specify epochs in config or via --epochs CLI flag."
        )

    best_auc = -1.0
    fake_class_idx = dataset_cfg.get("fake_class_index", 1)

    logger.info(f"Starting training for {total_epochs} epochs (from epoch {start_epoch})...")

    for epoch in range(start_epoch, total_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{total_epochs} (LR: {optimizer.param_groups[0]['lr']:.6e}) ---")

        # Train one epoch
        mean_losses = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            precision=precision,
            epoch=epoch,
            total_epochs=total_epochs,
            logger=logger,
        )
        logger.info(f"Epoch {epoch + 1} Train Losses: {mean_losses}")

        # Update LR scheduler once per epoch
        scheduler.step()

        # Evaluate on validation set
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            positive_label=fake_class_idx,
        )
        logger.info(f"Epoch {epoch + 1} Val Metrics: {val_metrics}")

        # Save checkpoint
        current_auc = val_metrics.get("auc", 0.0)
        is_best = current_auc > best_auc
        if is_best:
            best_auc = current_auc

        ckpt_state = create_checkpoint_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            metrics=val_metrics,
            config=config,
        )
        save_checkpoint(
            state=ckpt_state,
            checkpoint_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch + 1}.pt",
            is_best=is_best,
        )

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
