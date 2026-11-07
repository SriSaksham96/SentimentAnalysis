import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import yaml

from src.data.dataset import IndicMemeDataset
from src.models.indicsent_m import IndicSentM
from src.training.trainer import (
    build_optimizer_param_groups,
    compute_class_weights,
    evaluate,
    get_device,
    train_one_epoch,
)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train IndicSent-M with Masked Multi-Task Loss & Class Weights")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/experiment_2.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro-batch size per forward step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of gradient accumulation steps")
    parser.add_argument("--lr_text", type=float, default=None, help="Learning rate for text encoder (MuRIL)")
    parser.add_argument("--lr_fusion", type=float, default=None, help="Learning rate for fusion & heads")
    parser.add_argument("--lr_vision", type=float, default=None, help="Learning rate for vision encoder (CLIP)")
    parser.add_argument("--max_text_seq_len", type=int, default=None, help="Maximum text token sequence length")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio for scheduler")
    parser.add_argument("--freeze_vision", action=argparse.BooleanOptionalAction, default=None, help="Freeze vision encoder")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None, help="Use FP16 mixed precision")
    parser.add_argument("--dry_run", action="store_true", help="Quick smoke test on a small slice of batches")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to processed data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to resumable checkpoint to continue training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base configuration if provided
    cfg = {}
    if args.config:
        print(f"Loading configuration from: {args.config}")
        cfg = load_yaml_config(args.config)

    # Resolve settings with CLI arguments taking precedence over YAML config
    cfg_exp = cfg.get("experiment", {})
    cfg_model = cfg.get("model", {})
    cfg_text = cfg_model.get("text_encoder", {})
    cfg_vis = cfg_model.get("vision_encoder", {})
    cfg_arch = cfg_model.get("architecture", {})
    cfg_data = cfg.get("data", {})
    cfg_train = cfg.get("training", {})

    seed = args.seed if args.seed is not None else cfg_train.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    print(f"Using computing device: {device}")

    # Training parameters
    epochs = args.epochs if args.epochs is not None else cfg_train.get("epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else cfg_train.get("batch_size", 4)
    grad_accum_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else cfg_train.get("gradient_accumulation_steps", 4)
    lr_text = args.lr_text if args.lr_text is not None else cfg_text.get("lr", 1.0e-5)
    lr_fusion = args.lr_fusion if args.lr_fusion is not None else cfg_arch.get("fusion_lr", 2.0e-5)
    lr_vision = args.lr_vision if args.lr_vision is not None else cfg_vis.get("lr", 0.0)
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg_train.get("weight_decay", 0.01)
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else cfg_train.get("warmup_ratio", 0.1)
    max_grad_norm = cfg_train.get("max_grad_norm", 1.0)
    max_text_seq_len = args.max_text_seq_len if args.max_text_seq_len is not None else cfg_data.get("max_text_seq_len", 64)
    image_size = cfg_data.get("image_size", 224)
    freeze_vision = args.freeze_vision if args.freeze_vision is not None else cfg_vis.get("freeze", True)
    freeze_text = cfg_text.get("freeze", False)
    fp16 = args.fp16 if args.fp16 is not None else cfg_train.get("fp16", True)
    loss_weights = cfg_train.get("loss_weights", {"sentiment": 1.0, "offensive": 1.0})
    loss_weight_sentiment = float(loss_weights.get("sentiment", 1.0))
    loss_weight_offensive = float(loss_weights.get("offensive", 1.0))

    # Early stopping
    cfg_es = cfg_train.get("early_stopping", {})
    early_stopping_enabled = cfg_es.get("enabled", True)
    early_stopping_patience = cfg_es.get("patience", 3)

    data_dir_str = args.data_dir if args.data_dir is not None else cfg_data.get("data_dir", "data/processed")
    output_dir_str = args.output_dir if args.output_dir is not None else cfg_train.get("output_dir", "checkpoints/experiment_2")

    out_dir = Path(output_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Experiment Configuration ---")
    print(f"  Name: {cfg_exp.get('name', 'IndicSent-M')}")
    print(f"  Epochs: {epochs} | Micro-Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} (Effective Batch Size: {batch_size * grad_accum_steps})")
    print(f"  Max Sequence Length: {max_text_seq_len} | Image Size: {image_size}")
    print(f"  Learning Rates -> Text: {lr_text} | Fusion/Heads: {lr_fusion} | Vision: {lr_vision}")
    print(f"  Backbones -> Vision Frozen: {freeze_vision} | Text Frozen: {freeze_text}")
    print(f"  Precision -> FP16 Mixed Precision: {fp16}")
    print(f"  Loss Weights -> Sentiment: {loss_weight_sentiment} | Offensive: {loss_weight_offensive}")
    print(f"  Output Directory: {out_dir}")

    # 1. Load Data Splits
    data_dir = Path(data_dir_str)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training split not found at {train_csv}. Run preprocessing first.")

    train_df = pd.read_csv(train_csv)
    print(f"\nLoaded training split: {len(train_df)} samples")

    # 2. Compute Balanced Class Weights (ignoring -1)
    sent_weights = compute_class_weights(train_df, "sentiment", num_classes=3).to(device)
    off_weights = compute_class_weights(train_df, "offensive", num_classes=2).to(device)

    print("\n--- Balanced Class Weights ---")
    print(f"Sentiment weights (Neg: 0, Neu: 1, Pos: 2): {[round(w, 4) for w in sent_weights.tolist()]}")
    print(f"Offensive weights (Non-Hate: 0, Hate: 1):   {[round(w, 4) for w in off_weights.tolist()]}")

    # 3. Multi-Task Masked Loss Criteria (reduction='sum' for window-normalized gradient accumulation)
    criterion_sent = nn.CrossEntropyLoss(weight=sent_weights, ignore_index=-1, reduction="sum")
    criterion_off = nn.CrossEntropyLoss(weight=off_weights, ignore_index=-1, reduction="sum")

    # 4. Initialize Datasets & DataLoaders with Truncation Stats
    print("\nInitializing datasets, tokenizers, and image transforms...")
    text_model_name = cfg_text.get("name", "google/muril-base-cased")
    vision_model_name = cfg_vis.get("name", "openai/clip-vit-base-patch32")

    # Training dataset uses conservative augmentations; eval datasets are deterministic
    train_dataset = IndicMemeDataset(
        train_csv,
        tokenizer_name=text_model_name,
        max_length=max_text_seq_len,
        image_size=image_size,
        is_train=True,
    )
    val_dataset = IndicMemeDataset(
        val_csv,
        tokenizer=train_dataset.tokenizer,
        max_length=max_text_seq_len,
        image_size=image_size,
        is_train=False,
    )
    test_dataset = IndicMemeDataset(
        test_csv,
        tokenizer=train_dataset.tokenizer,
        max_length=max_text_seq_len,
        image_size=image_size,
        is_train=False,
    )

    t_stats = train_dataset.truncation_stats
    print(f"\n--- Text Truncation Statistics (Target: {max_text_seq_len} tokens) ---")
    print(f"  Total Samples:        {t_stats['total_samples']:,}")
    print(f"  Truncated Samples:    {t_stats['truncated_samples']:,} ({t_stats['truncation_percentage']}%)")
    print(f"  Median Token Length:  {t_stats['median_token_length']}")
    print(f"  95th-Percentile Len:  {t_stats['p95_token_length']}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if args.dry_run:
        out_dir = out_dir / "dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[DRY RUN] Checkpoints isolated to dedicated directory: {out_dir}")
        print("[DRY RUN] Truncating DataLoaders to 2 batches for fast pipeline validation...")
        train_loader = list(train_loader)[:2]
        val_loader = list(val_loader)[:2]
        test_loader = list(test_loader)[:2]
        epochs = 1

    # 5. Build IndicSent-M Model
    print("\nBuilding IndicSent-M Model...")
    hidden_dim = cfg_arch.get("hidden_dim", 512)
    num_heads = cfg_arch.get("num_heads", 8)
    dropout = cfg_arch.get("dropout", 0.3)
    num_sentiment_classes = cfg_arch.get("num_sentiment_classes", 3)
    num_offensive_classes = cfg_arch.get("num_offensive_classes", 2)

    model = IndicSentM(
        text_model_name=text_model_name,
        vision_model_name=vision_model_name,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_sentiment_classes=num_sentiment_classes,
        num_offensive_classes=num_offensive_classes,
        dropout=dropout,
        freeze_text_encoder=freeze_text,
        freeze_vision_encoder=freeze_vision,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,}")

    # 6. Optimizer with Parameter Groups (Single Lifecycle - No Re-instantiation)
    param_groups = build_optimizer_param_groups(
        model=model,
        lr_text=lr_text,
        lr_fusion=lr_fusion,
        lr_vision=lr_vision,
        weight_decay=weight_decay,
    )
    print(f"Configured {len(param_groups)} parameter groups for AdamW optimizer.")
    for g in param_groups:
        print(f"  -> Group: {g.get('name', 'unnamed')}, lr={g['lr']}, weight_decay={g['weight_decay']}, num_tensors={len(g['params'])}")

    optimizer = torch.optim.AdamW(param_groups)

    # Learning Rate Scheduler (uses math.ceil to match exact accumulation steps per epoch)
    effective_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = effective_steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda") if (fp16 and device.type == "cuda") else None

    # Checkpoint paths: separate resumable (full training state) and inference-only (clean weights)
    best_resumable_path = out_dir / "indicsent_m_resumable.pt"
    best_inference_path = out_dir / "indicsent_m_inference.pt"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"

    start_epoch = 1
    best_val_f1 = -1.0
    patience_counter = 0

    # Resume capability for Colab disconnections
    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.exists():
            raise FileNotFoundError(f"Checkpoint for resume not found at {resume_file}")
        print(f"\n--- Resuming Training from: {resume_file} ---")
        ckpt = torch.load(resume_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed successfully! Next Epoch: {start_epoch}/{epochs} | Best Val Macro-F1: {best_val_f1:.4f} | Patience: {patience_counter}")

    # 7. Training Loop with Early Stopping & Checkpoint Persistence
    print(f"\nStarting Training (Epochs {start_epoch} to {epochs}, Effective Optimizer Steps: {total_steps})...")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_sent=criterion_sent,
            criterion_off=criterion_off,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation_steps=grad_accum_steps,
            fp16=fp16,
            loss_weight_sentiment=loss_weight_sentiment,
            loss_weight_offensive=loss_weight_offensive,
            max_grad_norm=max_grad_norm,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion_sent=criterion_sent,
            criterion_off=criterion_off,
            device=device,
            loss_weight_sentiment=loss_weight_sentiment,
            loss_weight_offensive=loss_weight_offensive,
        )

        sent_f1 = val_metrics.get("sentiment_macro_f1", 0.0)
        off_f1 = val_metrics.get("offensive_macro_f1", 0.0)
        h_sent_f1 = val_metrics.get("hinglish_sentiment_macro_f1", 0.0)
        b_sent_f1 = val_metrics.get("bengali_sentiment_macro_f1", 0.0)
        selection_score = val_metrics.get("selection_score", (sent_f1 + off_f1) / 2.0)

        current_lrs = [round(g["lr"], 8) for g in optimizer.param_groups]
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(Sent: {train_metrics['sentiment_loss']:.4f}, Off: {train_metrics['offensive_loss']:.4f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Selection Score (Lang-Balanced Sent F1): {selection_score:.4f} "
            f"[Hinglish: {h_sent_f1:.4f}, Bengali: {b_sent_f1:.4f}] | "
            f"Overall Sent F1: {sent_f1:.4f} | "
            f"Off F1: {off_f1:.4f} | "
            f"LRs: {current_lrs}"
        )

        resumable_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_f1": best_val_f1,
            "selection_score": selection_score,
            "patience_counter": patience_counter,
            "val_metrics": val_metrics,
            "config": cfg,
        }
        # Save last completed epoch checkpoint for crash resumption
        torch.save(resumable_payload, last_checkpoint_path)

        # Checkpoint saving & early stopping check (Primary Metric: selection_score)
        if selection_score > best_val_f1:
            best_val_f1 = selection_score
            patience_counter = 0
            resumable_payload["best_val_f1"] = best_val_f1
            resumable_payload["patience_counter"] = patience_counter

            # Save full resumable checkpoint
            torch.save(resumable_payload, best_resumable_path)

            # Save lightweight inference-only checkpoint (model weights only, ~1.2GB instead of ~3GB)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_inference_path,
            )
            print(f"  -> Saved new best model checkpoints:")
            print(f"     [Resumable]: {best_resumable_path}")
            print(f"     [Inference]: {best_inference_path} (Selection Score: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement in Selection Score. Early stopping patience: {patience_counter}/{early_stopping_patience}")
            if early_stopping_enabled and patience_counter >= early_stopping_patience:
                print(f"\n[Early Stopping Triggered] Stopping training at epoch {epoch}.")
                break

    # 8. Final Test Evaluation using Best Checkpoint
    eval_checkpoint = best_inference_path if best_inference_path.exists() else best_resumable_path
    if eval_checkpoint.exists():
        print(f"\nLoading best checkpoint from {eval_checkpoint} for Test set evaluation...")
        checkpoint = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion_sent=criterion_sent,
        criterion_off=criterion_off,
        device=device,
        loss_weight_sentiment=loss_weight_sentiment,
        loss_weight_offensive=loss_weight_offensive,
    )

    print("\n================ FINAL TEST SET RESULTS ================")
    print(f"Test Loss:                          {test_metrics['val_loss']:.4f}")
    print(f"Language-Balanced Sent Macro-F1:    {test_metrics.get('selection_score', 0.0):.4f}")
    print(f"Hinglish Sentiment Macro-F1:        {test_metrics.get('hinglish_sentiment_macro_f1', 0.0):.4f}")
    print(f"Bengali Sentiment Macro-F1:         {test_metrics.get('bengali_sentiment_macro_f1', 0.0):.4f}")
    print(f"Overall Sentiment Accuracy:         {test_metrics.get('sentiment_accuracy', 0.0)*100:.2f}%")
    print(f"Overall Sentiment Macro-F1:         {test_metrics.get('sentiment_macro_f1', 0.0):.4f}")
    print(f"Overall Sentiment Weighted-F1:      {test_metrics.get('sentiment_weighted_f1', 0.0):.4f}")
    print(f"Offensive Accuracy (Memotion only): {test_metrics.get('offensive_accuracy', 0.0)*100:.2f}%")
    print(f"Offensive Macro-F1 (Memotion only): {test_metrics.get('offensive_macro_f1', 0.0):.4f}")
    print(f"Offensive Weighted-F1:              {test_metrics.get('offensive_weighted_f1', 0.0):.4f}")
    print("========================================================")


if __name__ == "__main__":
    main()
