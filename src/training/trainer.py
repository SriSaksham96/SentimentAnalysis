import contextlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Returns the best available computing device (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(df: pd.DataFrame, column: str, num_classes: int) -> torch.Tensor:
    """Computes balanced class weights ignoring masked entries (-1)."""
    valid_labels = df[df[column] != -1][column].values
    if len(valid_labels) == 0:
        return torch.ones(num_classes, dtype=torch.float)
    
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=valid_labels)
    return torch.tensor(weights, dtype=torch.float)


def build_optimizer_param_groups(
    model: nn.Module,
    lr_text: float = 1.0e-5,
    lr_fusion: float = 2.0e-5,
    lr_vision: float = 0.0,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """Constructs parameter groups with differential learning rates and weight-decay decay/no-decay splits.
    
    Categories:
    - Pretrained text encoder (MuRIL): lr_text (e.g. 1e-5)
    - Pretrained vision encoder (CLIP): lr_vision (0.0 if frozen, or low LR if unfrozen)
    - Multimodal fusion + heads (cross-attn, gate, LayerNorms, heads): lr_fusion (e.g. 2e-5)
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "norm.weight", "norm.bias", "norm_t.weight", "norm_t.bias", "fusion_norm.weight", "fusion_norm.bias"]

    text_params_decay, text_params_no_decay = [], []
    vision_params_decay, vision_params_no_decay = [], []
    fusion_params_decay, fusion_params_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in no_decay)

        if "text_encoder" in name:
            if is_no_decay:
                text_params_no_decay.append(param)
            else:
                text_params_decay.append(param)
        elif "vision_encoder" in name:
            if is_no_decay:
                vision_params_no_decay.append(param)
            else:
                vision_params_decay.append(param)
        else:
            # Projections, cross-attention, gate, LayerNorms, classification heads
            if is_no_decay:
                fusion_params_no_decay.append(param)
            else:
                fusion_params_decay.append(param)

    param_groups = []

    # 1. Text Encoder Group (MuRIL)
    if text_params_decay:
        param_groups.append({"params": text_params_decay, "lr": lr_text, "weight_decay": weight_decay, "name": "text_decay"})
    if text_params_no_decay:
        param_groups.append({"params": text_params_no_decay, "lr": lr_text, "weight_decay": 0.0, "name": "text_no_decay"})

    # 2. Vision Encoder Group (CLIP, if unfrozen)
    if vision_params_decay and lr_vision > 0:
        param_groups.append({"params": vision_params_decay, "lr": lr_vision, "weight_decay": weight_decay, "name": "vision_decay"})
    if vision_params_no_decay and lr_vision > 0:
        param_groups.append({"params": vision_params_no_decay, "lr": lr_vision, "weight_decay": 0.0, "name": "vision_no_decay"})

    # 3. Multimodal Fusion & Classification Heads
    if fusion_params_decay:
        param_groups.append({"params": fusion_params_decay, "lr": lr_fusion, "weight_decay": weight_decay, "name": "fusion_decay"})
    if fusion_params_no_decay:
        param_groups.append({"params": fusion_params_no_decay, "lr": lr_fusion, "weight_decay": 0.0, "name": "fusion_no_decay"})

    return param_groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_sent: nn.Module,
    criterion_off: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    loss_weight_sentiment: float = 1.0,
    loss_weight_offensive: float = 1.0,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Trains the model for one epoch using window-normalized multi-task gradient accumulation and FP16.
    
    Each accumulation window (including the final incomplete window) is normalized across the 
    actual valid sample counts for each task, ensuring bit-exact gradient equivalence with 
    processing the window in a single batch.
    """
    model.train()
    total_loss = 0.0
    sent_loss_total = 0.0
    off_loss_total = 0.0
    num_batches = len(loader)

    optimizer.zero_grad()
    use_cuda_autocast = fp16 and device.type == "cuda"
    use_mps_autocast = fp16 and device.type == "mps"

    window_batches = []

    for step, batch in enumerate(loader):
        window_batches.append(batch)
        is_window_end = (len(window_batches) == gradient_accumulation_steps) or (step + 1 == num_batches)
        if not is_window_end:
            continue

        # 1. Count valid labels across the entire accumulation window for each task
        window_valid_sent = sum((b["sentiment"] != -1).sum().item() for b in window_batches)
        window_valid_off = sum((b["offensive"] != -1).sum().item() for b in window_batches)

        # 2. Forward and backward pass for each microbatch in the window
        for b in window_batches:
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            pixel_values = b["pixel_values"].to(device)
            sent_labels = b["sentiment"].to(device)
            off_labels = b["offensive"].to(device)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda_autocast
                else torch.amp.autocast(device_type="mps", dtype=torch.float16)
                if use_mps_autocast
                else contextlib.nullcontext()
            )

            with autocast_ctx:
                sent_logits, off_logits = model(input_ids, attention_mask, pixel_values)

                micro_loss = torch.tensor(0.0, device=device)
                has_valid = False

                # Sentiment task: normalize microbatch sum-loss by window_valid_sent
                if window_valid_sent > 0 and (sent_labels != -1).any():
                    l_sent_sum = criterion_sent(sent_logits, sent_labels)
                    l_sent_norm = l_sent_sum / window_valid_sent
                    micro_loss = micro_loss + loss_weight_sentiment * l_sent_norm
                    sent_loss_total += l_sent_norm.item()
                    has_valid = True

                # Offensive task: normalize microbatch sum-loss by window_valid_off
                if window_valid_off > 0 and (off_labels != -1).any():
                    l_off_sum = criterion_off(off_logits, off_labels)
                    l_off_norm = l_off_sum / window_valid_off
                    micro_loss = micro_loss + loss_weight_offensive * l_off_norm
                    off_loss_total += l_off_norm.item()
                    has_valid = True

                if not has_valid:
                    continue

            # Backward pass on microbatch loss
            if scaler is not None and use_cuda_autocast:
                scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()

            total_loss += micro_loss.item()

        # 3. Optimizer Step after accumulating all microbatches in the window
        if scaler is not None and use_cuda_autocast:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad()
            if scale_before <= scale_after and scheduler is not None:
                scheduler.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # Reset window buffer for next accumulation window
        window_batches = []

    total_windows = max(1, math.ceil(num_batches / gradient_accumulation_steps))
    return {
        "loss": total_loss / total_windows,
        "sentiment_loss": sent_loss_total / total_windows,
        "offensive_loss": off_loss_total / total_windows,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_sent: nn.Module,
    criterion_off: nn.Module,
    device: torch.device,
    loss_weight_sentiment: float = 1.0,
    loss_weight_offensive: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates multi-task performance ignoring masked labels (-1), with language-balanced sentiment scoring."""
    model.eval()
    total_loss = 0.0
    sent_preds_all, sent_targets_all, sent_langs_all = [], [], []
    off_preds_all, off_targets_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            sent_labels = batch["sentiment"].to(device)
            off_labels = batch["offensive"].to(device)
            languages = batch.get("language", ["unknown"] * len(input_ids))

            sent_logits, off_logits = model(input_ids, attention_mask, pixel_values)

            loss = torch.tensor(0.0, device=device)
            if (sent_labels != -1).any():
                l_s = criterion_sent(sent_logits, sent_labels)
                n_s = (sent_labels != -1).sum().item()
                loss = loss + loss_weight_sentiment * (l_s / max(1, n_s))
            if (off_labels != -1).any():
                l_o = criterion_off(off_logits, off_labels)
                n_o = (off_labels != -1).sum().item()
                loss = loss + loss_weight_offensive * (l_o / max(1, n_o))

            total_loss += loss.item()

            # Sentiment predictions (ignore -1)
            sent_preds = torch.argmax(sent_logits, dim=-1).cpu().numpy()
            sent_targets = sent_labels.cpu().numpy()
            for p, t, l in zip(sent_preds, sent_targets, languages):
                if t != -1:
                    sent_preds_all.append(int(p))
                    sent_targets_all.append(int(t))
                    sent_langs_all.append(str(l).lower())

            # Offensive predictions (ignore -1)
            off_preds = torch.argmax(off_logits, dim=-1).cpu().numpy()
            off_targets = off_labels.cpu().numpy()
            for p, t in zip(off_preds, off_targets):
                if t != -1:
                    off_preds_all.append(int(p))
                    off_targets_all.append(int(t))

    metrics = {
        "val_loss": total_loss / max(len(loader), 1),
    }

    # 1. Overall Sentiment Metrics
    if len(sent_targets_all) > 0:
        metrics["sentiment_accuracy"] = float(accuracy_score(sent_targets_all, sent_preds_all))
        metrics["sentiment_macro_f1"] = float(f1_score(sent_targets_all, sent_preds_all, average="macro"))
        metrics["sentiment_weighted_f1"] = float(f1_score(sent_targets_all, sent_preds_all, average="weighted"))

    # 2. Language-Specific Sentiment Metrics & Class Support Validation
    sent_targets_arr = np.array(sent_targets_all)
    sent_preds_arr = np.array(sent_preds_all)
    sent_langs_arr = np.array(sent_langs_all)

    lang_macro_f1s = []
    for lang in ["hinglish", "bengali"]:
        mask = (sent_langs_arr == lang)
        if mask.sum() == 0:
            print(f"Warning: No validation samples found for language '{lang}'.")
            continue

        l_targets = sent_targets_arr[mask]
        l_preds = sent_preds_arr[mask]

        # Verify class support across all 3 classes (0: Neg, 1: Neu, 2: Pos)
        expected_classes = {0, 1, 2}
        present_classes = set(np.unique(l_targets))
        missing_classes = expected_classes - present_classes
        class_counts = {c: int(np.sum(l_targets == c)) for c in expected_classes}

        if missing_classes:
            print(f"WARNING: Language '{lang}' validation set lacks samples for class(es) {missing_classes}! Class counts: {class_counts}")
        
        l_acc = float(accuracy_score(l_targets, l_preds))
        l_f1 = float(f1_score(l_targets, l_preds, average="macro"))
        
        metrics[f"{lang}_sentiment_accuracy"] = l_acc
        metrics[f"{lang}_sentiment_macro_f1"] = l_f1
        metrics[f"{lang}_sentiment_class_counts"] = class_counts
        lang_macro_f1s.append(l_f1)

    # 3. Primary Selection Score: Language-Balanced Sentiment Macro-F1: mean(hinglish_f1, bengali_f1)
    if len(lang_macro_f1s) > 0:
        selection_score = float(np.mean(lang_macro_f1s))
    else:
        selection_score = metrics.get("sentiment_macro_f1", 0.0)
    metrics["selection_score"] = selection_score

    # 4. Overall Offensive Metrics (Memotion only; does NOT influence selection_score)
    if len(off_targets_all) > 0:
        metrics["offensive_accuracy"] = float(accuracy_score(off_targets_all, off_preds_all))
        metrics["offensive_macro_f1"] = float(f1_score(off_targets_all, off_preds_all, average="macro"))
        metrics["offensive_weighted_f1"] = float(f1_score(off_targets_all, off_preds_all, average="weighted"))

    return metrics
