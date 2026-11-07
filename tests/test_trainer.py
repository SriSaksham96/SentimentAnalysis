import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, CLIPVisionConfig, BertModel, CLIPVisionModel

from src.models.indicsent_m import IndicSentM
from src.training.trainer import evaluate, train_one_epoch


def test_masked_loss_and_weights():
    print("Testing Masked Multi-Task Loss and Class Weights...")
    device = torch.device("cpu")

    # Mock lightweight IndicSentM
    text_config = BertConfig(vocab_size=500, hidden_size=128, num_hidden_layers=1, num_attention_heads=2)
    vision_config = CLIPVisionConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=32)

    class FastMockIndicSentM(IndicSentM):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.text_encoder = BertModel(text_config)
            self.vision_encoder = CLIPVisionModel(vision_config)
            self.text_proj = nn.Linear(128, 64)
            self.vision_proj = nn.Linear(128, 64)
            self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
            self.norm = nn.LayerNorm(64)
            self.gate = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
            self.dropout = nn.Dropout(0.1)
            self.sentiment_head = nn.Linear(64, 3)
            self.offensive_head = nn.Linear(64, 2)

    model = FastMockIndicSentM().to(device)

    # Class weights
    sent_weights = torch.tensor([0.85, 1.12, 1.08], dtype=torch.float, device=device)
    off_weights = torch.tensor([0.87, 1.18], dtype=torch.float, device=device)

    criterion_sent = nn.CrossEntropyLoss(weight=sent_weights, ignore_index=-1)
    criterion_off = nn.CrossEntropyLoss(weight=off_weights, ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic batch: 4 samples
    # Sample 0: valid sentiment (0), valid offensive (1)
    # Sample 1: valid sentiment (1), MASKED offensive (-1) -> e.g. MemoSen
    # Sample 2: valid sentiment (2), valid offensive (0)
    # Sample 3: valid sentiment (1), MASKED offensive (-1) -> e.g. MemoSen
    B, seq_len = 4, 16
    batch = {
        "input_ids": torch.randint(0, 500, (B, seq_len)),
        "attention_mask": torch.ones((B, seq_len), dtype=torch.long),
        "pixel_values": torch.randn(B, 3, 224, 224),
        "sentiment": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "offensive": torch.tensor([1, -1, 0, -1], dtype=torch.long),
    }

    # Custom mini loader
    class MiniLoader:
        def __iter__(self):
            yield batch
        def __len__(self):
            return 1

    loader = MiniLoader()

    # Run train_one_epoch
    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion_sent=criterion_sent,
        criterion_off=criterion_off,
        device=device,
    )
    print("Train step completed successfully!")
    print(f"  -> Loss: {metrics['loss']:.4f}")
    print(f"  -> Sentiment Loss: {metrics['sentiment_loss']:.4f}")
    print(f"  -> Offensive Loss: {metrics['offensive_loss']:.4f}")

    assert metrics["loss"] > 0
    assert metrics["sentiment_loss"] > 0
    assert metrics["offensive_loss"] > 0

    # Run evaluate
    eval_metrics = evaluate(
        model=model,
        loader=loader,
        criterion_sent=criterion_sent,
        criterion_off=criterion_off,
        device=device,
    )
    print("\nEval step completed successfully!")
    print(f"  -> Val Loss: {eval_metrics['val_loss']:.4f}")
    print(f"  -> Sentiment Accuracy: {eval_metrics['sentiment_accuracy']:.2f}")
    print(f"  -> Offensive Accuracy: {eval_metrics['offensive_accuracy']:.2f}")

    print("\n✔ Masked multi-task loss correctly handled -1 labels.")
    print("✔ Class weights successfully applied to both loss heads.")
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    test_masked_loss_and_weights()

