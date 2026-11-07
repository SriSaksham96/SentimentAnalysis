import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IndicMemeDataset(Dataset):
    """PyTorch Dataset for Multilingual Multimodal Memes.
    
    Loads preprocessed meme samples, tokenizes multilingual text using MuRIL / XLM-R,
    formats images for CLIP ViT patch extraction, records truncation statistics,
    and applies conservative training-only image augmentations.
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        tokenizer = None,
        tokenizer_name: str = "google/muril-base-cased",
        max_length: int = 64,
        image_size: int = 224,
        is_train: bool = False,
        augment: Optional[bool] = None,
        transform = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.max_length = max_length
        self.is_train = is_train
        if augment is None:
            self.augment = is_train
        else:
            self.augment = augment

        # 1. Text Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 2. Vision Transforms
        if transform is not None:
            self.transform = transform
        elif self.augment:
            # Conservative training-only augmentations
            # Preserves embedded meme text (NO horizontal flips or aggressive crops)
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            # Deterministic evaluation transforms
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

        # 3. Compute and Record Truncation Statistics
        self.truncation_stats = self._compute_truncation_stats()

    def _compute_truncation_stats(self) -> Dict[str, Union[int, float]]:
        """Calculates exact truncation statistics against max_length."""
        total_samples = len(self.df)
        if total_samples == 0:
            return {"total_samples": 0, "truncated_samples": 0, "truncation_percentage": 0.0}

        # Vectorized tokenization check
        raw_texts = self.df["ocr_text"].astype(str).tolist()
        # Batch tokenization without truncation to measure natural length
        encoded = self.tokenizer(raw_texts, truncation=False, padding=False, add_special_tokens=True)
        token_lengths = [len(seq) for seq in encoded["input_ids"]]
        truncated_count = sum(1 for length in token_lengths if length > self.max_length)
        pct = (truncated_count / total_samples) * 100.0

        return {
            "total_samples": total_samples,
            "truncated_samples": truncated_count,
            "truncation_percentage": round(pct, 2),
            "max_configured_length": self.max_length,
            "median_token_length": round(float(pd.Series(token_lengths).median()), 1),
            "p95_token_length": round(float(pd.Series(token_lengths).quantile(0.95)), 1),
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        ocr_text = str(row["ocr_text"])
        sentiment = int(row["sentiment"])

        # Tokenize text with attention mask preserved
        encoded = self.tokenizer(
            ocr_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Load and transform image
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                pixel_values = self.transform(img_rgb)
        except Exception:
            # Fallback zero tensor in case of unexpected I/O issue
            pixel_values = torch.zeros(3, 224, 224)

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "language": row.get("language", "unknown"),
            "dataset_source": row.get("dataset_source", "unknown"),
        }

        # Offensive / hate label (defaults to -1 for masked multi-task loss)
        if "offensive" in row and pd.notnull(row["offensive"]):
            item["offensive"] = torch.tensor(int(row["offensive"]), dtype=torch.long)
        else:
            item["offensive"] = torch.tensor(-1, dtype=torch.long)

        return item
