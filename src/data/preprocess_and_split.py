import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml


def clean_ocr_text(text: str) -> str:
    """Cleans OCR noise, watermarks, URLs, and normalizes unicode."""
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Strip URLs and common watermark domains
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[a-zA-Z0-9.-]+\.(?:com|in|org|net|co|tv|xyz)\b(?:\/\S*)?", "", text, flags=re.IGNORECASE)
    
    # 3. Strip control characters and zero-width artifacts
    text = re.sub(r"[\u200b-\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # 4. Collapse whitespace (tabs, consecutive spaces, newlines)
    text = re.sub(r"\s+", " ", text)
    
    # 5. Strip surrounding quotes and whitespace
    text = text.strip(" \"'`“”‘’«»[]{}()<>\t\r\n")
    
    return text


def compute_image_hash(image_path: str) -> str:
    """Computes MD5 hash of an image file for binary deduplication."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DataPipeline:
    def __init__(self, config_path: str = "configs/dataset_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.target_cols = self.config["unified_schema"]["target_columns"]
        
    def process(
        self,
        input_data = None,
        output_dir: str = "data/processed",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        if input_data is None:
            # If not provided, merge fresh from raw datasets using DatasetMerger
            from src.data.merger import DatasetMerger
            merger = DatasetMerger()
            df = merger.merge_all()
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            df = pd.read_csv(str(input_data))

        initial_len = len(df)
        print(f"Loaded dataset: {initial_len} samples.")
        
        # --- 1. Verify Image Paths & Decodability ---
        print("\n[1/4] Verifying image paths and image file integrity...")
        valid_img_mask = []
        for path_str in df["image_path"]:
            p = Path(path_str)
            if not p.is_file():
                valid_img_mask.append(False)
                continue
            try:
                with Image.open(p) as img:
                    img.verify()
                valid_img_mask.append(True)
            except Exception:
                valid_img_mask.append(False)
                
        df = df[valid_img_mask].copy()
        broken_imgs = initial_len - len(df)
        print(f"  -> Removed {broken_imgs} broken/missing image links. Remaining: {len(df)}")
        
        # --- 2. Clean OCR Noise ---
        print("\n[2/4] Cleaning OCR noise and filtering placeholder texts...")
        df["ocr_text"] = df["ocr_text"].apply(clean_ocr_text)
        
        # Filter empty texts and generic placeholders
        placeholders = {"no caption", "nocaption", "none", "nan", "null", "meme"}
        meaningful_mask = (df["ocr_text"].str.len() > 0) & (~df["ocr_text"].str.lower().isin(placeholders))
        dropped_noisy_text = len(df) - meaningful_mask.sum()
        df = df[meaningful_mask].copy()
        print(f"  -> Removed {dropped_noisy_text} empty/placeholder texts. Remaining: {len(df)}")
        
        # --- 3. Deduplication ---
        print("\n[3/4] Deduplicating images and samples...")
        # Exact row deduplication
        df = df.drop_duplicates()
        
        # Compute image binary hashes to identify duplicate image files
        df["img_hash"] = df["image_path"].apply(compute_image_hash)
        dup_img_hashes = df.duplicated(subset=["img_hash"]).sum()
        df = df.drop_duplicates(subset=["img_hash"], keep="first")
        print(f"  -> Removed {dup_img_hashes} duplicate image files (MD5 matches). Remaining: {len(df)}")
        
        # Save cleaned unified dataset
        cleaned_unified_file = out_path / "unified_multilingual_memes.csv"
        df[self.target_cols].to_csv(cleaned_unified_file, index=False)
        print(f"  -> Saved cleaned unified dataset to: {cleaned_unified_file}")
        
        # --- 4. Stratified Split (80% Train, 10% Val, 10% Test) ---
        print(f"\n[4/4] Creating stratified splits ({int(train_ratio*100)}% train, {int(val_ratio*100)}% val, {int(test_ratio*100)}% test)...")
        # Composite stratum ensures balance across BOTH datasets AND ALL sentiment classes
        df["strata"] = df["dataset_source"].astype(str) + "_" + df["sentiment"].astype(str)
        print("  -> Stratification strata distribution:")
        for stratum, count in df["strata"].value_counts().items():
            print(f"       {stratum}: {count}")
        
        # Split 80% train, 20% temp (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df["strata"],
            random_state=seed,
        )
        
        # Split temp equally into 10% val and 10% test
        val_share_of_temp = val_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_share_of_temp),
            stratify=temp_df["strata"],
            random_state=seed,
        )
        
        # Export final splits with unified schema columns
        train_csv = out_path / "train.csv"
        val_csv = out_path / "val.csv"
        test_csv = out_path / "test.csv"
        
        train_df[self.target_cols].to_csv(train_csv, index=False)
        val_df[self.target_cols].to_csv(val_csv, index=False)
        test_df[self.target_cols].to_csv(test_csv, index=False)
        
        print("\nSplits successfully exported:")
        print(f"  -> Train: {train_csv} ({len(train_df)} samples, {len(train_df)/len(df)*100:.1f}%)")
        print(f"  -> Val:   {val_csv} ({len(val_df)} samples, {len(val_df)/len(df)*100:.1f}%)")
        print(f"  -> Test:  {test_csv} ({len(test_df)} samples, {len(test_df)/len(df)*100:.1f}%)")
        
        return train_df[self.target_cols], val_df[self.target_cols], test_df[self.target_cols]


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.process()
