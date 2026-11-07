# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages Using Text and Images

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**IndicSent-M** is a research framework for **multimodal sentiment analysis and offensive content classification** across Indian social media memes. In multimodal social media content, textual sarcasm and imagery often conflict (e.g., a cheerful visual template paired with biting satire or insult). IndicSent-M combines multilingual contextual language models with vision transformers through cross-modal attention and incongruity-aware gated fusion to analyze text embedded in meme images alongside visual semantics.

---

## Architecture

```
                    ┌───────────────────────────────┐
                    │  Multimodal Meme Input        │
                    │  (Embedded OCR Text + Image)  │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌──────────────────────────────┐                ┌──────────────────────────────┐
│ Text Encoder: MuRIL Base     │                │ Vision Encoder: CLIP ViT-B/32│
│ (google/muril-base-cased)    │                │ (openai/clip-vit-base-patch32)│
│ Hidden Size: 768             │                │ Hidden Size: 768             │
└──────────────┬───────────────┘                └──────────────┬───────────────┘
               ▼                                               ▼
         Linear Projection                               Linear Projection
         (768 -> 512)                                    (768 -> 512)
               │                                               │
               │            ┌──────────────────────────────────┤
               ▼            ▼                                  ▼
     ┌──────────────────────────────┐              ┌───────────────────────────┐
     │ Cross-Modal Multi-Head Attn  │              │ Global Visual Token       │
     │ Query: Text Tokens           │              │ (v_global: CLIP Pooler/CLS│
     │ Key/Value: Vision Patches    │              └─────────────┬─────────────┘
     └──────────────┬───────────────┘                            │
                    ▼                                            │
       Masked Mean Pooling                                       │
       (excludes [PAD] tokens)                                   │
                    │                                            │
                    ▼                                            ▼
               t_rep [B, 512]                              v_global [B, 512]
                    │                                            │
                    └──────────────────────┬─────────────────────┘
                                           ▼
                       ┌───────────────────────────────────────┐
                       │ Incongruity-Aware Gated Fusion Unit   │
                       │ diff = |t_rep - v_global|             │
                       │ g = Sigmoid(W_t*t + W_v*v + W_d*diff) │
                       │ fused = LayerNorm(g*t + (1-g)*v)      │
                       └───────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │ Sentiment Head: Linear(512,3│               │ Offensive Head: Linear(512,2│
     │ 0: Negative, 1: Neu, 2: Pos │               │ 0: Non-Hate, 1: Hate/Offens.│
     └─────────────────────────────┘               └─────────────────────────────┘
```

---

## Datasets Supported

| Dataset | Language | Modality | Tasks Annotated |
|---|---|---|---|
| **Memotion 3.0** | Hindi-English (Hinglish/English) | Text + Image | 3-Class Sentiment, 2-Class Offensive |
| **MemoSen** | Bengali | Text + Image | 3-Class Sentiment (Offensive masked with `-1`) |
| **TelMeMood** | Telugu (Schema ready) | Text + Image | 3-Class Sentiment |

### Unified Processed Splits (Stratified 80/10/10)
- **Train:** 10,230 samples (6,753 Memotion, 3,477 MemoSen)
- **Validation:** 1,279 samples (844 Memotion, 435 MemoSen)
- **Test:** 1,279 samples (844 Memotion, 435 MemoSen)
- **Total:** 12,788 multimodal samples

---

## Repository Structure

```
.
├── IndicSent_M_Colab_Training.ipynb  # End-to-end training notebook for Google Colab (Tesla T4)
├── PJ_10.pdf                          # Project specification & guidelines
├── configs/
│   ├── dataset_config.yaml            # Dataset schemas, mappings, and stratification
│   ├── experiment_2.yaml              # Production Experiment 2A/2B configuration
│   └── model_config.yaml              # Baseline model hyperparameters
├── data/
│   ├── processed/                     # Cleaned, deduplicated stratified splits
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── unified_multilingual_memes.csv
│   └── raw/                           # Dataset annotation CSVs (images ignored in git)
├── reports/
│   ├── experiment_2_implementation.md # Architectural implementation & audit report
│   ├── experiment_2_preflight.md      # Pre-training verification & test suite audit
│   ├── experiment_2a_error_investigation.md # Deep root cause analysis of Exp 2A
│   └── pre_experiment_2b_diagnostics.md    # Pre-Experiment 2B empirical diagnostic audit
├── src/
│   ├── data/
│   │   ├── dataset.py                 # IndicMemeDataset with truncation logging & augmentations
│   │   ├── merger.py                  # Multi-dataset standardizer and merger
│   │   └── preprocess_and_split.py    # OCR cleaning, hash deduplication, stratified splitting
│   ├── models/
│   │   └── indicsent_m.py             # IndicSentM PyTorch neural architecture
│   └── training/
│       └── trainer.py                 # Multi-task training loop, window gradient accumulation
├── tests/
│   ├── test_experiment_2.py           # 10-point automated mathematical verification suite
│   ├── test_indicsent_m.py            # Architectural unit tests
│   └── test_trainer.py                # Loss and trainer verification
├── train.py                           # CLI training entrypoint with AMP, resume & early stopping
└── requirements.txt                   # Environment dependencies
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/SriSaksham96/SentimentAnalysis.git
%cd SentimentAnalysis

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Verification & Unit Testing

Run the 10-point mathematical and architectural verification suite:

```bash
python3 tests/test_experiment_2.py
```

Run an end-to-end dry-run smoke test (2 batches per split):

```bash
python3 train.py --config configs/experiment_2.yaml --dry_run
```

---

## Training on Google Colab

Use the provided [`IndicSent_M_Colab_Training.ipynb`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/IndicSent_M_Colab_Training.ipynb):
1. Select **Runtime > Change runtime type > T4 GPU**.
2. Mount Google Drive and extract the codebase.
3. Execute full training with automatic mixed precision (FP16):
   ```bash
   !python train.py \
       --config configs/experiment_2.yaml \
       --output_dir /content/drive/MyDrive/IndicSentM/checkpoints/experiment_2a
   ```
4. If interrupted, automatically resume weights, AdamW optimizer moments, scheduler, and scaler:
   ```bash
   !python train.py \
       --config configs/experiment_2.yaml \
       --output_dir /content/drive/MyDrive/IndicSentM/checkpoints/experiment_2a \
       --resume /content/drive/MyDrive/IndicSentM/checkpoints/experiment_2a/checkpoint_last.pt
   ```

---

## Experimental Progression & Technical Reports

- **Run 1:** Baseline run. Diagnosed with unmasked pad-token pooling, discarded visual features, and optimizer re-instantiation. (Documented in [`reports/experiment_2_implementation.md`](reports/experiment_2_implementation.md)).
- **Experiment 2A:** Implemented masked pooling, genuine visual representation, persistent AdamW, window-normalized gradient accumulation, and differential learning rates. (Documented in [`reports/experiment_2_preflight.md`](reports/experiment_2_preflight.md)).
- **Error Investigation & Pre-2B Diagnostics:** Diagnosed the ~51% performance ceiling as arising from dataset semantic label mismatch (Memotion 41.8% Neutral vs MemoSen 62.4% Negative), multitask gradient interference (3.88:1 skew), scale disparity, and volatile minority-class checkpoint selection. (Documented in [`reports/experiment_2a_error_investigation.md`](reports/experiment_2a_error_investigation.md) and [`reports/pre_experiment_2b_diagnostics.md`](reports/pre_experiment_2b_diagnostics.md)).

---

## License

This project is licensed under the MIT License.
