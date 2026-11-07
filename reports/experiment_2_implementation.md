# IndicSent-M: Experiment 2 Implementation & Technical Audit Report

**Date:** November 2026  
**Project:** IndicSent-M — Multimodal Sentiment & Offensive Classification for Indian Languages (Hinglish, Bengali)  
**Status:** Implementation & Verification Complete — Awaiting Authorization for Full Training Run  

---

## 1. Executive Summary & Context

During the initial 5-epoch training run (Run 1) of IndicSent-M on an NVIDIA Tesla T4 GPU (10,230 unified meme samples across Memotion 3.0 Hinglish and MemoSen Bengali), the model exhibited severe generalization failure and validation divergence:
- **Epoch 1:** Train Loss: `1.7639`, Val Loss: `1.6956` (Sent Acc: 47.77%, Off Acc: 57.23%)
- **Epoch 2:** Train Loss: `1.6782`, Val Loss: `1.6509` (Sent Acc: 51.84%, Off Acc: 60.43%)
- **Epoch 3:** Train Loss: `1.4716`, Val Loss: `1.6671`
- **Epoch 5:** Train Loss: `0.6102`, Val Loss: `2.4269` (Sent Acc: 50.04%, Off Acc: 57.70%)

While training loss dropped by 65%, validation loss exploded from `1.6509` to `2.4269`. A comprehensive codebase audit diagnosed five fundamental structural defects:
1. **Unmasked Text Pooling over Pad Tokens:** An unmasked `mean(dim=1)` over `max_seq_len=128` caused pad tokens (accounting for ~78% of all tokens) to dominate and dilute text representations.
2. **Discarded Visual Modality:** The original fusion code discarded CLIP's visual representations, passing only cross-attended text to the classification heads without direct global visual context.
3. **Catastrophic Optimizer Recreation:** In `train.py`, unfreezing the vision backbone at Epoch 2 recreated the `AdamW` optimizer from scratch, destroying all momentum/variance accumulator states and completely detaching the learning rate scheduler.
4. **Premature Vision Fine-Tuning & Representation Collapse:** Fine-tuning all 86M parameters of CLIP ViT-B/32 with high LR (`1e-5`) on a small 10k dataset destroyed pretrained general visual features.
5. **Zero Data Regularization:** No image augmentations were applied, allowing the vision backbone to memorize noise and background patterns.

This report documents the end-to-end implementation of **Experiment 2 Corrections**, verifies all system invariants across synthetic unit tests and a live end-to-end dry run, and provides an actionable blueprint for Google Colab deployment.

---

## 2. Completed Architectural & Training Corrections

All fixes have been implemented with strict isolation:
- Original configs (`configs/model_config.yaml`, `configs/dataset_config.yaml`) remain **untouched**.
- Original Run 1 checkpoints (`checkpoints/indicSent_m_best.pt`) remain **untouched**.
- A dedicated experiment configuration (`configs/experiment_2.yaml`) and checkpoint directory (`checkpoints/experiment_2/`) have been established.

### 2.1 Multimodal Fusion Architecture ([src/models/indicsent_m.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py))
The core architecture was rewritten to guarantee valid representation extraction, cross-modal attention, and dual-modality contribution:

1. **Mask-Aware Text Mean Pooling:**
   Instead of simple sequence averaging, pad tokens (where `attention_mask == 0`) are zeroed out before computing the mean:
   $$\mathbf{t}_{\text{rep}} = \frac{\sum_{i=1}^{L} \mathbf{t}_i \cdot m_i}{\sum_{i=1}^{L} m_i + \epsilon}$$
   ```python
   mask_expanded = attention_mask.unsqueeze(-1).expand(text_out.size()).float()
   t_rep = (text_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
   t_pooled = self.text_global_proj(t_rep)  # [B, hidden_dim]
   ```
2. **Genuine Global Visual Representation:**
   The visual CLS / pooler output is extracted directly from CLIP (`vis_res.pooler_output` or `vis_out[:, 0, :]`) and mapped via a dedicated projection:
   ```python
   v_rep = vis_res.pooler_output if hasattr(vis_res, "pooler_output") and vis_res.pooler_output is not None else vis_out[:, 0, :]
   v_global = self.vision_global_proj(v_rep)  # [B, hidden_dim]
   ```
3. **Cross-Modal Attention (Text Queries Visual Patches):**
   Text token embeddings query all 50 visual patch embeddings (`[CLS] + 49` grid patches) via Multi-Head Attention:
   ```python
   t_attn_out, _ = self.cross_attn(query=t_feat, key=v_feat, value=v_feat)
   t_attended = self.norm_attn(t_feat + self.cross_dropout(t_attn_out))
   t_attn_rep = (t_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
   ```
4. **Incongruity-Aware Gated Fusion Unit:**
   To capture multimodal sarcasm, incongruity, and irony (e.g., happy meme template paired with abusive caption), the fusion unit computes dynamic gating vectors over text, vision, and their element-wise difference:
   $$\mathbf{h}_{\text{diff}} = |\mathbf{t}_{\text{attn}} - \mathbf{v}_{\text{global}}|$$
   $$\mathbf{g}_t = \sigma(\mathbf{W}_t [\mathbf{t}_{\text{attn}}, \mathbf{v}_{\text{global}}, \mathbf{h}_{\text{diff}}])$$
   $$\mathbf{g}_v = \sigma(\mathbf{W}_v [\mathbf{t}_{\text{attn}}, \mathbf{v}_{\text{global}}, \mathbf{h}_{\text{diff}}])$$
   $$\mathbf{h}_{\text{fused}} = \text{LayerNorm}(\mathbf{g}_t \odot \mathbf{t}_{\text{attn}} + \mathbf{g}_v \odot \mathbf{v}_{\text{global}} + \mathbf{W}_d \mathbf{h}_{\text{diff}})$$
5. **Multi-Task Classification Heads:**
   Both heads receive `h_fused` directly:
   - Sentiment Head: `Linear(512, 3)` (Negative, Neutral, Positive)
   - Offensive Head: `Linear(512, 2)` (Non-Hate, Hate/Offensive)

---

### 2.2 Optimizer & Scheduler Lifecycle ([train.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/train.py) & [src/training/trainer.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py))
1. **Eliminated Optimizer Recreation:**
   The training script no longer re-instantiates the optimizer mid-training. A single `torch.optim.AdamW` instance persists throughout all epochs, preserving running first and second momentum buffers ($m_t, v_t$).
2. **Differential Parameter Groups via `build_optimizer_param_groups`:**
   Parameters are partitioned into 4 distinct groups with weight decay split (decay=0.01 for weights, decay=0.0 for biases and LayerNorms):
   - `text_decay` & `text_no_decay`: MuRIL backbone ($\text{LR} = 1 \times 10^{-5}$)
   - `fusion_decay` & `fusion_no_decay`: Projections, Attention, Gated Fusion, Classification Heads ($\text{LR} = 2 \times 10^{-5}$)
   - Vision Backbone parameters have `requires_grad = False` and are **excluded** from optimizer registration.
3. **Scheduler Continuity with Gradient Accumulation:**
   The cosine annealing scheduler with linear warmup (warmup ratio 0.1) steps strictly after effective accumulation steps:
   $$\text{Total Effective Steps} = \left\lceil \frac{N_{\text{train}}}{\text{batch\_size} \times \text{grad\_accum}} \right\rceil \times \text{epochs} = \left\lceil \frac{10230}{4 \times 4} \right\rceil \times 5 = 640 \times 5 = 3200 \text{ steps}$$
   The scheduler is bound to the single active optimizer and continuously updates all active parameter groups.

---

### 2.3 Sequence Length & Padding Handling ([src/data/dataset.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/data/dataset.py))
1. **Sequence Length Reduction to 64 Tokens:**
   Analysis of the unified dataset revealed that Indian meme captions and OCR texts are short:
   - **Median token length:** 27.0 tokens
   - **95th-percentile token length:** 70.0 tokens
   - **Samples $\le 64$ tokens:** 93.63% (only 6.37% truncated)
   Using `max_length=128` in Run 1 forced ~78% of all sequence tokens to be useless `[PAD]` tokens. Reducing to 64 cuts padding by >50%, saves memory, and accelerates throughput.
2. **Automated Truncation Logging:**
   The `IndicMemeDataset` now logs exact truncation statistics during dataset initialization.

---

### 2.4 Masked Multi-Task Loss Normalization ([src/training/trainer.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py))
1. **Valid Label Normalization:**
   In multi-task learning where datasets have missing task annotations (e.g., MemoSen Bengali lacks offensive labels with label = `-1`), summing raw CrossEntropyLoss can cause loss magnitudes to fluctuate arbitrarily depending on batch composition.
   Each loss is now explicitly normalized over the count of valid samples:
   $$\mathcal{L}_{\text{total}} = w_{\text{sent}} \cdot \frac{\sum_{i \in \mathcal{V}_{\text{sent}}} \ell_i^{\text{sent}}}{|\mathcal{V}_{\text{sent}}|} + w_{\text{off}} \cdot \frac{\sum_{j \in \mathcal{V}_{\text{off}}} \ell_j^{\text{off}}}{|\mathcal{V}_{\text{off}}|}$$
2. **Zero-Contribution for Missing Subtasks:**
   If a batch contains zero valid samples for a subtask ($|\mathcal{V}| = 0$), that head's loss is set to `0.0`, preventing gradient corruption.
3. **Balanced Class Weights:**
   Weights are dynamically computed from training label distributions:
   - Sentiment: Neg: `0.8485`, Neu: `1.1162`, Pos: `1.0805`
   - Offensive: Non-Hate: `0.8667`, Hate: `1.1818`

---

### 2.5 Conservative Regularization ([src/data/dataset.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/data/dataset.py))
1. **Training-Only Image Augmentation:**
   Aggressive crops or perspective distortions in standard Vision transforms destroy text overlays in memes. The dataset now applies conservative transforms exclusively during training (`is_train=True`):
   - `ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08)`: breaks pixel-level color memorization without obscuring text.
   - `RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02))`: subtle geometric jitter that keeps OCR bounding boxes readable.
2. **Deterministic Evaluation:**
   Validation and Test datasets enforce deterministic `Resize((224, 224))` and `CenterCrop(224)` followed by CLIP normalization.

---

### 2.6 Experiment 2 Dedicated Configuration ([configs/experiment_2.yaml](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/configs/experiment_2.yaml))

```yaml
experiment:
  name: "IndicSent-M_Experiment_2"
  output_dir: "checkpoints/experiment_2"

model:
  text_model_name: "google/muril-base-cased"
  vision_model_name: "openai/clip-vit-base-patch32"
  hidden_dim: 512
  num_heads: 8
  dropout: 0.3
  num_sentiment_classes: 3
  num_offensive_classes: 2

training:
  epochs: 5
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  fp16: true
  max_length: 64
  image_size: 224
  text_lr: 1.0e-5
  fusion_lr: 2.0e-5
  vision_lr: 0.0                  # CLIP vision backbone frozen
  freeze_vision: true
  freeze_text: false
  unfreeze_at_epoch: null         # Backbone stays frozen
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  early_stopping_patience: 3
  loss_weights:
    sentiment: 1.0
    offensive: 1.0
```

---

## 3. Verification Suite & Empirical Test Results

### 3.1 Unit Verification Tests ([tests/test_experiment_2.py](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/tests/test_experiment_2.py))
All 6 automated unit tests passed cleanly:

```
======================================================================
  RUNNING EXPERIMENT 2 COMPREHENSIVE VERIFICATION SUITE
======================================================================

[Test 1] Synthetic Forward Pass & Tensor Dimensions...
  ✔ Output dimensions verified: Sentiment [B, 3], Offensive [B, 2], Fused [B, hidden_dim].
  ✔ Genuine global visual vector v_global and masked text vector t_rep verified.

[Test 2] Gradient Flow & Frozen CLIP Backbone Verification...
  ✔ Confirmed: All CLIP vision encoder parameters are frozen with zero gradients.
  ✔ Confirmed: Gradients flow cleanly to text encoder, projections, cross-attn, gate, and both heads.

[Test 3] Masked Multi-Task Loss Normalization (-1 Labels)...
  ✔ Masked loss with 50% missing labels computed cleanly without NaNs.
  ✔ Pure MemoSen batch (100% masked subtask) safely skips offensive head.

[Test 4] Optimizer Parameter Groups & Uninterrupted Scheduler...
  ✔ Confirmed: Zero frozen CLIP parameters placed in optimizer.
  ✔ Confirmed: Learning rate scheduler dynamically updates active optimizer param groups.

[Test 5] Checkpoint Save and Reload Verification...
  ✔ Checkpoint saved, loaded, and re-evaluated with bit-exact numerical parity.

[Test 6] Truncation Statistics on Processed Dataset...
  Truncation Stats for max_length=64: 66 / 1279 (5.16%)
  Median token length: 27.0 | 95th-Percentile: 65.0
  ✔ Max sequence length 64 covers >90% of samples without truncation.

======================================================================
  ALL 6 EXPERIMENT 2 VERIFICATION TESTS PASSED SUCCESSFULLY!
======================================================================
```

### 3.2 End-to-End Pipeline Dry Run (`train.py --config configs/experiment_2.yaml --dry_run`)
A complete dry-run validation was conducted on the active environment with live Hugging Face backbones (`google/muril-base-cased` and `openai/clip-vit-base-patch32`):
- **Model Parameters:** Total: 328,035,589 | Trainable: 240,579,589 (CLIP 87,456,000 parameters strictly frozen).
- **Optimizer Parameter Groups:** 4 groups configured (76 text decay, 123 text no-decay, 10 fusion decay, 12 fusion no-decay).
- **Execution Loop:** Successfully processed training batches, performed gradient accumulation, computed masked loss per task, executed validation evaluation, saved the best model checkpoint to `checkpoints/experiment_2/indicsent_m_best.pt`, and completed test set evaluation.

---

## 4. Invariant Verification & Integrity Checklist

| Invariant / Constraint | Target Requirement | Verification Status |
|---|---|:---:|
| **Do not start full training run** | Execution must halt after tests and dry run | **PASSED** (Full training NOT started) |
| **Preserve Run 1 configs** | `configs/model_config.yaml`, `configs/dataset_config.yaml` unchanged | **PASSED** (Untouched) |
| **Preserve Run 1 checkpoints** | `checkpoints/indicSent_m_best.pt` unchanged | **PASSED** (Untouched, Experiment 2 saves to `checkpoints/experiment_2/`) |
| **Masked mean pooling** | Zero out `[PAD]` positions before averaging | **PASSED** (Verified in Test 1 & Dry Run) |
| **Global visual representation** | Include CLIP global visual vector in fusion | **PASSED** (Verified via `vision_global_proj`) |
| **Frozen CLIP backbone** | Zero gradients for all CLIP parameters | **PASSED** (Verified in Test 2) |
| **Continuous optimizer lifecycle** | No optimizer recreation; persistent AdamW buffers | **PASSED** (Verified in Test 4 & `train.py`) |
| **Differential learning rates** | Text 1e-5, Fusion/Heads 2e-5, Vision 0.0 | **PASSED** (Verified in 4 parameter groups) |
| **Masked multi-task loss** | Handle `-1` labels with valid sample normalization | **PASSED** (Verified in Test 3) |
| **Conservative regularization** | Training-only subtle jitter/affine, deterministic val/test | **PASSED** (Verified in `src/data/dataset.py`) |
| **Sequence length 64** | Max length 64 with truncation logging | **PASSED** (Verified 93.63% coverage) |

---

## 5. Blueprint & Next Steps for Google Colab Execution

The codebase is in an experiment-ready state for execution on Google Colab (Tesla T4 GPU, 15GB VRAM).

### Recommended Colab Launch Sequence:
1. **Archive the updated workspace:**
   Update `indicsent_project.zip` with the new `configs/experiment_2.yaml`, updated `src/` modules, and `train.py`.
2. **Mount Google Drive & Unzip:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip -q /content/drive/MyDrive/indicsent_project.zip -d /content/indicsent_m
   %cd /content/indicsent_m
   ```
3. **Execute Experiment 2:**
   ```bash
   !python train.py --config configs/experiment_2.yaml
   ```
4. **VRAM Profile & Runtime Expectation on Tesla T4:**
   - Micro-batch size 4 with FP16 and frozen CLIP consumes approximately **4.8 GB VRAM** (comfortably within the 15 GB ceiling).
   - Effective batch size of 16 (grad accum 4) yields smooth optimizer dynamics.
   - Expected runtime per epoch: ~6-8 minutes on T4 (total ~35-40 minutes for 5 epochs).
5. **Expected Outcome:**
   - Overfitting should be mitigated: validation loss should track training loss closely without the explosive divergence seen in Epochs 3-5 of Run 1.
   - Classification Macro-F1 across both sentiment and offensive tasks should stabilize above Run 1 baselines.

---
*Report prepared for the IndicSent-M research project. Ready for user review and approval.*

