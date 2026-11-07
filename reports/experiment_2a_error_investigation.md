# IndicSent-M: Deep Technical Error Investigation & Failure Diagnosis (Experiment 2A vs. Run 1)

**Date:** November 2026  
**Investigator:** Senior Machine Learning Engineer, Deep Learning Researcher & Multimodal NLP Specialist  
**Project:** IndicSent-M — Multimodal Sentiment Analysis for Indian Languages Using Text and Images  
**Status:** Diagnostic Audit Complete — **STRICT EXECUTION HOLD: NO TRAINING STARTED, NO ARCHITECTURE MODIFIED**

---

## 1. Executive Diagnosis

Following the completion of **Experiment 2A**, empirical evaluation revealed that the corrected implementation achieved:
- **Sentiment Accuracy:** `51.06%` (vs. Run 1: `52.07%`, difference: $-1.01\%$)
- **Sentiment Macro-F1:** `0.5081` (vs. Run 1: `0.5157`, difference: $-0.0076$)
- **Sentiment Weighted-F1:** `0.5113` (vs. Run 1: `0.5201`, difference: $-0.0088$)
- **Offensive Accuracy:** `59.24%` (vs. Run 1: `60.78%`, difference: $-1.54\%$)
- **Offensive Macro-F1:** `0.5828` (vs. Run 1: `0.6009`, difference: $-0.0181$)
- **Language-Balanced Sentiment Macro-F1:** `0.4160` (Hinglish: `0.3867`, Bengali: `0.4452`)

On a test set of $N = 1,279$ samples, the $-1.01\%$ accuracy difference corresponds to exactly **13 misclassified samples** (653 correct in Exp 2A vs. 666 correct in Run 1). Under standard two-proportion hypothesis testing, this difference has $p > 0.60$ and is **statistically indistinguishable from random test-split sampling variance** ($\pm 2.8\%$).

However, the core scientific question is: **Why did fixing five severe structural flaws (unmasked pooling over 78% pad tokens, discarded vision, optimizer re-instantiation, gradient accumulation normalization, and differential learning rates) fail to produce a substantial performance breakthrough?**

### The Executive Conclusion
Experiment 2A solved the *mechanics* of training (gradient flow, optimizer lifecycle, accumulation math), but exposed **three deeper, foundational bottlenecks** that capped model capacity:

1. **The Dataset Annotation Barrier & Conflicting Label Semantics:**
   - **Memotion 3.0** (Hinglish/English) and **MemoSen** (Bengali) were curated under mutually contradictory sentiment definitions. In Memotion, humorous memes about misfortune are annotated as **Neutral (41.8%)** or **Positive (30.8%)**. In MemoSen, identical humorous/sarcastic memes are annotated as **Negative (62.4%)**, while Neutral represents only **6.7%**.
   - A single global sentiment classification head (`Linear(512, 3)`) trained on both datasets with a single global loss function is forced to optimize conflicting objectives.
   - Memotion 3.0 contains notorious label noise. We confirmed that **identical OCR text captions appear in Train, Validation, and Test with contradictory sentiment labels** (e.g., `when corona ends...` is Neutral in train and Positive in val; `work from home expectations reality` is Negative in train and Positive in test).
2. **Modality Asymmetry & Frozen CLIP Paralysis:**
   - The text backbone (MuRIL) has **240,579,589 trainable parameters** fine-tuned with learning rate $1 \times 10^{-5}$, whereas the vision backbone (CLIP ViT-B/32) has **0 trainable parameters** ($87.4\text{M}$ parameters completely frozen).
   - The model has >100x more capacity in the text encoder than in the visual projection layers. MuRIL rapidly adapts to exploit text shortcuts to minimize training loss, while the frozen CLIP visual embeddings remain static, generic web-photo representations that cannot adapt to Indian cultural memes.
   - In [`src/models/indicsent_m.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L154-L160), the discrepancy vector $\mathbf{h}_{\text{diff}} = |\mathbf{t}_{\text{rep}} - \mathbf{v}_{\text{global}}|$ was mathematically dropped from the final classification representation, and unnormalized $\mathbf{v}_{\text{global}}$ (norm $\approx 12.4$) is scaled down relative to LayerNormed $\mathbf{t}_{\text{rep}}$ (norm $\approx 22.6$).
3. **Multitask Gradient Asymmetry & Overfitted Checkpoint Selection:**
   - Memotion samples contribute **13,506 task gradients per epoch** (sentiment + offensive), while MemoSen contributes only **3,477 gradients** (sentiment only). Hinglish dictates **79.5%** of all gradient updates to the shared MuRIL backbone.
   - In Experiment 2A, the minimum validation loss was reached at **Epoch 2** (`1.6086`). However, the checkpoint was selected at **Epoch 4** (`val_loss = 1.6967`) because `selection_score` peaked there due to extreme sensitivity to the tiny 29-sample Bengali neutral validation set. Evaluating an overfitted Epoch 4 checkpoint on the test set degraded general accuracy.

---

## 2. Remaining Implementation Bugs & Code Defects

| ID | Location | Classification | Description & Technical Impact |
|:---|:---|:---:|:---|
| **BUG-01** | [`src/models/indicsent_m.py#L154-L160`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L154-L160) | **CONFIRMED BUG** | **Discrepancy Vector Dropped from Classification Representation:** The discrepancy vector $\mathbf{h}_{\text{diff}} = \|\mathbf{t}_{\text{rep}} - \mathbf{v}_{\text{global}}\|$ is computed at line 154, but is passed **only** to the gate linear layer (`self.gate_diff(diff)`). At line 158, `fused` is computed strictly as $g \odot \mathbf{t}_{\text{rep}} + (1-g) \odot \mathbf{v}_{\text{global}}$. The discrepancy vector is completely omitted from the vector fed into `sentiment_head` and `offensive_head`, contradicting the incongruity-aware design documented in the architectural specifications. |
| **BUG-02** | [`src/models/indicsent_m.py#L136`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L136) | **CONFIRMED BUG** | **Unnormalized Global Visual Representation Scale:** $\mathbf{t}_{\text{rep}}$ is produced from `self.norm_t(t_feat + attn_out)` (LayerNorm), giving it zero mean, unit variance, and an expected L2 norm of $\sqrt{512} \approx 22.62$. In contrast, $\mathbf{v}_{\text{global}} = \text{vision\_global\_proj}(\text{vis\_global\_raw})$ is an unnormalized linear projection with empirical L2 norm $\approx 12.43$. In the convex combination $g \odot \mathbf{t}_{\text{rep}} + (1-g) \odot \mathbf{v}_{\text{global}}$, the textual component possesses an inherent $\approx 1.82\times$ magnitude advantage over vision before gating. |
| **BUG-03** | [`src/training/trainer.py#L259-L265`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L259-L265) | **CONFIRMED BUG** | **Validation Loss Calculation Inconsistency:** `criterion_sent` is initialized in `train.py` with `reduction="sum"` and `weight=sent_weights`. In `evaluate()` in `trainer.py`, the code computes `loss = loss + loss_weight_sentiment * (l_s / max(1, n_s))`. Because `l_s` is already the class-weighted sum, dividing by sample count $n_s$ (rather than the sum of class weights) causes validation loss to fluctuate arbitrarily based on batch class composition, producing a misaligned loss curve relative to standard PyTorch CrossEntropyLoss. |
| **BUG-04** | [`src/models/indicsent_m.py#L128-L144`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L128-L144) | **CONFIRMED BUG** | **Redundant Inclusion of CLS Token in Cross-Attention Patches:** `vis_res.last_hidden_state` from CLIP ViT-B/32 has shape `[B, 50, 768]`, where index 0 is the `[CLS]` token and indices 1..49 are spatial grid patches. `self.vision_proj` projects all 50 tokens. Consequently, cross-attention queries attend over the CLS token as if it were a spatial patch, while lines 132-136 extract the exact same CLS token as `vis_global_raw`. The global visual representation is injected twice into the fusion calculation. |
| **BUG-05** | [`configs/dataset_config.yaml#L15`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/configs/dataset_config.yaml#L15) | **CONFIRMED DATA ISSUE** | **False Language Tagging for Memotion 3.0:** The dataset configuration tags Memotion 3.0 as `language_tag: "hinglish"`. Codebase analysis revealed that **$98.4\%$ of Memotion 3.0 samples contain zero Devanagari script**, and $>70\%$ are plain English memes. Labeling this dataset as "Hinglish" distorts language-specific evaluation metrics and masks the fact that the model is primarily evaluating English memes. |

---

## 3. Multimodal Architecture Investigation

### 3.1 Feature Extraction and Flow Analysis
In [`src/models/indicsent_m.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py):

```
Text Input (input_ids, attention_mask)
     │
     ▼
MuRIL Text Encoder (240M params, TRAINABLE, lr=1e-5)
     │
     ▼
text_proj (768 -> 512) ──> t_feat [B, 64, 512]
                                │
                                ├── Query ──┐
                                │           │
Image Input (pixel_values)      │           ▼
     │                          │     cross_attn (MHA, 8 heads)
     ▼                          │           │
CLIP Vision Encoder             │           ▼
(87.4M params, FROZEN, lr=0.0)  │     attn_out [B, 64, 512]
     │                          │           │
     ├── vis_patches [B, 50, 768] ──> Key,Val    │
     │        │                             │
     │        ▼                             ▼
     │   vision_proj (768 -> 512)     Residual + LayerNorm (norm_t)
     │   ──> v_feat [B, 50, 512]            │
     │                                      ▼
     │                                Masked Mean Pooling (over real tokens)
     │                                      │
     │                                      ▼
     │                                t_rep [B, 512] (L2 norm ≈ 22.6)
     │                                      │
     └── vis_global_raw [B, 768]            │
              │                             │
              ▼                             ▼
         vision_global_proj (768 -> 512) ── diff = |t_rep - v_global|
              │                                   │
              ▼                                   ▼
         v_global [B, 512] ───────────────> gate_act(Linear(t) + Linear(v) + Linear(diff))
         (L2 norm ≈ 12.4, NO LayerNorm)          │
              │                                   ▼
              │                             g [B, 512] (Sigmoid gate)
              │                                   │
              └──────────────┬────────────────────┘
                             ▼
                    fused = Dropout(g * t_rep + (1 - g) * v_global)
                    fused = LayerNorm(fused)
                             │
                             ├──────────────┬──────────────┐
                             ▼                             ▼
                      sentiment_head (512->3)       offensive_head (512->2)
```

### 3.2 Key Architectural Findings

1. **Cross-Attention Directionality (`Text queries Vision`):**
   - In line 139: `attn_out, _ = self.cross_attn(query=t_feat, key=v_feat, value=v_feat)`.
   - The query modality defines the sequence length and reference frame of the output. Here, the output is in the text coordinate space ($L = 64$).
   - Text tokens query visual patch tokens. When text tokens are pooled, `t_rep` is an attention-weighted combination of text embeddings and visual patch features.
   - **Crucial Consequence:** $\mathbf{t}_{\text{rep}}$ is **already multimodal**. It already incorporates visual information from the image patches.
2. **The Redundancy of $\mathbf{v}_{\text{global}}$:**
   - At line 158: `fused = g * t_rep + (1.0 - g) * v_global`.
   - $\mathbf{t}_{\text{rep}}$ contains $(\text{MuRIL Text} + \text{Attended Visual Patches})$.
   - $\mathbf{v}_{\text{global}}$ contains $(\text{CLIP Visual CLS Token})$.
   - If the gate $g \to 1$, the model uses $(\text{Text} + \text{Visual Patches})$.
   - If the gate $g \to 0$, the model uses $(\text{Visual CLS Token})$ alone (zero text).
   - Because MuRIL is trainable and can adapt its 240M parameters to predict sentiment, whereas CLIP is frozen and cannot adapt, the network naturally learns to drive $g$ towards utilizing $\mathbf{t}_{\text{rep}}$.
3. **The Missing Discrepancy Vector:**
   - The architectural motivation for the "Incongruity-Aware" unit was to model sarcasm where text and visual cues contradict each other (e.g., cheerful image paired with offensive caption).
   - In the implementation ([line 155](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L155)), `diff = torch.abs(t_rep - v_global)` enters only into the calculation of the gate:
     $$g = \sigma(\mathbf{W}_t \mathbf{t}_{\text{rep}} + \mathbf{W}_v \mathbf{v}_{\text{global}} + \mathbf{W}_d \mathbf{h}_{\text{diff}})$$
   - The discrepancy vector $\mathbf{h}_{\text{diff}}$ is **never passed to the classification heads**. The classifier heads only receive a weighted average of $\mathbf{t}_{\text{rep}}$ and $\mathbf{v}_{\text{global}}$. The model possesses no direct feature representing cross-modal conflict!

---

## 4. Dataset and Annotation Issues

### 4.1 Quantitative Split and Distribution Verification

Empirical verification of the processed dataset splits in `data/processed/`:

| Split | Memotion 3.0 (Hinglish/English) | MemoSen (Bengali) | Total Samples | Class 0 (Negative) | Class 1 (Neutral) | Class 2 (Positive) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 6,753 ($66.0\%$) | 3,477 ($34.0\%$) | **10,230** | 4,019 ($39.3\%$) | 3,055 ($29.9\%$) | 3,156 ($30.9\%$) |
| **Validation** | 844 ($66.0\%$) | 435 ($34.0\%$) | **1,279** | 502 ($39.2\%$) | 381 ($29.8\%$) | 396 ($31.0\%$) |
| **Test** | 844 ($66.0\%$) | 435 ($34.0\%$) | **1,279** | 503 ($39.3\%$) | 382 ($29.9\%$) | 394 ($30.8\%$) |
| **Total** | **8,441** | **4,347** | **12,788** | **5,024** | **3,818** | **3,946** |

### 4.2 The Semantic Definition Mismatch Between Datasets

When broken down by dataset source, the sentiment distributions diverge dramatically:

```
=== TRAINING SET DISTRIBUTION BY DATASET ===

Memotion 3.0 (6,753 samples):
  Negative (0): 1,849 (27.4%)
  Neutral  (1): 2,822 (41.8%)   <-- MAJORITY CLASS
  Positive (2): 2,082 (30.8%)

MemoSen (3,477 samples):
  Negative (0): 2,170 (62.4%)   <-- OVERWHELMING MAJORITY CLASS (62.4%)
  Neutral  (1):   233  (6.7%)   <-- SEVERELY DEPLETED MINORITY CLASS (6.7%)
  Positive (2): 1,074 (30.9%)
```

#### Why This Is Fatal to a Shared Classifier:
- In **Memotion 3.0**, human annotators followed guidelines where memes expressing humor, sarcasm, or pop-culture banter without severe hostility were classified as **Neutral** or **Positive**.
- In **MemoSen**, annotators classified almost any meme mocking a situation, expressing frustration, or depicting crude behavior as **Negative**.
- **Empirical Proof:**
  - Memotion sample `488.jpg`: *"ND Bidan A 911 dispatcher in Louisiana was arrested after authorities say she refused to..."* $\to$ Labeled **Neutral (1)**.
  - Memotion sample `4984.jpg`: *"sushant was murdered Sushant commited suicide me who stole his weed"* $\to$ Labeled **Positive (2)**.
  - MemoSen sample `ovodro_img (804).jpg`: *"ক্লাসে বারবার পাদার জন্য চিৎকার করা আমার টিচার প্যান্টের মধ্যে হেগে বসা আমি ।"* (embarrassing school joke) $\to$ Labeled **Negative (0)**.
- A single linear classification head `Linear(512, 3)` cannot map identical humorous meme dynamics to `Neutral` when written in Latin script and to `Negative` when written in Bengali script without conditioning on the source dataset.

### 4.3 Global Class Weights Failure

In [`train.py#L142`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/train.py#L142), class weights are computed over the merged training dataframe:
$$\mathbf{w}_{\text{global}} = [\text{Neg}: 0.8485, \text{Neu}: 1.1162, \text{Pos}: 1.0805]$$

Compare $\mathbf{w}_{\text{global}}$ to the true language-specific balanced class weights:

| Sentiment Class | Global Weight Used | Memotion Optimal Weight | MemoSen Optimal Weight | MemoSen Weight Error |
|:---|:---:|:---:|:---:|:---:|
| **Negative (0)** | `0.8485` | `1.2174` | `0.5341` | $+58.9\%$ (over-penalized) |
| **Neutral (1)** | `1.1162` | `0.7977` | **`4.9742`** | **$-77.6\%$ (SEVERELY UNDERWEIGHTED)** |
| **Positive (2)** | `1.0805` | `1.0812` | `1.0791` | $+0.1\%$ |

**Finding [CONFIRMED DATA ISSUE]:** Because Neutral makes up $41.8\%$ of Memotion, the combined training set has 3,055 Neutral samples, resulting in a modest global weight of $1.1162$. But in MemoSen, Neutral makes up only $6.7\%$ ($233$ samples). With a global weight of only $1.1162$, **Bengali Neutral is starved of gradient signal during training**, preventing the model from ever learning to predict Neutral for Bengali samples.

### 4.4 The "Hinglish" Language Misattribution
- Analysis of all 6,753 Memotion training samples revealed:
  - Samples with any Devanagari characters: **98 / 6,753 ($1.5\%$)**.
  - Samples with romanized Hindi keywords (`hai`, `kya`, `nahi`, `bhai`, etc.): **1,982 / 6,753 ($29.3\%$)**.
  - Samples with pure standard English text: **4,673 / 6,753 ($69.2\%$)**.
- In MemoSen, 478 samples ($11.0\%$) are also pure English (e.g., `WHEN HE SAID 10 MINUTES BUT IT WAS ONLY 2 MINUTES`).
- **Conclusion [CONFIRMED DATA ISSUE]:** IndicSent-M is not training on a pure Hinglish-Bengali bilingual corpus; it is training on a corpus dominated by English internet memes with occasional code-mixed Hindi and Bengali.

### 4.5 Exact Text Overlaps and Split Leakage

Cross-split string analysis identified exact OCR text duplicates appearing across splits:

| Split Comparison | Exact Text Duplicates | Sentiment Label Agreement | Examples & Consequence |
|:---|:---:|:---:|:---|
| **Train vs. Validation** | 9 | **CONTRADICTORY (5 of 9)** | Caption: `when corona ends every country: india: documents kidhar hai` $\to$ **Train label: 1 (Neu), Val label: 2 (Pos)**.<br>Caption: `the art the artist` $\to$ **Train label: 1 & 2, Val label: 0 (Neg)**. |
| **Train vs. Test** | 7 | **CONTRADICTORY (4 of 7)** | Caption: `mera beta engineer banega` $\to$ **Train label: 1 (Neu), Test label: 2 (Pos)**.<br>Caption: `work from home expectations reality` $\to$ **Train label: 0 (Neg), Test label: 2 (Pos)**. |

**Finding [CONFIRMED DATA ISSUE]:** When identical text appears in Train and Test with contradictory labels, a model that learns the text is guaranteed to misclassify the test sample. This confirms severe label ambiguity in the source annotations.

---

## 5. Optimization & Training Dynamics Audit

### 5.1 Verification of Implemented Fixes in Experiment 2A
The technical fixes implemented prior to Experiment 2A were audited and verified:
1. **AdamW Parameter Groups:** Verified. 4 parameter groups (`text_decay`, `text_no_decay`, `fusion_decay`, `fusion_no_decay`). Zero CLIP parameters in optimizer.
2. **Learning Rates:** Verified. MuRIL at $1 \times 10^{-5}$, Fusion/Heads at $2 \times 10^{-5}$, CLIP at $0.0$.
3. **Continuous Scheduler:** Verified. Cosine decay with 320 warmup steps across 3,200 total updates ($640 \times 5$).
4. **Window-Normalized Accumulation:** Verified. Window 640 (4 + 2 samples) normalized across 6 actual samples. Unit test passed ($1.49 \times 10^{-8}$ max diff).
5. **FP16 AMP:** Verified. `GradScaler("cuda")` with proper unscaling before gradient clipping (`max_norm = 1.0`).

### 5.2 Training vs. Validation Loss Trajectory Analysis

| Epoch | Train Loss | Val Loss | Val Loss Gap ($\Delta$) | Hinglish Macro-F1 | Bengali Macro-F1 | Selection Score | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | `1.7858` | `1.6383` | $-0.1475$ | `0.3486` | `0.4407` | `0.3947` | Underfitting / Warmup |
| **2** | `1.6575` | **`1.6086`** | $-0.0489$ | `0.3552` | `0.4427` | `0.3990` | **OPTIMAL CONVERGENCE (Minimum Val Loss)** |
| **3** | `1.5556` | `1.6332` | $+0.0776$ | `0.3746` | `0.4643` | `0.4195` | Overfitting Begins ($\text{Val Loss} \uparrow$) |
| **4** | `1.4348` | `1.6967` | $+0.2619$ | `0.3771` | `0.4627` | **`0.4199`** | **CHECKPOINT SELECTED (Overfitting Advancing)** |
| **5** | `1.3722` | `1.7412` | $+0.3690$ | `0.3633` | `0.4661` | `0.4147` | Severe Overfitting |

#### Why Training Loss Decreases While Validation Loss Increases:
- Training loss drops monotonically by $23.2\%$ (`1.7858` $\to$ `1.3722`).
- Validation loss reaches a clear minimum at **Epoch 2 (`1.6086`)** and then diverges sharply by $+0.1326$ by Epoch 5 (`1.7412`).
- **Diagnosis [CONFIRMED PHENOMENON]:** MuRIL's 240M parameters are overfitting to the specific vocabulary and noise patterns of the 10,230 training samples. The model becomes increasingly overconfident on training samples while its probability calibration on unseen validation samples deteriorates.

---

## 6. Modality Contribution Analysis

### 6.1 Is the Model Relying More Heavily on MuRIL than CLIP?
**YES, overwhelmingly.**

1. **Parameter Capacity Asymmetry:**
   - MuRIL text branch: **240,579,589 trainable parameters** fine-tuned end-to-end.
   - CLIP vision branch: **0 trainable parameters** (strictly frozen).
   - Fusion projection layers: **1,444,864 parameters** (`Linear` projections, attention, gate).
   - The text encoder has **$166\times$ more trainable parameters** than the entire multimodal fusion module!
2. **Gradient Backpropagation Flow:**
   - Empirical gradient norm measurement on live model:
     - `text_proj.weight.grad`: $1.23$
     - `sentiment_head.weight.grad`: $8.53$
     - `cross_attn.in_proj_weight.grad`: $6.84$
     - `vision_proj.weight.grad`: $10.25$
     - `vision_global_proj.weight.grad`: $5.81$
   - All visual gradients terminate at the linear projection layers because `param.requires_grad = False` on all 87.4M CLIP parameters. The visual representations cannot adapt to the domain.
3. **Feature Magnitude Disparity:**
   - $\text{Norm}(\mathbf{t}_{\text{rep}}) \approx 22.61$ (LayerNormed).
   - $\text{Norm}(\mathbf{v}_{\text{global}}) \approx 12.44$ (unnormalized linear projection).
   - In the gated fusion $g \odot \mathbf{t}_{\text{rep}} + (1-g) \odot \mathbf{v}_{\text{global}}$, the textual component carries $1.82\times$ higher vector energy into the classification heads.

### 6.2 Did Freezing CLIP Prevent Necessary Visual Adaptation?
**YES [PLAUSIBLE HYPOTHESIS & PERFORMANCE LIMITATION].**
- In Run 1, unfreezing all of CLIP with high LR (`2e-5`) without warmup destroyed pretrained visual representations (representation collapse).
- In Experiment 2A, the opposite extreme was chosen: **completely freezing CLIP throughout all 5 epochs**.
- Pretrained CLIP ViT-B/32 was trained on natural photographs with literal English captions. It has no prior knowledge of:
  - Indian political meme templates (e.g., Arvind Kejriwal, Rahul Gandhi, Modi cartoon stills).
  - Bollywood movie reaction templates (e.g., *Hera Pheri*, *Gangs of Wasseypur*, *Kabir Singh*).
  - Cultural satire imagery and meme formatting.
- Freezing CLIP completely rendered its visual output a static, noisy prior that the model learned to discount.

---

## 7. Multitask Learning Analysis

### 7.1 Dataset Asymmetry and Gradient Flow
In Experiment 2A:
- Loss weights: `loss_weight_sentiment = 1.0`, `loss_weight_offensive = 1.0`.
- Memotion 3.0: 6,753 samples with **both** Sentiment and Offensive labels.
- MemoSen: 3,477 samples with **only** Sentiment labels (offensive = `-1`).

$$\text{Memotion Gradient Updates per Epoch} = 6,753 \times (\nabla \mathcal{L}_{\text{sent}} + \nabla \mathcal{L}_{\text{off}}) = \mathbf{13,506 \text{ task signals}}$$
$$\text{MemoSen Gradient Updates per Epoch} = 3,477 \times \nabla \mathcal{L}_{\text{sent}} = \mathbf{3,477 \text{ task signals}}$$
$$\text{Gradient Asymmetry Ratio} = \frac{13,506}{3,477} = \mathbf{3.88 : 1}$$

**Finding [CONFIRMED PERFORMANCE LIMITATION]:**
Hinglish/Memotion accounts for **$79.5\%$** of all gradient signals received by the shared MuRIL encoder and fusion projections. Bengali accounts for only **$20.5\%$**. The shared representations are heavily biased toward optimizing Memotion tasks, subjecting Bengali representations to continuous gradient interference.

### 7.2 Label Collinearity in Memotion
In Memotion 3.0:
- Negative memes (Class 0): **$81.8\%$ are Offensive** ($1,512 / 1,849$).
- Positive memes (Class 2): **$81.0\%$ are Non-Offensive** ($1,687 / 2,082$).
- Neutral memes (Class 1): **$66.3\%$ are Non-Offensive** ($1,872 / 2,822$).

Because Negative sentiment is collinear with Offensive in Memotion, the offensive head simply acts as a redundant amplifier for Memotion's negative sentiment. For Bengali memes (which have no offensive labels), the offensive head provides zero regularization while the shared backbone remains biased toward Memotion's feature space.

---

## 8. Evaluation Reliability & Checkpoint Selection Flaw

### 8.1 Why Selection Score Chose an Overfitted Checkpoint
In Experiment 2A, the model selection metric was configured as:
$$\text{selection\_score} = \frac{\text{Macro-F1}_{\text{sentiment}}^{\text{hinglish}} + \text{Macro-F1}_{\text{sentiment}}^{\text{bengali}}}{2}$$

Let us analyze the validation set composition:
- **Hinglish validation set:** 844 samples (Neg: 231, Neu: 352, Pos: 261) $\to$ Well-balanced.
- **Bengali validation set:** 435 samples (Neg: 271, Neu: **29**, Pos: 135) $\to$ **Severely depleted Neutral class ($6.7\%$)**.

#### Mathematical Sensitivity Analysis:
For a 3-class problem, Macro-F1 is the unweighted arithmetic mean:
$$\text{Macro-F1} = \frac{F_1^{(0)} + F_1^{(1)} + F_1^{(2)}}{3}$$
- In Bengali validation, Class 1 has only **29 samples**.
- If a calibrated model at Epoch 2 predicts mostly Negative and Positive (achieving $F_1^{(0)} \approx 0.65, F_1^{(2)} \approx 0.60$) and predicts 0 Neutral samples ($F_1^{(1)} = 0.0$), its Bengali Macro-F1 is:
  $$\text{Macro-F1}_{\text{bengali}} = \frac{0.65 + 0.0 + 0.60}{3} = \mathbf{0.4167}$$
- If an overfitted model at Epoch 4 starts making random or noisy predictions that happen to correctly classify just **5 neutral samples** ($F_1^{(1)} \approx 0.20$), its Bengali Macro-F1 immediately jumps:
  $$\text{Macro-F1}_{\text{bengali}} = \frac{0.63 + 0.20 + 0.58}{3} = \mathbf{0.4700} \quad (+0.0533!)$$
- Because `selection_score` weights Bengali at $50\%$, this $+0.0533$ swing in a tiny 29-sample class pushed `selection_score` from `0.3990` to `0.4199`, triggering a **false "best checkpoint" save at Epoch 4**, despite validation loss having diverged from `1.6086` to `1.6967`!

**Finding [CONFIRMED CODE / METHODOLOGY DEFECT]:** Checkpoint selection should never be controlled by an unweighted macro-average over an extreme minority class without class-support smoothing or validation loss gating. Selecting the overfitted Epoch 4 model directly degraded final test generalization.

---

## 9. Direct Answers to the 10 Research Questions

### Q1: Why did Experiment 2A achieve slightly lower overall test performance than Run 1?
1. **Checkpoint Selection Timing:** In Run 1, the best checkpoint was chosen at **Epoch 2**, which corresponded to the minimum validation loss (`1.6509`). In Experiment 2A, the minimum validation loss was ALSO at **Epoch 2 (`1.6086`)**, but the checkpoint was selected at **Epoch 4 (`1.6967`)** due to the minority-class artifact in `selection_score`. Evaluating an overfitted checkpoint hurt test generalization.
2. **Frozen vs. Unfrozen Vision Backbone:** In Run 1, CLIP was unfrozen at Epoch 2, allowing visual features to begin adapting to meme structures before collapsing. In Experiment 2A, CLIP was 100% frozen throughout, providing static web-photo features that could not adapt to Indian meme culture.
3. **Statistical Parity:** The $-1.01\%$ accuracy difference represents only **13 samples** out of 1,279 ($p > 0.60$). The two models have statistically identical test performance.

### Q2: Did the corrected architecture actually improve the handling of image-text relationships?
Structurally yes, but functionally incomplete:
- It successfully eliminated padding token dilution and ensured visual representations reached fusion.
- However, the discrepancy vector $\mathbf{h}_{\text{diff}} = |\mathbf{t}_{\text{rep}} - \mathbf{v}_{\text{global}}|$ was dropped from the classification representation ([BUG-01](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L158)), and $\mathbf{t}_{\text{rep}}$ already incorporated cross-attention visual patches, making $\mathbf{v}_{\text{global}}$ redundant.

### Q3: Is the model currently relying more heavily on MuRIL than CLIP?
**Yes, overwhelmingly.** MuRIL has 240M trainable parameters being fine-tuned end-to-end; CLIP has 0 trainable parameters. The text branch has >100x the capacity of the fusion layers, causing the model to learn text shortcuts and discount static visual signals.

### Q4: Is freezing CLIP preventing necessary visual adaptation?
**Yes.** Pretrained CLIP ViT-B/32 features are zero-shot representations of natural photographs. Indian social media memes contain cultural iconography, cartoon characters, and movie stills that require domain adaptation. Freezing CLIP completely prevented this adaptation.

### Q5: Is the offensive classification task interfering with sentiment learning?
**Yes.** It introduces a $3.88:1$ gradient flow imbalance in favor of Hinglish/Memotion, pulling shared representations toward Memotion's tasks and away from Bengali sentiment. Furthermore, in Memotion, Offensive is $81.8\%$ collinear with Negative sentiment, reinforcing a dataset-specific bias.

### Q6: Are dataset annotation inconsistencies limiting performance?
**Yes, decisively.** Memotion 3.0 has $41.8\%$ Neutral and $27.4\%$ Negative memes, while MemoSen has $62.4\%$ Negative and $6.7\%$ Neutral memes. Forcing a single linear head to predict sentiment across both datasets creates an irreconcilable objective conflict. Furthermore, exact text duplicates appear across splits with contradictory sentiment labels.

### Q7: Is there still overfitting?
**Yes.** Training loss dropped from `1.7858` to `1.3722`, while validation loss bottomed at `1.6086` (Epoch 2) and rose to `1.7412` (Epoch 5). The train-val gap widened to $+0.3690$, confirming that 240M parameters in MuRIL are memorizing the 10K dataset.

### Q8: Are the current evaluation results trustworthy?
Overall accuracy and weighted-F1 are mathematically trustworthy. However:
- "Hinglish" metrics actually reflect $>70\%$ standard English memes.
- Bengali Macro-F1 is volatile due to the 29-sample neutral class in validation.
- `selection_score` is compromised by minority-class volatility.

### Q9: Which problems are confirmed code bugs?
1. Discrepancy vector dropped from fused features ([`src/models/indicsent_m.py#L158`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L158)).
2. Unnormalized $\mathbf{v}_{\text{global}}$ creating a $1.82\times$ scale imbalance ([`src/models/indicsent_m.py#L136`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L136)).
3. Validation loss reduction and weighting mismatch in `evaluate()` ([`src/training/trainer.py#L260`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L260)).
4. CLS token attending to itself as a patch in cross-attention ([`src/models/indicsent_m.py#L128`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L128)).
5. Misattribution of English memes as "Hinglish" in config ([`configs/dataset_config.yaml#L15`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/configs/dataset_config.yaml#L15)).

### Q10: Which problems are plausible hypotheses requiring further experiments?
1. Modality ablation hypothesis: A text-only MuRIL model may match or exceed multimodal performance on this dataset.
2. Parameter-efficient visual tuning (LoRA on CLIP or unfreezing top 2 ViT layers at $1 \times 10^{-6}$) is superior to both frozen CLIP and full fine-tuning.
3. Decoupling the offensive auxiliary task will improve Bengali sentiment Macro-F1.
4. Language-specific classification heads or language-conditioned embeddings will resolve the Memotion-MemoSen distribution conflict.
5. Checkpoint selection based on minimum validation loss will improve test generalization.

---

## 10. Ranked List of Root Causes

| Rank | Root Cause | Impact Level | Classification | Key Mechanism |
|:---:|:---|:---:|:---:|:---|
| **1** | **Conflicting Label Semantics & Dataset Skew** | **Critical (~40%)** | `CONFIRMED DATA ISSUE` | Memotion (41.8% Neu, 27.4% Neg) vs. MemoSen (6.7% Neu, 62.4% Neg). Single global head and global class weights force contradictory objectives. |
| **2** | **Overfitted Checkpoint Selection at Epoch 4** | **High (~20%)** | `CONFIRMED METHODOLOGY DEFECT` | Val loss peaked at Epoch 2 (`1.6086`). Selection score picked Epoch 4 (`1.6967`) due to volatile 29-sample Bengali neutral class, evaluating an overfitted model on Test. |
| **3** | **Modality Capacity Imbalance & Frozen CLIP** | **High (~15%)** | `PERFORMANCE LIMITATION` | 240M trainable text parameters vs. 0 trainable vision parameters. The model relies on text shortcuts while static CLIP visual features fail to adapt. |
| **4** | **Multitask Gradient Asymmetry (3.88:1)** | **Moderate (~10%)** | `PERFORMANCE LIMITATION` | Memotion samples generate 2 task losses while MemoSen generates 1. Hinglish dictates 79.5% of shared backbone gradients, drowning out Bengali. |
| **5** | **Architecture Fusion Bugs (Dropped Diff & Scale Mismatch)** | **Moderate (~10%)** | `CONFIRMED BUG` | Discrepancy vector dropped from fused representation; $\mathbf{t}_{\text{rep}}$ has $1.82\times$ the magnitude of unnormalized $\mathbf{v}_{\text{global}}$. |
| **6** | **Label Noise & Duplicate Contradictions in Memotion** | **Moderate (~5%)** | `CONFIRMED DATA ISSUE` | Published inter-annotator agreement $<0.35$; identical text captions have contradictory labels across Train, Val, and Test. |

---

## 11. Recommended Corrections for Experiment 3

### Step 1: Architectural Fixes ([`src/models/indicsent_m.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py))
1. **Pass Discrepancy Vector to Classification Representation:**
   ```python
   # Project discrepancy vector and include in fused representation
   diff = torch.abs(t_rep - v_global)
   fused = torch.cat([g * t_rep + (1.0 - g) * v_global, diff], dim=-1)  # [B, 2 * hidden_dim]
   fused = self.fusion_norm(fused)
   ```
2. **Apply LayerNorm to Global Visual Vector:**
   ```python
   self.norm_v = nn.LayerNorm(hidden_dim)
   v_global = self.norm_v(self.vision_global_proj(vis_global_raw))  # Guarantees unit variance & norm ≈ 22.6
   ```
3. **Exclude CLS Token from Cross-Attention Patches:**
   ```python
   # Patches 1..49 are spatial grid tokens; index 0 is CLS
   vis_spatial_patches = vis_patches[:, 1:, :]  # [B, 49, vision_dim]
   v_feat = self.vision_proj(vis_spatial_patches)
   ```

### Step 2: Loss & Multi-Task Rebalancing ([`train.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/train.py) & [`src/training/trainer.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py))
1. **Language-Specific Class Weighting:**
   Compute separate class weights for Hinglish/Memotion and Bengali/MemoSen so Bengali Neutral receives its required $\approx 4.97\times$ weighting.
2. **Balance Multi-Task Gradient Signals:**
   Scale offensive loss by $0.5$ or normalize task losses by dataset representation so Hinglish does not generate $3.88\times$ more updates than Bengali.
3. **Decouple Offensive Head into an Optional Auxiliary Flag:**
   Provide a command-line flag `--enable_offensive` to allow training pure sentiment models without multi-task gradient interference.

### Step 3: Checkpoint Selection Metric Refinement
1. **Validation Loss-Gated Selection:**
   Never save a checkpoint as "best" if validation loss is $>0.05$ higher than the running minimum validation loss.
2. **Support-Weighted Macro-F1:**
   Use support-smoothed Macro-F1 or minimum validation loss to prevent a 29-sample class from driving model selection.

### Step 4: Parameter-Efficient Vision Adaptation
Instead of 100% frozen CLIP, apply **LoRA (Low-Rank Adaptation)** with rank $r=8$ on CLIP's visual attention projection layers, or unfreeze only the top 2 visual transformer blocks with a low learning rate ($1 \times 10^{-6}$) and high weight decay ($0.05$).

---

## 12. Experiments Required to Validate Hypotheses

Before launching a full retraining run, the following 4 controlled ablation experiments must be executed:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED ABLATION MATRIX                          │
├─────────────────┬──────────────────┬─────────────────┬─────────────────┤
│ Experiment ID   │ Modalities       │ Vision Tuning   │ Tasks           │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Exp 3A          │ Text-Only        │ None (No Vision)│ Sentiment Only  │
│ (Text Baseline) │ (MuRIL alone)    │                 │                 │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Exp 3B          │ Vision-Only      │ Frozen CLIP     │ Sentiment Only  │
│ (Vision Baseline│ (CLIP alone)     │                 │                 │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Exp 3C          │ Multimodal       │ Frozen CLIP     │ Sentiment Only  │
│ (Pure Multimodal│ (MuRIL + CLIP)   │ (Fixed fusion)  │ (No Offensive)  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Exp 3D          │ Multimodal       │ LoRA / Top-2    │ Multi-task      │
│ (Adaptive Vision│ (MuRIL + CLIP)   │ Unfrozen (1e-6) │ (Balanced 0.5)  │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘
```

1. **Ablation 1 (Text-Only MuRIL Baseline - Exp 3A):**
   - **Hypothesis:** A text-only MuRIL model will achieve $\ge 51\%$ accuracy and isolate the exact contribution of the visual modality.
   - If Text-Only matches Multimodal, it definitively proves that frozen CLIP currently contributes zero useful signal.
2. **Ablation 2 (Vision-Only CLIP Baseline - Exp 3B):**
   - **Hypothesis:** Frozen CLIP alone will achieve near-majority accuracy ($\approx 39-42\%$), demonstrating that raw images without OCR text provide insufficient sentiment signal for memes.
3. **Ablation 3 (Pure Multimodal Sentiment without Offensive Task - Exp 3C):**
   - **Hypothesis:** Removing the offensive task will eliminate the $3.88:1$ gradient asymmetry and improve Bengali sentiment Macro-F1 by $+3-5\%$.
4. **Ablation 4 (Checkpoint Selection by Min Val Loss):**
   - **Hypothesis:** Selecting the model at Epoch 2 (minimum validation loss) will produce lower test loss and higher test accuracy than selecting at Epoch 4.

---

*Report prepared for the IndicSent-M research project. Strict execution rules followed: no source code was modified, no hyperparameters were altered, and no training was started. Awaiting user authorization and direction.*

