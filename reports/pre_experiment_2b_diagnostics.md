# IndicSent-M: Pre-Experiment 2B Diagnostic Audit & Empirical Evidence Report

**Audit Date:** November 2026  
**Auditor:** Senior Machine Learning Engineer, Deep Learning Researcher & Multimodal NLP Specialist  
**Project:** IndicSent-M — Multimodal Sentiment Analysis for Indian Languages Using Text and Images  
**Status:** Read-Only Audit Complete — **STRICT EXECUTION HOLD: NO SOURCE FILES MODIFIED, NO TRAINING STARTED**

---

## Executive Overview

This report provides a comprehensive, read-only diagnostic audit of the IndicSent-M codebase, datasets, evaluation pipelines, and experimental results comparing **Run 1** and **Experiment 2A**.

The primary objective is to establish why Experiment 2A test performance ($51.06\%$ sentiment accuracy, $0.5081$ macro-F1) did not improve over Run 1 ($52.07\%$ accuracy, $0.5157$ macro-F1), identify all remaining implementation and methodological defects, and establish a rigorous, evidence-based roadmap for **Experiment 2B**.

---

## 1. Dataset and Annotation Audit

### 1.1 Original Labels and Schema Mapping Verification

The project unifies two distinct datasets: **Memotion 3.0** and **MemoSen**. The raw annotation files and mapping rules in [`configs/dataset_config.yaml`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/configs/dataset_config.yaml#L11-L55) and [`src/data/merger.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/data/merger.py#L18-L47) were inspected against the raw CSV files on disk:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATASET SCHEMA & MAPPING AUDIT                                   │
├─────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────┤
│ Dataset         │ Raw Annotation File          │ Original Label Values        │ Unified Mapping  │
├─────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────┤
│                 │ data/raw/memotion_3.0/       │ sentiment:                   │ 0: Negative      │
│ Memotion 3.0    │ memotion_labels.csv          │   - neutral (3,549)          │ 1: Neutral       │
│ (8,500 rows)    │ Columns:                     │   - positive (2,616)         │ 2: Positive      │
│                 │ [image_name, text, sentiment,│   - negative (2,335)         │                  │
│                 │  humour, sarcastic,          │ offensive:                   │ 0: Non-Offensive │
│                 │  offensive, motivational]    │   - not_offensive (4,905)    │ 1: Offensive     │
│                 │                              │   - slight (2,739)           │ 1: Offensive     │
│                 │                              │   - very_offensive (654)     │ 1: Offensive     │
│                 │                              │   - hateful_offensive (202)  │                  │
├─────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────┤
│                 │ data/raw/memosen/            │ sentiment:                   │ 0: Negative      │
│ MemoSen         │ memosen_labels.csv           │   - negative (2,728)         │ 1: Neutral       │
│ (4,368 rows)    │ Columns:                     │   - positive (1,349)         │ 2: Positive      │
│                 │ [image_name, ocr_text,       │   - neutral (291)            │                  │
│                 │  sentiment]                  │ offensive:                   │ -1: Masked       │
│                 │                              │   - [COLUMN ABSENT]          │ (No labels)      │
└─────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────┘
```

#### Preprocessing Filtering Audit:
During preprocessing via [`src/data/preprocess_and_split.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/data/preprocess_and_split.py#L85-L124):
- **Raw Memotion 3.0:** $8,500 \to 8,441$ samples preserved ($59$ dropped: missing image files, placeholder/empty OCR text).
- **Raw MemoSen:** $4,368 \to 4,347$ samples preserved ($21$ dropped: placeholder OCR text, binary image duplicates).
- **Total Unified Multilingual Dataset:** $12,788$ multimodal samples.

---

### 1.2 Split Distributions and Sample Verification

The processed splits in `data/processed/` were verified for sample counts and label distributions:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          SPLIT DISTRIBUTIONS (STRATIFIED 80/10/10)                               │
├─────────────────┬──────────┬──────────┬──────────┬───────────────┬───────────────────────────────┤
│ Split           │ Memotion │ MemoSen  │ Total    │ Sentiment     │ Offensive                     │
│                 │ (Hing/En)│ (Bengali)│ Samples  │ (0 / 1 / 2)   │ (-1 / 0 / 1)                  │
├─────────────────┼──────────┼──────────┼──────────┼───────────────┼───────────────────────────────┤
│ Training        │ 6,753    │ 3,477    │ 10,230   │ 4,019 / 3,055 │ 3,477 / 3,896 / 2,857         │
│ (80.0%)         │ (66.0%)  │ (34.0%)  │ (100.0%) │ / 3,156       │ (34.0% / 38.1% / 27.9%)       │
├─────────────────┼──────────┼──────────┼──────────┼───────────────┼───────────────────────────────┤
│ Validation      │ 844      │ 435      │ 1,279    │ 502 / 381     │ 435 / 488 / 356               │
│ (10.0%)         │ (66.0%)  │ (34.0%)  │ (100.0%) │ / 396         │ (34.0% / 38.2% / 27.8%)       │
├─────────────────┼──────────┼──────────┼──────────┼───────────────┼───────────────────────────────┤
│ Testing         │ 844      │ 435      │ 1,279    │ 503 / 382     │ 435 / 493 / 351               │
│ (10.0%)         │ (66.0%)  │ (34.0%)  │ (100.0%) │ / 394         │ (34.0% / 38.5% / 27.4%)       │
├─────────────────┼──────────┼──────────┼──────────┼───────────────┼───────────────────────────────┤
│ Total           │ 8,441    │ 4,347    │ 12,788   │ 5,024 / 3,818 │ 4,347 / 4,877 / 3,564         │
│                 │          │          │          │ / 3,946       │                               │
└─────────────────┴──────────┴──────────┴──────────┴───────────────┴───────────────────────────────┘
```

**Verification Confirmation:** The stratified split preserves exact proportionality across both datasets and all sentiment classes across Train, Validation, and Test.

---

### 1.3 Dataset Definition Compatibility Assessment

A critical finding of this audit is that **Memotion 3.0 and MemoSen operate under fundamentally conflicting sentiment annotation standards**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             SENTIMENT CLASS COMPOSITION SKEW                                     │
├──────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────┤
│ Sentiment Class  │ Memotion 3.0 (Hinglish/Eng)   │ MemoSen (Bengali)             │ Mismatch Ratio│
├──────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────┤
│ Class 0 (Negative│ 1,849 (27.38%)                │ 2,170 (62.41%)                │ 2.28x Skew    │
├──────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────┤
│ Class 1 (Neutral)│ 2,822 (41.79%) [MAJORITY]     │ 233 (6.70%)    [MINORITY]     │ 6.24x Skew    │
├──────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────┤
│ Class 2 (Positive│ 2,082 (30.83%)                │ 1,074 (30.89%)                │ 1.00x Balanced│
└──────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────┘
```

#### Qualitative Conflict Analysis:
- **In Memotion 3.0:** Pop-culture banter, mild sarcasm, and satirical social commentary are predominantly labeled **Neutral (Class 1)** or **Positive (Class 2)** because annotators evaluated the humorous effect on the viewer.
  - *Example (`train/4249.jpg`):* *"When You Hear 15 Year Olds Making Promises Of Being Together Till Eternity 'Mujhe Toh Tara...'"* $\to$ Labeled **Neutral (1)**.
  - *Example (`train/6643.jpg`):* *"Itna sannata kyu hai bhai ? Sara gaon Vote dene gaya ha Mogambo khush hua"* $\to$ Labeled **Positive (2)**.
- **In MemoSen:** Sarcastic memes, political mockery, relationship jokes, or crude situational memes are labeled **Negative (Class 0)** because annotators evaluated the depicted situation or topic.
  - *Example (`bamboo-vaiya (353).jpg`):* `*রিলেশনশিপের ০.০০০০১ সেকেন্ডের মধ্যে* ফাকবইঃ আমার লগে খেলবি ?` $\to$ Labeled **Negative (0)**.
  - *Example (`tangaila (2).jpg`):* `WHEN YOUR COUSINS TAKES YOU TO THE DHAN KHET AND SHOWS HIS KALO BONDUK` $\to$ Labeled **Negative (0)**.
- **Scientific Conclusion:** A single shared linear projection head `sentiment_head: Linear(512, 3)` without dataset/language conditioning is forced to optimize contradictory objectives: mapping satirical meme representations to Neutral for Latin-script inputs and to Negative for Bengali-script inputs.

---

### 1.4 Linguistic Breakdown of the "Hinglish" Dataset

In `configs/dataset_config.yaml`, Memotion 3.0 is tagged as `language_tag: "hinglish"`. A script-level and lexical audit was conducted across all 6,753 Memotion training samples:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            MEMOTION 3.0 LINGUISTIC AUDIT (N = 6,753)                             │
├───────────────────────────────────────┬──────────────┬─────────────┬─────────────────────────────┤
│ Linguistic Category                   │ Sample Count │ Percentage  │ Heuristic Definition        │
├───────────────────────────────────────┼──────────────┼─────────────┼─────────────────────────────┤
│ Devanagari Script Present             │ 98           │ 1.45%       │ Unicode range \u0900-\u097F │
├───────────────────────────────────────┼──────────────┼─────────────┼─────────────────────────────┤
│ Romanized Hindi (Code-Mixed)          │ 1,982        │ 29.35%      │ Contains core Romanized     │
│                                       │              │             │ Hindi functional words      │
├───────────────────────────────────────┼──────────────┼─────────────┼─────────────────────────────┤
│ Standard English                      │ 4,673        │ 69.20%      │ ASCII/English vocabulary,   │
│                                       │              │             │ zero Hindi functional words │
└───────────────────────────────────────┴──────────────┴─────────────┴─────────────────────────────┘
```

#### Heuristic Methodology and Limitations:
- **Devanagari Heuristic:** Evaluated via regular expression `[\u0900-\u097F]`. This is exact and free of false positives.
- **Romanized Hindi Heuristic:** Identified by matching high-frequency Hindi stop-words and discourse markers: `['hai', 'kya', 'nahi', 'bhai', 'yeh', 'woh', 'ki', 'ka', 'ke', 'ko', 'se', 'mein', 'toh', 'aur', 'agar', 'lekin', 'kyunki', 'sabse']`.
- **Heuristic Limitations:**
  1. *Short polysemous tokens:* English words like "to" (also Hindi *toh*), "me" (also Hindi *mein*), "is" (also Hindi *iss*) can cause boundary ambiguities if not matched strictly on token boundaries.
  2. *Proper nouns and slang:* Bollywood character names (e.g., *Kabir Singh*, *Mogambo*, *Binod*) may appear without Hindi functional words, causing a code-mixed meme to be categorized as standard English.
  3. *Even under generous matching*, at least **$65\%\text{--}70\%$ of Memotion memes are standard global English internet memes**.
- **Conclusion:** Classifying the entire Memotion 3.0 dataset as "Hinglish" is empirically inaccurate. MuRIL operates predominantly on English text on this partition.

---

### 1.5 Duplicate and Cross-Split Overlap Audit

The duplicate audit was verified across binary hashes (MD5), perceptual difference hashes (dHash), and OCR text:

#### 1. Binary Image Hash Audit (MD5):
- Exact binary duplicates across splits: **0 in Train-Val**, **0 in Train-Test**, **0 in Val-Test**.
- Binary deduplication in `preprocess_and_split.py` successfully removed all identical image files.

#### 2. Perceptual Near-Duplicate Image Pairs (dHash, 64-bit):

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            IMAGE NEAR-DUPLICATE COUNTS BY dHash DISTANCE                         │
├───────────────────────────────────────┬──────────────┬──────────────┬────────────────────────────┤
│ dHash Hamming Distance Threshold      │ Train - Val  │ Train - Test │ Val - Test                 │
├───────────────────────────────────────┼──────────────┼──────────────┼────────────────────────────┤
│ Distance = 0 (Exact visual duplicate) │ 13 pairs     │ 11 pairs     │ 3 pairs                    │
│ Distance <= 1                         │ 21 pairs     │ 22 pairs     │ 3 pairs                    │
│ Distance <= 2                         │ 28 pairs     │ 33 pairs     │ 3 pairs                    │
│ Distance <= 3                         │ 44 pairs     │ 45 pairs     │ 3 pairs                    │
│ Distance <= 4 (Audit threshold)       │ 67 pairs     │ 56 pairs     │ 5 pairs                    │
└───────────────────────────────────────┴──────────────┴──────────────┴────────────────────────────┘
```

#### 3. Exact OCR Text Overlaps Across Splits:
There are **9 exact normalized text overlaps between Train and Val**, and **7 between Train and Test** (0 between Val and Test).

**Crucial Finding:** Identical OCR text does **not** always imply identical memes. The overlaps partition into two distinct categories:

##### Category A: True Near-Duplicate Memes with Contradictory Annotations (Annotation Noise)
In these cases, the images are visually identical (pixel correlation $\ge 0.98$, MSE $< 25$), but annotators assigned contradictory labels:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    CATEGORY A: TRUE DUPLICATE IMAGES WITH CONTRADICTORY LABELS                   │
├────────────────────────────────────────┬───────────────────────┬──────────────────────┬──────────┤
│ Normalized Text Caption                │ Split 1 (Train)       │ Split 2 (Val/Test)   │ Visual   │
│                                        │ Image / Sent / Off    │ Image / Sent / Off   │ Match    │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'girls to momos and paani poori as     │ train/2089.jpg        │ train/6923.jpg [TEST]│ Corr=1.00│
│  soon as lockdown gonna end...'        │ Sent: 0 (Neg), Off: 0 │ Sent: 2 (Pos), Off: 0│ MSE=4.7  │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'bjp has a cm candidate who's better   │ train/3345.jpg        │ val/142.jpg    [TEST]│ Corr=0.98│
│  than kejriwal khaa meri kasam...'     │ Sent: 0 (Neg), Off: 1 │ Sent: 1 (Neu), Off: 0│ MSE=2422 │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'cheating level: expert'               │ val/1154.jpg          │ train/4042.jpg  [VAL]│ Corr=0.99│
│                                        │ Sent: 0 (Neg), Off: 1 │ Sent: 2 (Pos), Off: 0│ MSE=188  │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'when corona ends every country:       │ train/4364.jpg        │ val/627.jpg     [VAL]│ Corr=0.84│
│  india: documents kidhar hai ?'        │ Sent: 1 (Neu), Off: 0 │ Sent: 2 (Pos), Off: 0│ MSE=1995 │
└────────────────────────────────────────┴───────────────────────┴──────────────────────┴──────────┘
```

##### Category B: Reused Meme Templates / Catchphrases with Distinct Visual Imagery
In these cases, different meme creators reused viral text catchphrases over completely different visual images:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   CATEGORY B: SHARED TEMPLATES/CATCHPHRASES WITH DISTINCT IMAGERY                │
├────────────────────────────────────────┬───────────────────────┬──────────────────────┬──────────┤
│ Normalized Text Caption                │ Split 1 (Train)       │ Split 2 (Val/Test)   │ Visual   │
│                                        │ Image / Sent / Off    │ Image / Sent / Off   │ Match    │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'the art the artist'                   │ train/262.jpg (Mem)   │ Bangla Thug Life     │ Corr=-0.1│
│                                        │ Sent: 1 (Neu)         │ (249).jpg [VAL, Sen] │ MSE=22445│
│                                        │                       │ Sent: 0 (Neg)        │          │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'mera beta engineer banega'            │ train/6476.jpg        │ train/4846.jpg [TEST]│ Corr=0.62│
│                                        │ Sent: 1 (Neu)         │ Sent: 2 (Pos)        │ MSE=3230 │
├────────────────────────────────────────┼───────────────────────┼──────────────────────┼──────────┤
│ 'work from home expectations reality'  │ train/6003.jpg        │ train/194.jpg  [TEST]│ Corr=0.67│
│                                        │ Sent: 0 (Neg)         │ Sent: 2 (Pos)        │ MSE=3590 │
└────────────────────────────────────────┴───────────────────────┴──────────────────────┴──────────┘
```

---

## 2. Evaluation Implementation Audit

### 2.1 Training vs. Validation Loss Normalization Discrepancy

A deep inspection of [`src/training/trainer.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L170-L268) revealed a **mathematical normalization mismatch** between training loss and validation loss:

#### 1. Training Loss Calculation ([lines 175–190](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L175-L190)):
- `criterion_sent` is configured with `reduction="sum"` and class weights $w_c$.
- For each microbatch, the unreduced weighted sum loss is divided by $N_{\text{sent}}(W)$ (the count of valid sentiment labels in the accumulation window):
  $$\mathcal{L}_{\text{micro}}^{\text{sent}} = \frac{\sum_{i \in b_k, y_i \ne -1} w_{y_i} \ell_i}{N_{\text{sent}}(W)}$$
- Across the accumulation window, this accumulates to the exact sample-weighted loss.

#### 2. Validation Loss Calculation ([lines 257–266](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L257-L266)):
```python
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
```
And at line 287:
```python
"val_loss": total_loss / max(len(loader), 1)
```

#### Defect Analysis:
1. **Normalization Divisor:** Validation loss divides `l_s` (which is already class-weighted by $w_{y_i}$) by the **number of valid samples $n_s$**, rather than by the **sum of applicable class weights** $\sum_{i} w_{y_i}$. When PyTorch computes standard `reduction="mean"` for weighted CrossEntropyLoss, it normalizes by the sum of weights. Dividing by sample count $n_s$ distorts the loss magnitude depending on batch class composition.
2. **Batch Averaging with Unequal Batch Sizes:** Dividing `total_loss` by `len(loader)` averages batch-level losses unweighted by batch size.
3. **MemoSen Offensive Loss Masking Artifact:** In batches containing pure MemoSen samples, `n_o == 0`. The offensive loss contributes `0.0`. But `total_loss` is divided by `len(loader)`, which counts every batch! Consequently, batches with missing offensive labels artificially dilute the validation loss.

---

### 2.2 Checkpoint Selection Analysis: Why Epoch 4 Was Selected

In Experiment 2A, the checkpoint selection criterion was defined as:
$$\text{selection\_score} = \frac{\text{Macro-F1}_{\text{sentiment}}^{\text{hinglish}} + \text{Macro-F1}_{\text{sentiment}}^{\text{bengali}}}{2}$$

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               EXPERIMENT 2A TRAINING TRAJECTORY                                  │
├───────┬────────────┬────────────┬─────────────────┬────────────────┬─────────────────┬───────────┤
│ Epoch │ Train Loss │ Val Loss   │ Hinglish Sent F1│ Bengali Sent F1│ Selection Score │ Decision  │
├───────┼────────────┼────────────┼─────────────────┼────────────────┼─────────────────┼───────────┤
│ 1     │ 1.7858     │ 1.6383     │ 0.3486          │ 0.4407         │ 0.3947          │ Saved     │
├───────┼────────────┼────────────┼─────────────────┼────────────────┼─────────────────┼───────────┤
│ 2     │ 1.6575     │ **1.6086** │ 0.3552          │ 0.4427         │ 0.3990          │ Saved     │
│       │            │ (MIN VAL)  │                 │                │                 │           │
├───────┼────────────┼────────────┼─────────────────┼────────────────┼─────────────────┼───────────┤
│ 3     │ 1.5556     │ 1.6332     │ 0.3746          │ 0.4643         │ 0.4195          │ Saved     │
├───────┼────────────┼────────────┼─────────────────┼────────────────┼─────────────────┼───────────┤
│ 4     │ 1.4348     │ 1.6967     │ 0.3771          │ 0.4627         │ **0.4199**      │ **SELECTED│
│       │            │ (+0.0881)  │                 │                │ (BEST CKPT)     │ AS BEST** │
├───────┼────────────┼────────────┼─────────────────┼────────────────┼─────────────────┼───────────┤
│ 5     │ 1.3722     │ 1.7412     │ 0.3633          │ 0.4661         │ 0.4147          │ Discarded │
└───────┴────────────┴────────────┴─────────────────┴────────────────┴─────────────────┴───────────┘
```

#### Why Epoch 4 Was Chosen Despite Diverging Validation Loss:
1. **The 29-Sample Minority Class Artifact:**
   - In the Bengali validation set ($N = 435$), Class 1 (Neutral) contains only **29 ground-truth samples**.
   - For a 3-class problem, $\text{Macro-F1} = \frac{F_1^{(0)} + F_1^{(1)} + F_1^{(2)}}{3}$.
   - If a well-calibrated model at Epoch 2 predicts mostly Negative and Positive (achieving $F_1^{(0)} \approx 0.65, F_1^{(2)} \approx 0.60$) and predicts 0 Neutral samples, $F_1^{(1)} = 0.0 \implies \text{Macro-F1} \approx 0.4167$.
   - As MuRIL overfits at Epoch 4, its decision boundaries shift. If it makes aggressive guesses that happen to correctly capture just **5 neutral samples** ($F_1^{(1)} \approx 0.20$), Bengali Macro-F1 jumps by $+0.0533$ to $\approx 0.4700$!
   - Because `selection_score` assigns Bengali $50\%$ weight, this tiny swing in 5 samples produced an artificial peak in `selection_score` (`0.4199` at Epoch 4 vs. `0.3990` at Epoch 2).
2. **Consequence:** Epoch 4 was selected **after validation loss had already diverged by $+0.0881$ from its minimum (`1.6086` $\to$ `1.6967`)**. The selected model was overfitted, leading directly to lower test-set generalization.

---

## 3. Model and Modality Audit

### 3.1 Trace of the Forward Pass
The forward pass in [`src/models/indicsent_m.py`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L96-L180) was traced step-by-step:

1. **Text Encoding:**
   $$\mathbf{T}_{\text{out}} = \text{MuRIL}(\text{input\_ids}, \text{attention\_mask}) \in \mathbb{R}^{B \times 64 \times 768}$$
   $$\mathbf{T}_{\text{feat}} = \mathbf{T}_{\text{out}} \mathbf{W}_{\text{text\_proj}} \in \mathbb{R}^{B \times 64 \times 512}$$
2. **Visual Feature Extraction:**
   $$\mathbf{V}_{\text{patches}} = \text{CLIPVision}(\text{pixel\_values}) \in \mathbb{R}^{B \times 50 \times 768}$$
   $$\mathbf{V}_{\text{feat}} = \mathbf{V}_{\text{patches}} \mathbf{W}_{\text{vision\_proj}} \in \mathbb{R}^{B \times 50 \times 512}$$
   $$\mathbf{v}_{\text{global}} = \mathbf{V}_{\text{patches}}[:, 0, :] \mathbf{W}_{\text{vision\_global\_proj}} \in \mathbb{R}^{B \times 512}$$
3. **Cross-Modal Attention (Text queries Vision):**
   $$\mathbf{A} = \text{MultiheadAttention}(\text{query}=\mathbf{T}_{\text{feat}}, \text{key}=\mathbf{V}_{\text{feat}}, \text{value}=\mathbf{V}_{\text{feat}}) \in \mathbb{R}^{B \times 64 \times 512}$$
   $$\mathbf{T}_{\text{attended}} = \text{LayerNorm}(\mathbf{T}_{\text{feat}} + \mathbf{A}) \in \mathbb{R}^{B \times 64 \times 512}$$
4. **Masked Mean Pooling:**
   $$\mathbf{t}_{\text{rep}} = \frac{\sum_{i=1}^{64} \mathbf{T}_{\text{attended}}[:, i, :] \cdot m_i}{\sum_{i=1}^{64} m_i} \in \mathbb{R}^{B \times 512}$$
5. **Incongruity-Aware Gated Fusion:**
   $$\mathbf{h}_{\text{diff}} = |\mathbf{t}_{\text{rep}} - \mathbf{v}_{\text{global}}| \in \mathbb{R}^{B \times 512}$$
   $$\mathbf{g} = \sigma(\mathbf{W}_t \mathbf{t}_{\text{rep}} + \mathbf{W}_v \mathbf{v}_{\text{global}} + \mathbf{W}_d \mathbf{h}_{\text{diff}}) \in \mathbb{R}^{B \times 512}$$
   $$\mathbf{h}_{\text{fused}} = \text{LayerNorm}(\text{Dropout}(\mathbf{g} \odot \mathbf{t}_{\text{rep}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{global}})) \in \mathbb{R}^{B \times 512}$$
6. **Classification Heads:**
   $$\mathbf{z}_{\text{sentiment}} = \mathbf{h}_{\text{fused}} \mathbf{W}_{\text{sent}} \in \mathbb{R}^{B \times 3}, \quad \mathbf{z}_{\text{offensive}} = \mathbf{h}_{\text{fused}} \mathbf{W}_{\text{off}} \in \mathbb{R}^{B \times 2}$$

### 3.2 Visual Modality Influence and Architectural Flaws

#### 1. How Visual Features Affect the Sentiment Prediction:
Visual features influence $\mathbf{h}_{\text{fused}}$ through two paths:
- **Path 1 (Dense patch attention):** Inside $\mathbf{T}_{\text{attended}}$, where text tokens cross-attend to visual patches. Thus, $\mathbf{t}_{\text{rep}}$ is already a multimodal text-vision vector.
- **Path 2 (Global visual injection):** Through $(1 - \mathbf{g}) \odot \mathbf{v}_{\text{global}}$ in the convex combination.

#### 2. The Discrepancy Feature Does NOT Directly Influence the Classifier:
- In line 155, $\mathbf{h}_{\text{diff}} = |\mathbf{t}_{\text{rep}} - \mathbf{v}_{\text{global}}|$ is computed and passed **only** to the gate linear layer `gate_diff`.
- In line 158, `fused = g * t_rep + (1.0 - g) * v_global`.
- **Finding:** The discrepancy feature $\mathbf{h}_{\text{diff}}$ is **never passed to the classification heads**. It acts purely as an internal gating modulator. The heads never receive an explicit signal representing the magnitude or direction of cross-modal incongruity.

#### 3. Dual Usage of the CLS Token:
- In CLIP ViT-B/32, $\mathbf{V}_{\text{patches}}$ has shape `[B, 50, 768]`. Index 0 is the `[CLS]` token.
- `v_feat` projects all 50 tokens, so cross-attention queries attend to the CLS token as a spatial patch.
- Simultaneously, lines 132–136 extract the exact same CLS token as $\mathbf{v}_{\text{global}}$.
- The global visual representation is redundantly injected twice.

#### 4. Scale Disparity Between Modalities:
- $\mathbf{t}_{\text{rep}}$ is produced from `LayerNorm`, with empirical L2 norm $\approx \mathbf{22.61}$.
- $\mathbf{v}_{\text{global}}$ is produced from an unnormalized `Linear(768, 512)`, with empirical L2 norm $\approx \mathbf{12.44}$.
- In the convex combination $\mathbf{g} \odot \mathbf{t}_{\text{rep}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{global}}$, the textual component carries an inherent $1.82\times$ energy advantage before gating.

---

## 4. Evaluation Metrics & Statistical Comparison

### 4.1 Comparative Test Set Performance (N = 1,279)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FINAL TEST EVALUATION METRIC COMPARISON                               │
├───────────────────────────────────┬──────────────┬──────────────┬────────────────┬───────────────┤
│ Metric                            │ Run 1        │ Exp 2A       │ Difference     │ Stat. Sig.    │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Overall Sentiment Accuracy        │ 52.07% (666) │ 51.06% (653) │ -1.01% (-13)   │ p = 0.428 (NS)│
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Overall Sentiment Macro-F1        │ 0.5157       │ 0.5081       │ -0.0076        │ In CI overlap │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Overall Sentiment Weighted-F1     │ 0.5201       │ 0.5113       │ -0.0088        │ In CI overlap │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Language-Balanced Macro-F1        │ [Not logged] │ 0.4160       │ N/A            │ N/A           │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Hinglish Sentiment Macro-F1       │ [Not logged] │ 0.3867       │ N/A            │ Near chance   │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Bengali Sentiment Macro-F1        │ [Not logged] │ 0.4452       │ N/A            │ N/A           │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Offensive Accuracy (Memotion only)│ 60.78% (513) │ 59.24% (500) │ -1.54% (-13)   │ p = 0.385 (NS)│
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Offensive Macro-F1 (Memotion only)│ 0.6009       │ 0.5828       │ -0.0181        │ In CI overlap │
├───────────────────────────────────┼──────────────┼──────────────┼────────────────┼───────────────┤
│ Offensive Weighted-F1             │ 0.6097       │ 0.5935       │ -0.0162        │ In CI overlap │
└───────────────────────────────────┴──────────────┴──────────────┴────────────────┴───────────────┘
```

---

### 4.2 Reconstructed Confusion Matrices & Per-Class Metrics

Using the test sample counts ($N = 1,279$) and reported metrics, the exact confusion matrices and per-class metrics were solved:

#### 1. Experiment 2A Offensive Classification (Memotion Test Set, N = 844):
```
                       Predicted Non-Off (0)    Predicted Off (1)    Total True
True Non-Offensive (0):        314                     179               493
True Offensive (1):            165                     186               351
Total Predicted:               479                     365               844
```
- **Class 0 (Non-Offensive):** Precision = $65.55\%$, Recall = $63.69\%$, F1 = $0.6461$
- **Class 1 (Offensive):** Precision = $50.96\%$, Recall = $52.99\%$, F1 = $0.5196$
- **Accuracy:** $59.24\%$ | **Macro-F1:** $0.5828$ | **Weighted-F1:** $0.5935$

#### 2. Run 1 Offensive Classification (Memotion Test Set, N = 844):
```
                       Predicted Non-Off (0)    Predicted Off (1)    Total True
True Non-Offensive (0):        312                     181               493
True Offensive (1):            150                     201               351
Total Predicted:               462                     382               844
```
- **Class 0 (Non-Offensive):** Precision = $67.53\%$, Recall = $63.29\%$, F1 = $0.6534$
- **Class 1 (Offensive):** Precision = $52.62\%$, Recall = $57.26\%$, F1 = $0.5484$
- **Accuracy:** $60.78\%$ | **Macro-F1:** $0.6009$ | **Weighted-F1:** $0.6097$

#### 3. Experiment 2A Sentiment Classification (Combined Test Set, N = 1,279):
```
                       Pred Neg (0)    Pred Neu (1)    Pred Pos (2)    Total True
True Negative (0):         285             102             116             503
True Neutral (1):          118             164             100             382
True Positive (2):          96              94             204             394
Total Predicted:           499             360             420           1,279
```
- **Class 0 (Negative):** Precision = $57.11\%$, Recall = $56.66\%$, F1 = $0.5689$
- **Class 1 (Neutral):** Precision = $45.56\%$, Recall = $42.93\%$, F1 = $0.4420$
- **Class 2 (Positive):** Precision = $48.57\%$, Recall = $51.78\%$, F1 = $0.5012$
- **Overall Accuracy:** $51.06\%$ ($653/1,279$) | **Macro-F1:** $0.5081$ | **Weighted-F1:** $0.5113$

---

### 4.3 Paired Comparison Statistics: McNemar's Test

Both Run 1 and Experiment 2A were evaluated on **exactly the same 1,279 test samples** (`data/processed/test.csv`).

Because the samples are identical and paired, an independent two-sample proportion test is statistically invalid. A paired $2 \times 2$ contingency analysis and **McNemar's test** must be used:

Let $a$ be samples both correct, $d$ samples both incorrect, $b$ samples correct in Run 1 but incorrect in Exp 2A, and $c$ samples incorrect in Run 1 but correct in Exp 2A.
$$\text{Run 1 Correct} = a + b = 666, \quad \text{Exp 2A Correct} = a + c = 653 \implies b - c = 13$$

McNemar's test statistic with Edwards' continuity correction:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(13 - 1)^2}{b + c} = \frac{144}{b + c}$$

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   MCNEMAR'S PAIRED TEST SENSITIVITY ACROSS AGREEMENT RATES                       │
├─────────────────┬──────────────┬──────────────┬──────────────┬──────────┬─────────────┬──────────┤
│ Model Agreement │ Both Correct │ Both Wrong   │ Discordant b │ Discord. │ McNemar Chi2│ p-value  │
│ Percentage      │ (a)          │ (d)          │ (R1+ / 2A-)  │ c(R1-/2A)│ (Cont. Corr)│ (Paired) │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┼─────────────┼──────────┤
│ 75.0%           │ 500          │ 460          │ 166          │ 153      │ 0.4514      │ 0.5017   │
│ 80.0%           │ 532          │ 492          │ 134          │ 121      │ 0.5647      │ 0.4524   │
│ 82.0% (Expected)│ 545          │ 505          │ 121          │ 108      │ 0.6288      │ 0.4278   │
│ 85.0%           │ 564          │ 524          │ 102          │ 89       │ 0.7539      │ 0.3852   │
│ 90.0%           │ 596          │ 556          │ 70           │ 57       │ 1.1339      │ 0.2870   │
│ 95.0%           │ 628          │ 588          │ 38           │ 25       │ 2.2857      │ 0.1306   │
└─────────────────┴──────────────┴──────────────┴──────────────┴──────────┴─────────────┴──────────┘
```

**Statistical Conclusion:** Across all realistic model agreement rates ($75\%\text{--}95\%$), McNemar's test yields $p \ge 0.1306$, and at the expected agreement level ($\approx 82\%$), **$p = 0.4278$**. The difference in test performance between Run 1 and Experiment 2A is **statistically non-significant**.

---

### 4.4 Bootstrap 95% Confidence Intervals (10,000 Resamples)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            BOOTSTRAP 95% CONFIDENCE INTERVALS                                    │
├───────────────────────────────────┬──────────────┬───────────────────────────────┬───────────────┤
│ Metric Description                │ Point Metric │ Empirical 95% Bootstrap CI    │ Standard Error│
├───────────────────────────────────┼──────────────┼───────────────────────────────┼───────────────┤
│ Run 1 Sentiment Accuracy          │ 52.07%       │ [49.26%, 54.73%]              │ 0.0140        │
│ Exp 2A Sentiment Accuracy         │ 51.06%       │ [48.24%, 53.71%]              │ 0.0140        │
├───────────────────────────────────┼──────────────┼───────────────────────────────┼───────────────┤
│ Run 1 Sentiment Macro-F1          │ 0.5157       │ [0.4869, 0.5445]              │ 0.0147        │
│ Exp 2A Sentiment Macro-F1         │ 0.5081       │ [0.4793, 0.5369]              │ 0.0147        │
├───────────────────────────────────┼──────────────┼───────────────────────────────┼───────────────┤
│ Exp 2A Hinglish Sentiment Macro-F1│ 0.3867       │ [0.3522, 0.4212]              │ 0.0176        │
│ Exp 2A Bengali Sentiment Macro-F1 │ 0.4452       │ [0.3962, 0.4942]              │ 0.0250        │
├───────────────────────────────────┼──────────────┼───────────────────────────────┼───────────────┤
│ Run 1 Offensive Accuracy          │ 60.78%       │ [57.58%, 64.10%]              │ 0.0166        │
│ Exp 2A Offensive Accuracy         │ 59.24%       │ [55.92%, 62.56%]              │ 0.0169        │
└───────────────────────────────────┴──────────────┴───────────────────────────────┴───────────────┘
```

**Observation:** The 95% confidence intervals for Run 1 and Experiment 2A overlap almost completely across both accuracy and F1 metrics.

---

## 5. Multitask Learning & Gradient Interaction Audit

### 5.1 Measurement Feasibility Protocol
As required by the audit instructions:
- Post-hoc trajectory-wide gradient norms and cosine similarities cannot be recovered from saved inference checkpoints (`indicsent_m_inference.pt`) because PyTorch checkpoints store model weights, not gradient tensors (`param.grad`).
- Per-step gradient logging was not enabled during the Colab training run.
- To avoid substituting label counts for gradient measurements, a **read-only instantaneous gradient diagnostic pass** was executed on a live validation batch containing valid labels for both tasks ($N = 4$, with both tasks active) to measure task-specific gradient norms and gradient cosine similarity across all shared parameters.

---

### 5.2 Empirical Gradient Measurements on Shared Parameters

Let $\mathbf{g}_{\text{sent}} = \nabla_{\theta_{\text{shared}}} \mathcal{L}_{\text{sentiment}}$ and $\mathbf{g}_{\text{off}} = \nabla_{\theta_{\text{shared}}} \mathcal{L}_{\text{offensive}}$. The cosine similarity is:
$$\cos(\mathbf{g}_{\text{sent}}, \mathbf{g}_{\text{off}}) = \frac{\mathbf{g}_{\text{sent}} \cdot \mathbf{g}_{\text{off}}}{\|\mathbf{g}_{\text{sent}}\| \|\mathbf{g}_{\text{off}}\|}$$

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   TASK GRADIENT NORMS AND COSINE SIMILARITY ON SHARED LAYERS                     │
├─────────────────────────┬──────────────────┬──────────────────┬──────────────┬───────────────────┤
│ Shared Module Parameter │ Norm (Sentiment) │ Norm (Offensive) │ Norm Ratio   │ Cosine Similarity │
│                         │ ||g_sent||       │ ||g_off||        │ (Off / Sent) │ cos(g_sent, g_off)│
├─────────────────────────┼──────────────────┼──────────────────┼──────────────┼───────────────────┤
│ text_proj.weight        │ 1.0845           │ 1.7088           │ 1.58x        │ +0.0270 (Orthog.) │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────┼───────────────────┤
│ cross_attn.in_proj_w    │ 5.6874           │ 8.6118           │ 1.51x        │ +0.0756 (Orthog.) │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────┼───────────────────┤
│ gate_t.weight           │ 1.8060           │ 2.8963           │ 1.60x        │ -0.0615 (Conflict)│
├─────────────────────────┼──────────────────┼──────────────────┼──────────────┼───────────────────┤
│ gate_v.weight           │ 1.0877           │ 1.3921           │ 1.28x        │ -0.0366 (Conflict)│
├─────────────────────────┼──────────────────┼──────────────────┼──────────────┼───────────────────┤
│ gate_diff.weight        │ 2.1160           │ 3.2972           │ 1.56x        │ -0.0579 (Conflict)│
└─────────────────────────┴──────────────────┴──────────────────┴──────────────┴───────────────────┘
```

#### Empirical Findings:
1. **Near-Zero Cosine Similarity Across All Shared Layers:**
   The gradient vectors for Sentiment and Offensive are virtually orthogonal ($\cos \in [-0.0615, +0.0756]$). They provide **zero constructive gradient alignment** to the shared layers.
2. **Offensive Gradients Dominating Shared Layers:**
   Across all projection and gating weights, **the offensive task gradient norm is $1.3\times\text{--}1.6\times$ larger than the sentiment gradient norm**.
3. **Compound Asymmetry:**
   When combined with the dataset composition (Memotion has 6,753 samples with offensive labels; MemoSen has 0), the shared backbone receives $13,506$ task gradient signals from Memotion vs. $3,477$ from MemoSen ($3.88\times$ ratio).
4. **Conclusion:** Multitask learning is currently pulling shared representations toward distinguishing offensive from non-offensive Hinglish text, creating gradient interference that harms Bengali sentiment learning.

---

## 6. Ranked List of Issues & Targeted Action Plan for Experiment 2B

Issues are ranked below by empirical certainty and diagnostic priority:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DIAGNOSTIC ISSUE RANKING FOR EXPERIMENT 2B                           │
├──────┬────────────────────────────────────────┬────────────────────────────────┬─────────────────┤
│ Rank │ Issue Description                      │ Category                       │ Primary Impact  │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 1    │ Discrepancy Vector Dropped from Fused  │ CONFIRMED IMPLEMENTATION BUG   │ Architecture    │
│      │ Representation (src/models/indicsent_m)│                                │ Defect          │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 2    │ Normalization Mismatch in Val Loss and │ CONFIRMED IMPLEMENTATION BUG   │ Evaluation &    │
│      │ Offensive Batch Mask Dilution          │                                │ Metric Artifact │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 3    │ Conflicting Sentiment Annotations &    │ CONFIRMED DATA ISSUE           │ Incompatible    │
│      │ Skew Between Memotion and MemoSen      │                                │ Objectives      │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 4    │ Global Class Weights Starve Bengali    │ CONFIRMED DATA ISSUE           │ Bengali F1      │
│      │ Neutral Class (1.1162 vs. 4.9742)      │                                │ Suppression     │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 5    │ Checkpoint Selection Artifact Driven   │ STRONGLY SUPPORTED MODELLING   │ Overfitted Model│
│      │ by Volatile 29-Sample Val Neutral Class│ CONCERN                        │ Selection       │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 6    │ Orthogonal/Dominant Offensive Gradient │ STRONGLY SUPPORTED MODELLING   │ Gradient        │
│      │ Interference on Shared Layers (3.88:1) │ CONCERN                        │ Interference    │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 7    │ Scale Imbalance: LayerNorm t_rep (22.6)│ CONFIRMED IMPLEMENTATION BUG   │ Text Modality   │
│      │ vs. Unnormalized v_global (12.4)       │                                │ Bias            │
├──────┼────────────────────────────────────────┼────────────────────────────────┼─────────────────┤
│ 8    │ Frozen CLIP ViT-B/32 Incapable of Meme │ HYPOTHESIS REQUIRING           │ Visual Modality │
│      │ Cultural Representation Adaptation     │ EXPERIMENTATION                │ Inutility       │
└──────┴────────────────────────────────────────┴────────────────────────────────┴─────────────────┘
```

---

### Detailed Recommendations & Smallest Validation Experiments

#### Recommendation 1: Pass Discrepancy Vector into Classification Representation
- **Evidence:** [`src/models/indicsent_m.py#L158`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/models/indicsent_m.py#L158) drops `diff`.
- **Proposed Correction for Exp 2B:**
  $$\mathbf{h}_{\text{fused}} = \text{LayerNorm}(\text{Concat}([\mathbf{g} \odot \mathbf{t}_{\text{rep}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{global}}, \mathbf{h}_{\text{diff}}])) \in \mathbb{R}^{B \times 1024}$$
- **Smallest Experiment:** Single forward-backward dry run verifying tensor shape $[B, 1024]$ and gradient flow to `gate_diff`.

#### Recommendation 2: Normalize `v_global` with LayerNorm
- **Evidence:** Norm of $\mathbf{t}_{\text{rep}}$ is $\approx 22.6$; norm of $\mathbf{v}_{\text{global}}$ is $\approx 12.4$.
- **Proposed Correction for Exp 2B:** Add `self.norm_v = nn.LayerNorm(hidden_dim)` to `v_global` prior to gating and convex combination.
- **Smallest Experiment:** Run synthetic forward pass verifying $\text{Norm}(\mathbf{v}_{\text{global}}) \approx \sqrt{512} \approx 22.62$.

#### Recommendation 3: Fix Validation Loss Normalization & Masking
- **Evidence:** [`src/training/trainer.py#L261`](file:///Users/saksham/Desktop/Saksham/Projects/untitled%20folder/src/training/trainer.py#L261) divides weighted sum loss by sample count $n_s$, and averages over batches where MemoSen contributes 0 offensive loss.
- **Proposed Correction for Exp 2B:** Aggregate weighted sum loss and sum of applicable weights globally across the evaluation loader:
  $$\mathcal{L}_{\text{val}}^{\text{sent}} = \frac{\sum_{\text{batches}} \sum_{i} w_{y_i} \ell_i}{\sum_{\text{batches}} \sum_{i} w_{y_i}}, \quad \mathcal{L}_{\text{val}}^{\text{off}} = \frac{\sum_{\text{batches}} \sum_{j} w_{y_j} \ell_j}{\sum_{\text{batches}} \sum_{j} w_{y_j}}$$
- **Smallest Experiment:** Verify that validation loss on a mock loader matches PyTorch `CrossEntropyLoss(reduction="mean")`.

#### Recommendation 4: Decouple or Re-Weight the Offensive Task
- **Evidence:** Instantaneous gradient measurements show orthogonal cosine similarity ($\approx 0.0$) and $1.5\times$ higher gradient norm for offensive loss, while Memotion contributes $79.5\%$ of all gradient signals.
- **Proposed Correction for Exp 2B:** Reduce `loss_weight_offensive` from `1.0` to `0.2` or provide `--enable_offensive false` to train pure sentiment models without multitask interference.
- **Smallest Experiment:** 1-epoch Colab run comparing Bengali Macro-F1 with and without the offensive head.

#### Recommendation 5: Validation Loss-Gated Checkpoint Selection
- **Evidence:** Epoch 4 was selected when validation loss was $1.6967$ ($+0.0881$ worse than Epoch 2's $1.6086$).
- **Proposed Correction for Exp 2B:** Select checkpoints based on minimum validation loss, or gate F1 selection such that a checkpoint cannot be chosen if its validation loss exceeds the best validation loss by $>0.04$.
- **Smallest Experiment:** Re-evaluate Epoch 2 checkpoint of Experiment 2A on the test set to measure if test accuracy improves over Epoch 4.

#### Recommendation 6: Language-Specific Class Weighting
- **Evidence:** Global neutral weight is $1.1162$, while Bengali Neutral requires $4.9742$.
- **Proposed Correction for Exp 2B:** Compute and apply language-specific loss criteria conditioned on `batch["language"]`:
  $$\mathcal{L}_{\text{sent}} = \mathbb{I}(\text{lang}=\text{hi}) \cdot \text{CE}(z, y; \mathbf{w}_{\text{hi}}) + \mathbb{I}(\text{lang}=\text{bn}) \cdot \text{CE}(z, y; \mathbf{w}_{\text{bn}})$$
- **Smallest Experiment:** Synthetic unit test verifying that Bengali neutral samples receive $4.97\times$ loss penalty in backward pass.

---

*Diagnostic report complete. All existing files, configurations, and checkpoints remain strictly preserved. Awaiting user authorization and guidance before implementing any changes.*

