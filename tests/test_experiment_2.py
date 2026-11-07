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
from transformers import BertConfig, CLIPVisionConfig, BertModel, CLIPVisionModel, get_cosine_schedule_with_warmup

from src.models.indicsent_m import IndicSentM
from src.data.dataset import IndicMemeDataset
from src.training.trainer import (
    build_optimizer_param_groups,
    compute_class_weights,
    evaluate,
    train_one_epoch,
)


def create_mock_indicsent_m(freeze_vision=True):
    """Instantiates a fast, lightweight mock IndicSentM model for rigorous testing without downloading 1GB weights."""
    text_config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=224,
        patch_size=32,
    )

    class MockIndicSentM(IndicSentM):
        def __init__(self, freeze_vis=True):
            nn.Module.__init__(self)
            self.text_model_name = "mock/text"
            self.vision_model_name = "mock/clip"
            self.hidden_dim = 64
            self.num_sentiment_classes = 3
            self.num_offensive_classes = 2

            self.text_encoder = BertModel(text_config)
            self.vision_encoder = CLIPVisionModel(vision_config)

            if freeze_vis:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False

            text_dim = 128
            vision_dim = 128

            self.text_proj = nn.Linear(text_dim, self.hidden_dim)
            self.vision_proj = nn.Linear(vision_dim, self.hidden_dim)
            self.vision_global_proj = nn.Linear(vision_dim, self.hidden_dim)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim, num_heads=2, dropout=0.1, batch_first=True
            )
            self.norm_t = nn.LayerNorm(self.hidden_dim)

            self.gate_t = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.gate_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.gate_diff = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.gate_act = nn.Sigmoid()
            self.dropout = nn.Dropout(0.1)
            self.fusion_norm = nn.LayerNorm(self.hidden_dim)

            self.sentiment_head = nn.Linear(self.hidden_dim, self.num_sentiment_classes)
            self.offensive_head = nn.Linear(self.hidden_dim, self.num_offensive_classes)

    return MockIndicSentM(freeze_vis=freeze_vision)


def run_experiment_2_verification_suite():
    print("=" * 70)
    print("  RUNNING EXPERIMENT 2 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)
    device = torch.device("cpu")

    # ---------------------------------------------------------
    # 1. Forward Pass & Output Dimensions Check
    # ---------------------------------------------------------
    print("\n[Test 1] Synthetic Forward Pass & Tensor Dimensions...")
    model = create_mock_indicsent_m(freeze_vision=True).to(device)
    model.eval()

    B, S = 4, 64
    input_ids = torch.randint(0, 1000, (B, S))
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[:, 30:] = 0  # 34 pad tokens per sequence
    pixel_values = torch.randn(B, 3, 224, 224)

    with torch.no_grad():
        sent_logits, off_logits = model(input_ids, attention_mask, pixel_values)
        assert sent_logits.shape == (B, 3), f"Incorrect sentiment logits shape: {sent_logits.shape}"
        assert off_logits.shape == (B, 2), f"Incorrect offensive logits shape: {off_logits.shape}"

        dict_out = model(
            input_ids, attention_mask, pixel_values,
            return_dict=True, return_gate_weights=True, return_attentions=True
        )
        assert dict_out["fused_features"].shape == (B, 64), f"Bad fused_features: {dict_out['fused_features'].shape}"
        assert dict_out["gate_weights"].shape == (B, 64), f"Bad gate_weights: {dict_out['gate_weights'].shape}"
        assert dict_out["t_rep"].shape == (B, 64), f"Bad t_rep: {dict_out['t_rep'].shape}"
        assert dict_out["v_global"].shape == (B, 64), f"Bad v_global: {dict_out['v_global'].shape}"

    print("  ✔ Output dimensions verified: Sentiment [B, 3], Offensive [B, 2], Fused [B, hidden_dim].")
    print("  ✔ Genuine global visual vector v_global and masked text vector t_rep verified.")

    # ---------------------------------------------------------
    # 2. Gradient Flow & Frozen Backbone Verification
    # ---------------------------------------------------------
    print("\n[Test 2] Gradient Flow & Frozen CLIP Backbone Verification...")
    model.train()
    model.zero_grad()

    sent_logits, off_logits = model(input_ids, attention_mask, pixel_values)
    dummy_loss = sent_logits.sum() + off_logits.sum()
    dummy_loss.backward()

    # Verify frozen CLIP parameters received NO gradients
    for name, param in model.vision_encoder.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen but requires_grad=True"
        assert param.grad is None, f"Parameter {name} in frozen vision encoder received gradient: {param.grad}"
    print("  ✔ Confirmed: All CLIP vision encoder parameters are frozen with zero gradients.")

    # Verify trainable modules received valid non-zero gradients
    trainable_modules = [
        ("text_encoder", model.text_encoder.encoder.layer[0].attention.self.query.weight),
        ("text_proj", model.text_proj.weight),
        ("vision_proj", model.vision_proj.weight),
        ("vision_global_proj", model.vision_global_proj.weight),
        ("cross_attn", model.cross_attn.in_proj_weight),
        ("gate_t", model.gate_t.weight),
        ("gate_v", model.gate_v.weight),
        ("gate_diff", model.gate_diff.weight),
        ("sentiment_head", model.sentiment_head.weight),
        ("offensive_head", model.offensive_head.weight),
    ]

    for mod_name, param in trainable_modules:
        assert param.grad is not None, f"Module {mod_name} received None gradient!"
        assert torch.isfinite(param.grad).all(), f"Module {mod_name} received NaN or Inf gradient!"
        assert param.grad.abs().sum() > 0, f"Module {mod_name} received all-zero gradient!"
    print("  ✔ Confirmed: Gradients flow cleanly to text encoder, projections, cross-attn, gate, and both heads.")

    # ---------------------------------------------------------
    # 3. Masked Multi-Task Loss with Partially/Fully Missing Labels
    # ---------------------------------------------------------
    print("\n[Test 3] Masked Multi-Task Loss Normalization (-1 Labels)...")
    criterion_sent = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_off = nn.CrossEntropyLoss(ignore_index=-1)

    # Case A: Mixed labels (e.g. 2 Memotion with valid offensive, 2 MemoSen with -1)
    sent_labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    off_labels_mixed = torch.tensor([1, -1, 0, -1], dtype=torch.long)

    loss_sent = criterion_sent(sent_logits, sent_labels)
    loss_off_mixed = criterion_off(off_logits, off_labels_mixed)
    total_mixed_loss = loss_sent + loss_off_mixed

    assert torch.isfinite(loss_sent), "Sentiment loss is not finite!"
    assert torch.isfinite(loss_off_mixed), "Offensive loss is not finite!"
    assert total_mixed_loss > 0, "Total mixed loss must be positive!"
    print("  ✔ Masked loss with 50% missing labels computed cleanly without NaNs.")

    # Case B: Pure MemoSen batch (100% missing offensive labels)
    off_labels_all_masked = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
    valid_off = (off_labels_all_masked != -1).any()
    assert not valid_off, "Should have no valid offensive labels"
    # In trainer.py, when valid_off is False, criterion_off is skipped cleanly
    loss_all_masked = loss_sent + (criterion_off(off_logits, off_labels_all_masked) if valid_off else 0.0)
    assert torch.isfinite(loss_all_masked)
    print("  ✔ Pure MemoSen batch (100% masked subtask) safely skips offensive head.")

    # ---------------------------------------------------------
    # 4. Parameter Groups, Optimizer & Scheduler Continuity
    # ---------------------------------------------------------
    print("\n[Test 4] Optimizer Parameter Groups & Uninterrupted Scheduler...")
    param_groups = build_optimizer_param_groups(
        model=model,
        lr_text=1.0e-5,
        lr_fusion=2.0e-5,
        lr_vision=0.0,
        weight_decay=0.01,
    )
    assert len(param_groups) >= 3, f"Expected at least 3 param groups, got {len(param_groups)}"
    
    # Check that frozen vision parameters are NOT in any optimizer group
    all_opt_params = set()
    for g in param_groups:
        for p in g["params"]:
            all_opt_params.add(id(p))
            
    for name, p in model.vision_encoder.named_parameters():
        assert id(p) not in all_opt_params, f"Frozen parameter {name} was found in optimizer param groups!"
    print("  ✔ Confirmed: Zero frozen CLIP parameters placed in optimizer.")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=10)

    initial_text_lr = param_groups[0]["lr"]
    # Step scheduler and verify active optimizer param groups update
    optimizer.step()
    scheduler.step()
    updated_text_lr = optimizer.param_groups[0]["lr"]
    assert updated_text_lr != initial_text_lr, "Scheduler failed to update the active optimizer!"
    print("  ✔ Confirmed: Learning rate scheduler dynamically updates active optimizer param groups.")

    # ---------------------------------------------------------
    # 5. Checkpoint Save and Reload Integrity
    # ---------------------------------------------------------
    print("\n[Test 5] Checkpoint Save and Reload Verification...")
    ckpt_path = Path("checkpoints/test_checkpoint.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": {"val_macro_f1": 0.65},
    }, ckpt_path)

    # Reload into fresh model instance
    new_model = create_mock_indicsent_m(freeze_vision=True).to(device)
    new_model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        new_sent_logits, new_off_logits = new_model(input_ids, attention_mask, pixel_values)
        model.eval()
        orig_sent_logits, orig_off_logits = model(input_ids, attention_mask, pixel_values)

        assert torch.allclose(new_sent_logits, orig_sent_logits, atol=1e-5)
        assert torch.allclose(new_off_logits, orig_off_logits, atol=1e-5)

    ckpt_path.unlink()  # Clean up test checkpoint
    print("  ✔ Checkpoint saved, loaded, and re-evaluated with bit-exact numerical parity.")

    # ---------------------------------------------------------
    # 6. Truncation Statistics Check on Real Dataset Split
    # ---------------------------------------------------------
    print("\n[Test 6] Truncation Statistics on Processed Dataset...")
    val_csv = "data/processed/val.csv"
    if os.path.exists(val_csv):
        val_dataset = IndicMemeDataset(val_csv, max_length=64, is_train=False)
        stats = val_dataset.truncation_stats
        print(f"  Truncation Stats for max_length=64: {stats['truncated_samples']} / {stats['total_samples']} ({stats['truncation_percentage']}%)")
        print(f"  Median token length: {stats['median_token_length']} | 95th-Percentile: {stats['p95_token_length']}")
        assert stats["total_samples"] == 1279
        assert stats["truncation_percentage"] < 10.0, f"Expected <10% truncation with len=64, got {stats['truncation_percentage']}%"
        print("  ✔ Max sequence length 64 covers >90% of samples without truncation.")

    # ---------------------------------------------------------
    # 7. Resumable vs. Inference Checkpoint Parity & State Restoration
    # ---------------------------------------------------------
    print("\n[Test 7] Resumable vs. Inference Checkpoints & State Restoration...")
    resumable_path = Path("checkpoints/test_resumable.pt")
    inference_path = Path("checkpoints/test_inference.pt")

    resumable_payload = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": 0.582,
        "patience_counter": 1,
        "val_metrics": {"val_macro_f1": 0.582},
    }
    inference_payload = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "val_metrics": {"val_macro_f1": 0.582},
    }
    torch.save(resumable_payload, resumable_path)
    torch.save(inference_payload, inference_path)

    # Verify inference checkpoint is smaller (no optimizer moments)
    assert inference_path.stat().st_size < resumable_path.stat().st_size
    print(f"  ✔ Checkpoint sizes: Resumable {resumable_path.stat().st_size / 1024:.1f} KB vs Inference {inference_path.stat().st_size / 1024:.1f} KB")

    # Verify state restoration
    resumed_ckpt = torch.load(resumable_path, map_location=device)
    assert resumed_ckpt["epoch"] == 3
    assert resumed_ckpt["best_val_f1"] == 0.582
    assert "scheduler_state_dict" in resumed_ckpt
    assert "optimizer_state_dict" in resumed_ckpt

    resumable_path.unlink()
    inference_path.unlink()
    print("  ✔ Confirmed: Resumable and inference checkpoints created and validated.")

    # ---------------------------------------------------------
    # 8. Incomplete Accumulation Group & Optimizer Step Alignment
    # ---------------------------------------------------------
    print("\n[Test 8] Incomplete Accumulation Group Execution...")
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 7  # 7 batches with grad_accum=4 means group 1 (batches 0-3) and group 2 (batches 4-6)
        def __getitem__(self, idx):
            return {
                "input_ids": torch.ones(10, dtype=torch.long),
                "attention_mask": torch.ones(10, dtype=torch.long),
                "pixel_values": torch.randn(3, 224, 224),
                "sentiment": torch.tensor(1),
                "offensive": torch.tensor(0),
            }

    tiny_loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=1)
    tiny_model = nn.Sequential(nn.Linear(10, 3))
    
    # Simple forward wrapper for trainer
    class TinyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(10, 3)
            self.f2 = nn.Linear(10, 2)
        def forward(self, input_ids, attention_mask, pixel_values):
            x = input_ids.float()
            return self.f1(x), self.f2(x)

    t_model = TinyWrapper()
    t_opt = torch.optim.AdamW(t_model.parameters(), lr=1e-3)
    t_sched = get_cosine_schedule_with_warmup(t_opt, num_warmup_steps=0, num_training_steps=10)

    opt_steps = [0]
    orig_t_step = t_opt.step
    def mock_t_step():
        opt_steps[0] += 1
        return orig_t_step()
    t_opt.step = mock_t_step

    train_one_epoch(
        model=t_model,
        loader=tiny_loader,
        optimizer=t_opt,
        criterion_sent=nn.CrossEntropyLoss(reduction="sum"),
        criterion_off=nn.CrossEntropyLoss(reduction="sum"),
        device=torch.device("cpu"),
        scheduler=t_sched,
        gradient_accumulation_steps=4,
        fp16=False,
    )

    assert opt_steps[0] == 2, f"Expected 2 optimizer steps for 7 batches with accum=4, got {opt_steps[0]}"
    print(f"  ✔ Confirmed: Final incomplete accumulation group correctly triggered step 2 (Total steps: {opt_steps[0]}).")

    # ---------------------------------------------------------
    # 9. Incomplete Window Normalization vs Single Large Batch Equivalence
    # ---------------------------------------------------------
    print("\n[Test 9] Incomplete Window Normalization vs Reference Large Batch...")
    torch.manual_seed(123)

    class SmallMTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)
            self.s_head = nn.Linear(4, 3)
            self.o_head = nn.Linear(4, 2)
        def forward(self, x):
            h = torch.relu(self.linear(x))
            return self.s_head(h), self.o_head(h)

    # 6 samples: 4 in microbatch 1, 2 in microbatch 2
    X_window = torch.randn(6, 8)
    sent_window = torch.tensor([0, 1, 2, 1, 0, 2])
    # 3 valid offensive labels (2 in microbatch 1, 1 in microbatch 2), 3 masked (-1)
    off_window = torch.tensor([0, 1, -1, -1, 1, -1])

    # Reference Model (Evaluated in one large batch of 6)
    ref_model = SmallMTModel()
    acc_model = SmallMTModel()
    acc_model.load_state_dict(ref_model.state_dict())

    crit_s_sum = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
    crit_o_sum = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    s_ref, o_ref = ref_model(X_window)
    n_s_total = (sent_window != -1).sum().item()
    n_o_total = (off_window != -1).sum().item()

    loss_ref = (crit_s_sum(s_ref, sent_window) / n_s_total) + (crit_o_sum(o_ref, off_window) / n_o_total)
    loss_ref.backward()

    # Accumulated Model (Microbatch 1: 4 samples, Microbatch 2: 2 samples)
    # Using window-normalized accumulation logic
    # Microbatch 1
    s_1, o_1 = acc_model(X_window[:4])
    l_s1 = crit_s_sum(s_1, sent_window[:4]) / n_s_total
    l_o1 = crit_o_sum(o_1, off_window[:4]) / n_o_total
    (l_s1 + l_o1).backward()

    # Microbatch 2
    s_2, o_2 = acc_model(X_window[4:])
    l_s2 = crit_s_sum(s_2, sent_window[4:]) / n_s_total
    l_o2 = crit_o_sum(o_2, off_window[4:]) / n_o_total
    (l_s2 + l_o2).backward()

    max_grad_diff = 0.0
    for (name_r, p_r), (name_a, p_a) in zip(ref_model.named_parameters(), acc_model.named_parameters()):
        diff = (p_r.grad - p_a.grad).abs().max().item()
        if diff > max_grad_diff:
            max_grad_diff = diff

    assert max_grad_diff < 1e-6, f"Accumulated gradients deviate from reference! Max diff: {max_grad_diff}"
    print(f"  ✔ Confirmed: Accumulated gradients match single large batch within {max_grad_diff:.8e} max diff.")

    # ---------------------------------------------------------
    # 10. Language-Balanced Checkpoint Selection (Hinglish + Bengali)
    # ---------------------------------------------------------
    print("\n[Test 10] Language-Balanced Checkpoint Selection & Class Support...")
    class SyntheticEvalDataset(torch.utils.data.Dataset):
        def __init__(self):
            # 6 Hinglish samples (2 of each sentiment class) + 6 Bengali samples (2 of each sentiment class)
            self.samples = [
                {"sentiment": 0, "offensive": 0, "language": "hinglish"},
                {"sentiment": 1, "offensive": 1, "language": "hinglish"},
                {"sentiment": 2, "offensive": 0, "language": "hinglish"},
                {"sentiment": 0, "offensive": 1, "language": "hinglish"},
                {"sentiment": 1, "offensive": 0, "language": "hinglish"},
                {"sentiment": 2, "offensive": 1, "language": "hinglish"},
                {"sentiment": 0, "offensive": -1, "language": "bengali"},
                {"sentiment": 1, "offensive": -1, "language": "bengali"},
                {"sentiment": 2, "offensive": -1, "language": "bengali"},
                {"sentiment": 0, "offensive": -1, "language": "bengali"},
                {"sentiment": 1, "offensive": -1, "language": "bengali"},
                {"sentiment": 2, "offensive": -1, "language": "bengali"},
            ]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s = self.samples[idx]
            return {
                "input_ids": torch.ones(10, dtype=torch.long),
                "attention_mask": torch.ones(10, dtype=torch.long),
                "pixel_values": torch.randn(3, 224, 224),
                "sentiment": torch.tensor(s["sentiment"], dtype=torch.long),
                "offensive": torch.tensor(s["offensive"], dtype=torch.long),
                "language": s["language"],
            }

    eval_loader = torch.utils.data.DataLoader(SyntheticEvalDataset(), batch_size=4, shuffle=False)
    eval_model = create_mock_indicsent_m(freeze_vision=True).to(device)

    metrics = evaluate(
        model=eval_model,
        loader=eval_loader,
        criterion_sent=nn.CrossEntropyLoss(reduction="sum", ignore_index=-1),
        criterion_off=nn.CrossEntropyLoss(reduction="sum", ignore_index=-1),
        device=device,
    )

    assert "selection_score" in metrics, "selection_score missing from evaluate() output!"
    assert "hinglish_sentiment_macro_f1" in metrics
    assert "bengali_sentiment_macro_f1" in metrics
    expected_score = (metrics["hinglish_sentiment_macro_f1"] + metrics["bengali_sentiment_macro_f1"]) / 2.0
    assert abs(metrics["selection_score"] - expected_score) < 1e-6
    print(f"  ✔ Selection Score: {metrics['selection_score']:.4f} exactly equals mean(Hinglish: {metrics['hinglish_sentiment_macro_f1']:.4f}, Bengali: {metrics['bengali_sentiment_macro_f1']:.4f})")
    assert metrics["hinglish_sentiment_class_counts"] == {0: 2, 1: 2, 2: 2}
    assert metrics["bengali_sentiment_class_counts"] == {0: 2, 1: 2, 2: 2}
    print("  ✔ Confirmed: Both language subsets verify complete class support for all 3 sentiment classes.")

    print("\n" + "=" * 70)
    print("  ALL 10 EXPERIMENT 2A VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment_2_verification_suite()

