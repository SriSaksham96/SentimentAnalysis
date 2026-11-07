import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import BertConfig, CLIPVisionConfig, BertModel, CLIPVisionModel
from src.models.indicsent_m import IndicSentM


def test_indicsent_m_shapes():
    """Validates the input-output tensor contracts of IndicSent-M."""
    print("Initializing mock encoders for fast unit testing...")
    text_config = BertConfig(
        vocab_size=1000,
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=768,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=224,
        patch_size=32,
    )

    class FastMockIndicSentM(IndicSentM):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.text_model_name = "mock/muril"
            self.vision_model_name = "mock/clip"
            self.hidden_dim = 512
            self.num_sentiment_classes = 3
            self.num_offensive_classes = 2

            self.text_encoder = BertModel(text_config)
            self.vision_encoder = CLIPVisionModel(vision_config)
            self.text_proj = torch.nn.Linear(768, 512)
            self.vision_proj = torch.nn.Linear(768, 512)
            self.cross_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self.norm = torch.nn.LayerNorm(512)
            self.gate = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.Sigmoid())
            self.dropout = torch.nn.Dropout(0.3)
            self.sentiment_head = torch.nn.Linear(512, 3)
            self.offensive_head = torch.nn.Linear(512, 2)

    model = FastMockIndicSentM()
    model.eval()

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224)

    print(f"Testing forward pass with batch_size={batch_size}, seq_len={seq_len}...")
    with torch.no_grad():
        # Test standard tuple output
        sentiment_logits, offensive_logits = model(input_ids, attention_mask, pixel_values)
        assert sentiment_logits.shape == (batch_size, 3), f"Bad sentiment shape: {sentiment_logits.shape}"
        assert offensive_logits.shape == (batch_size, 2), f"Bad offensive shape: {offensive_logits.shape}"

        # Test dictionary output with gate weights and cross-attention maps
        out = model(
            input_ids,
            attention_mask,
            pixel_values,
            return_dict=True,
            return_gate_weights=True,
            return_attentions=True,
        )
        assert out["fused_features"].shape == (batch_size, 512)
        assert out["gate_weights"].shape == (batch_size, 512)
        # 32 text query tokens, 50 visual key patches (49 grid + 1 class token)
        assert out["cross_attention_weights"].shape == (batch_size, seq_len, 50)

    print("✔ Standard forward pass returned correct (sentiment_logits, offensive_logits).")
    print("✔ Gated fusion correctly returned gate weights [B, 512].")
    print("✔ Cross-modal multi-head attention weights verified: [B, seq_len, num_patches].")
    print("\nAll unit tests passed successfully!")


if __name__ == "__main__":
    test_indicsent_m_shapes()
