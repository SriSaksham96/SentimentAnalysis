import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Handle macOS OpenMP runtime duplication
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel


class IndicSentM(nn.Module):
    """IndicSent-M Architecture: Multilingual Multimodal Sentiment & Sarcasm Network.
    
    Combines multilingual contextual text encoders (MuRIL / XLM-RoBERTa) with
    CLIP vision transformer encoders via Cross-Modal Multi-Head Attention and
    Incongruity-Aware Gated Multimodal Fusion.
    
    Fixes in Experiment 2:
    1. Masked mean pooling for text using attention_mask (eliminates padding token dilution).
    2. Genuine global visual representation extraction via CLIP visual pooler/CLS token.
    3. Direct dual-modality contribution to classification heads via gated fusion.
    4. Sarcasm & incongruity awareness via explicit cross-modal difference modeling.
    """

    def __init__(
        self,
        text_model_name: str = "google/muril-base-cased",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_sentiment_classes: int = 3,
        num_offensive_classes: int = 2,
        dropout: float = 0.3,
        freeze_text_encoder: bool = False,
        freeze_vision_encoder: bool = True,
    ):
        super().__init__()
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.hidden_dim = hidden_dim
        self.num_sentiment_classes = num_sentiment_classes
        self.num_offensive_classes = num_offensive_classes

        # 1. Encoders
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size      # 768 for base models
        vision_dim = self.vision_encoder.config.hidden_size  # 768 for ViT-B/32

        # 2. Linear Projections into a shared d-dimensional space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)            # Projects visual patches
        self.vision_global_proj = nn.Linear(vision_dim, hidden_dim)     # Projects global visual token/pooler

        # 3. Cross-Modal Multi-Head Attention (Text queries Vision patches)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_t = nn.LayerNorm(hidden_dim)

        # 4. Incongruity-Aware Gated Multimodal Fusion Unit
        # Evaluates text, genuine vision, and visual-textual discrepancy: |t - v|
        self.gate_t = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_diff = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # 5. Multi-Task Classification Heads
        # Sentiment Head: 3-class (Negative: 0, Neutral: 1, Positive: 2)
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        # Offensive / Hate Head: 2-class (Non-Hate: 0, Hate/Offensive: 1)
        self.offensive_head = nn.Linear(hidden_dim, num_offensive_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_dict: bool = False,
        return_gate_weights: bool = False,
        return_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass through IndicSent-M.
        
        Args:
            input_ids: [B, seq_len] Text token IDs
            attention_mask: [B, seq_len] Text attention mask (1=real token, 0=pad)
            pixel_values: [B, 3, H, W] Preprocessed image tensors
            return_dict: Whether to return a dict containing intermediate representations
            return_gate_weights: Whether to return the fusion gate values
            return_attentions: Whether to return cross-modal attention maps
            
        Returns:
            Tuple of (sentiment_logits, offensive_logits) by default,
            or dictionary if return_dict=True.
        """
        # 1. Text token embeddings: [B, seq_len, text_dim] -> [B, seq_len, hidden_dim]
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        t_feat = self.text_proj(text_out)

        # 2. Vision extraction: genuine visual global representation and patches
        vis_res = self.vision_encoder(pixel_values=pixel_values)
        vis_patches = vis_res.last_hidden_state  # [B, num_patches, vision_dim]
        v_feat = self.vision_proj(vis_patches)   # [B, num_patches, hidden_dim]

        # Extract genuine global visual representation (pooler_output if present, else CLS token)
        if hasattr(vis_res, "pooler_output") and vis_res.pooler_output is not None:
            vis_global_raw = vis_res.pooler_output
        else:
            vis_global_raw = vis_patches[:, 0, :]
        v_global = self.vision_global_proj(vis_global_raw)  # [B, hidden_dim]

        # 3. Cross-Modal Attention: Text queries Visual patches
        attn_out, attn_weights = self.cross_attn(
            query=t_feat,
            key=v_feat,
            value=v_feat,
            need_weights=return_attentions,
        )
        t_attended = self.norm_t(t_feat + attn_out)  # [B, seq_len, hidden_dim]

        # 4. Masked Mean Pooling: exclude [PAD] tokens from sequence representation
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        t_rep = (t_attended * mask_expanded).sum(dim=1) / sum_mask  # [B, hidden_dim]

        # 5. Incongruity-Aware Gated Multimodal Fusion
        # Measures direct difference between textual sentiment cue and visual global signal
        diff = torch.abs(t_rep - v_global)
        g = self.gate_act(self.gate_t(t_rep) + self.gate_v(v_global) + self.gate_diff(diff))  # [B, hidden_dim]
        
        # Convex combination ensuring direct contribution from both modalities
        fused = self.dropout(g * t_rep + (1.0 - g) * v_global)
        fused = self.fusion_norm(fused)  # [B, hidden_dim]

        # 6. Multi-Task Classification Heads
        sentiment_logits = self.sentiment_head(fused)
        offensive_logits = self.offensive_head(fused)

        if return_dict:
            output = {
                "sentiment_logits": sentiment_logits,
                "offensive_logits": offensive_logits,
                "fused_features": fused,
                "t_rep": t_rep,
                "v_global": v_global,
            }
            if return_gate_weights:
                output["gate_weights"] = g
            if return_attentions:
                output["cross_attention_weights"] = attn_weights
            return output

        return sentiment_logits, offensive_logits


if __name__ == "__main__":
    print("IndicSentM architecture module loaded successfully.")
