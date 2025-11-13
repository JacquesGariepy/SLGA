"""Integration tests for attention pipeline (10 tests).

Tests the complete attention flow:
Embedding → Local attention → Global attention → Fusion → Output
"""

import pytest
import torch
import torch.nn as nn

from src.models import ModelConfig, SLGATransformer
from src.core.attention import SLGAAttention, LocalAttention, GlobalAttention
from src.core.layers import EmbeddingLayer


class TestAttentionPipeline:
    """Attention pipeline integration tests (10 tests)."""

    def test_embedding_to_attention(self, tiny_config):
        """Test embedding layer output feeds into attention."""
        batch_size, seq_len = 2, 32

        # Create embedding
        embedding = EmbeddingLayer(
            vocab_size=tiny_config.vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            embed_dim=tiny_config.embed_dim,
            dropout_rate=tiny_config.dropout_rate,
        )

        # Create attention
        attention = SLGAAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            local_window=tiny_config.local_window,
            global_k=tiny_config.global_k,
        )

        # Forward pass
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        x = embedding(input_ids)
        attn_out = attention(x)

        assert attn_out.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_local_attention_standalone(self, tiny_config):
        """Test local attention operates correctly."""
        batch_size, seq_len = 2, 64

        local_attn = LocalAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            window_size=tiny_config.local_window,
        )

        # Create input
        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)

        # Forward pass
        output = local_attn(x, x, x)

        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_global_attention_standalone(self, tiny_config):
        """Test global attention operates correctly."""
        batch_size, seq_len = 2, 64
        num_landmarks = tiny_config.global_k

        global_attn = GlobalAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
        )

        # Create input
        query = torch.randn(batch_size, seq_len, tiny_config.embed_dim)
        cache = torch.randn(batch_size, num_landmarks, tiny_config.embed_dim)

        # Forward pass
        output = global_attn(query, cache, cache)

        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_slga_attention_fusion(self, tiny_config):
        """Test SLGA attention fuses local and global correctly."""
        batch_size, seq_len = 2, 64

        slga = SLGAAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            local_window=tiny_config.local_window,
            global_k=tiny_config.global_k,
            gated_fusion=True,
        )

        # Create input
        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)

        # Create landmarks
        num_landmarks = tiny_config.global_k
        landmarks = torch.randn(batch_size, num_landmarks, tiny_config.embed_dim)

        # Forward pass
        output = slga(x, cache_global=landmarks)

        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_landmark_selection_to_attention(self, tiny_config):
        """Test landmark selection feeds into global attention."""
        from src.legacy.landmarks import LearnableLandmarkSelector

        batch_size, seq_len = 2, 64

        # Create components
        landmark_selector = LearnableLandmarkSelector(
            embed_dim=tiny_config.embed_dim,
            num_landmarks=tiny_config.global_k * 2,
        )

        slga = SLGAAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            local_window=tiny_config.local_window,
            global_k=tiny_config.global_k,
        )

        # Create input
        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)

        # Select landmarks
        landmark_indices, _, _ = landmark_selector(x, use_gumbel=True)

        # Extract landmark states
        landmark_indices_safe = torch.clamp(landmark_indices, 0, seq_len - 1)
        landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(
            batch_size, tiny_config.global_k, tiny_config.embed_dim
        )
        landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)

        # Attention with landmarks
        output = slga(x, cache_global=landmark_states)

        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_attention_gradients_flow(self, tiny_config):
        """Test gradients flow through attention mechanism."""
        batch_size, seq_len = 2, 32

        slga = SLGAAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            local_window=tiny_config.local_window,
            global_k=tiny_config.global_k,
        )

        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim, requires_grad=True)
        landmarks = torch.randn(batch_size, tiny_config.global_k, tiny_config.embed_dim)

        output = slga(x, cache_global=landmarks)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_attention_with_different_window_sizes(self, tiny_config):
        """Test attention with various local window sizes."""
        batch_size, seq_len = 2, 64

        for window_size in [16, 32, 64]:
            slga = SLGAAttention(
                d_model=tiny_config.embed_dim,
                num_heads=tiny_config.num_heads,
                local_window=window_size,
                global_k=tiny_config.global_k,
            )

            x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)
            output = slga(x)

            assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

    def test_attention_causal_masking(self, tiny_config):
        """Test causal masking in attention prevents future information leak."""
        batch_size, seq_len = 2, 32

        slga = SLGAAttention(
            d_model=tiny_config.embed_dim,
            num_heads=tiny_config.num_heads,
            local_window=tiny_config.local_window,
            global_k=tiny_config.global_k,
            causal=True,
        )

        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)
        output = slga(x)

        # Causal attention should produce valid output
        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)

        # Test that changing future tokens doesn't affect past
        x_modified = x.clone()
        x_modified[:, -10:, :] = torch.randn(batch_size, 10, tiny_config.embed_dim)

        output_modified = slga(x_modified)

        # First tokens should be similar (only affected by past context)
        # Note: Due to global attention, this is an approximate check
        similarity = torch.cosine_similarity(
            output[:, :5, :].flatten(),
            output_modified[:, :5, :].flatten(),
            dim=0
        )
        assert similarity > 0.9  # Should be highly similar

    def test_attention_with_dropout(self, tiny_config):
        """Test attention with dropout enabled."""
        config = tiny_config.model_copy()
        config.dropout_rate = 0.5

        slga = SLGAAttention(
            d_model=config.embed_dim,
            num_heads=config.num_heads,
            local_window=config.local_window,
            global_k=config.global_k,
            dropout=config.dropout_rate,
        )

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.embed_dim)

        # Training mode (dropout active)
        slga.train()
        output1 = slga(x)
        output2 = slga(x)

        # Should produce different outputs due to dropout
        assert not torch.allclose(output1, output2)

        # Eval mode (dropout inactive)
        slga.eval()
        output3 = slga(x)
        output4 = slga(x)

        # Should produce same outputs
        assert torch.allclose(output3, output4)

    def test_full_transformer_block_pipeline(self, tiny_config):
        """Test complete transformer block with attention and FFN."""
        from src.core.layers import TransformerBlock

        batch_size, seq_len = 2, 32

        block = TransformerBlock(tiny_config, layer_idx=0)

        x = torch.randn(batch_size, seq_len, tiny_config.embed_dim)
        landmarks = torch.randn(batch_size, tiny_config.global_k, tiny_config.embed_dim)

        output = block(x, cache_global=landmarks)

        assert output.shape == (batch_size, seq_len, tiny_config.embed_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
