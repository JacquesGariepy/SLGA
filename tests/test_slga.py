"""
Test suite for SLGA module
Tests the three main bug fixes:
1. Parameter validation
2. Mask caching
3. Deterministic top-k selection
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legacy.slga import SLGAModule


class TestSLGAModule:
    """Test SLGA module core functionality"""

    def test_parameter_validation(self):
        """Test que les paramètres invalides lèvent des erreurs"""
        # embed_dim non-divisible par num_heads
        with pytest.raises(AssertionError, match="embed_dim.*divisible"):
            SLGAModule(embed_dim=513, num_heads=8, local_window=128, global_k=24)

        # local_window négatif
        with pytest.raises(AssertionError, match="local_window"):
            SLGAModule(embed_dim=512, num_heads=8, local_window=-1, global_k=24)

        # dropout invalide
        with pytest.raises(AssertionError, match="attn_drop"):
            SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24, attn_drop=1.5)

        # global_k négatif
        with pytest.raises(AssertionError, match="global_k"):
            SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=-1)

    def test_valid_parameters(self):
        """Test que les paramètres valides ne lèvent pas d'erreurs"""
        # Ces configurations doivent toutes fonctionner
        valid_configs = [
            {"embed_dim": 384, "num_heads": 6, "local_window": 128, "global_k": 24},
            {"embed_dim": 512, "num_heads": 8, "local_window": 256, "global_k": 32},
            {"embed_dim": 768, "num_heads": 12, "local_window": 64, "global_k": 16},
        ]

        for config in valid_configs:
            module = SLGAModule(**config)
            assert module.D == config["embed_dim"]
            assert module.H == config["num_heads"]

    def test_mask_caching(self):
        """Test que le cache de masques fonctionne"""
        module = SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24)

        device = torch.device("cpu")

        # Premier appel: calcul
        mask1 = module._create_local_causal_mask_vectorized(256, 128, device)

        # Deuxième appel: depuis cache
        mask2 = module._create_local_causal_mask_vectorized(256, 128, device)

        # Doivent être identiques
        assert torch.equal(mask1, mask2)

        # Vérifier que c'est bien le même objet (cache)
        assert mask1 is mask2

    def test_mask_cache_different_sizes(self):
        """Test que des tailles différentes ne partagent pas le cache"""
        module = SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24)

        device = torch.device("cpu")

        mask1 = module._create_local_causal_mask_vectorized(256, 128, device)
        mask2 = module._create_local_causal_mask_vectorized(512, 128, device)

        # Shapes différentes
        assert mask1.shape != mask2.shape
        # Objets différents
        assert mask1 is not mask2

    def test_deterministic_topk(self):
        """Test que la sélection landmarks est déterministe"""
        torch.manual_seed(42)
        module = SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24)

        x = torch.randn(2, 256, 512)

        # Deux forwards avec même seed
        torch.manual_seed(42)
        out1 = module(x)

        torch.manual_seed(42)
        out2 = module(x)

        # Sorties doivent être identiques
        assert torch.allclose(out1, out2, atol=1e-6), "Forward pass not deterministic"

    def test_deterministic_with_different_seeds(self):
        """Test que différentes seeds donnent des résultats différents"""
        module = SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24)
        x = torch.randn(2, 256, 512)

        torch.manual_seed(42)
        out1 = module(x)

        torch.manual_seed(123)
        out2 = module(x)

        # Sorties doivent être différentes
        assert not torch.allclose(out1, out2, atol=1e-6), "Different seeds should produce different outputs"

    def test_forward_shapes(self):
        """Test que les shapes sont préservées"""
        module = SLGAModule(embed_dim=512, num_heads=8, local_window=128, global_k=24)

        B, L, D = 4, 256, 512
        x = torch.randn(B, L, D)

        out = module(x)

        assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"

    def test_forward_various_sequence_lengths(self):
        """Test avec différentes longueurs de séquence"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)

        test_lengths = [64, 128, 256, 512, 1024]

        for length in test_lengths:
            x = torch.randn(2, length, 384)
            out = module(x)
            assert out.shape == (2, length, 384), f"Failed for length {length}"

    def test_backward_pass(self):
        """Test que les gradients se propagent correctement"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)

        x = torch.randn(2, 256, 384, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()

        # Vérifier que les gradients existent
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_training_mode(self):
        """Test comportement en mode train vs eval"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24, attn_drop=0.1)
        x = torch.randn(2, 256, 384)

        # Mode train
        module.train()
        out_train = module(x)

        # Mode eval
        module.eval()
        with torch.no_grad():
            out_eval = module(x)

        # Les sorties peuvent être légèrement différentes à cause du dropout
        assert out_train.shape == out_eval.shape

    def test_device_compatibility(self):
        """Test que le module fonctionne sur CPU"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)
        x = torch.randn(2, 128, 384)

        out = module(x)
        assert out.device == x.device


class TestSLGAModuleEdgeCases:
    """Test edge cases et cas limites"""

    def test_single_token_sequence(self):
        """Test avec une séquence d'un seul token"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)
        x = torch.randn(1, 1, 384)

        out = module(x)
        assert out.shape == (1, 1, 384)

    def test_large_batch_size(self):
        """Test avec un grand batch size"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=128, global_k=24)
        x = torch.randn(32, 128, 384)

        out = module(x)
        assert out.shape == (32, 128, 384)

    def test_window_larger_than_sequence(self):
        """Test quand local_window > sequence_length"""
        module = SLGAModule(embed_dim=384, num_heads=6, local_window=256, global_k=24)
        x = torch.randn(2, 128, 384)

        # Ne doit pas crasher
        out = module(x)
        assert out.shape == (2, 128, 384)

    def test_minimal_configuration(self):
        """Test configuration minimale"""
        module = SLGAModule(embed_dim=64, num_heads=2, local_window=8, global_k=4)
        x = torch.randn(1, 32, 64)

        out = module(x)
        assert out.shape == (1, 32, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
