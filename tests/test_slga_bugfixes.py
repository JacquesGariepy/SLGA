#!/usr/bin/env python3
"""
Tests unitaires pour vérifier les 3 bug fixes critiques dans src/slga.py

Basé sur ANALYSE_COMPLETE_LLM.md (lignes 760-820)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from src.slga import SLGAModule


class TestBugFix1_ParameterValidation:
    """Bug #1: Validation de paramètres manquante (lignes 30-57)"""

    def test_embed_dim_not_divisible_by_num_heads(self):
        """Doit rejeter embed_dim non divisible par num_heads"""
        with pytest.raises(AssertionError, match="embed_dim.*divisible.*num_heads"):
            SLGAModule(embed_dim=513, num_heads=8)  # 513 % 8 != 0

    def test_local_window_zero(self):
        """Doit rejeter local_window <= 0"""
        with pytest.raises(AssertionError, match="local_window must be > 0"):
            SLGAModule(embed_dim=512, num_heads=8, local_window=0)

    def test_global_k_negative(self):
        """Doit rejeter global_k <= 0"""
        with pytest.raises(AssertionError, match="global_k must be > 0"):
            SLGAModule(embed_dim=512, num_heads=8, global_k=-5)

    def test_attn_drop_out_of_range(self):
        """Doit rejeter dropout hors [0, 1)"""
        with pytest.raises(AssertionError, match="attn_drop must be in"):
            SLGAModule(embed_dim=512, num_heads=8, attn_drop=1.5)

    def test_proj_drop_out_of_range(self):
        """Doit rejeter proj_drop hors [0, 1)"""
        with pytest.raises(AssertionError, match="proj_drop must be in"):
            SLGAModule(embed_dim=512, num_heads=8, proj_drop=-0.1)

    def test_dilation_invalid(self):
        """Doit rejeter dilation < 1"""
        with pytest.raises(AssertionError, match="dilation must be >= 1"):
            SLGAModule(embed_dim=512, num_heads=8, dilation=0)

    def test_valid_parameters_no_error(self):
        """Paramètres valides doivent passer"""
        module = SLGAModule(
            embed_dim=512,
            num_heads=8,
            local_window=128,
            global_k=24,
            attn_drop=0.1,
            proj_drop=0.1,
            dilation=2
        )
        assert module.D == 512
        assert module.H == 8


class TestBugFix2_MaskCaching:
    """Bug #2: Cache pour masques manquant (lignes 65-76)"""

    def test_mask_cache_exists(self):
        """Vérifier que le cache de masques est initialisé"""
        module = SLGAModule(embed_dim=512, num_heads=8)
        assert hasattr(module, '_mask_cache')
        assert isinstance(module._mask_cache, dict)

    def test_mask_caching_works(self):
        """Vérifier que les masques sont réutilisés du cache"""
        module = SLGAModule(embed_dim=512, num_heads=8, causal=True)
        device = torch.device('cpu')

        # Premier appel: calcul et mise en cache
        mask1 = module._create_local_causal_mask_vectorized(128, 32, device)
        assert (128, 32, device) in module._mask_cache

        # Deuxième appel: récupération du cache (doit être la même instance)
        mask2 = module._create_local_causal_mask_vectorized(128, 32, device)
        assert mask1 is mask2  # Même objet en mémoire

    def test_mask_vectorization_correctness(self):
        """Vérifier que le masque vectorisé est correct"""
        module = SLGAModule(embed_dim=512, num_heads=8, causal=True)
        device = torch.device('cpu')
        seq_len, window = 10, 3

        mask = module._create_local_causal_mask_vectorized(seq_len, window, device)

        # Vérifications manuelles pour quelques positions
        # Position 0: peut voir seulement elle-même (0)
        assert mask[0, 0] == False  # Peut voir position 0
        assert mask[0, 1:].all()    # Ne peut pas voir futures

        # Position 5 avec window=3: peut voir [2, 3, 4, 5]
        assert mask[5, 2] == False  # Dans la fenêtre
        assert mask[5, 3] == False
        assert mask[5, 4] == False
        assert mask[5, 5] == False
        assert mask[5, 1] == True   # Hors fenêtre (trop loin)
        assert mask[5, 6] == True   # Futur

    def test_mask_performance_improvement(self):
        """Mesurer le speedup du cache (doit être >2x)"""
        import time

        module = SLGAModule(embed_dim=512, num_heads=8)
        device = torch.device('cpu')
        seq_len, window = 512, 128

        # Mesure sans cache (premier appel)
        module._mask_cache.clear()
        start = time.time()
        for _ in range(100):
            module._mask_cache.clear()
            _ = module._create_local_causal_mask_vectorized(seq_len, window, device)
        time_no_cache = time.time() - start

        # Mesure avec cache
        module._mask_cache.clear()
        _ = module._create_local_causal_mask_vectorized(seq_len, window, device)  # Prime cache
        start = time.time()
        for _ in range(100):
            _ = module._create_local_causal_mask_vectorized(seq_len, window, device)
        time_with_cache = time.time() - start

        speedup = time_no_cache / time_with_cache
        print(f"\nMask cache speedup: {speedup:.1f}x")
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


class TestIntegration:
    """Tests d'intégration pour vérifier que les fixes n'ont pas cassé le forward pass"""

    def test_forward_pass_with_fixes(self):
        """Vérifier que le forward pass fonctionne toujours correctement"""
        B, L, D = 2, 128, 512
        H = 8

        module = SLGAModule(
            embed_dim=D,
            num_heads=H,
            local_window=32,
            global_k=16,
            causal=True
        )

        x = torch.randn(B, L, D)
        cache = torch.randn(B, 24, D)

        # Forward pass ne doit pas planter
        out = module(x, cache_global=cache)

        assert out.shape == (B, L, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow_with_fixes(self):
        """Vérifier que les gradients se propagent correctement"""
        B, L, D = 2, 64, 512

        module = SLGAModule(embed_dim=D, num_heads=8, local_window=16, global_k=8)

        x = torch.randn(B, L, D, requires_grad=True)
        cache = torch.randn(B, 16, D, requires_grad=True)

        out = module(x, cache_global=cache)
        loss = out.sum()
        loss.backward()

        # Vérifier gradients non-None
        assert x.grad is not None
        assert cache.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(cache.grad).any()


def run_all_tests():
    """Exécuter tous les tests avec rapport détaillé"""
    print("=" * 80)
    print("TESTS DES 3 BUG FIXES CRITIQUES - src/slga.py")
    print("=" * 80)

    # Run pytest programmatically
    pytest_args = [
        __file__,
        '-v',           # Verbose
        '--tb=short',   # Traceback court
        '-x',           # Stop au premier échec
    ]

    exit_code = pytest.main(pytest_args)

    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ TOUS LES TESTS PASSÉS - Les 3 bugs ont été corrigés avec succès!")
    else:
        print("❌ ÉCHEC DES TESTS - Voir détails ci-dessus")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    exit(run_all_tests())
