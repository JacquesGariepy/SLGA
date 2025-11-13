"""
Factory integration tests for PositionalLandmarkSelector and HybridLandmarkSelector.

Tests cover:
- Factory creation of positional selector
- Factory creation of hybrid selector
- Parameter passing through factory
- Registry integration
- Error handling
- Compatibility with existing selectors
"""

import pytest
import torch

from src.core.landmarks.factory import LandmarkSelectorFactory
from src.core.landmarks.learned import (
    PositionalLandmarkSelector,
    HybridLandmarkSelector,
    LearnedLandmarkSelector,
)
from src.core.landmarks.gumbel import GumbelSoftmaxSelector
from src.core.landmarks.heuristic import HeuristicLandmarkSelector


class TestFactoryPositionalSelector:
    """Test factory creation of PositionalLandmarkSelector."""

    def test_create_positional_selector(self):
        """Test basic positional selector creation."""
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=2048,
            num_landmarks=64,
            embed_dim=512,
        )
        assert isinstance(selector, PositionalLandmarkSelector)
        assert selector.num_landmarks == 64
        assert selector.max_seq_len == 2048

    def test_positional_selector_parameters(self):
        """Test that parameters are correctly passed."""
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=1024,
            num_landmarks=32,
            embed_dim=256,
        )
        assert selector.max_seq_len == 1024
        assert selector.num_landmarks == 32
        assert selector.pos_embeddings.shape == (1024, 256)

    def test_positional_selector_functional(self):
        """Test that factory-created selector works correctly."""
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=2048,
            num_landmarks=64,
            embed_dim=512,
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 64)
        assert states.shape == (2, 64, 512)
        assert scores.shape == (2, 256)

    @pytest.mark.parametrize("max_seq_len,num_landmarks,embed_dim", [
        (512, 16, 128),
        (1024, 32, 256),
        (2048, 64, 512),
        (4096, 128, 768),
    ])
    def test_various_positional_configurations(
        self, max_seq_len, num_landmarks, embed_dim
    ):
        """Test various positional selector configurations."""
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=max_seq_len,
            num_landmarks=num_landmarks,
            embed_dim=embed_dim,
        )
        assert isinstance(selector, PositionalLandmarkSelector)
        assert selector.num_landmarks == num_landmarks


class TestFactoryHybridSelector:
    """Test factory creation of HybridLandmarkSelector."""

    def test_create_hybrid_selector(self):
        """Test basic hybrid selector creation."""
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=512,
            max_seq_len=2048,
            num_landmarks=64,
        )
        assert isinstance(selector, HybridLandmarkSelector)
        assert selector.num_landmarks == 64

    def test_hybrid_selector_parameters(self):
        """Test that parameters are correctly passed."""
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=256,
            max_seq_len=1024,
            num_landmarks=32,
        )
        assert selector.num_landmarks == 32
        assert selector.content_selector.embed_dim == 256
        assert selector.position_selector.max_seq_len == 1024

    def test_hybrid_selector_functional(self):
        """Test that factory-created selector works correctly."""
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=512,
            max_seq_len=2048,
            num_landmarks=64,
        )
        x = torch.randn(2, 256, 512)

        indices, states, scores = selector(x)

        assert indices.shape == (2, 64)
        assert states.shape == (2, 64, 512)
        assert scores.shape == (2, 256)

    @pytest.mark.parametrize("embed_dim,max_seq_len,num_landmarks", [
        (128, 512, 16),
        (256, 1024, 32),
        (512, 2048, 64),
        (768, 4096, 128),
    ])
    def test_various_hybrid_configurations(
        self, embed_dim, max_seq_len, num_landmarks
    ):
        """Test various hybrid selector configurations."""
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_landmarks=num_landmarks,
        )
        assert isinstance(selector, HybridLandmarkSelector)
        assert selector.num_landmarks == num_landmarks

    def test_hybrid_has_both_sub_selectors(self):
        """Test that hybrid contains both content and position selectors."""
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=512,
            max_seq_len=2048,
            num_landmarks=64,
        )
        assert isinstance(selector.content_selector, LearnedLandmarkSelector)
        assert isinstance(selector.position_selector, PositionalLandmarkSelector)


class TestFactoryRegistryIntegration:
    """Test registry integration for new selectors."""

    def test_positional_in_registry(self):
        """Test that 'positional' is in registry."""
        available = LandmarkSelectorFactory.list_available()
        assert "positional" in available

    def test_hybrid_in_registry(self):
        """Test that 'hybrid' is in registry."""
        available = LandmarkSelectorFactory.list_available()
        assert "hybrid" in available

    def test_positional_retrievable_from_registry(self):
        """Test that positional can be retrieved from registry."""
        assert "positional" in LandmarkSelectorFactory._registry
        assert LandmarkSelectorFactory._registry["positional"] == PositionalLandmarkSelector

    def test_hybrid_retrievable_from_registry(self):
        """Test that hybrid can be retrieved from registry."""
        assert "hybrid" in LandmarkSelectorFactory._registry
        assert LandmarkSelectorFactory._registry["hybrid"] == HybridLandmarkSelector


class TestFactoryErrorHandling:
    """Test error handling in factory."""

    def test_missing_required_parameter_positional(self):
        """Test that missing required parameters raise errors."""
        with pytest.raises(TypeError):
            # Missing max_seq_len
            LandmarkSelectorFactory.create(
                "positional",
                num_landmarks=64,
                embed_dim=512,
            )

    def test_missing_required_parameter_hybrid(self):
        """Test that missing required parameters raise errors."""
        with pytest.raises(TypeError):
            # Missing max_seq_len
            LandmarkSelectorFactory.create(
                "hybrid",
                embed_dim=512,
                num_landmarks=64,
            )

    def test_invalid_selector_type(self):
        """Test that invalid selector type raises error."""
        with pytest.raises(ValueError, match="Unknown selector type"):
            LandmarkSelectorFactory.create(
                "nonexistent_selector",
                embed_dim=512,
                num_landmarks=64,
            )


class TestFactoryCompatibilityWithExistingSelectors:
    """Test compatibility with existing selector types."""

    def test_all_selector_types_creatable(self):
        """Test that all selector types can be created."""
        selectors = {
            "gumbel": LandmarkSelectorFactory.create(
                "gumbel", embed_dim=512, num_landmarks=64, temperature=1.0
            ),
            "learned": LandmarkSelectorFactory.create(
                "learned", embed_dim=512, num_landmarks=64
            ),
            "positional": LandmarkSelectorFactory.create(
                "positional", max_seq_len=2048, num_landmarks=64, embed_dim=512
            ),
            "hybrid": LandmarkSelectorFactory.create(
                "hybrid", embed_dim=512, max_seq_len=2048, num_landmarks=64
            ),
            "uniform": LandmarkSelectorFactory.create(
                "uniform", num_landmarks=64
            ),
            "random": LandmarkSelectorFactory.create(
                "random", num_landmarks=64, seed=42
            ),
            "first": LandmarkSelectorFactory.create(
                "first", num_landmarks=64
            ),
        }

        assert len(selectors) == 7
        assert all(selector is not None for selector in selectors.values())

    def test_all_selectors_have_correct_types(self):
        """Test that all selectors have correct types."""
        gumbel = LandmarkSelectorFactory.create(
            "gumbel", embed_dim=512, num_landmarks=64, temperature=1.0
        )
        learned = LandmarkSelectorFactory.create(
            "learned", embed_dim=512, num_landmarks=64
        )
        positional = LandmarkSelectorFactory.create(
            "positional", max_seq_len=2048, num_landmarks=64, embed_dim=512
        )
        hybrid = LandmarkSelectorFactory.create(
            "hybrid", embed_dim=512, max_seq_len=2048, num_landmarks=64
        )
        uniform = LandmarkSelectorFactory.create(
            "uniform", num_landmarks=64
        )

        assert isinstance(gumbel, GumbelSoftmaxSelector)
        assert isinstance(learned, LearnedLandmarkSelector)
        assert isinstance(positional, PositionalLandmarkSelector)
        assert isinstance(hybrid, HybridLandmarkSelector)
        assert isinstance(uniform, HeuristicLandmarkSelector)

    def test_all_selectors_functional(self):
        """Test that all selectors work correctly."""
        x = torch.randn(2, 128, 512)

        selectors = {
            "gumbel": LandmarkSelectorFactory.create(
                "gumbel", embed_dim=512, num_landmarks=32, temperature=1.0
            ),
            "learned": LandmarkSelectorFactory.create(
                "learned", embed_dim=512, num_landmarks=32
            ),
            "positional": LandmarkSelectorFactory.create(
                "positional", max_seq_len=2048, num_landmarks=32, embed_dim=512
            ),
            "hybrid": LandmarkSelectorFactory.create(
                "hybrid", embed_dim=512, max_seq_len=2048, num_landmarks=32
            ),
            "uniform": LandmarkSelectorFactory.create(
                "uniform", num_landmarks=32
            ),
        }

        for name, selector in selectors.items():
            indices, states, scores = selector(x)
            assert indices.shape == (2, 32), f"{name} failed"
            assert states.shape == (2, 32, 512), f"{name} failed"
            assert scores.shape == (2, 128), f"{name} failed"


class TestFactoryListAvailable:
    """Test list_available functionality."""

    def test_list_available_includes_new_selectors(self):
        """Test that list_available includes positional and hybrid."""
        available = LandmarkSelectorFactory.list_available()

        assert "positional" in available
        assert "hybrid" in available

    def test_list_available_includes_all_selectors(self):
        """Test that list_available includes all selector types."""
        available = LandmarkSelectorFactory.list_available()

        expected = [
            "gumbel", "learned", "positional", "hybrid",
            "uniform", "random", "first"
        ]

        for selector_type in expected:
            assert selector_type in available

    def test_list_available_sorted(self):
        """Test that list is sorted."""
        available = LandmarkSelectorFactory.list_available()
        assert available == sorted(available)


class TestFactoryCustomRegistration:
    """Test custom selector registration (integration test)."""

    def test_can_unregister_and_reregister_positional(self):
        """Test unregistering and re-registering positional."""
        # Unregister
        LandmarkSelectorFactory.unregister("positional")
        assert "positional" not in LandmarkSelectorFactory._registry

        # Re-register
        LandmarkSelectorFactory.register("positional", PositionalLandmarkSelector)
        assert "positional" in LandmarkSelectorFactory._registry

        # Verify it works
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=2048,
            num_landmarks=64,
            embed_dim=512,
        )
        assert isinstance(selector, PositionalLandmarkSelector)

    def test_can_unregister_and_reregister_hybrid(self):
        """Test unregistering and re-registering hybrid."""
        # Unregister
        LandmarkSelectorFactory.unregister("hybrid")
        assert "hybrid" not in LandmarkSelectorFactory._registry

        # Re-register
        LandmarkSelectorFactory.register("hybrid", HybridLandmarkSelector)
        assert "hybrid" in LandmarkSelectorFactory._registry

        # Verify it works
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=512,
            max_seq_len=2048,
            num_landmarks=64,
        )
        assert isinstance(selector, HybridLandmarkSelector)


class TestFactoryEndToEnd:
    """End-to-end integration tests."""

    def test_complete_workflow_positional(self):
        """Test complete workflow with positional selector."""
        # Create
        selector = LandmarkSelectorFactory.create(
            "positional",
            max_seq_len=2048,
            num_landmarks=64,
            embed_dim=512,
        )

        # Use
        x = torch.randn(4, 256, 512)
        indices, states, scores = selector(x)

        # Train (use scores for gradient flow to positional embeddings)
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = scores.sum()
        loss.backward()
        optimizer.step()

        # Verify training worked
        assert selector.pos_embeddings.grad is not None

    def test_complete_workflow_hybrid(self):
        """Test complete workflow with hybrid selector."""
        # Create
        selector = LandmarkSelectorFactory.create(
            "hybrid",
            embed_dim=512,
            max_seq_len=2048,
            num_landmarks=64,
        )

        # Use
        x = torch.randn(4, 256, 512)
        indices, states, scores = selector(x)

        # Train
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = states.sum()
        loss.backward()
        optimizer.step()

        # Verify all components trained
        assert any(p.grad is not None for p in selector.content_selector.parameters())
        assert selector.position_selector.pos_embeddings.grad is not None
        assert selector.gate.weight.grad is not None

    def test_switching_between_selector_types(self):
        """Test creating multiple selector types in sequence."""
        x = torch.randn(2, 128, 512)

        # Create different selectors
        positional = LandmarkSelectorFactory.create(
            "positional", max_seq_len=2048, num_landmarks=32, embed_dim=512
        )
        hybrid = LandmarkSelectorFactory.create(
            "hybrid", embed_dim=512, max_seq_len=2048, num_landmarks=32
        )
        learned = LandmarkSelectorFactory.create(
            "learned", embed_dim=512, num_landmarks=32
        )

        # All should work
        idx_pos, _, _ = positional(x)
        idx_hyb, _, _ = hybrid(x)
        idx_lrn, _, _ = learned(x)

        assert idx_pos.shape == (2, 32)
        assert idx_hyb.shape == (2, 32)
        assert idx_lrn.shape == (2, 32)
