"""
Test suite for KV Cache implementation.

Tests:
1. KVCache creation and basic operations
2. Integration with SLGAModule
3. Integration with LLMTransformer.generate()
4. Performance comparison (cached vs non-cached)
5. Correctness: cached generation == non-cached generation
"""

import pytest
import torch
from src.kv_cache import KVCache
from src.model import Config, LLMTransformer


class TestKVCacheBasics:
    """Test basic KVCache functionality."""

    def test_cache_creation(self):
        """Test cache can be created with correct dimensions."""
        cache = KVCache.create(
            batch_size=2,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=128,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert cache.seq_len == 0
        assert cache.max_seq_len == 128
        assert cache.get_batch_size() == 2
        assert cache.get_num_layers() == 4
        assert cache.get_num_heads() == 8
        assert cache.get_head_dim() == 64

    def test_cache_update(self):
        """Test updating cache with new K, V."""
        cache = KVCache.create(
            batch_size=2,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=100,
            device=torch.device("cpu"),
        )

        # Add first sequence
        k1 = torch.randn(2, 4, 5, 32)  # 5 tokens
        v1 = torch.randn(2, 4, 5, 32)
        cache.update(0, k1, v1)
        cache.increment_seq_len(5)

        assert cache.seq_len == 5

        # Add second sequence
        k2 = torch.randn(2, 4, 3, 32)  # 3 more tokens
        v2 = torch.randn(2, 4, 3, 32)
        cache.update(0, k2, v2)
        cache.increment_seq_len(3)

        assert cache.seq_len == 8

        # Retrieve cached values
        k_cached, v_cached = cache.get(0)
        assert k_cached.shape == (2, 4, 8, 32)
        assert v_cached.shape == (2, 4, 8, 32)

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = KVCache.create(
            batch_size=1,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            max_seq_len=50,
            device=torch.device("cpu"),
        )

        k = torch.randn(1, 2, 10, 16)
        v = torch.randn(1, 2, 10, 16)
        cache.update(0, k, v)
        cache.increment_seq_len(10)

        assert cache.seq_len == 10

        cache.clear()
        assert cache.seq_len == 0

    def test_cache_overflow(self):
        """Test that cache raises error on overflow."""
        cache = KVCache.create(
            batch_size=1,
            num_layers=1,
            num_heads=2,
            head_dim=16,
            max_seq_len=10,
            device=torch.device("cpu"),
        )

        k = torch.randn(1, 2, 15, 16)  # Too many tokens
        v = torch.randn(1, 2, 15, 16)

        with pytest.raises(ValueError, match="Cache overflow"):
            cache.update(0, k, v)


class TestKVCacheWithModel:
    """Test KV cache integration with the full model."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        cfg = Config(
            vocab_size=256,
            max_seq_len=128,
            embed_dim=128,
            num_heads=4,
            ff_hidden_multiplier=2,
            n_layers=2,
            dropout_rate=0.0,
            local_window=16,
            global_k=8,
            learned_landmarks=False,  # Simpler for testing
        )
        model = LLMTransformer(cfg)
        model.eval()
        return model

    def test_generate_with_cache(self, small_model):
        """Test generation with KV cache enabled."""
        torch.manual_seed(42)
        prompt = torch.randint(0, 256, (1, 10))

        # Generate with cache
        output_cached = small_model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,  # Greedy for determinism
            use_kv_cache=True,
            stop_on_eos=False,
        )

        assert output_cached.shape[1] == 30  # 10 prompt + 20 new tokens

    def test_generate_without_cache(self, small_model):
        """Test generation without KV cache."""
        torch.manual_seed(42)
        prompt = torch.randint(0, 256, (1, 10))

        # Generate without cache
        output_no_cache = small_model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            use_kv_cache=False,
            stop_on_eos=False,
        )

        assert output_no_cache.shape[1] == 30

    def test_cache_correctness(self, small_model):
        """
        Test that cached generation produces same results as non-cached.

        This is the critical correctness test.
        """
        # Ensure model is in eval mode
        small_model.eval()

        torch.manual_seed(123)
        prompt = torch.randint(0, 256, (2, 8))

        # Generate without cache - use separate runs to avoid state leakage
        torch.manual_seed(456)
        output_no_cache = small_model.generate(
            prompt.clone(),
            max_new_tokens=15,
            temperature=0.0,  # Greedy for determinism
            use_kv_cache=False,
            stop_on_eos=False,
        )

        # Reset RNG and generate with cache
        torch.manual_seed(456)
        output_cached = small_model.generate(
            prompt.clone(),
            max_new_tokens=15,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=False,
        )

        # Results should be identical
        if not torch.equal(output_no_cache, output_cached):
            # Print debug info
            print("\nNo cache:", output_no_cache)
            print("Cached:  ", output_cached)
            diff_pos = (output_no_cache != output_cached).nonzero(as_tuple=True)
            print(f"Differences at positions: {diff_pos}")

        assert torch.equal(output_no_cache, output_cached), \
            "Cached and non-cached generation produced different results!"

    def test_cache_with_sampling(self, small_model):
        """Test cache with temperature-based sampling."""
        torch.manual_seed(456)
        prompt = torch.randint(0, 256, (1, 5))

        output = small_model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            use_kv_cache=True,
            stop_on_eos=False,
        )

        assert output.shape[1] == 15

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cache_batch_sizes(self, small_model, batch_size):
        """Test cache works with different batch sizes."""
        torch.manual_seed(789)
        prompt = torch.randint(0, 256, (batch_size, 6))

        output = small_model.generate(
            prompt,
            max_new_tokens=12,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=False,
        )

        assert output.shape == (batch_size, 18)


class TestKVCachePerformance:
    """Test KV cache performance improvements."""

    @pytest.fixture
    def medium_model(self):
        """Create a medium-sized model for performance testing."""
        cfg = Config(
            vocab_size=512,
            max_seq_len=512,
            embed_dim=256,
            num_heads=8,
            ff_hidden_multiplier=4,
            n_layers=4,
            dropout_rate=0.0,
            local_window=32,
            global_k=16,
            learned_landmarks=False,
        )
        model = LLMTransformer(cfg)
        model.eval()
        return model

    @pytest.mark.slow
    def test_cache_speedup(self, medium_model):
        """
        Test that cached generation is faster than non-cached.

        Note: This test is marked as slow and may be skipped in CI.
        """
        import time

        prompt = torch.randint(0, 512, (1, 20))
        max_new_tokens = 50

        # Warmup
        _ = medium_model.generate(
            prompt.clone(),
            max_new_tokens=5,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=False,
        )

        # Time without cache
        start = time.time()
        _ = medium_model.generate(
            prompt.clone(),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_kv_cache=False,
            stop_on_eos=False,
        )
        time_no_cache = time.time() - start

        # Time with cache
        start = time.time()
        _ = medium_model.generate(
            prompt.clone(),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=False,
        )
        time_cached = time.time() - start

        # Cache should be faster
        speedup = time_no_cache / time_cached
        print(f"\nSpeedup: {speedup:.2f}x (cached: {time_cached:.3f}s, "
              f"no cache: {time_no_cache:.3f}s)")

        # Expect at least 1.5x speedup (conservative)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"


class TestKVCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_with_eos_stopping(self):
        """Test cache works correctly with EOS stopping."""
        cfg = Config(
            vocab_size=100,
            max_seq_len=64,
            embed_dim=64,
            num_heads=4,
            n_layers=2,
            learned_landmarks=False,
        )
        model = LLMTransformer(cfg)
        model.eval()

        prompt = torch.randint(0, 100, (1, 5))
        eos_token_id = 99

        output = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=True,
            eos_token_id=eos_token_id,
        )

        # Should stop early or at max_new_tokens
        assert output.shape[1] <= 25

    def test_cache_with_long_sequence(self):
        """Test cache handles sequences approaching max_seq_len."""
        cfg = Config(
            vocab_size=100,
            max_seq_len=100,
            embed_dim=64,
            num_heads=4,
            n_layers=2,
            learned_landmarks=False,
        )
        model = LLMTransformer(cfg)
        model.eval()

        # Long prompt close to max
        prompt = torch.randint(0, 100, (1, 80))

        output = model.generate(
            prompt,
            max_new_tokens=15,
            temperature=0.0,
            use_kv_cache=True,
            stop_on_eos=False,
        )

        # Should truncate to max_seq_len
        assert output.shape[1] <= 100


if __name__ == "__main__":
    # Run basic tests
    print("Running KV Cache Tests...")

    # Test 1: Basic cache operations
    print("\n1. Testing cache creation and operations...")
    test_basics = TestKVCacheBasics()
    test_basics.test_cache_creation()
    test_basics.test_cache_update()
    test_basics.test_cache_clear()
    print("✓ Basic cache tests passed")

    # Test 2: Model integration
    print("\n2. Testing model integration...")
    cfg = Config(
        vocab_size=256,
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        learned_landmarks=False,
    )
    model = LLMTransformer(cfg)
    model.eval()

    test_model = TestKVCacheWithModel()
    test_model.test_generate_with_cache(model)
    test_model.test_generate_without_cache(model)
    print("✓ Model integration tests passed")

    # Test 3: Correctness
    print("\n3. Testing correctness (cached == non-cached)...")
    test_model.test_cache_correctness(model)
    print("✓ Correctness test passed - outputs match!")

    print("\n✅ All KV Cache tests passed!")
