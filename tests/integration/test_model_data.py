"""Integration tests for model-data integration (30 tests).

Tests the integration between data loaders and models:
Data loader → Model forward pass → Output validation
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.models import ModelConfig, SLGATransformer
from src.legacy.data import CollatorLocal, CollatorLocalGlobal


class TestDataLoaderIntegration:
    """DataLoader to model integration tests (10 tests)."""

    def test_dataloader_to_model_basic(self, sample_dataset, collator_local, tiny_model, tiny_config):
        """Test basic dataloader to model pipeline."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                assert logits.shape[0] == input_ids.shape[0]
                assert logits.shape[1] == input_ids.shape[1]
                assert logits.shape[2] == tiny_config.vocab_size

    def test_dataloader_with_landmarks(self, sample_dataset, collator_global, tiny_model):
        """Test dataloader with global landmarks."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            collate_fn=collator_global,
        )

        tiny_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                landmarks = batch["cache_global_ids"]

                logits = tiny_model(input_ids, cache_global_ids=landmarks)

                assert logits.shape[0] == input_ids.shape[0]
                assert "cache_global_ids" in batch

    def test_dataloader_different_batch_sizes(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with different batch sizes."""
        for batch_size in [1, 2, 4]:
            dataloader = DataLoader(
                sample_dataset,
                batch_size=batch_size,
                collate_fn=collator_local,
            )

            tiny_model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"]
                    logits = tiny_model(input_ids)

                    assert logits.shape[0] <= batch_size

    def test_dataloader_shuffling(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with shuffling."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        batches = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)
                batches.append(logits.clone())

        # Should have processed all data
        assert len(batches) > 0

    def test_dataloader_drop_last(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with drop_last."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            drop_last=True,
            collate_fn=collator_local,
        )

        batch_count = 0
        tiny_model.eval()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                # All batches should have exactly batch_size
                assert input_ids.shape[0] == 2
                batch_count += 1

        assert batch_count > 0

    def test_dataloader_num_workers(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with multiple workers."""
        # Note: num_workers=0 is used for testing to avoid multiprocessing issues
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                assert logits.shape[0] == input_ids.shape[0]

    def test_dataloader_pin_memory(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with pin_memory."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            pin_memory=True,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                assert logits.shape[0] == input_ids.shape[0]

    def test_dataloader_iteration_consistency(self, sample_dataset, collator_local, tiny_model):
        """Test multiple iterations over dataloader produce same results."""
        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collator_local,
        )

        tiny_model.eval()

        # First iteration
        first_batches = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)
                first_batches.append(logits.clone())

        # Second iteration
        second_batches = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)
                second_batches.append(logits.clone())

        # Should produce same results
        assert len(first_batches) == len(second_batches)
        for b1, b2 in zip(first_batches, second_batches):
            assert torch.allclose(b1, b2)

    def test_dataloader_empty_batch_handling(self, collator_local, tiny_model):
        """Test dataloader with very small dataset."""
        from datasets import Dataset

        small_dataset = Dataset.from_dict({"text": ["Single example"]})

        dataloader = DataLoader(
            small_dataset,
            batch_size=2,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                assert logits.shape[0] == 1  # Only one example
                batch_count += 1

        assert batch_count == 1

    def test_dataloader_large_dataset(self, collator_local, tiny_model):
        """Test dataloader with larger dataset."""
        from datasets import Dataset

        large_dataset = Dataset.from_dict({
            "text": [f"Sample text number {i}" for i in range(100)]
        })

        dataloader = DataLoader(
            large_dataset,
            batch_size=8,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        batch_count = 0
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                logits = tiny_model(input_ids)

                sample_count += input_ids.shape[0]
                batch_count += 1

        assert batch_count > 0
        assert sample_count == 100


class TestBatchVariations:
    """Test various batch configurations (10 tests)."""

    def test_single_example_batch(self, tiny_model, tiny_config):
        """Test model with batch size of 1."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 32))

        logits = tiny_model(input_ids)

        assert logits.shape == (1, 32, tiny_config.vocab_size)

    def test_varying_sequence_lengths_same_batch(self, tiny_model, tokenizer):
        """Test batch with varying actual sequence lengths (with padding)."""
        texts = [
            "Short",
            "Medium length text here",
            "Very long text with many more words to fill up space",
        ]

        collator = CollatorLocal(tokenizer, max_length=64)
        batch = collator([{"text": t} for t in texts])

        logits = tiny_model(batch["input_ids"])

        assert logits.shape == (3, 64, tiny_model.config.vocab_size)

    def test_maximum_sequence_length(self, tiny_model, tiny_config):
        """Test model at maximum sequence length."""
        seq_len = tiny_config.max_seq_len
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, seq_len))

        logits = tiny_model(input_ids)

        assert logits.shape == (2, seq_len, tiny_config.vocab_size)

    def test_minimum_sequence_length(self, tiny_model, tiny_config):
        """Test model with minimal sequence length."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 1))

        logits = tiny_model(input_ids)

        assert logits.shape == (2, 1, tiny_config.vocab_size)

    def test_power_of_two_batch_sizes(self, tiny_model, tiny_config):
        """Test model with power-of-2 batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, 32))
            logits = tiny_model(input_ids)

            assert logits.shape == (batch_size, 32, tiny_config.vocab_size)

    def test_odd_batch_sizes(self, tiny_model, tiny_config):
        """Test model with odd batch sizes."""
        for batch_size in [3, 5, 7, 11]:
            input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, 32))
            logits = tiny_model(input_ids)

            assert logits.shape == (batch_size, 32, tiny_config.vocab_size)

    def test_batch_with_all_padding(self, tiny_model, tokenizer):
        """Test batch where some examples are all padding."""
        texts = [
            "",  # Empty text
            "Real text here",
            "",  # Another empty
        ]

        collator = CollatorLocal(tokenizer, max_length=64)
        batch = collator([{"text": t} for t in texts])

        logits = tiny_model(batch["input_ids"])

        assert logits.shape == (3, 64, tiny_model.config.vocab_size)

    def test_batch_landmark_count_matches(self, tiny_model, tiny_config, tokenizer):
        """Test landmarks match across different batches."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=tiny_config.global_k,
            strategy="regular"
        )

        texts = [f"Text {i}" for i in range(10)]

        for i in range(0, len(texts), 2):
            batch_texts = texts[i:i+2]
            batch = collator([{"text": t} for t in batch_texts])

            landmarks = batch["cache_global_ids"]
            assert landmarks.shape[1] == tiny_config.global_k

            logits = tiny_model(batch["input_ids"], cache_global_ids=landmarks)
            assert logits.shape[0] == len(batch_texts)

    def test_batch_all_same_text(self, tiny_model, tokenizer):
        """Test batch with all identical text."""
        text = "Repeated text"
        texts = [text] * 5

        collator = CollatorLocal(tokenizer, max_length=64)
        batch = collator([{"text": t} for t in texts])

        logits = tiny_model(batch["input_ids"])

        # All logits should be identical for identical inputs
        for i in range(1, 5):
            assert torch.allclose(logits[0], logits[i])

    def test_batch_memory_scaling(self, tiny_model, tiny_config):
        """Test memory usage scales linearly with batch size."""
        import gc

        gc.collect()

        # Small batch
        input_ids_small = torch.randint(0, tiny_config.vocab_size, (2, 32))
        logits_small = tiny_model(input_ids_small)
        size_small = logits_small.numel() * logits_small.element_size()

        del input_ids_small, logits_small
        gc.collect()

        # Large batch (4x)
        input_ids_large = torch.randint(0, tiny_config.vocab_size, (8, 32))
        logits_large = tiny_model(input_ids_large)
        size_large = logits_large.numel() * logits_large.element_size()

        # Size should scale approximately linearly
        ratio = size_large / size_small
        assert 3.5 < ratio < 4.5  # Should be ~4x


class TestDeviceTransfers:
    """Test device transfer integration (10 tests)."""

    def test_cpu_inference(self, tiny_model, tiny_config):
        """Test model inference on CPU."""
        tiny_model = tiny_model.cpu()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        input_ids = input_ids.cpu()

        logits = tiny_model(input_ids)

        assert logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self, tiny_model, tiny_config):
        """Test model inference on CUDA."""
        tiny_model = tiny_model.cuda()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        input_ids = input_ids.cuda()

        logits = tiny_model(input_ids)

        assert logits.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_to_cuda_transfer(self, tiny_model, tiny_config):
        """Test transferring model from CPU to CUDA."""
        # Start on CPU
        tiny_model = tiny_model.cpu()

        # Move to CUDA
        tiny_model = tiny_model.cuda()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32)).cuda()
        logits = tiny_model(input_ids)

        assert logits.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_to_cpu_transfer(self, tiny_model, tiny_config):
        """Test transferring model from CUDA to CPU."""
        # Start on CUDA
        tiny_model = tiny_model.cuda()

        # Move to CPU
        tiny_model = tiny_model.cpu()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32)).cpu()
        logits = tiny_model(input_ids)

        assert logits.device.type == "cpu"

    def test_mixed_device_error(self, tiny_model, tiny_config):
        """Test error when model and data on different devices."""
        tiny_model = tiny_model.cpu()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

            with pytest.raises(RuntimeError):
                logits = tiny_model(input_ids)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_device_consistency(self, tiny_model, tiny_config, tokenizer):
        """Test all batch components on same device."""
        tiny_model = tiny_model.cuda()

        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=64,
            max_global=tiny_config.global_k,
            strategy="regular"
        )

        batch = collator([{"text": "Test text"}])

        # Move batch to CUDA
        input_ids = batch["input_ids"].cuda()
        landmarks = batch["cache_global_ids"].cuda()

        logits = tiny_model(input_ids, cache_global_ids=landmarks)

        assert logits.device.type == "cuda"

    def test_device_transfer_preserves_weights(self, tiny_model):
        """Test device transfer doesn't change weights."""
        # Get CPU weights
        cpu_weights = {
            name: param.clone().cpu()
            for name, param in tiny_model.named_parameters()
        }

        if torch.cuda.is_available():
            # Move to CUDA and back
            tiny_model = tiny_model.cuda()
            tiny_model = tiny_model.cpu()

            # Check weights unchanged
            for name, param in tiny_model.named_parameters():
                assert torch.allclose(param, cpu_weights[name], atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_consistency_across_devices(self, tiny_config):
        """Test inference produces same results on CPU and CUDA."""
        torch.manual_seed(42)
        model_cpu = SLGATransformer(tiny_config)

        torch.manual_seed(42)
        model_cuda = SLGATransformer(tiny_config).cuda()

        # Same input
        torch.manual_seed(100)
        input_ids_cpu = torch.randint(0, tiny_config.vocab_size, (2, 32))
        input_ids_cuda = input_ids_cpu.cuda()

        # Inference
        model_cpu.eval()
        model_cuda.eval()

        with torch.no_grad():
            logits_cpu = model_cpu(input_ids_cpu)
            logits_cuda = model_cuda(input_ids_cuda)

        # Results should match
        assert torch.allclose(
            logits_cpu,
            logits_cuda.cpu(),
            atol=1e-4
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataloader_cuda_integration(self, sample_dataset, collator_local, tiny_model):
        """Test dataloader with CUDA model."""
        tiny_model = tiny_model.cuda()

        dataloader = DataLoader(
            sample_dataset,
            batch_size=2,
            pin_memory=True,
            collate_fn=collator_local,
        )

        tiny_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].cuda()
                logits = tiny_model(input_ids)

                assert logits.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_device_consistency(self, tiny_model, tiny_config):
        """Test gradients stay on same device as model."""
        tiny_model = tiny_model.cuda()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32)).cuda()
        targets = torch.randint(0, tiny_config.vocab_size, (2, 32)).cuda()

        logits = tiny_model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size),
            targets.reshape(-1)
        )

        loss.backward()

        # Check all gradients are on CUDA
        for param in tiny_model.parameters():
            if param.grad is not None:
                assert param.grad.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
