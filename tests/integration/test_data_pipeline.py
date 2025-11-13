"""Integration tests for data pipeline (50 tests).

Tests the complete data flow:
Dataset loading → Text processing → Tokenization → Collation → Batching
"""

import pytest
import torch
import gc
import os
from datasets import Dataset, load_dataset
from typing import List, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.legacy.data import (
    get_tokenizer,
    CollatorLocal,
    CollatorLocalGlobal,
    CollatorWithTFIDF,
    DataLoaderFactory,
)


class TestDataPipelineBasics:
    """Basic data pipeline integration tests (10 tests)."""

    def test_full_pipeline_local_collator(self, sample_texts, tokenizer):
        """Test complete pipeline with local-only collator."""
        # Create dataset
        dataset = Dataset.from_dict({"text": sample_texts})

        # Create collator
        collator = CollatorLocal(tokenizer, max_length=128)

        # Process batch
        batch = collator(list(dataset))

        # Verify output
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == len(sample_texts)
        assert batch["input_ids"].shape[1] == 128
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

        # Verify no padding tokens in labels (should be -100)
        pad_positions = (batch["input_ids"] == tokenizer.pad_token_id)
        assert (batch["labels"][pad_positions] == -100).all()

    def test_full_pipeline_global_collator(self, sample_texts, tokenizer):
        """Test complete pipeline with global collator."""
        dataset = Dataset.from_dict({"text": sample_texts})

        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            global_every=32,
            max_global=8,
            strategy="regular"
        )

        batch = collator(list(dataset))

        assert "input_ids" in batch
        assert "labels" in batch
        assert "cache_global_ids" in batch
        assert batch["cache_global_ids"].shape == (len(sample_texts), 8)

        # Verify landmarks are positions, not tokens
        assert batch["cache_global_ids"].max() < 128
        assert batch["cache_global_ids"].min() >= 0

    def test_pipeline_with_empty_text(self, tokenizer):
        """Test pipeline handles empty text gracefully."""
        texts = [{"text": ""}]
        collator = CollatorLocal(tokenizer, max_length=128)

        batch = collator(texts)

        assert batch["input_ids"].shape == (1, 128)
        # Should be all padding or special tokens
        assert (batch["input_ids"][0] == tokenizer.pad_token_id).sum() > 0

    def test_pipeline_with_very_long_text(self, tokenizer):
        """Test pipeline truncates very long text correctly."""
        long_text = " ".join(["word"] * 1000)
        texts = [{"text": long_text}]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        # Should be exactly max_length
        assert batch["input_ids"].shape[1] == 128
        # Should not contain padding (all real tokens)
        assert (batch["input_ids"][0] != tokenizer.pad_token_id).sum() == 128

    def test_pipeline_batch_consistency(self, sample_texts, tokenizer):
        """Test batch processing is consistent with individual processing."""
        collator = CollatorLocal(tokenizer, max_length=128)

        # Process as batch
        batch_all = collator([{"text": t} for t in sample_texts])

        # Process individually
        batches_individual = [
            collator([{"text": t}]) for t in sample_texts
        ]

        # Compare (shapes should match)
        for i, batch_single in enumerate(batches_individual):
            assert torch.equal(
                batch_all["input_ids"][i],
                batch_single["input_ids"][0]
            )

    def test_pipeline_with_special_characters(self, tokenizer):
        """Test pipeline handles special characters correctly."""
        texts = [
            {"text": "Hello! How are you? I'm fine."},
            {"text": "Test @#$%^&*() symbols."},
            {"text": "Unicode: 你好, مرحبا, שלום"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (3, 128)
        assert torch.isfinite(batch["input_ids"].float()).all()

    def test_pipeline_mixed_lengths(self, tokenizer):
        """Test pipeline handles mixed text lengths correctly."""
        texts = [
            {"text": "Short."},
            {"text": "Medium length sentence here."},
            {"text": " ".join(["Long"] * 50)},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        # All should be padded to same length
        assert batch["input_ids"].shape == (3, 128)

        # Verify padding is applied correctly
        pad_counts = (batch["input_ids"] == tokenizer.pad_token_id).sum(dim=1)
        # Shorter texts should have more padding
        assert pad_counts[0] > pad_counts[2]

    def test_pipeline_labels_shifted_correctly(self, tokenizer):
        """Test labels are shifted by 1 position for language modeling."""
        text = "The quick brown fox"
        collator = CollatorLocal(tokenizer, max_length=128)

        batch = collator([{"text": text}])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]

        # Find non-padding tokens
        non_pad_mask = (input_ids != tokenizer.pad_token_id)
        non_pad_positions = torch.where(non_pad_mask)[0]

        if len(non_pad_positions) > 1:
            # Check shift: labels[i] should equal input_ids[i+1]
            for i in range(len(non_pad_positions) - 1):
                pos = non_pad_positions[i].item()
                next_pos = non_pad_positions[i + 1].item()
                if next_pos == pos + 1:
                    # Labels should contain the next token
                    assert labels[pos] == input_ids[next_pos]

    def test_pipeline_deterministic(self, sample_texts, tokenizer):
        """Test pipeline produces deterministic results."""
        collator = CollatorLocal(tokenizer, max_length=128)

        batch1 = collator([{"text": t} for t in sample_texts])
        batch2 = collator([{"text": t} for t in sample_texts])

        assert torch.equal(batch1["input_ids"], batch2["input_ids"])
        assert torch.equal(batch1["labels"], batch2["labels"])

    def test_pipeline_different_max_lengths(self, tokenizer):
        """Test pipeline works with different max_length settings."""
        text = " ".join(["word"] * 100)

        for max_len in [32, 64, 128, 256]:
            collator = CollatorLocal(tokenizer, max_length=max_len)
            batch = collator([{"text": text}])

            assert batch["input_ids"].shape[1] == max_len


class TestGlobalLandmarkStrategies:
    """Test different landmark selection strategies (10 tests)."""

    def test_regular_landmark_strategy(self, sample_texts, tokenizer):
        """Test regular spacing landmark strategy."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch = collator([{"text": t} for t in sample_texts])
        landmarks = batch["cache_global_ids"]

        # Should have exactly max_global landmarks
        assert landmarks.shape[1] == 8

        # Should be evenly spaced (approximately)
        for i in range(len(sample_texts)):
            lm = landmarks[i].tolist()
            # Check monotonically increasing
            assert all(lm[j] <= lm[j+1] for j in range(len(lm)-1))

    def test_random_landmark_strategy(self, sample_texts, tokenizer):
        """Test random landmark selection strategy."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="random"
        )

        batch1 = collator([{"text": t} for t in sample_texts])
        batch2 = collator([{"text": t} for t in sample_texts])

        # Random strategy should produce different results
        # (with very high probability)
        assert not torch.equal(
            batch1["cache_global_ids"],
            batch2["cache_global_ids"]
        )

    def test_paragraph_landmark_strategy(self, tokenizer):
        """Test paragraph-based landmark strategy."""
        texts = [
            {"text": "Para 1.\n\nPara 2.\n\nPara 3."},
            {"text": "First section.\n\nSecond section."},
        ]

        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="paragraph"
        )

        batch = collator(texts)
        landmarks = batch["cache_global_ids"]

        # Should have detected paragraph boundaries
        assert landmarks.shape == (2, 8)
        # First landmark should be at position 0
        assert (landmarks[:, 0] == 0).all()

    def test_landmark_count_validation(self, sample_texts, tokenizer):
        """Test exactly max_global landmarks are selected."""
        for num_landmarks in [4, 8, 16, 32]:
            collator = CollatorLocalGlobal(
                tokenizer,
                max_length=128,
                max_global=num_landmarks,
                strategy="regular"
            )

            batch = collator([{"text": t} for t in sample_texts])
            assert batch["cache_global_ids"].shape[1] == num_landmarks

    def test_landmarks_within_sequence_bounds(self, sample_texts, tokenizer):
        """Test all landmark positions are within valid range."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch = collator([{"text": t} for t in sample_texts])
        landmarks = batch["cache_global_ids"]

        # All landmarks must be within [0, max_length)
        assert (landmarks >= 0).all()
        assert (landmarks < 128).all()

    def test_landmarks_are_positions_not_tokens(self, sample_texts, tokenizer):
        """Test landmarks are position indices, not token IDs."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch = collator([{"text": t} for t in sample_texts])
        landmarks = batch["cache_global_ids"]
        input_ids = batch["input_ids"]

        # Landmarks should be small integers (positions)
        # Token IDs can be much larger (up to vocab_size)
        assert landmarks.max() < 128  # Max position

        # Verify by gathering - should work without errors
        batch_size, seq_len = input_ids.shape
        num_landmarks = landmarks.shape[1]

        landmarks_safe = torch.clamp(landmarks, 0, seq_len - 1)
        landmarks_exp = landmarks_safe.unsqueeze(-1).expand(batch_size, num_landmarks, 1)
        gathered = torch.gather(input_ids.unsqueeze(-1), dim=1, index=landmarks_exp)

        assert gathered.shape == (batch_size, num_landmarks, 1)

    def test_landmark_strategy_consistency(self, sample_texts, tokenizer):
        """Test regular strategy produces consistent results."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch1 = collator([{"text": t} for t in sample_texts])
        batch2 = collator([{"text": t} for t in sample_texts])

        # Regular strategy should be deterministic
        assert torch.equal(
            batch1["cache_global_ids"],
            batch2["cache_global_ids"]
        )

    def test_landmark_distribution(self, tokenizer):
        """Test landmarks are well-distributed across sequence."""
        text = " ".join(["word"] * 50)
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch = collator([{"text": text}])
        landmarks = batch["cache_global_ids"][0].tolist()

        # Calculate spacing between landmarks
        spacings = [landmarks[i+1] - landmarks[i] for i in range(len(landmarks)-1)]

        # Should have similar spacing (allowing some variance)
        avg_spacing = sum(spacings) / len(spacings)
        assert all(abs(s - avg_spacing) < avg_spacing * 0.5 for s in spacings)

    def test_landmarks_with_short_sequences(self, tokenizer):
        """Test landmark selection with very short sequences."""
        texts = [{"text": "Hi"}]

        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        batch = collator(texts)
        landmarks = batch["cache_global_ids"]

        # Should still produce max_global landmarks (with repetition)
        assert landmarks.shape == (1, 8)
        assert (landmarks >= 0).all()

    def test_landmarks_preserved_across_batches(self, sample_texts, tokenizer):
        """Test same text produces same landmarks across batches."""
        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        # Process same text in different batches
        batch1 = collator([{"text": sample_texts[0]}])
        batch2 = collator([{"text": sample_texts[0]}, {"text": sample_texts[1]}])

        # First example should have same landmarks
        assert torch.equal(
            batch1["cache_global_ids"][0],
            batch2["cache_global_ids"][0]
        )


class TestMemoryAndPerformance:
    """Memory efficiency and performance tests (10 tests)."""

    @pytest.mark.slow
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_memory_efficiency_many_batches(self, sample_texts, tokenizer):
        """Test memory doesn't grow unbounded during batch processing."""
        process = psutil.Process(os.getpid())
        collator = CollatorLocal(tokenizer, max_length=128)

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process many batches
        for _ in range(100):
            batch = collator([{"text": t} for t in sample_texts])
            del batch
            if _ % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal (<50MB)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB"

    def test_batch_processing_time(self, sample_texts, tokenizer):
        """Test batch processing is reasonably fast."""
        import time

        collator = CollatorLocal(tokenizer, max_length=128)

        start = time.time()
        for _ in range(100):
            batch = collator([{"text": t} for t in sample_texts])
        duration = time.time() - start

        # Should process 100 batches in under 5 seconds
        assert duration < 5.0, f"Took {duration:.2f}s for 100 batches"

    def test_large_batch_handling(self, tokenizer):
        """Test handling of large batch sizes."""
        texts = [{"text": f"Sample text {i}"} for i in range(100)]
        collator = CollatorLocal(tokenizer, max_length=128)

        batch = collator(texts)

        assert batch["input_ids"].shape == (100, 128)
        assert torch.isfinite(batch["input_ids"].float()).all()

    def test_long_sequence_handling(self, tokenizer):
        """Test handling of very long sequences."""
        long_text = " ".join(["word"] * 5000)
        collator = CollatorLocal(tokenizer, max_length=512)

        batch = collator([{"text": long_text}])

        # Should truncate to max_length
        assert batch["input_ids"].shape[1] == 512

    def test_cache_consistency(self, sample_texts, tokenizer):
        """Test tokenizer caching produces consistent results."""
        collator = CollatorLocal(tokenizer, max_length=128)

        # Process same batch multiple times
        results = []
        for _ in range(5):
            batch = collator([{"text": t} for t in sample_texts])
            results.append(batch["input_ids"].clone())

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i])

    def test_parallel_collator_safety(self, sample_texts, tokenizer):
        """Test collator can be used safely in parallel (no state corruption)."""
        collator = CollatorLocal(tokenizer, max_length=128)

        # Simulate parallel calls with different data
        batch1 = collator([{"text": sample_texts[0]}])
        batch2 = collator([{"text": sample_texts[1]}])
        batch3 = collator([{"text": sample_texts[0]}])

        # batch1 and batch3 should be identical (same input)
        assert torch.equal(batch1["input_ids"], batch3["input_ids"])

        # batch1 and batch2 should be different
        assert not torch.equal(batch1["input_ids"], batch2["input_ids"])

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_memory_usage_global_collator(self, sample_texts, tokenizer):
        """Test global collator doesn't use excessive memory."""
        import tracemalloc

        tracemalloc.start()

        collator = CollatorLocalGlobal(
            tokenizer,
            max_length=128,
            max_global=8,
            strategy="regular"
        )

        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(50):
            batch = collator([{"text": t} for t in sample_texts])
            del batch

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate memory difference
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in stats)

        # Should not leak significant memory
        assert abs(total_diff) < 10 * 1024 * 1024  # <10MB

    def test_collator_with_streaming(self, tokenizer):
        """Test collator works with streaming dataset."""
        collator = CollatorLocal(tokenizer, max_length=128)

        # Simulate streaming by yielding examples one at a time
        def stream_examples():
            for i in range(10):
                yield {"text": f"Streaming example {i}"}

        batches = []
        for example in stream_examples():
            batch = collator([example])
            batches.append(batch)

        assert len(batches) == 10
        assert all(b["input_ids"].shape == (1, 128) for b in batches)

    def test_collator_garbage_collection(self, sample_texts, tokenizer):
        """Test collator releases memory properly."""
        import sys

        collator = CollatorLocal(tokenizer, max_length=128)

        # Create batch and get reference count
        batch = collator([{"text": t} for t in sample_texts])
        initial_refcount = sys.getrefcount(batch)

        # Delete batch
        del batch
        gc.collect()

        # Create new batch - should not accumulate references
        batch2 = collator([{"text": t} for t in sample_texts])
        new_refcount = sys.getrefcount(batch2)

        # Reference counts should be similar (no leaks)
        assert abs(new_refcount - initial_refcount) <= 1

    def test_extreme_batch_size(self, tokenizer):
        """Test handling of extreme batch sizes."""
        # Very large batch
        texts = [{"text": f"Text {i}"} for i in range(500)]
        collator = CollatorLocal(tokenizer, max_length=64)

        batch = collator(texts)

        assert batch["input_ids"].shape == (500, 64)
        assert torch.isfinite(batch["input_ids"].float()).all()


class TestEdgeCases:
    """Edge case handling tests (10 tests)."""

    def test_unicode_text_handling(self, tokenizer):
        """Test pipeline handles various Unicode characters."""
        texts = [
            {"text": "Hello 世界"},
            {"text": "Привет мир"},
            {"text": "مرحبا بالعالم"},
            {"text": "שלום עולם"},
            {"text": "🌍🌎🌏"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (5, 128)
        assert torch.isfinite(batch["input_ids"].float()).all()

    def test_very_short_text(self, tokenizer):
        """Test handling of extremely short text."""
        texts = [
            {"text": "a"},
            {"text": "."},
            {"text": ""},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (3, 128)
        # Should be mostly padding
        assert (batch["input_ids"] == tokenizer.pad_token_id).float().mean() > 0.9

    def test_whitespace_only_text(self, tokenizer):
        """Test handling of whitespace-only text."""
        texts = [
            {"text": "   "},
            {"text": "\n\n\n"},
            {"text": "\t\t"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (3, 128)

    def test_repeated_text(self, tokenizer):
        """Test handling of highly repetitive text."""
        text = "repeat " * 1000
        collator = CollatorLocal(tokenizer, max_length=128)

        batch = collator([{"text": text}])

        assert batch["input_ids"].shape == (1, 128)

    def test_special_token_in_text(self, tokenizer):
        """Test text containing special tokens is handled correctly."""
        # Text that might look like special tokens
        texts = [
            {"text": "<|endoftext|>"},
            {"text": "[PAD] [SEP] [CLS]"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (2, 128)

    def test_numbers_and_symbols(self, tokenizer):
        """Test handling of numbers and mathematical symbols."""
        texts = [
            {"text": "1234567890"},
            {"text": "x + y = z"},
            {"text": "f(x) = ∫₀¹ x² dx"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (3, 128)

    def test_mixed_language_text(self, tokenizer):
        """Test handling of mixed-language text."""
        text = "Hello world 你好世界 مرحبا العالم"
        collator = CollatorLocal(tokenizer, max_length=128)

        batch = collator([{"text": text}])

        assert batch["input_ids"].shape == (1, 128)

    def test_code_snippets(self, tokenizer):
        """Test handling of code snippets."""
        texts = [
            {"text": "def hello():\n    print('world')"},
            {"text": "import torch\nimport numpy as np"},
            {"text": "x = [1, 2, 3]\ny = x[0]"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (3, 128)

    def test_single_token_text(self, tokenizer):
        """Test text that tokenizes to exactly one token."""
        # Find a single-token word
        text = "Hello"
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Use single-token text
        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator([{"text": text}])

        assert batch["input_ids"].shape == (1, 128)

    def test_newline_preservation(self, tokenizer):
        """Test newlines are preserved or handled correctly."""
        texts = [
            {"text": "Line1\nLine2\nLine3"},
            {"text": "Para1\n\nPara2"},
        ]

        collator = CollatorLocal(tokenizer, max_length=128)
        batch = collator(texts)

        assert batch["input_ids"].shape == (2, 128)


class TestDatasetLoading:
    """Dataset loading integration tests (10 tests)."""

    def test_load_from_dict(self):
        """Test loading dataset from dictionary."""
        data = {
            "text": ["Sample 1", "Sample 2", "Sample 3"]
        }
        dataset = Dataset.from_dict(data)

        assert len(dataset) == 3
        assert dataset[0]["text"] == "Sample 1"

    def test_dataloader_factory(self):
        """Test DataLoaderFactory creates loaders."""
        loader = DataLoaderFactory.create("text")
        assert loader is not None

    def test_load_from_list_of_dicts(self):
        """Test loading dataset from list of dictionaries."""
        data = [
            {"text": "Sample 1"},
            {"text": "Sample 2"},
        ]
        dataset = Dataset.from_list(data)

        assert len(dataset) == 2

    def test_dataset_iteration(self, sample_dataset):
        """Test iterating over dataset."""
        count = 0
        for example in sample_dataset:
            assert "text" in example
            count += 1

        assert count == len(sample_dataset)

    def test_dataset_slicing(self, sample_dataset):
        """Test dataset slicing operations."""
        subset = sample_dataset[:3]

        assert len(subset) == 3
        assert all("text" in ex for ex in subset)

    def test_dataset_filtering(self, sample_dataset):
        """Test dataset filtering."""
        filtered = sample_dataset.filter(
            lambda x: len(x["text"]) > 50
        )

        assert len(filtered) <= len(sample_dataset)
        assert all(len(ex["text"]) > 50 for ex in filtered)

    def test_dataset_mapping(self, sample_dataset, tokenizer):
        """Test dataset mapping with tokenization."""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        tokenized = sample_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        assert "input_ids" in tokenized.column_names
        assert len(tokenized) == len(sample_dataset)

    def test_dataset_train_test_split(self, sample_dataset):
        """Test splitting dataset into train/test."""
        split = sample_dataset.train_test_split(test_size=0.4)

        assert "train" in split
        assert "test" in split
        assert len(split["train"]) + len(split["test"]) == len(sample_dataset)

    def test_dataset_shuffle(self, sample_dataset):
        """Test dataset shuffling."""
        shuffled = sample_dataset.shuffle(seed=42)

        assert len(shuffled) == len(sample_dataset)
        # At least some examples should be in different positions
        different = sum(
            sample_dataset[i]["text"] != shuffled[i]["text"]
            for i in range(len(sample_dataset))
        )
        assert different > 0

    def test_dataset_select(self, sample_dataset):
        """Test selecting specific indices from dataset."""
        indices = [0, 2, 4]
        selected = sample_dataset.select(indices)

        assert len(selected) == 3
        for i, idx in enumerate(indices):
            assert selected[i]["text"] == sample_dataset[idx]["text"]

    def test_dataset_add_column(self, sample_dataset):
        """Test adding new column to dataset."""
        lengths = [len(ex["text"]) for ex in sample_dataset]
        dataset_with_length = sample_dataset.add_column("length", lengths)

        assert "length" in dataset_with_length.column_names
        assert all(
            dataset_with_length[i]["length"] == len(dataset_with_length[i]["text"])
            for i in range(len(dataset_with_length))
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
