"""
Tests for refactored data module.

Validates loaders, processors, tokenizers, collators, and factories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from datasets import Dataset
from src.legacy.data import (
    DataLoaderFactory,
    TokenizerFactory,
    CollatorFactory,
    ProcessorFactory,
    TextDatasetLoader,
    WikipediaDatasetLoader,
    TextProcessor,
    DataCleaner,
    CleanedDataset,
    TokenizerWrapper,
    CollatorLocal,
    CollatorLocalGlobal,
    CollatorWithTFIDF,
    get_tokenizer,
)


# ============================================================================
# Test Factories
# ============================================================================

def test_data_loader_factory():
    """Test DataLoaderFactory."""
    # Text loader
    loader = DataLoaderFactory.create_loader("text", dataset_name="test")
    assert isinstance(loader, TextDatasetLoader)

    # Wikipedia loader
    wiki_loader = DataLoaderFactory.create_loader("wikipedia", language="en")
    assert isinstance(wiki_loader, WikipediaDatasetLoader)

    # Unknown loader
    with pytest.raises(ValueError):
        DataLoaderFactory.create_loader("unknown")


def test_tokenizer_factory():
    """Test TokenizerFactory."""
    wrapper = TokenizerFactory.create_tokenizer("gpt2")
    assert isinstance(wrapper, TokenizerWrapper)
    assert wrapper.tokenizer is not None


def test_collator_factory():
    """Test CollatorFactory."""
    tokenizer = get_tokenizer("gpt2")

    # Local collator
    collator = CollatorFactory.create_collator("local", tokenizer, max_length=128)
    assert isinstance(collator, CollatorLocal)

    # Local-global collator
    collator_global = CollatorFactory.create_collator(
        "local_global", tokenizer, max_length=128, strategy="regular"
    )
    assert isinstance(collator_global, CollatorLocalGlobal)

    # TF-IDF collator
    collator_tfidf = CollatorFactory.create_collator("tfidf", tokenizer, max_length=128)
    assert isinstance(collator_tfidf, CollatorWithTFIDF)


def test_processor_factory():
    """Test ProcessorFactory."""
    # Text processor
    processor = ProcessorFactory.create_processor("text")
    assert isinstance(processor, TextProcessor)

    # Cleaner
    cleaner = ProcessorFactory.create_processor("cleaner")
    assert isinstance(cleaner, DataCleaner)


# ============================================================================
# Test Text Processor
# ============================================================================

def test_text_processor_clean():
    """Test TextProcessor cleaning."""
    processor = TextProcessor()

    # Test whitespace normalization
    text = "Hello    world   test"
    cleaned = processor._clean_text(text)
    assert cleaned == "Hello world test"

    # Test control character removal
    text = "Hello\x00world\x01test"
    cleaned = processor._clean_text(text)
    assert cleaned == "Helloworldtest"

    # Test newline normalization
    text = "Line1\n\n\n\nLine2"
    cleaned = processor._clean_text(text)
    assert cleaned == "Line1\n\nLine2"


def test_text_processor_dataset():
    """Test TextProcessor on dataset."""
    processor = TextProcessor(text_key="text")

    # Create mock dataset
    data = {"text": ["Hello   world", "Test   text"]}
    dataset = Dataset.from_dict(data)

    # Process dataset
    processed = processor.process_dataset(dataset, batched=True, num_proc=None)
    assert len(processed) == 2
    assert processed[0]["text"] == "Hello world"


# ============================================================================
# Test Data Cleaner
# ============================================================================

def test_data_cleaner_unicode():
    """Test DataCleaner Unicode handling."""
    cleaner = DataCleaner()

    # Unicode replacement characters
    text = "Hello \ufffd world � test"
    cleaned = cleaner.clean_text(text)
    assert "\ufffd" not in cleaned
    assert "�" not in cleaned


def test_data_cleaner_control_chars():
    """Test DataCleaner control character removal."""
    cleaner = DataCleaner()

    # Control characters
    text = "Hello\x00\x01\x02world"
    cleaned = cleaner.clean_text(text)
    assert "\x00" not in cleaned
    assert cleaned == "Helloworld"


def test_data_cleaner_spaces():
    """Test DataCleaner space normalization."""
    cleaner = DataCleaner()

    # Multiple spaces
    text = "Hello    world     test"
    cleaned = cleaner.clean_text(text)
    assert cleaned == "Hello  world  test"


def test_cleaned_dataset_wrapper():
    """Test CleanedDataset wrapper."""
    # Create mock dataset
    data = {"text": ["Hello � world", "Test\x00text"]}
    dataset = Dataset.from_dict(data)

    # Wrap with cleaner
    cleaned = CleanedDataset(dataset, text_key="text")
    assert len(cleaned) == 2

    # Check cleaned text
    example = cleaned[0]
    assert "�" not in example["text"]


# ============================================================================
# Test Tokenizer Wrapper
# ============================================================================

def test_tokenizer_wrapper():
    """Test TokenizerWrapper."""
    wrapper = TokenizerWrapper("gpt2")

    # Check tokenizer loaded
    assert wrapper.tokenizer is not None
    assert wrapper.tokenizer.pad_token is not None

    # Test special tokens
    special = wrapper.get_special_tokens()
    assert "eos" in special
    assert "pad" in special

    # Test vocab size
    vocab_size = wrapper.get_vocab_size()
    assert vocab_size > 0

    # Test encode/decode
    text = "Hello world"
    token_ids = wrapper.encode(text)
    decoded = wrapper.decode(token_ids)
    assert "Hello" in decoded


# ============================================================================
# Test Collators
# ============================================================================

def test_collator_local():
    """Test CollatorLocal."""
    tokenizer = get_tokenizer("gpt2")
    collator = CollatorLocal(tokenizer, max_length=128)

    # Create examples
    examples = [
        {"text": "This is a test sentence. " * 10},
        {"text": "Another example text. " * 10},
    ]

    # Collate
    batch = collator(examples)

    # Check shapes
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)

    # Check labels masked
    assert -100 in batch["labels"]


def test_collator_local_global():
    """Test CollatorLocalGlobal."""
    tokenizer = get_tokenizer("gpt2")
    collator = CollatorLocalGlobal(
        tokenizer, max_length=128, max_global=8, strategy="regular"
    )

    # Create examples
    examples = [
        {"text": "This is a test sentence. " * 10},
        {"text": "Another example text. " * 10},
    ]

    # Collate
    batch = collator(examples)

    # Check shapes
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    assert batch["cache_global_ids"].shape == (2, 8)

    # Check landmarks are positions
    assert batch["cache_global_ids"].min() >= 0
    assert batch["cache_global_ids"].max() < 128


def test_collator_strategies():
    """Test different landmark strategies."""
    tokenizer = get_tokenizer("gpt2")

    examples = [{"text": "Test sentence. " * 20}]

    # Regular strategy
    collator_regular = CollatorLocalGlobal(
        tokenizer, max_length=128, max_global=8, strategy="regular"
    )
    batch_regular = collator_regular(examples)
    assert batch_regular["cache_global_ids"].shape == (1, 8)

    # Random strategy
    collator_random = CollatorLocalGlobal(
        tokenizer, max_length=128, max_global=8, strategy="random"
    )
    batch_random = collator_random(examples)
    assert batch_random["cache_global_ids"].shape == (1, 8)

    # Paragraph strategy
    examples_para = [{"text": "Para1.\n\nPara2.\n\nPara3." * 10}]
    collator_para = CollatorLocalGlobal(
        tokenizer, max_length=128, max_global=8, strategy="paragraph"
    )
    batch_para = collator_para(examples_para)
    assert batch_para["cache_global_ids"].shape == (1, 8)


# ============================================================================
# Test Backward Compatibility
# ============================================================================

def test_backward_compatibility():
    """Test backward compatible interface."""
    # get_tokenizer
    tokenizer = get_tokenizer("gpt2")
    assert tokenizer is not None

    # Collators
    collator = CollatorLocal(tokenizer, max_length=128)
    assert collator is not None

    collator_global = CollatorLocalGlobal(tokenizer, max_length=128)
    assert collator_global is not None


# ============================================================================
# Integration Test
# ============================================================================

def test_full_pipeline():
    """Test complete data pipeline."""
    # 1. Create mock dataset
    data = {"text": ["Test text " * 20, "Example text " * 20]}
    dataset = Dataset.from_dict(data)

    # 2. Process with cleaner
    cleaned = CleanedDataset(dataset, text_key="text")

    # 3. Create tokenizer
    wrapper = TokenizerFactory.create_tokenizer("gpt2")
    tokenizer = wrapper.tokenizer

    # 4. Create collator
    collator = CollatorFactory.create_collator(
        "local_global", tokenizer, max_length=128, max_global=8, strategy="regular"
    )

    # 5. Create batch
    examples = [cleaned[0], cleaned[1]]
    batch = collator(examples)

    # 6. Verify batch
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    assert batch["cache_global_ids"].shape == (2, 8)

    print("Full pipeline test passed!")


if __name__ == "__main__":
    # Run all tests
    test_data_loader_factory()
    test_tokenizer_factory()
    test_collator_factory()
    test_processor_factory()
    test_text_processor_clean()
    test_data_cleaner_unicode()
    test_data_cleaner_control_chars()
    test_data_cleaner_spaces()
    test_cleaned_dataset_wrapper()
    test_tokenizer_wrapper()
    test_collator_local()
    test_collator_local_global()
    test_collator_strategies()
    test_backward_compatibility()
    test_full_pipeline()

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
