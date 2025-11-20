"""
Test script to verify vocab_size fix for tokenizers with special tokens.
Addresses the issue where max token ID can equal vocab_size for some tokenizers.
"""

import sys
sys.path.insert(0, '/mnt/d/ai/SLGA2')

from transformers import AutoTokenizer


def test_qwen_tokenizer_vocab_size():
    """Test that Qwen2 tokenizer vocabulary size is correctly handled."""
    print("=" * 80)
    print("Testing Qwen2 Tokenizer Vocabulary Size")
    print("=" * 80)

    tokenizer_name = "Qwen/Qwen2-7B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        # Get different size metrics
        vocab_size = tokenizer.vocab_size
        len_tokenizer = len(tokenizer)
        max_token_id = max(tokenizer.get_vocab().values())

        print(f"\n📊 Tokenizer Metrics for {tokenizer_name}:")
        print(f"   • tokenizer.vocab_size: {vocab_size}")
        print(f"   • len(tokenizer):       {len_tokenizer}")
        print(f"   • max(token IDs):       {max_token_id}")
        print(f"   • pad_token_id:         {tokenizer.pad_token_id}")
        print(f"   • eos_token_id:         {tokenizer.eos_token_id}")
        print(f"   • bos_token_id:         {tokenizer.bos_token_id}")

        # Check if there's a mismatch
        if max_token_id >= vocab_size:
            print(f"\n⚠️  WARNING: max_token_id ({max_token_id}) >= vocab_size ({vocab_size})")
            print(f"   This causes the assertion error!")
            print(f"\n✅ SOLUTION: Use len(tokenizer) = {len_tokenizer} as model vocab_size")
            recommended_vocab_size = len_tokenizer
        else:
            print(f"\n✅ No mismatch detected")
            recommended_vocab_size = vocab_size

        # Test encoding
        test_text = "Hello world! This is a test."
        encoded = tokenizer.encode(test_text)
        max_encoded = max(encoded)

        print(f"\n🧪 Test Encoding:")
        print(f"   • Input: '{test_text}'")
        print(f"   • Encoded length: {len(encoded)}")
        print(f"   • Max token ID in encoding: {max_encoded}")

        if max_encoded >= vocab_size:
            print(f"   ⚠️  Token ID {max_encoded} >= vocab_size {vocab_size}")

        # Recommendations
        print(f"\n📋 Recommendations:")
        print(f"   1. Set model vocab_size = len(tokenizer) = {recommended_vocab_size}")
        print(f"   2. Ensure embedding layer uses vocab_size = {recommended_vocab_size}")
        print(f"   3. Update config to use recommended vocab_size")

        return True

    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}")
        print(f"\nℹ️  If this is a gated model, you may need to:")
        print(f"   1. Accept the license at https://huggingface.co/{tokenizer_name}")
        print(f"   2. Login with: huggingface-cli login")
        return False


def test_alternative_tokenizers():
    """Test other common tokenizers to show the pattern."""
    print("\n" + "=" * 80)
    print("Testing Other Tokenizers")
    print("=" * 80)

    tokenizers = [
        "gpt2",
        "EleutherAI/gpt-neo-125M",
    ]

    for name in tokenizers:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            vocab_size = tokenizer.vocab_size
            len_tok = len(tokenizer)

            print(f"\n{name}:")
            print(f"   vocab_size: {vocab_size}, len: {len_tok}, diff: {len_tok - vocab_size}")

            if len_tok != vocab_size:
                print(f"   ⚠️  Mismatch detected!")
            else:
                print(f"   ✅ Consistent")

        except Exception as e:
            print(f"\n{name}: ❌ {e}")


if __name__ == "__main__":
    success = test_qwen_tokenizer_vocab_size()
    test_alternative_tokenizers()

    print("\n" + "=" * 80)
    if success:
        print("✅ Vocabulary size analysis complete!")
    else:
        print("⚠️  Some tests could not be completed")
    print("=" * 80)
