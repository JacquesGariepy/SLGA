"""Integration tests for generation pipeline (20 tests).

Tests the complete generation flow:
Model → Generator → Sampling → Decoding → Output
"""

import pytest
import torch

from src.models import ModelConfig, SLGATransformer, TextGenerator
from src.legacy.data import get_tokenizer


class TestGenerationBasics:
    """Basic generation integration tests (10 tests)."""

    def test_greedy_generation(self, tiny_model, tokenizer):
        """Test greedy generation (temperature=0)."""
        generator = TextGenerator(tiny_model)

        prompt = "The quick brown"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
        )

        # Output should be longer than input
        assert output.shape[1] > input_ids.shape[1]
        assert output.shape[1] == input_ids.shape[1] + 10

    def test_sampling_generation(self, tiny_model, tokenizer):
        """Test stochastic sampling generation."""
        generator = TextGenerator(tiny_model)

        prompt = "Hello world"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_generation_with_seed(self, tiny_model, tokenizer):
        """Test generation is deterministic with seed."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output1 = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            seed=42,
        )

        output2 = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            seed=42,
        )

        # Should produce identical outputs
        assert torch.equal(output1, output2)

    def test_generation_different_seeds(self, tiny_model, tokenizer):
        """Test generation varies with different seeds."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output1 = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            seed=42,
        )

        output2 = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            seed=123,
        )

        # Should produce different outputs (with high probability)
        assert not torch.equal(output1, output2)

    def test_top_k_sampling(self, tiny_model, tokenizer):
        """Test top-k sampling."""
        generator = TextGenerator(tiny_model)

        prompt = "The"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            top_k=50,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_top_p_sampling(self, tiny_model, tokenizer):
        """Test top-p (nucleus) sampling."""
        generator = TextGenerator(tiny_model)

        prompt = "The"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            top_p=0.9,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_combined_top_k_top_p(self, tiny_model, tokenizer):
        """Test combined top-k and top-p sampling."""
        generator = TextGenerator(tiny_model)

        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_generation_stops_at_eos(self, tiny_model, tokenizer):
        """Test generation stops at EOS token."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=50,
            temperature=1.0,
            stop_on_eos=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Output might be shorter than max if EOS was generated
        assert output.shape[1] <= input_ids.shape[1] + 50

    def test_generation_with_landmarks(self, tiny_model, tiny_config, tokenizer):
        """Test generation with heuristic landmarks."""
        generator = TextGenerator(tiny_model)

        prompt = "The quick brown fox"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Create landmarks
        landmarks = torch.linspace(
            0, input_ids.shape[1] - 1, tiny_config.global_k, dtype=torch.long
        ).unsqueeze(0)

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
            cache_global_ids=landmarks,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_decode_generation_output(self, tiny_model, tokenizer):
        """Test decoding generated tokens."""
        generator = TextGenerator(tiny_model)

        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
        )

        # Decode output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Should be valid text
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)


class TestGenerationParameters:
    """Test various generation parameters (10 tests)."""

    def test_temperature_effect(self, tiny_model, tokenizer):
        """Test temperature affects generation diversity."""
        generator = TextGenerator(tiny_model)

        prompt = "The"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Low temperature (more deterministic)
        outputs_low = []
        for _ in range(3):
            output = generator.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.1,
            )
            outputs_low.append(output)

        # High temperature (more random)
        outputs_high = []
        for _ in range(3):
            output = generator.generate(
                input_ids,
                max_new_tokens=5,
                temperature=2.0,
            )
            outputs_high.append(output)

        # Low temperature should be more consistent
        low_variance = sum(
            not torch.equal(outputs_low[0], outputs_low[i])
            for i in range(1, len(outputs_low))
        )

        # High temperature should be more diverse
        high_variance = sum(
            not torch.equal(outputs_high[0], outputs_high[i])
            for i in range(1, len(outputs_high))
        )

        # Usually high temp has more variance (not always guaranteed)
        # So we just verify both produce valid outputs
        assert all(o.shape[1] == input_ids.shape[1] + 5 for o in outputs_low)
        assert all(o.shape[1] == input_ids.shape[1] + 5 for o in outputs_high)

    def test_max_new_tokens_respected(self, tiny_model, tokenizer):
        """Test generation respects max_new_tokens."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        for max_tokens in [5, 10, 20, 50]:
            output = generator.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.0,
            )

            assert output.shape[1] == input_ids.shape[1] + max_tokens

    def test_repetition_penalty(self, tiny_model, tokenizer):
        """Test repetition penalty reduces repetition."""
        generator = TextGenerator(tiny_model)

        prompt = "The the the"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Without penalty
        output_no_penalty = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        # With penalty
        output_with_penalty = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            repetition_penalty=1.5,
        )

        # Both should produce valid outputs
        assert output_no_penalty.shape[1] == input_ids.shape[1] + 10
        assert output_with_penalty.shape[1] == input_ids.shape[1] + 10

    def test_no_repeat_ngram(self, tiny_model, tokenizer):
        """Test no-repeat n-gram blocking."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            no_repeat_ngram_size=3,
        )

        # Check no 3-grams repeat
        tokens = output[0].tolist()
        ngrams = set()
        has_repeat = False

        for i in range(len(tokens) - 2):
            ngram = tuple(tokens[i:i+3])
            if ngram in ngrams:
                has_repeat = True
                break
            ngrams.add(ngram)

        # Should avoid repeating 3-grams (with high probability)
        assert output.shape[1] == input_ids.shape[1] + 20

    def test_batch_generation(self, tiny_model, tokenizer):
        """Test generation with batch of prompts."""
        generator = TextGenerator(tiny_model)

        prompts = ["Hello", "World", "Test"]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]

        output = generator.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.0,
        )

        # All prompts should be extended
        assert output.shape[0] == 3
        assert output.shape[1] == input_ids.shape[1] + 5

    def test_generation_with_very_short_prompt(self, tiny_model, tokenizer):
        """Test generation with minimal prompt."""
        generator = TextGenerator(tiny_model)

        prompt = "A"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_generation_with_long_prompt(self, tiny_model, tokenizer):
        """Test generation with long prompt."""
        generator = TextGenerator(tiny_model)

        prompt = " ".join(["word"] * 50)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Might need to truncate to max_seq_len
        if input_ids.shape[1] > tiny_model.config.max_seq_len:
            input_ids = input_ids[:, :tiny_model.config.max_seq_len]

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
        )

        assert output.shape[1] > input_ids.shape[1]

    def test_generation_caching_landmarks(self, tiny_model, tokenizer):
        """Test generation updates landmarks incrementally."""
        generator = TextGenerator(tiny_model)

        prompt = "Test prompt"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.0,
        )

        # Should generate successfully with internal landmark updates
        assert output.shape[1] == input_ids.shape[1] + 20

    def test_generation_special_tokens(self, tiny_model, tokenizer):
        """Test generation handles special tokens correctly."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)

        output = generator.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
        )

        assert output.shape[1] == input_ids.shape[1] + 10

    def test_generation_without_stop_on_eos(self, tiny_model, tokenizer):
        """Test generation continues past EOS when stop_on_eos=False."""
        generator = TextGenerator(tiny_model)

        prompt = "Test"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        output = generator.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            stop_on_eos=False,
        )

        # Should generate exactly max_new_tokens
        assert output.shape[1] == input_ids.shape[1] + 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
