"""
Text generation engine for SLGA models.
Implements the main generation loop with sampling, penalties, and stopping criteria.
"""

import torch
from typing import Optional
from transformers import AutoTokenizer

from .config import GenerationConfig
from .sampling import sample_next_token
from .penalties import apply_repetition_penalty, apply_no_repeat_ngram
from .stopping import create_default_stopping_criteria


class TextGenerator:
    """Text generator for SLGA models.

    Handles the complete generation pipeline including:
    - Input preparation and encoding
    - Generation loop with sampling
    - Repetition penalties and n-gram blocking
    - Stopping criteria
    - Output decoding
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        """Initialize text generator.

        Args:
            model: SLGA model to use for generation
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Set model to eval mode
        self.model.eval()

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses defaults if None)
            verbose: Whether to print generation progress

        Returns:
            Generated text (includes prompt)

        Raises:
            ValueError: If prompt or config is invalid
            RuntimeError: If generation fails
        """
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate config
        config.validate()

        # Set EOS token from tokenizer if not specified
        if config.eos_token_id is None:
            config.eos_token_id = self.tokenizer.eos_token_id

        # Prepare input
        input_ids = self._prepare_input(prompt)
        initial_length = input_ids.size(1)

        if verbose:
            print(f"Prompt length: {initial_length} tokens")
            print(f"Generating {config.max_new_tokens} new tokens...")
            print()

        # Run generation loop
        with torch.no_grad():
            output_ids = self._generation_loop(
                input_ids,
                config,
                initial_length,
                verbose,
            )

        # Decode output
        generated_text = self._postprocess(output_ids)

        return generated_text

    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """Encode prompt to input IDs.

        Args:
            prompt: Text prompt

        Returns:
            Input token IDs [1, seq_len]

        Raises:
            RuntimeError: If encoding fails
        """
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.device)
            return input_ids
        except Exception as e:
            raise RuntimeError(f"Failed to encode prompt: {e}") from e

    def _generation_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        initial_length: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Main generation loop.

        Args:
            input_ids: Initial token IDs [1, seq_len]
            config: Generation configuration
            initial_length: Length of initial prompt
            verbose: Whether to print progress

        Returns:
            Generated token IDs [1, total_len]

        Raises:
            RuntimeError: If generation fails
        """
        # Create stopping criteria
        stopping_criteria = create_default_stopping_criteria(
            max_new_tokens=config.max_new_tokens,
            initial_length=initial_length,
            eos_token_id=config.eos_token_id,
            stop_on_eos=config.stop_on_eos,
        )

        # Generation loop
        generated_tokens = input_ids.clone()

        try:
            while not stopping_criteria.should_stop(generated_tokens):
                # Get logits from model
                outputs = self.model(generated_tokens)

                # Get logits for next token (last position)
                if hasattr(outputs, 'logits'):
                    next_token_logits = outputs.logits[:, -1, :]
                else:
                    # Assume outputs are logits directly
                    next_token_logits = outputs[:, -1, :]

                # Apply penalties
                if config.has_repetition_penalty():
                    next_token_logits = apply_repetition_penalty(
                        next_token_logits,
                        generated_tokens,
                        config.repetition_penalty,
                    )

                if config.has_no_repeat_ngram():
                    next_token_logits = apply_no_repeat_ngram(
                        next_token_logits,
                        generated_tokens,
                        config.no_repeat_ngram_size,
                    )

                # Sample next token
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=config.temperature,
                    top_k=config.top_k if config.has_top_k() else None,
                    top_p=config.top_p if config.has_top_p() else None,
                )

                # Append to sequence
                generated_tokens = torch.cat(
                    [generated_tokens, next_token.unsqueeze(-1)],
                    dim=-1,
                )

                # Progress indicator
                if verbose:
                    num_new = generated_tokens.size(1) - initial_length
                    if num_new % 10 == 0:
                        print(f"Generated {num_new}/{config.max_new_tokens} tokens...")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"GPU out of memory during generation. "
                    f"Try reducing max_new_tokens or using CPU."
                ) from e
            raise RuntimeError(f"Model generation failed: {e}") from e

        except ValueError as e:
            raise ValueError(f"Invalid generation parameters: {e}") from e

        if verbose:
            print(f"\nGeneration stopped: {stopping_criteria.get_reason()}")

        return generated_tokens

    def _postprocess(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Generated token IDs [1, seq_len]

        Returns:
            Decoded text

        Raises:
            RuntimeError: If decoding fails
        """
        try:
            text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to decode output: {e}") from e

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        return self.tokenizer.vocab_size

    def get_generation_info(self, config: GenerationConfig) -> dict:
        """Get human-readable generation configuration.

        Args:
            config: Generation configuration

        Returns:
            Dictionary with configuration details
        """
        info = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "sampling_strategy": "greedy" if config.is_greedy() else "stochastic",
            "top_k": config.top_k if config.has_top_k() else "disabled",
            "top_p": config.top_p if config.has_top_p() else "disabled",
            "repetition_penalty": (
                config.repetition_penalty if config.has_repetition_penalty()
                else "disabled"
            ),
            "no_repeat_ngram_size": (
                config.no_repeat_ngram_size if config.has_no_repeat_ngram()
                else "disabled"
            ),
            "stop_on_eos": config.stop_on_eos,
            "eos_token_id": config.eos_token_id,
        }

        return info
