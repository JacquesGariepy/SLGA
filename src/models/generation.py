"""Text generation with various sampling strategies.

Separates generation logic from model definition.
Breaks down the monolithic 171-line generate() method into focused sub-methods.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from src.models.slga_model import SLGATransformer


class TextGenerator:
    """Text generation with various sampling strategies.

    Separates generation logic from model definition.

    Args:
        model: SLGA Transformer model

    Example:
        >>> from src.models.slga_model import SLGATransformer
        >>> from src.models.config import ModelConfig
        >>> config = ModelConfig(vocab_size=50257, embed_dim=512)
        >>> model = SLGATransformer(config)
        >>> generator = TextGenerator(model)
        >>> input_ids = torch.randint(0, 50257, (1, 5))
        >>> output = generator.generate(input_ids, max_new_tokens=10)
    """

    def __init__(self, model: SLGATransformer) -> None:
        self.model = model
        self.config = model.config

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cache_global_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        stop_on_eos: bool = True,
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with deterministic and corrected EOS handling.

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >0.0 = stochastic)
            top_k: Top-K filtering threshold
            top_p: Nucleus (top-P) filtering threshold
            cache_global_ids: User-provided landmark indices (if learned=False)
            seed: Random seed for reproducibility
            stop_on_eos: Stop when all sequences generate EOS
            eos_token_id: EOS token ID (default: GPT-2 EOS)
            repetition_penalty: Penalty for repeating tokens (>1.0 penalizes repeats)
            no_repeat_ngram_size: Size of n-grams to avoid repeating

        Returns:
            Generated token IDs of shape (batch, seq_len + generated_tokens)
        """
        # Force evaluation mode
        self.model.eval()

        # Prepare generation state
        gen_state = self._prepare_generation(input_ids, cache_global_ids, seed, eos_token_id)

        # Generation loop
        for step in range(max_new_tokens):
            # Truncate if sequence exceeds max length
            gen_state.input_ids = self._truncate_if_needed(gen_state.input_ids)

            # Update landmarks if needed
            gen_state.cache_global_ids = self._update_landmarks(
                gen_state.input_ids, gen_state.cache_global_ids, gen_state.user_provided_landmarks
            )

            # Generate single token
            next_token = self._generate_step(
                gen_state,
                temperature,
                top_k,
                top_p,
                stop_on_eos,
                repetition_penalty,
                no_repeat_ngram_size,
            )

            # Append new token
            gen_state.input_ids = torch.cat([gen_state.input_ids, next_token], dim=1)

            # Check stopping criteria
            if self._check_stopping(gen_state, next_token, stop_on_eos):
                break

        # Post-process to clean up EOS tokens
        if stop_on_eos:
            gen_state.input_ids = self._postprocess_eos(gen_state.input_ids, eos_token_id)

        return gen_state.input_ids

    def _prepare_generation(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor],
        seed: Optional[int],
        eos_token_id: int,
    ) -> GenerationState:
        """Prepare generation state and set random seed."""
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Batch-aware EOS tracking
        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Track if user provided landmarks
        user_provided_landmarks = cache_global_ids is not None

        return GenerationState(
            input_ids=input_ids,
            cache_global_ids=cache_global_ids,
            finished=finished,
            user_provided_landmarks=user_provided_landmarks,
            eos_token_id=eos_token_id,
        )

    def _truncate_if_needed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Truncate sequence if it exceeds max length."""
        if input_ids.size(1) > self.config.max_seq_len:
            return input_ids[:, -self.config.max_seq_len :]
        return input_ids

    def _update_landmarks(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor],
        user_provided: bool,
    ) -> Optional[torch.Tensor]:
        """Recalculate heuristic landmarks for each step."""
        if not self.config.learned_landmarks and not user_provided:
            seq_len = input_ids.size(1)
            # Use linspace to ensure exactly global_k evenly spaced landmarks
            landmark_positions = torch.linspace(
                0, seq_len - 1, self.config.global_k, device=input_ids.device
            ).long()
            return landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
        return cache_global_ids

    def _generate_step(
        self,
        gen_state: GenerationState,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        stop_on_eos: bool,
        repetition_penalty: float,
        no_repeat_ngram_size: Optional[int],
    ) -> torch.Tensor:
        """Generate single token with sampling."""
        # Forward pass to get logits
        logits = self.model(
            gen_state.input_ids,
            cache_global_ids=gen_state.cache_global_ids,
        )  # (batch, seq_len, vocab_size)

        # Take logits for the last token
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Freeze finished sequences to only generate EOS
        if stop_on_eos and gen_state.finished.any():
            logits[gen_state.finished] = float("-inf")
            logits[gen_state.finished, gen_state.eos_token_id] = 1e4

        # Apply anti-repetition controls
        logits = self._apply_repetition_penalty(logits, gen_state.input_ids, repetition_penalty)
        logits = self._apply_no_repeat_ngram(logits, gen_state.input_ids, no_repeat_ngram_size)

        # Apply sampling strategy
        next_token = self._apply_sampling(logits, temperature, top_k, top_p)

        # Replace finished sequences with EOS
        if stop_on_eos and gen_state.finished.any():
            next_token[gen_state.finished] = gen_state.eos_token_id

        return next_token

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens."""
        if penalty is None or penalty == 1.0:
            return logits

        penalty = max(penalty, 1e-6)
        for b_idx in range(logits.size(0)):
            seen_tokens = input_ids[b_idx].unique()
            if seen_tokens.numel() == 0:
                continue

            token_logits = logits[b_idx, seen_tokens]
            penalized = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )
            logits[b_idx, seen_tokens] = penalized

        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        n_size: Optional[int],
    ) -> torch.Tensor:
        """Block n-grams that have already appeared."""
        if n_size is None or n_size <= 0:
            return logits

        for b_idx in range(logits.size(0)):
            prev_tokens = input_ids[b_idx]
            seq_len = prev_tokens.size(0)

            if seq_len < n_size:
                continue

            # Build n-gram dictionary
            ngram_dict: Dict[Tuple, Set[int]] = {}
            prev_tokens_list = prev_tokens.tolist()

            for idx in range(seq_len - n_size + 1):
                ngram = tuple(prev_tokens_list[idx : idx + n_size])
                prefix = ngram[:-1]
                next_token = ngram[-1]

                if prefix not in ngram_dict:
                    ngram_dict[prefix] = set()
                ngram_dict[prefix].add(next_token)

            # Get current prefix and banned tokens
            current_prefix = tuple(prev_tokens_list[-(n_size - 1) :]) if n_size > 1 else tuple()
            banned_tokens = ngram_dict.get(current_prefix, set())

            if banned_tokens:
                banned_indices = torch.tensor(
                    list(banned_tokens), device=logits.device, dtype=torch.long
                )
                logits[b_idx, banned_indices] = float("-inf")

        return logits

    def _apply_sampling(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Apply sampling strategy: temperature -> top-k -> top-p -> sample."""
        if temperature == 0.0:
            # Greedy sampling
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K filtering
        if top_k is not None and top_k > 0:
            logits = self._apply_top_k(logits, top_k)

        # Top-P (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            logits = self._apply_top_p(logits, top_p)

        # Sample with NaN protection
        probs = F.softmax(logits, dim=-1)

        # Fallback to uniform if all logits are -inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)

        # Clamp and renormalize probabilities
        probs = torch.clamp(probs, min=1e-10)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    def _apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering."""
        topk_vals, topk_idxs = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, topk_idxs, topk_vals)
        return logits_filtered

    def _apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens above cumulative probability threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False  # Keep at least the best token

        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

    def _check_stopping(
        self,
        gen_state: GenerationState,
        next_token: torch.Tensor,
        stop_on_eos: bool,
    ) -> bool:
        """Check if generation should stop."""
        if not stop_on_eos:
            return False

        # Update finished mask
        eos_mask = next_token.squeeze(-1) == gen_state.eos_token_id
        gen_state.finished = gen_state.finished | eos_mask

        # Stop if all sequences have generated EOS
        return gen_state.finished.all().item()

    def _postprocess_eos(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Truncate at first EOS to avoid repeated EOS."""
        for b_idx in range(input_ids.size(0)):
            tokens = input_ids[b_idx]
            eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                first_eos_pos = eos_positions[0].item()
                # Pad everything after first EOS with EOS for consistent batch length
                input_ids[b_idx, first_eos_pos + 1 :] = eos_token_id

        return input_ids


class GenerationState:
    """State container for generation process."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        cache_global_ids: Optional[torch.Tensor],
        finished: torch.Tensor,
        user_provided_landmarks: bool,
        eos_token_id: int,
    ) -> None:
        self.input_ids = input_ids
        self.cache_global_ids = cache_global_ids
        self.finished = finished
        self.user_provided_landmarks = user_provided_landmarks
        self.eos_token_id = eos_token_id


__all__ = ["GenerationState", "TextGenerator"]
