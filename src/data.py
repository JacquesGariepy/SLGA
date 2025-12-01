"""
Data loading utilities for SLGA training.

Includes:
- Tokenizer loading with fallback
- Dataset loading from HuggingFace (Wikipedia, SlimPajama, The Pile, FineWeb)
- Collators for local and local+global attention

Supported Datasets:
- wikimedia/wikipedia (subset: "20231101.en")
- cerebras/SlimPajama-627B
- EleutherAI/pile
- HuggingFaceFW/fineweb-edu (subset: "sample-10BT")
- togethercomputer/RedPajama-Data-1T
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
from transformers import AutoTokenizer, PreTrainedTokenizer


# ============================================================================
# Dataset field mappings
# ============================================================================
DATASET_TEXT_FIELDS = {
    # Format: dataset_name -> list of possible text field names
    "wikimedia/wikipedia": ["text"],
    "cerebras/SlimPajama-627B": ["text"],
    "EleutherAI/pile": ["text"],
    "HuggingFaceFW/fineweb-edu": ["text"],
    "togethercomputer/RedPajama-Data-1T": ["text"],
    "openwebtext": ["text"],
    "wikitext": ["text"],
    # Instruction/Q&A datasets
    "OpenAssistant/oasst1": ["text"],
    "tatsu-lab/alpaca": ["text", "instruction", "output"],
    "databricks/dolly-15k": ["context", "instruction", "response"],
    "HuggingFaceH4/ultrachat_200k": ["messages"],
    "timdettmers/openassistant-guanaco": ["text"],
    # Fallback fields to try
    "_default": ["text", "content", "document", "raw", "passage", "article"],
}

# Datasets with special conversation format
INSTRUCTION_DATASETS = {
    "OpenAssistant/oasst1",
    "tatsu-lab/alpaca",
    "databricks/dolly-15k",
    "HuggingFaceH4/ultrachat_200k",
    "timdettmers/openassistant-guanaco",
}


def get_text_field(dataset_name: str, example: Dict) -> str:
    """
    Determine the text field name for a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        example: A sample example from the dataset
    
    Returns:
        Name of the text field
    """
    # Try known dataset fields first
    known_fields = DATASET_TEXT_FIELDS.get(dataset_name, [])
    for field in known_fields:
        if field in example and isinstance(example[field], str):
            return field
    
    # Try default fields
    for field in DATASET_TEXT_FIELDS["_default"]:
        if field in example and isinstance(example[field], str):
            return field
    
    # Last resort: find any string field with substantial content
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 50:
            return key
    
    raise KeyError(
        f"Cannot find text field in dataset {dataset_name}. "
        f"Available keys: {list(example.keys())}"
    )


def get_tokenizer(config: Union[str, Dict[str, Any]]) -> PreTrainedTokenizer:
    """
    Load tokenizer from config.
    
    Args:
        config: Either a string (tokenizer name) or dict with 'name' key
    
    Returns:
        PreTrainedTokenizer instance
    """
    if isinstance(config, str):
        tokenizer_name = config
    else:
        tokenizer_name = config.get("name", config.get("path", "gpt2"))
    
    print(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Warning: Could not load {tokenizer_name}: {e}")
        print("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_text_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
):
    """
    Load a text dataset from HuggingFace with optional streaming.
    
    Following HuggingFace best practices:
    https://huggingface.co/docs/datasets/stream
    
    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset/config name
        split: Dataset split (train, validation, test)
        streaming: If True, use streaming mode (IterableDataset)
        shuffle_buffer_size: Buffer size for shuffling in streaming mode
        seed: Random seed for shuffling
    
    Returns:
        HuggingFace Dataset or IterableDataset object
    """
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
    print(f"  Split: {split}")
    print(f"  Streaming: {streaming}")
    
    # Wikipedia doesn't have validation split
    if "wikipedia" in dataset_name.lower() and split == "validation":
        print("  ⚠ Wikipedia has no validation split, using train subset")
        split = "train"
    
    # Load dataset
    try:
        if subset:
            dataset = load_dataset(
                dataset_name, 
                subset, 
                split=split, 
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name, 
                split=split, 
                streaming=streaming,
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if subset:
            print(f"  Retrying without subset...")
            dataset = load_dataset(
                dataset_name, 
                split=split, 
                streaming=streaming,
            )
        else:
            raise
    
    # Streaming: apply shuffle with buffer
    # HF best practice: shuffle() on IterableDataset, not in load_dataset
    if streaming and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        print(f"  ✓ Shuffle buffer: {shuffle_buffer_size}")
    
    # Report info
    if hasattr(dataset, '__len__'):
        print(f"  ✓ Loaded {len(dataset):,} examples")
    else:
        print(f"  ✓ Streaming mode (IterableDataset)")
    
    return dataset


class CollatorLocal:
    """
    Collator for local attention only (learned landmarks).
    
    Tokenizes text and creates input_ids/labels pairs.
    Labels are shifted by 1 (labels[i] = target for input_ids[i]).
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        dataset_name: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._text_field = None  # Cache for text field name
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of dicts with 'text' or 'input_ids' field
        
        Returns:
            Dict with 'input_ids', 'labels', 'cache_global_ids'
        """
        if not examples:
            raise ValueError("Empty batch")
        
        ex0 = examples[0]
        
        # Case 1: Pre-tokenized
        if isinstance(ex0, dict) and "input_ids" in ex0:
            return self._collate_pretokenized(examples)
        
        # Case 2: Raw text - use cached field or detect
        if self._text_field is None and isinstance(ex0, dict):
            try:
                self._text_field = get_text_field(self.dataset_name, ex0)
            except KeyError:
                # Fallback detection
                for k in ("text", "content", "document", "raw"):
                    if k in ex0 and isinstance(ex0[k], str):
                        self._text_field = k
                        break
        
        # Extract texts
        if isinstance(ex0, str):
            texts = list(examples)
        elif self._text_field and self._text_field in ex0:
            texts = [ex[self._text_field] for ex in examples]
        else:
            # Try any string field
            for k, v in ex0.items():
                if isinstance(v, str) and len(v) > 10:
                    texts = [ex[k] for ex in examples]
                    self._text_field = k
                    break
            else:
                raise KeyError(f"Cannot find text field. Keys: {list(ex0.keys())}")
        
        return self._collate_texts(texts)
    
    def _collate_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Collate raw texts."""
        # Tokenize with +1 for label shifting
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"]  # (B, L+1)
        
        # Split into input and labels
        input_ids_final = input_ids[:, :-1].contiguous()  # (B, L)
        labels = input_ids[:, 1:].clone().contiguous()     # (B, L)
        
        # Mask padding in labels
        labels[labels == self.pad_id] = -100
        
        # Don't include cache_global_ids: None - Accelerate can't concatenate None
        return {
            "input_ids": input_ids_final,
            "labels": labels,
        }
    
    def _collate_pretokenized(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate pre-tokenized examples."""
        batch_ids = []
        
        for ex in examples:
            ids = ex["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)
            ids = ids.view(-1)
            
            # Truncate or pad to max_length + 1
            if ids.numel() >= self.max_length + 1:
                ids = ids[:self.max_length + 1]
            else:
                pad = torch.full(
                    (self.max_length + 1 - ids.numel(),),
                    self.pad_id,
                    dtype=torch.long
                )
                ids = torch.cat([ids, pad])
            
            batch_ids.append(ids)
        
        input_ids = torch.stack(batch_ids, dim=0)  # (B, L+1)
        
        input_ids_final = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].clone().contiguous()
        labels[labels == self.pad_id] = -100
        
        # Don't include cache_global_ids: None - Accelerate can't concatenate None
        return {
            "input_ids": input_ids_final,
            "labels": labels,
        }


class CollatorLocalGlobal(CollatorLocal):
    """
    Collator for local + global attention (heuristic landmarks).
    
    Extends CollatorLocal to also provide cache_global_ids
    with evenly spaced landmark positions.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        global_every: int = 64,
        max_global: int = 48,
        dataset_name: str = "",
    ):
        super().__init__(tokenizer, max_length, dataset_name)
        self.global_every = global_every
        self.max_global = max_global
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with heuristic global landmarks."""
        batch = super().__call__(examples)
        
        B, L = batch["input_ids"].shape
        
        # Create evenly spaced landmark positions
        num_landmarks = min(self.max_global, L // self.global_every)
        num_landmarks = max(1, num_landmarks)
        
        landmark_positions = torch.linspace(0, L - 1, num_landmarks).long()
        cache_global_ids = landmark_positions.unsqueeze(0).expand(B, -1)
        
        batch["cache_global_ids"] = cache_global_ids
        
        return batch


class CollatorInstruction(CollatorLocal):
    """
    Collator for instruction/Q&A fine-tuning datasets.

    Handles various instruction dataset formats:
    - OpenAssistant/oasst1: {"text": "..."} (pre-formatted conversations)
    - tatsu-lab/alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - databricks/dolly-15k: {"instruction": "...", "context": "...", "response": "..."}
    - timdettmers/openassistant-guanaco: {"text": "..."} (pre-formatted)

    Formats into: "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        dataset_name: str = "",
    ):
        super().__init__(tokenizer, max_length, dataset_name)
        self.dataset_name = dataset_name

    def _format_instruction(self, example: Dict[str, Any]) -> str:
        """Format an instruction example into a single text string."""
        dataset = self.dataset_name.lower()

        # OpenAssistant/oasst1 - already has 'text' field with conversation
        if "oasst" in dataset or "openassistant" in dataset:
            if "text" in example:
                return example["text"]
            # Fallback: use role/content format if available
            if "role" in example and "content" in example:
                return f"### {example['role'].title()}:\n{example['content']}"

        # Guanaco format - pre-formatted text
        if "guanaco" in dataset:
            return example.get("text", "")

        # Alpaca format
        if "alpaca" in dataset:
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")

            if inp:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return text

        # Dolly format
        if "dolly" in dataset:
            instruction = example.get("instruction", "")
            context = example.get("context", "")
            response = example.get("response", "")

            if context:
                text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return text

        # UltraChat format (messages list)
        if "ultrachat" in dataset and "messages" in example:
            messages = example["messages"]
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"### {role.title()}:\n{content}")
            return "\n\n".join(parts)

        # Generic fallback: try common fields
        if "text" in example:
            return example["text"]
        if "instruction" in example:
            instruction = example["instruction"]
            response = example.get("output", example.get("response", ""))
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        # Last resort: concatenate all string fields
        parts = []
        for k, v in example.items():
            if isinstance(v, str) and len(v) > 5:
                parts.append(v)
        return "\n\n".join(parts)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate instruction examples."""
        if not examples:
            raise ValueError("Empty batch")

        # Format all examples
        texts = [self._format_instruction(ex) for ex in examples]

        # Filter empty texts
        texts = [t for t in texts if t and len(t.strip()) > 10]

        if not texts:
            # Return dummy batch if all texts are empty
            dummy = torch.zeros(1, self.max_length, dtype=torch.long)
            return {
                "input_ids": dummy,
                "labels": torch.full_like(dummy, -100),
            }

        return self._collate_texts(texts)


__all__ = [
    "get_tokenizer",
    "load_text_dataset",
    "get_text_field",
    "CollatorLocal",
    "CollatorLocalGlobal",
    "CollatorInstruction",
    "DATASET_TEXT_FIELDS",
    "INSTRUCTION_DATASETS",
]
