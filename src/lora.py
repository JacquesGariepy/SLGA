"""LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Reference: https://arxiv.org/abs/2106.09685
QLoRA: https://arxiv.org/abs/2305.14314

LoRA enables parameter-efficient fine-tuning by decomposing weight updates into
low-rank matrices: W' = W + BA where rank << min(d_in, d_out).

This reduces trainable parameters by 100-1000x while maintaining performance.
"""

from __future__ import annotations
import math
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, List, Set, Dict, Any, Union
from dataclasses import dataclass, asdict, field


# ======================================================================
# LoRA Linear Layer
# ======================================================================

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Implements: W' = W + (alpha/r) * BA where:
    - W: frozen original weights (d_out, d_in)
    - B: trainable matrix (d_out, rank)
    - A: trainable matrix (rank, d_in)
    - rank << min(d_in, d_out)
    - alpha: scaling factor

    The scaling factor alpha/r controls the magnitude of the LoRA contribution.
    Higher alpha = stronger adaptation. Default alpha=16, rank=8 gives 2x scaling.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """Initialize LoRA linear layer.

        Args:
            original_layer: Pretrained nn.Linear to adapt
            rank: Rank of adaptation matrices (lower = fewer params)
            alpha: Scaling factor for LoRA contribution
            dropout: Dropout probability for LoRA path
            merge_weights: Whether to merge LoRA into base weights on init
        """
        super().__init__()

        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices
        d_out, d_in = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming (as if it's a linear layer), B with zeros
        # This ensures W' = W at initialization (B @ A = 0 initially)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if merge_weights:
            self.merge()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor (..., d_in)

        Returns:
            Output tensor (..., d_out)
        """
        # Original forward pass
        result = self.original(x)

        # Add LoRA contribution if not merged
        if not self.merged:
            lora_out = self.lora_dropout(x) @ self.lora_A.T  # (..., rank)
            lora_out = lora_out @ self.lora_B.T  # (..., d_out)
            result = result + self.scaling * lora_out

        return result

    def merge(self) -> None:
        """Merge LoRA weights into original weights for inference.

        After merging:
        - No computational overhead during forward pass
        - LoRA matrices are added to original weights
        - Can't be unmerged (one-way operation)

        Call this before deployment to optimize inference speed.
        """
        if self.merged:
            return

        # Compute LoRA contribution: (alpha/r) * B @ A
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)

        # Add to original weights (in-place)
        with torch.no_grad():
            self.original.weight.data += lora_weight

        self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from original weights.

        Reverses the merge operation. Useful for:
        - Switching between different LoRA adapters
        - Continuing training after merging
        """
        if not self.merged:
            return

        # Subtract LoRA contribution
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)

        with torch.no_grad():
            self.original.weight.data -= lora_weight

        self.merged = False

    def __repr__(self) -> str:
        d_out, d_in = self.original.weight.shape
        return (
            f"LoRALinear(in={d_in}, out={d_out}, rank={self.rank}, "
            f"alpha={self.alpha}, merged={self.merged})"
        )


# ======================================================================
# QLoRA with 4-bit Quantization
# ======================================================================

class QLoRALinear(LoRALinear):
    """LoRA with 4-bit quantized base weights (QLoRA).

    QLoRA combines:
    - 4-bit NormalFloat (NF4) quantization for base weights
    - LoRA adapters in full precision
    - Double quantization for quantization constants

    This achieves 16GB -> 3GB memory reduction for 7B models while
    maintaining full fine-tuning performance.

    Requires: bitsandbytes library
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        quantize_base: bool = True,
    ):
        """Initialize QLoRA linear layer.

        Args:
            original_layer: Pretrained nn.Linear to adapt
            rank: Rank of LoRA matrices
            alpha: LoRA scaling factor
            dropout: LoRA dropout probability
            quantize_base: Whether to quantize base weights to 4-bit
        """
        # Initialize LoRA first
        super().__init__(original_layer, rank, alpha, dropout, merge_weights=False)

        # Try to quantize base weights
        self.is_quantized = False
        if quantize_base:
            try:
                import bitsandbytes as bnb
                self._quantize_weights()
                self.is_quantized = True
            except ImportError:
                print("⚠️  bitsandbytes not found. Install with: pip install bitsandbytes")
                print("   Falling back to standard LoRA (no quantization)")
            except Exception as e:
                print(f"⚠️  Quantization failed: {e}")
                print("   Falling back to standard LoRA")

    def _quantize_weights(self) -> None:
        """Quantize original weights to 4-bit NF4 format.

        NormalFloat 4-bit (NF4):
        - Optimized for normally distributed weights
        - Information-theoretically optimal for Gaussian data
        - Better than uniform 4-bit quantization

        Double quantization:
        - Quantizes the quantization constants themselves
        - Further memory reduction with minimal quality loss
        """
        import bitsandbytes as bnb

        # Get original weight
        weight = self.original.weight.data
        device = weight.device

        # Quantize to 4-bit NF4
        # This replaces the weight tensor with a quantized version
        quant_weight, quant_state = bnb.functional.quantize_4bit(
            weight,
            quant_type='nf4',  # NormalFloat 4-bit
            compress_statistics=True,  # Double quantization
        )

        # Store quantized weight and state
        self.original.weight.data = quant_weight
        self.quant_state = quant_state

        # Mark as quantized
        self.original.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization + LoRA.

        For QLoRA:
        1. Dequantize base weights on-the-fly (4-bit -> FP16/BF16)
        2. Compute base output: x @ W^T
        3. Add LoRA contribution: scaling * x @ A^T @ B^T
        """
        if self.is_quantized:
            import bitsandbytes as bnb

            # Dequantize weights for computation
            weight_deq = bnb.functional.dequantize_4bit(
                self.original.weight.data,
                self.quant_state,
            )

            # Manual linear: x @ W^T + bias
            result = x @ weight_deq.T
            if self.original.bias is not None:
                result = result + self.original.bias

            # Add LoRA
            if not self.merged:
                lora_out = self.lora_dropout(x) @ self.lora_A.T
                lora_out = lora_out @ self.lora_B.T
                result = result + self.scaling * lora_out

            return result
        else:
            # Fallback to standard LoRA
            return super().forward(x)

    def __repr__(self) -> str:
        d_out, d_in = self.original.weight.shape
        quant_str = "4-bit" if self.is_quantized else "full precision"
        return (
            f"QLoRALinear(in={d_in}, out={d_out}, rank={self.rank}, "
            f"alpha={self.alpha}, base={quant_str})"
        )


# ======================================================================
# Model Adaptation Utilities
# ======================================================================

def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ["qkv_proj", "out_proj", "fc1", "fc2"],
    dropout: float = 0.0,
    use_qlora: bool = False,
    quantize_base: bool = True,
) -> nn.Module:
    """Apply LoRA to specified modules in the model.

    This function walks the model tree and replaces target linear layers
    with LoRA-adapted versions.

    Args:
        model: Model to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to adapt
        dropout: LoRA dropout probability
        use_qlora: Use QLoRA (4-bit quantization) instead of LoRA
        quantize_base: Whether to quantize base weights (QLoRA only)

    Returns:
        Modified model with LoRA layers

    Example:
        >>> model = LLMTransformer(config)
        >>> model = apply_lora_to_model(
        ...     model,
        ...     rank=8,
        ...     target_modules=["qkv_proj", "out_proj"],
        ... )
        >>> # Only LoRA parameters are trainable now
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """
    replaced = 0

    for name, module in model.named_modules():
        # Check if this module should be adapted
        if not isinstance(module, nn.Linear):
            continue

        # Check if name matches any target pattern
        should_adapt = False
        for pattern in target_modules:
            if pattern in name:
                should_adapt = True
                break

        if not should_adapt:
            continue

        # Get parent module and attribute name
        *parent_names, attr_name = name.split('.')
        parent = model
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)

        # Replace with LoRA/QLoRA
        if use_qlora:
            lora_layer = QLoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                quantize_base=quantize_base,
            )
        else:
            lora_layer = LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        setattr(parent, attr_name, lora_layer)
        replaced += 1

    print(f"✅ Applied LoRA to {replaced} modules")
    print(f"   Rank: {rank}, Alpha: {alpha}")
    print(f"   Target modules: {target_modules}")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA trainable parameters.

    Use this to create an optimizer that only updates LoRA weights:

    Example:
        >>> lora_params = get_lora_parameters(model)
        >>> optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    """
    params = []
    for module in model.modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable vs frozen parameters.

    Returns:
        Dict with 'trainable', 'frozen', 'total' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': total,
        'trainable_pct': 100 * trainable / total if total > 0 else 0,
    }


def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA weights into base model for deployment.

    After merging:
    - No runtime overhead from LoRA
    - Model can be saved as standard checkpoint
    - Can't switch LoRA adapters anymore

    Call this before saving for production deployment.
    """
    merged = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            if not module.merged:
                module.merge()
                merged += 1

    print(f"✅ Merged {merged} LoRA layers into base model")


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerge LoRA weights to enable adapter switching.

    Use this to:
    - Switch between different LoRA adapters
    - Continue training after merging
    - Load a different LoRA checkpoint
    """
    unmerged = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            if module.merged:
                module.unmerge()
                unmerged += 1

    print(f"✅ Unmerged {unmerged} LoRA layers from base model")


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA weights (not base model).

    This creates a small checkpoint (~1-10MB) containing only the
    LoRA adaptation, not the full model weights.

    Args:
        model: Model with LoRA layers
        path: Path to save LoRA weights
    """
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data
            lora_state[f"{name}.rank"] = module.rank
            lora_state[f"{name}.alpha"] = module.alpha
            lora_state[f"{name}.scaling"] = module.scaling

    torch.save(lora_state, path)

    # Report size
    size_mb = sum(v.numel() * v.element_size() for v in lora_state.values() if torch.is_tensor(v)) / 1e6
    print(f"✅ Saved LoRA weights to {path} ({size_mb:.2f} MB)")


def load_lora_weights(model: nn.Module, path: str, strict: bool = True) -> None:
    """Load LoRA weights into model.

    Args:
        model: Model with LoRA layers
        path: Path to LoRA weights
        strict: Whether to require exact match of layer names
    """
    lora_state = torch.load(path, map_location='cpu')

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"

            if key_A in lora_state and key_B in lora_state:
                module.lora_A.data.copy_(lora_state[key_A])
                module.lora_B.data.copy_(lora_state[key_B])
                loaded += 1
            elif strict:
                raise KeyError(f"LoRA weights not found for {name}")

    print(f"✅ Loaded LoRA weights from {path} ({loaded} layers)")


# ======================================================================
# LoRA Configuration Dataclass
# ======================================================================

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        enabled: Whether to use LoRA
        rank: Rank of adaptation matrices
        alpha: Scaling factor
        target_modules: List of module name patterns to adapt
        dropout: Dropout probability for LoRA path
        use_qlora: Use 4-bit quantization (QLoRA)
        quantize_base: Whether to quantize base weights
        merge_weights: Merge weights after loading
    """
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = None
    dropout: float = 0.0
    use_qlora: bool = False
    quantize_base: bool = True
    merge_weights: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            # Default: adapt attention and FFN layers
            self.target_modules = ["qkv_proj", "out_proj", "fc1", "fc2"]


# ======================================================================
# MultiLoRA Compatibility - Adapter Configuration
# ======================================================================

@dataclass
class MultiLoRAAdapterConfig:
    """MultiLoRA-compatible adapter configuration.

    This configuration format is compatible with the MultiLoRA inference server
    at https://github.com/yourusername/MultiLoRA

    Format matches: MultiLoRA/adapters/*.json

    Example:
        {
            "adapter_name": "creative-style",
            "rank": 8,
            "alpha": 16.0,
            "target_modules": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
            "base_model": "llama-7b",
            "lora_dropout": 0.05,
            "metadata": {...}
        }
    """
    adapter_name: str
    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=lambda: ["qkv_proj", "out_proj", "fc1", "fc2"])
    base_model: str = "slga-v2"
    lora_dropout: float = 0.0
    # Extended metadata for MultiLoRA composition
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Training metadata
    training_steps: int = 0
    final_loss: float = 0.0
    dataset: str = ""
    # Version info
    version: str = "1.0"
    framework: str = "slga"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_name": self.adapter_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "base_model": self.base_model,
            "lora_dropout": self.lora_dropout,
            "metadata": self.metadata,
            "training_steps": self.training_steps,
            "final_loss": self.final_loss,
            "dataset": self.dataset,
            "version": self.version,
            "framework": self.framework,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiLoRAAdapterConfig":
        """Create from dictionary."""
        return cls(
            adapter_name=data.get("adapter_name", "unnamed"),
            rank=data.get("rank", 8),
            alpha=data.get("alpha", 16.0),
            target_modules=data.get("target_modules", ["qkv_proj", "out_proj", "fc1", "fc2"]),
            base_model=data.get("base_model", "slga-v2"),
            lora_dropout=data.get("lora_dropout", 0.0),
            metadata=data.get("metadata", {}),
            training_steps=data.get("training_steps", 0),
            final_loss=data.get("final_loss", 0.0),
            dataset=data.get("dataset", ""),
            version=data.get("version", "1.0"),
            framework=data.get("framework", "slga"),
        )

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MultiLoRAAdapterConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ======================================================================
# MultiLoRA Export/Import Functions
# ======================================================================

def save_multilora_adapter(
    model: nn.Module,
    adapter_dir: str,
    adapter_name: str,
    base_model: str = "slga-v2",
    metadata: Optional[Dict[str, Any]] = None,
    training_steps: int = 0,
    final_loss: float = 0.0,
    dataset: str = "",
) -> str:
    """Save LoRA adapter in MultiLoRA-compatible format.

    Creates an adapter directory with:
    - config.json: MultiLoRA adapter configuration
    - adapter_weights.pt: PyTorch weights (lora_A, lora_B for each layer)
    - adapter_weights.safetensors: SafeTensors format (if safetensors installed)

    This format is directly loadable by MultiLoRA server for inference.

    Args:
        model: Model with LoRA layers
        adapter_dir: Directory to save adapter (will create subdirectory)
        adapter_name: Name for the adapter
        base_model: Base model name (e.g., "llama-7b", "slga-v2")
        metadata: Optional metadata dict for composition hints
        training_steps: Number of training steps completed
        final_loss: Final training loss
        dataset: Dataset used for training

    Returns:
        Path to the created adapter directory

    Example:
        >>> save_multilora_adapter(
        ...     model,
        ...     "adapters/",
        ...     "code-assistant",
        ...     base_model="slga-v2",
        ...     metadata={"task": "code-generation", "quality": 0.92}
        ... )
        'adapters/code-assistant/'
    """
    # Create adapter directory
    adapter_path = Path(adapter_dir) / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights and config
    lora_weights = {}
    target_modules = set()
    rank = None
    alpha = None
    dropout = 0.0

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            # Get config from first layer
            if rank is None:
                rank = module.rank
                alpha = module.alpha
                if hasattr(module, 'lora_dropout') and isinstance(module.lora_dropout, nn.Dropout):
                    dropout = module.lora_dropout.p

            # Extract target module pattern
            parts = name.split('.')
            for part in parts:
                if any(pattern in part for pattern in ['proj', 'fc', 'attn', 'mlp', 'qkv', 'out']):
                    target_modules.add(part)

            # Save weights in MultiLoRA format
            # MultiLoRA expects: adapter_name.layer_name.lora_A/lora_B
            lora_weights[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_weights[f"{name}.lora_B"] = module.lora_B.data.cpu()

    if rank is None:
        raise ValueError("No LoRA layers found in model")

    # Create config
    config = MultiLoRAAdapterConfig(
        adapter_name=adapter_name,
        rank=rank,
        alpha=alpha,
        target_modules=list(target_modules) if target_modules else ["qkv_proj", "out_proj", "fc1", "fc2"],
        base_model=base_model,
        lora_dropout=dropout,
        metadata=metadata or {},
        training_steps=training_steps,
        final_loss=final_loss,
        dataset=dataset,
    )

    # Save config
    config.to_json(str(adapter_path / "config.json"))

    # Save weights in PyTorch format
    torch.save(lora_weights, str(adapter_path / "adapter_weights.pt"))

    # Try to save in SafeTensors format (preferred by MultiLoRA)
    try:
        from safetensors.torch import save_file
        save_file(lora_weights, str(adapter_path / "adapter_weights.safetensors"))
        print(f"✅ Saved SafeTensors weights: {adapter_path / 'adapter_weights.safetensors'}")
    except ImportError:
        print("ℹ️  safetensors not installed. Saved PyTorch format only.")
        print("   Install with: pip install safetensors")

    # Calculate size
    size_mb = sum(v.numel() * v.element_size() for v in lora_weights.values()) / 1e6

    print(f"✅ Saved MultiLoRA adapter: {adapter_path}")
    print(f"   Name: {adapter_name}")
    print(f"   Rank: {rank}, Alpha: {alpha}")
    print(f"   Layers: {len(lora_weights) // 2}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Compatible with: MultiLoRA server")

    return str(adapter_path)


def load_multilora_adapter(
    model: nn.Module,
    adapter_path: str,
    strict: bool = True,
    device: Optional[str] = None,
) -> MultiLoRAAdapterConfig:
    """Load a MultiLoRA-compatible adapter into model.

    Loads adapter from directory format:
    - config.json: Adapter configuration
    - adapter_weights.pt or adapter_weights.safetensors: Weights

    Args:
        model: Model with LoRA layers (must match adapter architecture)
        adapter_path: Path to adapter directory
        strict: Require all layers to be found
        device: Device to load weights to (auto-detect if None)

    Returns:
        MultiLoRAAdapterConfig with adapter metadata

    Example:
        >>> config = load_multilora_adapter(model, "adapters/code-assistant")
        >>> print(f"Loaded {config.adapter_name} trained for {config.training_steps} steps")
    """
    adapter_path = Path(adapter_path)

    # Load config
    config_path = adapter_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = MultiLoRAAdapterConfig.from_json(str(config_path))

    # Try SafeTensors first, then PyTorch
    safetensors_path = adapter_path / "adapter_weights.safetensors"
    pytorch_path = adapter_path / "adapter_weights.pt"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            lora_weights = load_file(str(safetensors_path), device=device or "cpu")
            print(f"✅ Loaded SafeTensors weights from {safetensors_path}")
        except ImportError:
            if pytorch_path.exists():
                lora_weights = torch.load(str(pytorch_path), map_location=device or "cpu")
            else:
                raise ImportError("safetensors required but not installed. pip install safetensors")
    elif pytorch_path.exists():
        lora_weights = torch.load(str(pytorch_path), map_location=device or "cpu")
    else:
        raise FileNotFoundError(f"No weights found in {adapter_path}")

    # Load weights into model
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, QLoRALinear)):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"

            if key_A in lora_weights and key_B in lora_weights:
                module.lora_A.data.copy_(lora_weights[key_A])
                module.lora_B.data.copy_(lora_weights[key_B])
                loaded += 1
            elif strict:
                raise KeyError(f"LoRA weights not found for {name}")

    print(f"✅ Loaded MultiLoRA adapter: {config.adapter_name}")
    print(f"   Loaded {loaded} LoRA layers")
    print(f"   Base model: {config.base_model}")
    print(f"   Training: {config.training_steps} steps, loss={config.final_loss:.4f}")

    return config


def list_multilora_adapters(adapter_dir: str) -> List[MultiLoRAAdapterConfig]:
    """List all MultiLoRA adapters in a directory.

    Args:
        adapter_dir: Directory containing adapter subdirectories

    Returns:
        List of adapter configurations

    Example:
        >>> adapters = list_multilora_adapters("adapters/")
        >>> for a in adapters:
        ...     print(f"{a.adapter_name}: rank={a.rank}, steps={a.training_steps}")
    """
    adapter_dir = Path(adapter_dir)
    adapters = []

    if not adapter_dir.exists():
        return adapters

    for subdir in adapter_dir.iterdir():
        if subdir.is_dir():
            config_path = subdir / "config.json"
            if config_path.exists():
                try:
                    config = MultiLoRAAdapterConfig.from_json(str(config_path))
                    adapters.append(config)
                except Exception as e:
                    print(f"⚠️  Failed to load {subdir}: {e}")

    return adapters


def convert_to_multilora_format(
    lora_weights_path: str,
    output_dir: str,
    adapter_name: str,
    rank: int = 8,
    alpha: float = 16.0,
    base_model: str = "slga-v2",
    target_modules: Optional[List[str]] = None,
) -> str:
    """Convert existing LoRA weights to MultiLoRA format.

    Use this to convert adapters saved with save_lora_weights() to
    the MultiLoRA-compatible directory format.

    Args:
        lora_weights_path: Path to existing .pt LoRA weights
        output_dir: Output directory for MultiLoRA adapter
        adapter_name: Name for the adapter
        rank: LoRA rank (from training config)
        alpha: LoRA alpha (from training config)
        base_model: Base model name
        target_modules: Target module patterns

    Returns:
        Path to created adapter directory
    """
    # Load existing weights
    lora_state = torch.load(lora_weights_path, map_location='cpu')

    # Filter to just the weight tensors
    lora_weights = {}
    detected_modules = set()

    for key, value in lora_state.items():
        if torch.is_tensor(value):
            lora_weights[key] = value
            # Extract module patterns
            parts = key.replace('.lora_A', '').replace('.lora_B', '').split('.')
            for part in parts:
                if any(p in part for p in ['proj', 'fc', 'attn', 'mlp', 'qkv', 'out']):
                    detected_modules.add(part)

    # Create adapter directory
    adapter_path = Path(output_dir) / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Create config
    config = MultiLoRAAdapterConfig(
        adapter_name=adapter_name,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules or list(detected_modules) or ["qkv_proj", "out_proj"],
        base_model=base_model,
    )

    # Save config and weights
    config.to_json(str(adapter_path / "config.json"))
    torch.save(lora_weights, str(adapter_path / "adapter_weights.pt"))

    # Try SafeTensors
    try:
        from safetensors.torch import save_file
        save_file(lora_weights, str(adapter_path / "adapter_weights.safetensors"))
    except ImportError:
        pass

    print(f"✅ Converted to MultiLoRA format: {adapter_path}")
    return str(adapter_path)


__all__ = [
    # Core LoRA classes
    "LoRALinear",
    "QLoRALinear",
    "LoRAConfig",
    # Model adaptation
    "apply_lora_to_model",
    "get_lora_parameters",
    "count_trainable_parameters",
    "merge_lora_weights",
    "unmerge_lora_weights",
    # Standard save/load
    "save_lora_weights",
    "load_lora_weights",
    # MultiLoRA compatibility
    "MultiLoRAAdapterConfig",
    "save_multilora_adapter",
    "load_multilora_adapter",
    "list_multilora_adapters",
    "convert_to_multilora_format",
]
