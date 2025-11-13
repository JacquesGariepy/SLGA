"""
Core constants for SLGA model.

Centralizes magic numbers and configuration constants used throughout the codebase.
This improves maintainability and makes it easy to tune hyperparameters.
"""

# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Epsilon for numerical stability in divisions and logarithms
EPS_STABILITY = 1e-10

# Epsilon for large operations (less strict)
EPS_LARGE = 1e-8

# Minimum/maximum values for clamping to prevent overflow
SCORE_CLAMP_MIN = -20.0
SCORE_CLAMP_MAX = 20.0


# ============================================================================
# Landmark Selection Constants
# ============================================================================

# Gumbel-Softmax temperature settings
GUMBEL_INITIAL_TEMPERATURE = 1.0
GUMBEL_TEMPERATURE_DECAY = 0.999  # 10x faster than original 0.9999
GUMBEL_MIN_TEMPERATURE = 0.3  # More discriminative than original 0.5

# Straight-through estimator temperature
STRAIGHT_THROUGH_TEMPERATURE = 0.1

# Default dropout rate for landmark scorer
LANDMARK_SCORER_DROPOUT = 0.1


# ============================================================================
# Attention Constants
# ============================================================================

# Default attention dropout rate
ATTENTION_DROPOUT = 0.1

# Default projection dropout rate
PROJECTION_DROPOUT = 0.1

# Diversity penalty for top-K selection (inter-head diversity)
DIVERSITY_PENALTY = 0.1


# ============================================================================
# Loss Function Constants
# ============================================================================

# Regularization weights for auxiliary losses
LAMBDA_SPACING_LOSS = 0.01  # Landmark spacing uniformity
LAMBDA_DIVERSITY_LOSS = 0.01  # Entropy-based diversity (deprecated)
LAMBDA_SPARSITY_LOSS = 0.001  # Concentration in top-K

# Sparsity loss target thresholds
SPARSITY_BASE_TARGET = 0.60  # Base target: 60% mass in top-K
SPARSITY_RATIO_WEIGHT = 0.40  # Bonus based on G/L ratio

# Label masking value for padding tokens
LABEL_MASK_VALUE = -100


# ============================================================================
# Architecture Defaults
# ============================================================================

# Default local window size
DEFAULT_LOCAL_WINDOW = 128

# Default global landmarks count
DEFAULT_GLOBAL_K = 24

# Default max global landmarks for heuristics
DEFAULT_MAX_GLOBAL = 64

# Default heuristic spacing
DEFAULT_GLOBAL_EVERY = 128

# Default dilation factor (1=dense, 2=skip every 2)
DEFAULT_DILATION = 1


# ============================================================================
# Training Constants
# ============================================================================

# Default learning rate
DEFAULT_LEARNING_RATE = 3e-4

# Default batch size
DEFAULT_BATCH_SIZE = 8

# Default gradient accumulation steps
DEFAULT_GRAD_ACCUM_STEPS = 4

# Default max sequence length
DEFAULT_MAX_SEQ_LENGTH = 512

# Default warmup steps for learning rate schedule
DEFAULT_WARMUP_STEPS = 500


# ============================================================================
# Generation Constants
# ============================================================================

# Default generation temperature
DEFAULT_GEN_TEMPERATURE = 0.8

# Default top-k for sampling
DEFAULT_GEN_TOP_K = 50

# Default top-p (nucleus sampling)
DEFAULT_GEN_TOP_P = 0.95

# Default max new tokens
DEFAULT_MAX_NEW_TOKENS = 100


# ============================================================================
# Data Processing Constants
# ============================================================================

# Default span length for TF-IDF collator
DEFAULT_SPAN_LENGTH = 8

# Text processing constants
PARAGRAPH_SEPARATOR = "\n\n"

# Default text key in dataset
DEFAULT_TEXT_KEY = "text"


# ============================================================================
# Model Architecture Constants
# ============================================================================

# Minimum guard for landmark spacing loss (avoid NaN when G < 2)
MIN_LANDMARKS_FOR_SPACING = 2

# Default hidden dimension ratio (relative to embed_dim)
HIDDEN_DIM_RATIO = 0.5  # hidden_dim = embed_dim // 2


# ============================================================================
# Cache and Optimization
# ============================================================================

# Maximum cache size for local attention masks
MAX_MASK_CACHE_SIZE = 100

# Enable/disable various optimizations
ENABLE_MASK_CACHING = True
ENABLE_DIVERSE_TOPK = True
ENABLE_GATED_FUSION = True
ENABLE_JOINT_NORMALIZATION = False  # Experimental


# ============================================================================
# Validation and Monitoring
# ============================================================================

# Threshold for detecting attention pattern collapse
ATTENTION_COLLAPSE_THRESHOLD = 0.95

# Threshold for NaN detection warnings
NAN_WARNING_THRESHOLD = 0.0

# Performance monitoring thresholds
SLOW_FORWARD_THRESHOLD_MS = 100.0
HIGH_MEMORY_THRESHOLD_GB = 10.0


# ============================================================================
# Testing Constants
# ============================================================================

# Test batch sizes
TEST_BATCH_SIZE = 2
TEST_SEQ_LENGTH = 128
TEST_EMBED_DIM = 256

# Test tolerances
TEST_RTOL = 1e-5
TEST_ATOL = 1e-8


# ============================================================================
# Deprecation Flags
# ============================================================================

# Use Gumbel-Softmax (True) or Straight-Through (False)
PREFER_GUMBEL_OVER_STRAIGHT_THROUGH = True

# Use spacing loss (True) or diversity loss (False)
USE_SPACING_OVER_DIVERSITY = True
