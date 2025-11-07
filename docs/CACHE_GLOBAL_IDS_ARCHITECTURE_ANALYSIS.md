# Critical Architecture Analysis: cache_global_ids Data Flow in SLGA

## Executive Summary

**CRITICAL BUG IDENTIFIED**: There is a **semantic mismatch** in the `cache_global_ids` data flow between the collator and the model.

- **Collator returns**: TOKEN IDs (actual vocabulary indices)
- **Model expects**: POSITION indices (sequence positions 0 to L-1)
- **Impact**: Complete corruption of landmark attention mechanism during training

---

## 1. Data Flow Architecture

```
CollatorLocalGlobal.__call__()
    ↓ Computes POSITION indices (0 to L-1)
    ↓
    ↓ [BUG HERE] Converts positions → tokens via torch.gather
    ↓
model.forward(cache_global_ids=TOKEN_IDS)  ← Expects POSITION indices
    ↓
    ↓ Uses as indices to gather from sequence
    ↓
[CRASH or WRONG LANDMARKS]
```

---

## 2. Detailed Analysis: CollatorLocalGlobal.__call__()

### Location: `/mnt/d/ai/SLGA/src/data.py` lines 256-292

### What It Does:

```python
# Step 1: Select landmark POSITIONS (0 to L-1)
cache_global_ids_list = []
for i, text in enumerate(texts):
    L = self.max_length

    if self.strategy == "regular":
        landmarks = self._select_landmarks_regular(L)  # Returns positions [0, 10, 20, ...]
    elif self.strategy == "random":
        landmarks = self._select_landmarks_random(L)   # Returns positions
    elif self.strategy == "paragraph":
        landmarks = self._select_landmarks_paragraph(text, tokens_i)  # Returns positions

    cache_global_ids_list.append(landmarks[:self.max_global])

cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)  # (B, G)
# At this point: cache_global_ids contains POSITIONS

# Step 2: [BUG] Convert positions to tokens
cache_global_tokens = torch.gather(
    input_ids,           # (B, L) - token IDs
    dim=1,
    index=cache_global_ids.clamp(0, self.max_length - 1),  # Use positions as indices
)

# Step 3: Return TOKEN IDs instead of positions
return {
    "input_ids": input_ids,
    "labels": labels,
    "cache_global_ids": cache_global_tokens,  # ❌ TOKENS, not positions!
}
```

### Example with Real Data:

```python
# Input sequence
input_ids = [101, 2023, 2003, 1037, 3231, 102, 0, 0]  # Token IDs
#            [CLS] this   is    a    test  [SEP] PAD PAD

# Landmark positions selected (e.g., regular strategy with max_global=3)
positions = [0, 3, 6]  # First token, middle token, last real token

# What collator SHOULD return:
cache_global_ids = [0, 3, 6]  # Positions

# What collator ACTUALLY returns:
cache_global_tokens = torch.gather(input_ids, 1, positions)
cache_global_ids = [101, 1037, 0]  # TOKEN IDs at those positions ❌
```

---

## 3. Detailed Analysis: model.forward()

### Location: `/mnt/d/ai/SLGA/src/model.py` lines 257-272

### What It Expects:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    cache_global_ids: Optional[torch.Tensor] = None,  # ✅ Should be POSITION indices
    return_aux: bool = False,
    global_weight: float = 1.0,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Args:
        cache_global_ids: (B, G) indices de landmarks globaux (si learned_landmarks=False)
                          ↑ Documentation says "indices" = positions
    """

    # ...

    elif cache_global_ids is not None:
        # Landmarks heuristiques - utiliser les indices fournis
        landmark_indices = cache_global_ids  # (B, G)

    # Passer par les blocs Transformer
    for block in self.blocks:
        # Extraire les états actuels des landmarks depuis x
        if landmark_indices is not None:
            B_cur, L_cur, D = x.shape
            G = landmark_indices.size(1)
            # ✅ FIX: Clamp indices pour éviter out-of-bounds
            landmark_indices_safe = torch.clamp(landmark_indices, 0, L_cur - 1)
            landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B_cur, G, D)

            # ❌ BUG: Using TOKEN IDs as POSITION indices!
            landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)  # (B, G, D)
```

### The Problem:

When `cache_global_ids` contains TOKEN IDs (e.g., `[101, 1037, 0]`), the model tries to use them as **position indices**:

```python
# Given:
x.shape = (1, 512, 768)  # (batch, seq_len, hidden_dim)
landmark_indices = [101, 1037, 0]  # TOKEN IDs (not positions!)

# torch.gather tries to gather positions:
# Position 101: ✅ Valid (if seq_len >= 102)
# Position 1037: ❌ OUT OF BOUNDS if seq_len = 512!
# Position 0: ✅ Valid

# Even if it doesn't crash (due to clamping), it gathers WRONG positions!
```

---

## 4. Detailed Analysis: model.generate()

### Location: `/mnt/d/ai/SLGA/src/model.py` lines 350-355

### What It Does (CORRECTLY):

```python
# Recompute landmarks for current context (if using heuristic landmarks)
if not self.cfg.learned_landmarks and cache_global_ids is None:
    L = input_ids.size(1)
    # ✅ CORRECT: Use linspace to get POSITION indices
    landmark_positions = torch.linspace(0, L-1, self.cfg.global_k, device=input_ids.device).long()
    cache_global_ids = landmark_positions.unsqueeze(0).expand(input_ids.size(0), -1)
    # cache_global_ids contains POSITIONS: [0, 64, 128, 192, ...]

# Forward
logits = self(input_ids, cache_global_ids=cache_global_ids)  # ✅ Passes positions
```

**This is CORRECT** - it creates position indices and passes them to `forward()`.

---

## 5. Root Cause Analysis

### The Bug Chain:

1. **Collator Design Flaw** (`src/data.py` line 282-291):
   - Landmark selection methods return **POSITION indices** (correct)
   - But collator then **converts positions → tokens** via `torch.gather`
   - Returns **TOKEN IDs** labeled as `cache_global_ids`

2. **Model Expectation** (`src/model.py` line 259, 271):
   - Model expects `cache_global_ids` to contain **POSITION indices**
   - Uses them directly in `torch.gather` to extract landmark states
   - **Receives TOKEN IDs instead**

3. **Consequences**:
   - TOKEN IDs used as position indices
   - Causes out-of-bounds access (mitigated by clamping)
   - Even with clamping, selects **completely wrong landmarks**
   - Training sees corrupted global attention patterns

### Why It Doesn't Crash Immediately:

```python
# In model.forward() line 270:
landmark_indices_safe = torch.clamp(landmark_indices, 0, L_cur - 1)

# This prevents crashes but doesn't fix the semantic error:
# - Token ID 101 → clamped to position 101 (wrong landmark)
# - Token ID 50257 → clamped to position 511 (always last position!)
# - Token ID 0 → position 0 (always first token)
```

---

## 6. Impact Assessment

### Severity: **CRITICAL** 🔴

### Affected Components:
- ✅ **Training**: Uses collator → sends TOKEN IDs → model uses wrong landmarks
- ✅ **Validation**: Same issue
- ⚠️ **Generation**: Bypasses collator → computes positions directly → WORKS CORRECTLY

### Why Training Still Works (Poorly):
1. Token IDs get clamped to valid sequence positions
2. Model still trains, but with **systematically wrong landmarks**
3. Performance degradation is subtle but significant
4. Explains poor generation quality even after training

### Smoking Gun Evidence:

```python
# In training logs, you'd see:
# - Landmarks always clustering at certain positions (0, last, vocab_size % seq_len)
# - No actual strategic landmark placement working
# - Global attention not learning proper long-range dependencies
```

---

## 7. Comparison: Generation vs Training

| Aspect | Generation (generate.py) | Training (train.py) |
|--------|--------------------------|---------------------|
| Landmark source | Computed in generate() | Computed in collator |
| Data type | POSITIONS ✅ | TOKEN IDs ❌ |
| Method | `linspace(0, L-1, G)` | `gather(input_ids, positions)` |
| Correctness | **CORRECT** | **BROKEN** |
| Why it works/fails | Direct position computation | Semantic type confusion |

---

## 8. The Fix

### Option 1: Remove the gather() (Simplest) ✅ RECOMMENDED

```python
# In src/data.py, line 288-292:
return {
    "input_ids": input_ids,
    "labels": labels,
    "cache_global_ids": cache_global_ids,  # ✅ Return POSITIONS, not tokens
}

# Remove lines 280-286 (the gather operation)
```

### Option 2: Change Model to Expect Tokens (Complex) ❌ NOT RECOMMENDED

Would require:
- Embedding token IDs to get landmark representations
- Matching them back to sequence positions
- Much more complex and inefficient

---

## 9. Verification Steps

### Before Fix:
```python
# Add debug logging in train.py before forward():
print(f"cache_ids range: {cache_ids.min()}-{cache_ids.max()}")
print(f"sequence length: {input_ids.size(1)}")
# Expect to see: cache_ids can be > seq_len (TOKEN IDs, not positions)
```

### After Fix:
```python
# Should see:
# cache_ids range: 0-511 (for seq_len=512)
# All values < sequence length
```

### Test Script:
```python
# tests/test_cache_global_ids_semantics.py
import torch
from src.data import CollatorLocalGlobal
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
collator = CollatorLocalGlobal(
    tokenizer=tokenizer,
    max_length=512,
    max_global=64,
    strategy="regular"
)

examples = [{"text": "This is a test sentence." * 50}]
batch = collator(examples)

# Check semantic correctness
cache_ids = batch["cache_global_ids"]
seq_len = batch["input_ids"].size(1)

print(f"cache_global_ids type: {cache_ids.dtype}")
print(f"cache_global_ids range: [{cache_ids.min()}, {cache_ids.max()}]")
print(f"sequence_length: {seq_len}")

# BEFORE FIX: cache_ids.max() can be >> seq_len (TOKEN IDs)
# AFTER FIX: cache_ids.max() < seq_len (POSITIONS)
assert cache_ids.max() < seq_len, "Bug: cache_global_ids contains TOKEN IDs instead of positions!"
assert cache_ids.min() >= 0, "Bug: negative indices"
print("✅ Test passed: cache_global_ids contains valid position indices")
```

---

## 10. Historical Context

### Why This Bug Exists:

Looking at the code comments and structure:

```python
# Line 280-286 comment:
# "Gather les tokens correspondants"
# "input_ids: (B, L), cache_global_ids: (B, G) -> (B, G)"
```

**Someone thought the model needed landmark TOKENS, not positions.**

This is a common confusion in transformer implementations:
- Positions are used for gathering states
- Tokens are used for predictions
- The collator confused these two concepts

### Why It Wasn't Caught:

1. **Clamping hides the error**: Model clamps indices, preventing crashes
2. **Training still converges**: With wrong landmarks, just slower/worse
3. **Generate works**: Bypasses collator entirely
4. **Documentation ambiguity**: "cache_global_ids" name doesn't clarify position vs token

---

## 11. Recommendations

### Immediate Actions (P0):

1. ✅ **Fix the collator**: Remove the `torch.gather` operation
2. ✅ **Add validation**: Assert `cache_global_ids.max() < seq_len` in model
3. ✅ **Test thoroughly**: Verify landmarks are at correct positions
4. ✅ **Retrain model**: Previous checkpoints learned with wrong landmarks

### Code Quality Improvements (P1):

1. **Rename for clarity**:
   ```python
   cache_global_ids → cache_landmark_positions
   ```

2. **Add type hints**:
   ```python
   def forward(
       self,
       input_ids: torch.Tensor,  # (B, L) token IDs
       cache_landmark_positions: Optional[torch.Tensor] = None,  # (B, G) position indices [0, L-1]
       ...
   ```

3. **Add assertions**:
   ```python
   if cache_landmark_positions is not None:
       assert cache_landmark_positions.max() < input_ids.size(1), \
           "Landmark positions must be < sequence length"
   ```

### Testing Requirements (P1):

1. Unit test for collator output semantics
2. Integration test for model forward with landmarks
3. End-to-end test comparing training vs generation landmark behavior
4. Regression test to prevent future semantic type confusion

---

## 12. Conclusion

**The bug is clear and critical:**

- ❌ Collator returns **TOKEN IDs** (vocabulary indices 0-50257)
- ❌ Model expects **POSITION indices** (sequence positions 0-511)
- ❌ Training uses completely wrong landmarks
- ✅ Generation works because it computes positions correctly

**The fix is simple:**

Remove the `torch.gather` operation in the collator and return positions directly.

**Impact:**

This bug explains:
- Why generation quality is poor despite training
- Why global attention seems ineffective
- Why model doesn't learn long-range dependencies well
- Why landmarks don't follow expected patterns

**Priority:**

This should be fixed **immediately** before any further training runs. All previous checkpoints were trained with corrupted landmark attention.

---

## Appendix: Code References

### Key Files:
- `/mnt/d/ai/SLGA/src/data.py` - Lines 156-292 (CollatorLocalGlobal)
- `/mnt/d/ai/SLGA/src/model.py` - Lines 219-272 (forward method)
- `/mnt/d/ai/SLGA/src/model.py` - Lines 350-358 (generate method)
- `/mnt/d/ai/SLGA/scripts/train.py` - Lines 358-361, 658-661 (usage)

### Related Issues:
- Generation quality problems (README_GENERATION_ANALYSIS.md)
- Training health concerns (TRAINING_HEALTH_DIAGNOSIS_STEP32000.md)
- Landmark selection issues (LINSPACE_LANDMARK_FIX_REPORT.md)
