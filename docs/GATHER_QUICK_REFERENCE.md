# Torch Gather: Quick Reference & Best Practices

## ⚠️ The Problem

```python
# ❌ DANGEROUS - Can crash if indices >= L
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

**Error**:
```
RuntimeError: index 105 is out of bounds for dimension 1 with size 100
```

---

## ✅ The Solution

```python
# ✅ SAFE - Always clamp before gather
landmark_indices_safe = torch.clamp(landmark_indices, 0, L - 1)
landmark_indices_exp = landmark_indices_safe.unsqueeze(-1).expand(B, G, D)
landmark_states = torch.gather(x, dim=1, index=landmark_indices_exp)
```

---

## 📋 Quick Checklist

Before every `torch.gather()`:

- [ ] Clamp indices to valid range `[0, max_valid_index]`
- [ ] Verify dimensions match for gathering
- [ ] Check that clamp doesn't hide logic errors
- [ ] Test with edge cases (short sequences, boundary indices)

---

## 🔧 Common Patterns

### Pattern 1: Basic Gather with Clamp
```python
# Safe gathering along sequence dimension
indices_safe = torch.clamp(indices, 0, seq_len - 1)
result = torch.gather(tensor, dim=1, index=indices_safe)
```

### Pattern 2: Expanded Indices
```python
# For multi-dimensional gathering
indices_safe = torch.clamp(indices, 0, L - 1)  # (B, G)
indices_exp = indices_safe.unsqueeze(-1).expand(B, G, D)  # (B, G, D)
result = torch.gather(x, dim=1, index=indices_exp)  # (B, G, D)
```

### Pattern 3: Inline Clamp
```python
# Concise one-liner (OK for simple cases)
result = torch.gather(tensor, dim=1, index=indices.clamp(0, L-1))
```

### Pattern 4: Multi-head Attention
```python
# For attention with multiple heads
topk_idxs_safe = torch.clamp(topk_idxs, 0, G - 1)  # (B, H, L, k)
topk_idxs_exp = topk_idxs_safe.unsqueeze(-1).expand(B, H, L, k, D)
vg_topk = torch.gather(vg_exp, dim=3, index=topk_idxs_exp)
```

---

## 🧪 Testing Your Gather

```python
def safe_gather_test(tensor, indices, dim=1):
    """Template for testing gather operations"""
    B, L, D = tensor.shape

    # 1. Normal case
    indices_normal = torch.randint(0, L, (B, 10))
    result = torch.gather(tensor, dim=1, index=indices_normal)

    # 2. Boundary case
    indices_boundary = torch.tensor([[0, L-1, L-1, 0]])
    result_boundary = torch.gather(tensor, dim=1, index=indices_boundary)

    # 3. With clamp (handles invalid)
    indices_invalid = torch.tensor([[0, L, L+10, -1]])  # Invalid!
    indices_safe = torch.clamp(indices_invalid, 0, L-1)
    result_safe = torch.gather(tensor, dim=1, index=indices_safe)  # ✓ Works

    # 4. Gradient flow
    tensor.requires_grad = True
    result.sum().backward()
    assert tensor.grad is not None, "No gradients!"
```

---

## 📊 All Gather Operations in SLGA

| File | Line | Status | Protection Method |
|------|------|--------|-------------------|
| `model.py` | 271 | ✅ Protected | `torch.clamp(landmark_indices, 0, L_cur-1)` |
| `slga.py` | 431 | ✅ Protected | `torch.clamp(topk_idxs, 0, G-1)` |
| `landmarks.py` | 169 | ✅ Protected | `torch.clamp(landmark_indices, 0, L-1)` |
| `landmarks.py` | 231 | ✅ Protected | `torch.clamp(landmark_indices, 0, L-1)` |
| `landmarks.py` | 281 | ✅ Protected | `torch.clamp(landmark_indices, 0, L-1)` |
| `data.py` | 285 | ✅ Protected | Inline: `index=cache_ids.clamp(0, max_len-1)` |
| `data.py` | 404 | ✅ Protected | Inline: `index=cache_ids.clamp(0, max_len-1)` |

---

## 🚨 When to Use Clamp

### Always Clamp When:

✅ Indices come from external sources (user input, config)
✅ Indices are computed programmatically (top-K, sampling)
✅ Sequence length is dynamic (variable batches, truncation)
✅ Using pre-computed landmarks with different sequence lengths

### Clamp May Not Be Needed When:

⚠️ Indices are hard-coded and provably valid
⚠️ You have explicit assertions checking bounds
⚠️ Indices come from `torch.arange(L)` with no modifications

**But**: When in doubt, **always clamp**. The overhead is negligible.

---

## 💡 Why Clamp is Better Than Alternatives

### ❌ Bad: Using position 0 as fallback
```python
# Creates artificial concentration at position 0
indices_safe = torch.where(indices >= L, 0, indices)
```

### ❌ Bad: Modulo wrapping
```python
# Wraps indices, loses semantic meaning
indices_safe = indices % L
```

### ✅ Good: Clamping to boundary
```python
# Preserves semantic proximity, differentiable
indices_safe = torch.clamp(indices, 0, L-1)
```

**Why clamp wins**:
- Preserves meaning (boundary positions still relevant)
- Fully differentiable (clean gradients)
- No artificial bias (no concentration at position 0)
- GPU-optimized (fast vectorized operation)

---

## 🔍 Debugging Gather Issues

### Common Errors

1. **RuntimeError: index X is out of bounds**
   - **Cause**: Forgot to clamp indices
   - **Fix**: Add `indices_safe = torch.clamp(indices, 0, max_valid_index)`

2. **Shape mismatch in gather**
   - **Cause**: Index tensor dimensions don't match gather rules
   - **Fix**: Verify `index.shape` aligns with `tensor.shape` on gather dimension

3. **Gradients not flowing**
   - **Cause**: Using non-differentiable operations before gather
   - **Fix**: Ensure all index computations are differentiable (clamp is OK!)

4. **Unexpected values after gather**
   - **Cause**: Indices clamped, reading wrong positions
   - **Fix**: This is expected behavior - inspect upstream index generation

### Debug Template

```python
# Add before gather operation for debugging
print(f"Indices shape: {indices.shape}")
print(f"Indices range: [{indices.min()}, {indices.max()}]")
print(f"Tensor dim {dim} size: {tensor.size(dim)}")
print(f"Any invalid indices? {torch.any(indices >= tensor.size(dim))}")

# Clamp and report
indices_safe = torch.clamp(indices, 0, tensor.size(dim) - 1)
if torch.any(indices != indices_safe):
    print(f"⚠️ Clamped {(indices != indices_safe).sum()} invalid indices")
```

---

## 📚 Additional Resources

- **PyTorch gather docs**: https://pytorch.org/docs/stable/generated/torch.gather.html
- **Full implementation guide**: `/docs/GATHER_CLAMP_PROTECTION.md`
- **Test suite**: `/tests/test_gather_protection.py`
- **Fix summary**: `/docs/GATHER_FIX_SUMMARY_2025-10-28.md`

---

## 🎯 TL;DR

```python
# The one thing to remember:
# ALWAYS CLAMP BEFORE GATHER

indices_safe = torch.clamp(indices, 0, valid_max)
result = torch.gather(tensor, dim=dim, index=indices_safe)
```

**Run tests**: `python tests/test_gather_protection.py`

---

**Last Updated**: 2025-10-28
**Status**: ✅ All 7 gather operations protected
