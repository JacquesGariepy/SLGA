# Quick Reference - PyTorch Memory Management

**For SLGA Training Code**

---

## 🚨 Common Memory Leak Patterns (AVOID)

### ❌ Pattern 1: Using `.any()` in conditionals
```python
# BAD - Retains computational graph
with torch.no_grad():
    if (tensor < 0).any():
        handle_invalid()
```

**Fix**:
```python
# GOOD - Cut graph with .item()
with torch.no_grad():
    invalid_count = (tensor < 0).sum().item()
    if invalid_count > 0:
        handle_invalid()
```

---

### ❌ Pattern 2: Indexing without detach
```python
# BAD - Graph retained through indexing
with torch.no_grad():
    mask = (labels < 0)
    invalid_vals = labels[mask].cpu().tolist()
```

**Fix**:
```python
# GOOD - Detach before CPU transfer
with torch.no_grad():
    mask = (labels < 0)
    invalid_vals = labels[mask].detach().cpu().tolist()
```

---

### ❌ Pattern 3: Metrics without .item()
```python
# BAD - Tensor retained in print
print(f"Mean: {tensor.mean()}")
```

**Fix**:
```python
# GOOD - Extract scalar with .item()
print(f"Mean: {tensor.mean().item()}")
```

---

### ❌ Pattern 4: No tensor cleanup in loops
```python
# BAD - Tensors accumulate
for batch in dataloader:
    input_ids = batch["input_ids"].to(device)
    output = model(input_ids)
    loss = criterion(output, labels)
    # Tensors accumulate in memory!
```

**Fix**:
```python
# GOOD - Explicit cleanup
for i, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    output = model(input_ids)
    loss = criterion(output, labels)

    # Clean up
    del input_ids, output, loss
    if i % 10 == 0:
        torch.cuda.empty_cache()
```

---

### ❌ Pattern 5: Auxiliary tensors with backward hooks
```python
# BAD - aux tensors may have backward hooks
landmark_indices = aux['landmark_indices']
gaps = landmark_indices[:, 1:] - landmark_indices[:, :-1]
spacing = gaps.mean().item()
```

**Fix**:
```python
# GOOD - Detach auxiliary tensors first
landmark_indices = aux['landmark_indices'].detach()
gaps = landmark_indices[:, 1:] - landmark_indices[:, :-1]
spacing = gaps.mean().item()
```

---

## ✅ Memory Management Checklist

### Before Validation
```python
import gc

# Synchronize all GPUs
if hasattr(accelerator, 'wait_for_everyone'):
    accelerator.wait_for_everyone()

# Set model to eval
model.eval()

# Force cleanup
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

### During Validation Loop
```python
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        # ... validation logic ...

        # Extract scalars properly
        loss_val = loss.item()  # ✅
        num_tokens = (labels != -100).sum().item()  # ✅

        # Clean up tensors
        del input_ids, labels, logits, loss

        # Periodic cache clearing
        if i % 5 == 0:
            torch.cuda.empty_cache()
```

### After Validation
```python
# Delete result dict
del val_metrics

# Force cleanup
gc.collect()
torch.cuda.empty_cache()

# Back to training
model.train()
```

---

## 📋 Code Review Checklist

When reviewing PyTorch code for memory leaks, check:

- [ ] All `.any()` replaced with `.sum().item() > 0`
- [ ] All tensor extractions use `.detach().cpu()`
- [ ] All metrics use `.item()` or `.cpu().tolist()`
- [ ] Tensors deleted with `del` after use
- [ ] `torch.cuda.empty_cache()` called periodically
- [ ] `gc.collect()` before/after memory-intensive ops
- [ ] Auxiliary tensors detached before operations
- [ ] No `torch.where()` in no_grad contexts
- [ ] Indexing operations followed by `.detach()`
- [ ] No accumulation of intermediate tensors

---

## 🔍 Debugging Memory Leaks

### Step 1: Profile memory usage
```python
import torch
import gc

# Reset stats
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()

# Run suspect code
suspect_function()

# Check stats
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_mem:.2f} GB")
```

### Step 2: Count tensor retention
```python
def count_tensors():
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                count += 1
        except:
            pass
    return count

# Before
count_before = count_tensors()

# Run code
suspect_function()

# After
count_after = count_tensors()
print(f"Tensor retention: {count_after - count_before}")
```

### Step 3: Track memory over time
```python
mem_readings = []

for i in range(100):
    suspect_function()

    mem = torch.cuda.memory_allocated() / 1e6
    mem_readings.append(mem)

    if i % 10 == 0:
        print(f"Iteration {i}: {mem:.2f} MB")

# Check trend
trend = mem_readings[-10:] - mem_readings[:10]
if trend > 50:  # 50 MB increase
    print("⚠️  Memory leak detected!")
```

---

## 🎯 Quick Fix Templates

### Template 1: Fix tensor validation
```python
# Replace this pattern:
if (tensor < 0).any():
    # ...

# With this pattern:
invalid_count = (tensor < 0).sum().item()
if invalid_count > 0:
    # ...
```

### Template 2: Fix tensor extraction
```python
# Replace this pattern:
values = tensor[mask].cpu().tolist()

# With this pattern:
values = tensor[mask].detach().cpu().tolist()
```

### Template 3: Fix metric logging
```python
# Replace this pattern:
writer.add_scalar("metric", tensor.mean(), step)

# With this pattern:
writer.add_scalar("metric", tensor.mean().item(), step)
```

### Template 4: Fix loop cleanup
```python
# Add after loop body:
del tensor1, tensor2, tensor3
if i % 10 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

---

## 📞 Need Help?

**Memory leak still present after fixes?**

1. Run verification test: `python tests/verify_memory_leak_fix.py`
2. Check all `.any()`, `.where()`, `.nonzero()` calls
3. Verify all metrics use `.item()`
4. Add explicit `del` statements
5. Profile with `torch.cuda.memory_summary()`

**Reference Documentation**:
- Full fix details: `/mnt/d/ai/SLGA/docs/MEMORY_LEAK_FIXES_2025-10-28.md`
- Summary report: `/mnt/d/ai/SLGA/docs/MEMORY_LEAK_FIX_SUMMARY.md`

---

**Last Updated**: 2025-10-28
**Status**: All memory leaks fixed and verified ✅
