## Float8 Support

When `torch.float8_e4m3fn` is available (CUDA >= 8.0, H100+), Pynerve
operations can accept float8 inputs for memory-efficient training.

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.float8_e4m3fn):
    diagram = nt.vr_persistence(points_fp8, max_dim=1)
```


### Memory savings

Float8 uses 1 byte per value compared to 4 bytes for float32 and 2 bytes for float16, making it one quarter the size of float32 and half the size of float16. Float8 reduces diagram storage by 4x vs float32.

### Precision behavior

- Persistence computation internally upcasts to float32 for arithmetic
- Stores result in float8 when the input is float8
- Distance computation remains in float32 for accuracy
- Backward pass uses float32 accumulation regardless of input dtype

### Limitations

- Float8 supported only on CUDA (H100 and later via `torch.float8_e4m3fn`)
- Backward pass uses float32 accumulation regardless of input dtype
- Diagrams with very close birth/death values (< 1e-3) may lose precision
- TorchScript export not supported with float8 inputs


## FP8 formats

The e4m3 format uses 4 exponent bits and 3 mantissa bits, giving a range of +/-448 and relative precision of approximately 0.05. The e5m2 format uses 5 exponent bits and 2 mantissa bits, giving a range of +/-57344 and relative precision of approximately 0.25.

e4m3 is preferred for activations (better precision). e5m2 is preferred for gradients (wider range).

## When to use FP8

For large model training, FP8 offers a 4x memory reduction and is recommended for models with over 1 billion parameters. When batch size is limited, FP8 enables roughly 2x larger batches by freeing up memory. For inference deployment, FP8 produces models about 4x smaller and should be applied via quantization after training. For use cases with sensitivity to precision, FP8 provides minimal benefit and FP16 or FP32 remain the better choice.

## Usage patterns

```python
# Pattern 1: Automatic mixed precision
with torch.amp.autocast(device_type="cuda", dtype=torch.float8_e4m3fn):
    diagram = nt.vr_persistence(points_fp8, max_dim=1)

# Pattern 2: Explicit FP8 storage
points_fp8 = points.to(torch.float8_e4m3fn)
diagram = nt.vr_persistence(points_fp8, max_dim=1)
# Diagram stored in FP8, computation upcast to FP32

# Pattern 3: FP8 for intermediate features
with torch.amp.autocast(dtype=torch.float8_e4m3fn):
    diagram = nt.vr_persistence(points, max_dim=1)
    features = diagram.to_persistence_image(resolution=64)
    # Image computed in FP8
```

## Implementation

```cpp
// Internally upcasts FP8 to FP32 for computation
at::Tensor vr_persistence_fp8(at::Tensor points) {
    TORCH_CHECK(points.is_cuda(), "FP8 requires CUDA");
    TORCH_CHECK(points.scalar_type() == at::kFloat8_e4m3fn,
                "Expected float8_e4m3fn input");

    // Upcast to float32 for arithmetic
    at::Tensor points_f32 = points.to(at::kFloat);

    // Compute persistence at float32 precision
    auto result = compute_persistence(points_f32);

    // Downcast result to float8
    if (store_fp8) {
        return result.to(at::kFloat8_e4m3fn);
    }
    return result;
}
```

## Precision benchmarks

```python
# Benchmark precision vs memory trade-off
from pynerve.validation import benchmark_fp_precision

for dtype in [torch.float32, torch.float16, torch.float8_e4m3fn]:
    bm = benchmark_fp_precision(
        n_points=5000, max_dim=2, dtype=dtype
    )
    print(f"{dtype}:")
    print(f"  Memory: {bm.memory_mb:.1f} MB")
    print(f"  Time: {bm.time_ms:.1f} ms")
    print(f"  Max error vs FP32: {bm.max_error:.2e}")
```

Expected results:

Float32 uses hundreds of megabytes of memory, takes 12 ms on GPU, and serves as the baseline for error. Float16 uses hundreds of megabytes of memory, takes 8 ms, with a max error of 1e-3 relative to float32. Float8 uses tens of megabytes of memory, takes 6 ms, with a max error of 1e-2.


## FAQ

**Q: Does FP8 work on all GPU architectures?**
A: No. FP8 (e4m3 and e5m2) require CUDA compute capability >= 8.9 (Ada Lovelace / Hopper). On older GPUs, fall back to FP16.

**Q: Does FP8 affect the quality of persistence results?**
A: The computation is internally upcast to FP32, so the result quality is identical to FP32. Only storage uses FP8. If you also want FP8 computation (for speed), expect ~1e-2 relative error.

**Q: Can I use FP8 with TorchScript?**
A: No. TorchScript does not support FP8 dtypes. Use FP16 or FP32 for TorchScript exported models. This limitation is tracked in the PyTorch roadmap.


### Cross-references

- `pynerve.cuda`: GPU infrastructure
- `pynerve.torch`: PyTorch integration
- `pynerve.optimization.gpu`: GPU optimizer with mixed precision
- `pynerve.encoders.gpu`: Mixed precision encoder kernels
