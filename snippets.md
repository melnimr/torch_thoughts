## Asserts
```python
 D = _gemm_fwd(A, B)
D_ref = torch.matmul(A, B)
assert torch.allclose(D, D_ref, atol=1e-03)
```


## Tensor Asserts
```python

if __name__ == "__main__":
    device = torch.device("cuda")

    # check correctness
    torch.manual_seed(1)
    GMEM_M, GMEM_N, GMEM_K = 8192, 8192, 8192

    A = torch.rand(GMEM_M, GMEM_K, device=device, dtype=torch.float16)
    B = torch.rand(GMEM_K, GMEM_N, device=device, dtype=torch.float16)
    D = _gemm_fwd(A, B)
    D_ref = torch.matmul(A, B)
    assert torch.allclose(D, D_ref, atol=1e-03)

    # quick benchmark
    for i in range(10):
        _ = _gemm_fwd(A, B)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(10):
        _ = _gemm_fwd(A, B)
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000
    tflops = (2 * GMEM_M * GMEM_N * GMEM_K) / 1e12
    tflops_per_sec = tflops / (elapsed_s / 10)
    print(f"TFLOPS: {tflops_per_sec:.1f}, time: {elapsed_ms:.4f} ms")


```
