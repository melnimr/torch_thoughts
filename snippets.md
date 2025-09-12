## Asserts
```python
 D = _gemm_fwd(A, B)
D_ref = torch.matmul(A, B)
assert torch.allclose(D, D_ref, atol=1e-03)
```
