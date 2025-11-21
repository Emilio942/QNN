import torch
import time
import numpy as np
from qnn.thermal_adapter import ThermalAttention

def test_thermal_attention_performance():
    print("Testing ThermalAttention Performance...")
    
    batch_size = 32
    seq_len = 128
    d_model = 64
    n_samples = 5
    
    # Create module
    attention = ThermalAttention(temperature=1.0, n_samples=n_samples)
    
    # Create dummy inputs
    query = torch.randn(batch_size, 1, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
    
    # Warmup (Compilation)
    print("Warmup (JIT Compilation)...")
    start_time = time.time()
    out = attention(query, key, value)
    end_time = time.time()
    print(f"Warmup time: {end_time - start_time:.4f}s")
    
    assert out.shape == (batch_size, 1, d_model), f"Output shape mismatch: {out.shape}"
    
    # Benchmark
    n_iters = 10
    print(f"Running {n_iters} iterations...")
    start_time = time.time()
    for _ in range(n_iters):
        out = attention(query, key, value)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iters
    print(f"Average time per batch: {avg_time:.4f}s")
    print(f"Throughput: {batch_size / avg_time:.2f} samples/s")
    
    print("Success!")

if __name__ == "__main__":
    test_thermal_attention_performance()
