import torch
import numpy as np
from qnn.thermal_adapter import ThermalLinear, ThermalAttention, TransformerToThermalAdapter, ThermalContext

def test_global_annealing():
    print("Testing Global Annealing (Central Context)...")
    
    # Setup Context
    ctx = ThermalContext(temperature=100.0) # Start Hot
    
    # Setup Layers
    adapter = TransformerToThermalAdapter() # Temp here doesn't matter for runtime
    
    # Linear Layer: y = x
    linear_base = torch.nn.Linear(1, 1, bias=False)
    linear_base.weight.data = torch.tensor([[1.0]])
    thermal_linear = ThermalLinear(linear_base, adapter, n_samples=10, context=ctx)
    
    # Attention Layer
    thermal_attn = ThermalAttention(n_samples=10, context=ctx)
    
    # Inputs
    x_linear = torch.tensor([[1.0]]) # Positive input
    
    # Attention Inputs (Key 1 is slightly better)
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[1.0], [2.0]]]) # Scores: 1, 2
    v = torch.tensor([[[0.0], [10.0]]]) # Values: 0, 10
    
    # --- Phase 1: High Temperature (T=100) ---
    print("\nPhase 1: High Temperature (T=100)")
    ctx.set_temperature(100.0)
    
    # Linear: Should be noisy/zero-mean (tanh(1/100) ~ 0)
    out_lin_hot = thermal_linear(x_linear)
    print(f"Linear Output (Hot): {out_lin_hot.item():.4f} (Expected ~0.0)")
    
    # Attention: Should be uniform (average of 0 and 10 -> 5)
    out_attn_hot = thermal_attn(q, k, v)
    print(f"Attention Output (Hot): {out_attn_hot.item():.4f} (Expected ~5.0)")
    
    # --- Phase 2: Low Temperature (T=0.1) ---
    print("\nPhase 2: Low Temperature (T=0.1)")
    ctx.set_temperature(0.1)
    
    # Linear: Should be saturated (tanh(1/0.1) = tanh(10) ~ 1)
    out_lin_cold = thermal_linear(x_linear)
    print(f"Linear Output (Cold): {out_lin_cold.item():.4f} (Expected ~1.0)")
    
    # Attention: Should be argmax (value 10)
    out_attn_cold = thermal_attn(q, k, v)
    print(f"Attention Output (Cold): {out_attn_cold.item():.4f} (Expected ~10.0)")
    
    # Validation
    success = True
    if abs(out_lin_hot.item()) > 0.5:
        print("FAILURE: Linear Hot not random enough.")
        success = False
    if abs(out_lin_cold.item() - 1.0) > 0.2:
        print("FAILURE: Linear Cold not saturated.")
        success = False
        
    if abs(out_attn_hot.item() - 5.0) > 2.0:
        print("FAILURE: Attention Hot not uniform.")
        success = False
    if abs(out_attn_cold.item() - 10.0) > 1.0:
        print("FAILURE: Attention Cold not argmax.")
        success = False
        
    if success:
        print("\nSUCCESS: Global Annealing works for both layers.")

if __name__ == "__main__":
    test_global_annealing()
