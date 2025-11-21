import torch
import numpy as np
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter

def test_thermal_linear_fidelity():
    print("Testing ThermalLinear Input Fidelity...")
    
    # Create a dummy linear layer
    # y = x (Identity)
    linear = torch.nn.Linear(1, 1, bias=False)
    linear.weight.data = torch.tensor([[1.0]])
    
    adapter = TransformerToThermalAdapter(temperature=0.1) # Low temp -> Deterministic-ish
    thermal_layer = ThermalLinear(linear, adapter, n_samples=1000)
    
    # Test Case 1: Large Positive Input
    # x = 10.0 -> h_eff = 10.0 -> P(s=1) ~ 1.0 -> Output ~ 1.0
    x_large_pos = torch.tensor([[10.0]])
    out_large_pos = thermal_layer(x_large_pos)
    print(f"Input: 10.0 -> Output: {out_large_pos.item():.4f} (Expected ~1.0)")
    
    # Test Case 2: Large Negative Input
    # x = -10.0 -> h_eff = -10.0 -> P(s=-1) ~ 1.0 -> Output ~ -1.0
    x_large_neg = torch.tensor([[-10.0]])
    out_large_neg = thermal_layer(x_large_neg)
    print(f"Input: -10.0 -> Output: {out_large_neg.item():.4f} (Expected ~-1.0)")
    
    # Test Case 3: Small Input (Linear Region)
    # x = 0.5. T=0.1 -> beta=10. h=0.5. 2*beta*h = 10. Sigmoid(10) ~ 1.
    # Wait, T=0.1 is very cold. It acts like sign(x).
    # Let's try T=1.0.
    
    print("\nSwitching to T=1.0 for linear region test...")
    thermal_layer.adapter.temperature = 1.0
    # x = 0.5 -> h=0.5. beta=1. 2*beta*h = 1.0.
    # P(s=1) = sigmoid(1.0) = 0.731
    # Mean s = 2*p - 1 = 2*0.731 - 1 = 0.462
    
    x_small = torch.tensor([[0.5]])
    out_small = thermal_layer(x_small)
    print(f"Input: 0.5 -> Output: {out_small.item():.4f} (Expected ~0.462)")
    
    # Test Case 4: Zero Input
    # x = 0.0 -> h=0.0 -> P(s=1) = 0.5 -> Output ~ 0.0
    x_zero = torch.tensor([[0.0]])
    out_zero = thermal_layer(x_zero)
    print(f"Input: 0.0 -> Output: {out_zero.item():.4f} (Expected ~0.0)")
    
    # Validation
    if abs(out_large_pos.item() - 1.0) < 0.1 and abs(out_large_neg.item() + 1.0) < 0.1:
        print("SUCCESS: Large inputs saturate correctly.")
    else:
        print("FAILURE: Saturation check failed.")
        
    if abs(out_zero.item()) < 0.1:
        print("SUCCESS: Zero input yields zero mean.")
    else:
        print("FAILURE: Zero input check failed.")

if __name__ == "__main__":
    test_thermal_linear_fidelity()
