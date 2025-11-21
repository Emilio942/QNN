import torch
import numpy as np
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def test_determinism():
    print("Testing Deterministic Sampling (RNG State Management)...")
    
    # Setup
    linear = torch.nn.Linear(1, 1, bias=False)
    linear.weight.data = torch.tensor([[0.0]]) # h=0 -> 50/50 chance
    adapter = TransformerToThermalAdapter(temperature=1.0)
    
    # Case 1: Same Seed -> Same Output
    print("\nCase 1: Same Seed (42)")
    ctx1 = ThermalContext(seed=42)
    layer1 = ThermalLinear(linear, adapter, n_samples=10, context=ctx1)
    
    ctx2 = ThermalContext(seed=42)
    layer2 = ThermalLinear(linear, adapter, n_samples=10, context=ctx2)
    
    x = torch.tensor([[0.0]])
    
    out1 = layer1(x)
    out2 = layer2(x)
    
    print(f"Out1: {out1.item():.4f}")
    print(f"Out2: {out2.item():.4f}")
    
    if abs(out1.item() - out2.item()) < 1e-6:
        print("SUCCESS: Outputs match.")
    else:
        print("FAILURE: Outputs differ despite same seed.")
        
    # Case 2: Different Seed -> Different Output (likely)
    print("\nCase 2: Different Seed (42 vs 99)")
    ctx3 = ThermalContext(seed=99)
    layer3 = ThermalLinear(linear, adapter, n_samples=10, context=ctx3)
    
    out3 = layer3(x)
    print(f"Out3: {out3.item():.4f}")
    
    if abs(out1.item() - out3.item()) > 1e-6:
        print("SUCCESS: Outputs differ as expected.")
    else:
        print("WARNING: Outputs match (could be chance, but unlikely with 10 samples).")
        
    # Case 3: Sequential Calls -> Different Output (State Update)
    print("\nCase 3: Sequential Calls (State Update)")
    # Increase samples to reduce chance of collision
    # Or check internal key state
    
    key_before = ctx1.key
    out1_next = layer1(x)
    key_after = ctx1.key
    
    print(f"Out1 (Call 2): {out1_next.item():.4f}")
    
    # Check if key changed
    if not np.array_equal(key_before, key_after):
         print("SUCCESS: Internal RNG key updated.")
    else:
         print("FAILURE: Internal RNG key did not update.")
         
    if abs(out1.item() - out1_next.item()) > 1e-6:
        print("SUCCESS: Sequential calls produce different results.")
    else:
        print("WARNING: Sequential calls match (could be chance).")

if __name__ == "__main__":
    test_determinism()
