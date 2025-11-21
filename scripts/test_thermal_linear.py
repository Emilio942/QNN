import torch
import torch.nn as nn
import numpy as np
from qnn.thermal_adapter import TransformerToThermalAdapter, ThermalLinear

def test_thermal_linear():
    print("Testing ThermalLinear Layer...")
    
    # 1. Setup PyTorch Layer
    n_in = 4
    n_out = 2
    layer = nn.Linear(n_in, n_out)
    
    # Set specific weights
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))
        layer.bias.zero_()
        
    # 2. Create ThermalLinear
    adapter = TransformerToThermalAdapter(temperature=1.0)
    thermal_layer = ThermalLinear(layer, adapter, n_samples=100)
    
    # 3. Run Forward Pass
    # Input: [1, -1, 1, -1] -> [1, 0, 1, 0] (boolean)
    # Note: ThermalLinear currently thresholds input > 0.
    # So -1 becomes False (0).
    # W @ x_bool = [1*1 + 0*0 + 1*1 + 0*0, ...] = [2, 0]
    # Wait, if input is 0/1.
    # W = [[1, 0, 1, 0], [0, 1, 0, 1]]
    # x = [1, 0, 1, 0]
    # Wx = [2, 0]
    # P(h=1) = sigmoid(2*beta*2) ~ 0.98 for h0.
    # P(h=1) = sigmoid(2*beta*0) = 0.5 for h1.
    
    # Let's use input [1, 1, 1, 1] -> [1, 1, 1, 1]
    # Wx = [2, 2]
    # Both should be high prob.
    
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
    # Batch size 2
    
    print(f"Input:\n{x}")
    
    output = thermal_layer(x)
    
    print(f"Output:\n{output}")
    
    # Check Batch 0
    # x[0] -> [1, 0, 1, 0] (boolean) -> [+1, -1, +1, -1] (spin)
    # Wx[0] = 1*1 + 0*-1 + 1*1 + 0*-1 = 2 -> Sigmoid(4) ~ 0.98 -> Scaled ~ 0.96
    # Wx[1] = 0*1 + 1*-1 + 0*1 + 1*-1 = -2 -> Sigmoid(-4) ~ 0.02 -> Scaled ~ -0.96
    out0 = output[0]
    print(f"Batch 0: {out0} (Expected ~0.96, ~-0.96)")
    
    assert out0[0] > 0.8
    assert out0[1] < -0.8
    
    # Check Batch 1
    # x[1] -> [1, 1, 1, 1]
    # Wx = [2, 2]
    # h0 ~ 0.98, h1 ~ 0.98 -> Scaled ~ 0.96
    out1 = output[1]
    print(f"Batch 1: {out1} (Expected ~0.96, ~0.96)")
    
    assert out1[0] > 0.8
    assert out1[1] > 0.8
    
    print("Test Passed!")

if __name__ == "__main__":
    test_thermal_linear()
