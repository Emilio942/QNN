import torch
import numpy as np
from qnn.thermal_adapter import ThermalAttention

def test_thermal_attention():
    print("Testing ThermalAttention...")
    
    # 1. Setup Data
    d = 2
    query = torch.tensor([[[1.0, 0.0]]]) # (1, 1, 2)
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # (1, 2, 2)
    value = torch.tensor([[[10.0, 10.0], [20.0, 20.0]]]) # (1, 2, 2)
    
    # 2. Standard Attention (Softmax)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    expected_context = torch.matmul(probs, value)
    
    print(f"Scores: {scores}")
    print(f"Probs: {probs}")
    print(f"Expected Context: {expected_context}")
    
    # 3. Thermal Attention
    # Use high n_samples for convergence
    attention = ThermalAttention(temperature=1.0, n_samples=500)
    
    thermal_context = attention(query, key, value)
    
    print(f"Thermal Context: {thermal_context}")
    
    # 4. Compare
    diff = torch.abs(expected_context - thermal_context)
    print(f"Difference: {diff}")
    
    # Allow some noise
    assert torch.all(diff < 2.0)
    print("Test Passed!")

if __name__ == "__main__":
    test_thermal_attention()
