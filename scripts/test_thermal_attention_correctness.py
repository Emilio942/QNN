import torch
import numpy as np
from qnn.thermal_adapter import ThermalAttention

def test_thermal_attention_correctness():
    print("Testing ThermalAttention Correctness...")
    
    # Setup
    d_model = 4
    seq_len = 3
    batch_size = 1
    
    # We want the model to attend to the 2nd element (index 1)
    # Q = [1, 0, 0, 0]
    # K = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]
    # Q @ K.T = [0, 1, 0] -> Index 1 has highest score.
    
    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]) # (1, 1, 4)
    
    key = torch.tensor([[
        [0.0, 1.0, 0.0, 0.0], # Score 0
        [10.0, 0.0, 0.0, 0.0], # Score 10 (High match)
        [0.0, 0.0, 1.0, 0.0]  # Score 0
    ]]) # (1, 3, 4)
    
    value = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0], # V0
        [2.0, 2.0, 2.0, 2.0], # V1 (Target)
        [3.0, 3.0, 3.0, 3.0]  # V2
    ]]) # (1, 3, 4)
    
    # Use low temperature to make it almost deterministic (argmax)
    # T=0.1
    attention = ThermalAttention(temperature=0.1, n_samples=50)
    
    print("Running ThermalAttention...")
    output = attention(query, key, value)
    
    print(f"Output: {output}")
    
    expected = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]])
    
    # Check if close
    diff = torch.abs(output - expected).mean()
    print(f"Difference from expected (V1): {diff.item()}")
    
    if diff < 0.1:
        print("SUCCESS: Attended to the correct key (V1).")
    else:
        print("FAILURE: Did not attend to the correct key.")
        # Print probabilities/logits for debugging if we could access them
        # But we can infer from output.
        
if __name__ == "__main__":
    test_thermal_attention_correctness()
