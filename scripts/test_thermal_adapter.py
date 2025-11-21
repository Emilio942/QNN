import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, sample_states, SamplingSchedule
from qnn.thermal_adapter import TransformerToThermalAdapter

def test_linear_adapter():
    print("Testing Linear Layer Adapter...")
    
    # 1. Setup PyTorch Layer
    n_in = 4
    n_out = 2
    layer = nn.Linear(n_in, n_out)
    
    # Set specific weights for predictability
    # W = [[1, 0, 1, 0], [0, 1, 0, 1]]
    # b = [0, 0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))
        layer.bias.zero_()
        
    print(f"PyTorch Weights:\n{layer.weight}")
    
    # 2. Setup Adapter
    adapter = TransformerToThermalAdapter(temperature=1.0)
    
    # Create Nodes
    input_nodes = [SpinNode() for i in range(n_in)]
    output_nodes = [SpinNode() for i in range(n_out)]
    
    # Register Nodes
    # Input is clamped (condition), Output is free (sampled)
    adapter.register_nodes(input_nodes, is_clamped=True)
    adapter.register_nodes(output_nodes, is_clamped=False)
    
    # Convert Layer
    adapter.convert_linear_layer(layer, input_nodes, output_nodes)
    
    # 3. Build Program
    program = adapter.build_sampling_program()
    print("Program built successfully.")
    
    # 4. Run Sampling
    # Define Input State (Clamped)
    # Let's set input to [1, -1, 1, -1] (True, False, True, False)
    input_vals = jnp.array([True, False, True, False]) # Boolean for SpinNode
    
    # Define Init State for Output (Free)
    # Random init
    output_init = jnp.array([True, True]) # Dummy init
    
    # Schedule
    schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=1)
    
    # Run
    key = jax.random.PRNGKey(0)
    
    # sample_states expects lists of arrays
    # init_state: [output_init]
    # clamped_state: [input_vals]
    # observe_blocks: [output_block] -> adapter.free_blocks[0]
    
    # We need to handle batching if we want multiple samples.
    # Let's run a single chain first.
    
    samples_list = sample_states(
        key,
        program,
        schedule,
        [output_init],
        [input_vals],
        adapter.free_blocks
    )
    
    # samples_list is a list of arrays (one per observed block)
    # Shape: (n_samples, nodes)
    output_samples = samples_list[0]
    
    print(f"Output Samples Shape: {output_samples.shape}")
    
    # Calculate mean
    # Convert boolean to float (+1/-1)
    # True -> 1.0, False -> 0.0 (Wait, thrml uses 0/1 for boolean?)
    # SpinNode uses boolean.
    # Usually True=+1, False=-1 in Ising context, but let's check.
    # If we cast to float, True is 1.0, False is 0.0.
    # If we want +1/-1, we do 2*x - 1.
    
    output_float = output_samples.astype(jnp.float32)
    mean_val = jnp.mean(output_float, axis=0)
    
    print(f"Mean Output (0/1): {mean_val}")
    
    # Expected:
    # Input: [1, -1, 1, -1] -> [1, 0, 1, 0] in 0/1?
    # Wait, if SpinNode is boolean, how does IsingEBM interpret it?
    # Usually IsingEBM maps True->+1, False->-1.
    # My input [True, False, True, False] corresponds to [+1, -1, +1, -1].
    # W @ x = [2, -2].
    # P(h=1) ~ sigmoid(2*beta*2) ~ 0.98.
    # So Output 0 should be True (1.0).
    # Output 1 should be False (0.0).
    
    out0 = mean_val[0]
    out1 = mean_val[1]
    
    print(f"Output 0: {out0} (Expected ~1.0)")
    print(f"Output 1: {out1} (Expected ~0.0)")
    
    assert out0 > 0.9
    assert out1 < 0.1
    print("Test Passed!")

if __name__ == "__main__":
    test_linear_adapter()
