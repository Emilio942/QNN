import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def main():
    print("Setting up Ising Model...")
    # 1. Define the Graph
    # A simple chain of 5 spins: 0-1-2-3-4
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i+1]) for i in range(4)]
    
    # 2. Define Parameters
    # Zero bias (no preference for +1 or -1 locally)
    biases = jnp.zeros((5,))
    # Ferromagnetic coupling (neighbors want to align)
    weights = jnp.ones((4,)) * 1.0 
    beta = jnp.array(1.0) # Inverse temperature
    
    model = IsingEBM(nodes, edges, biases, weights, beta)

    # 3. Define Sampling Program
    # Checkerboard pattern for parallel sampling (Block Gibbs)
    # Block 0: Nodes 0, 2, 4
    # Block 1: Nodes 1, 3
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    # 4. Initialize State
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    
    # Hinton initialization (random start)
    # Try with batch size 10
    init_state = hinton_init(k_init, model, free_blocks, (10,))
    print(f"Init state shape (batch=10): {init_state[0].shape}")
    
    # 5. Run Sampling
    print("Starting sampling...")
    schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

    # We want to observe all nodes together
    # We need to replicate init_state if we want to run multiple chains?
    # Or does sample_states handle the batch dimension automatically?
    
    # If init_state has batch dim, sample_states should handle it.
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # samples[0] shape should be (n_samples, nodes, batch) or similar
    spins = samples[0]
    print(f"Sampling done. Shape: {spins.shape}")
    
    # 6. Analyze Results
    # Calculate mean magnetization
    mean_mag = jnp.mean(spins, axis=0)
    print(f"Mean Magnetization per site: {mean_mag}")
    
    # Calculate correlation between neighbors (should be high for ferromagnetic)
    # spins is boolean (True/False), convert to +1/-1
    spins_pm = jnp.where(spins, 1.0, -1.0)
    
    print("\nNeighbor Correlations:")
    for i in range(4):
        corr = jnp.mean(spins_pm[:, i] * spins_pm[:, i+1])
        print(f"Spin {i} - Spin {i+1}: {corr:.4f}")

if __name__ == "__main__":
    main()
