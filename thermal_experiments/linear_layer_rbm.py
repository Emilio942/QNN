import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def main():
    print("Experiment: Linear Layer as RBM")
    
    # 1. Define Architecture
    # Input: 4 neurons (Visible)
    # Output: 2 neurons (Hidden)
    n_in = 4
    n_out = 2
    
    visible_nodes = [SpinNode() for _ in range(n_in)]
    hidden_nodes = [SpinNode() for _ in range(n_out)]
    all_nodes = visible_nodes + hidden_nodes
    
    # 2. Define Weights (Couplings) and Biases
    key = jax.random.key(123)
    k_w, k_b = jax.random.split(key, 2)
    
    # Random weights between -1 and 1
    W = jax.random.uniform(k_w, (n_in, n_out), minval=-1.0, maxval=1.0)
    # Random biases for hidden units
    b = jax.random.uniform(k_b, (n_out,), minval=-0.5, maxval=0.5)
    
    print(f"Weights W:\n{W}")
    print(f"Biases b:\n{b}")
    
    # 3. Construct Ising Model
    # Edges connect every visible node to every hidden node (Bipartite)
    edges = []
    weights_list = []
    
    # Flatten weights for the edge list
    for i in range(n_in):
        for j in range(n_out):
            edges.append((visible_nodes[i], hidden_nodes[j]))
            weights_list.append(W[i, j])
            
    weights_flat = jnp.array(weights_list)
    
    # Biases: First n_in are 0 (we clamp them anyway), last n_out are b
    biases_flat = jnp.concatenate([jnp.zeros(n_in), b])
    
    beta = jnp.array(1.0)
    model = IsingEBM(all_nodes, edges, biases_flat, weights_flat, beta)
    
    # 4. Define Input Vector (Clamped)
    # Let's pick a random binary input
    x_input = jnp.array([1, -1, 1, 1]) # Spin values (+1/-1)
    # Convert to boolean for thrml (True=+1, False=-1)
    x_bool = x_input > 0
    
    print(f"Input x: {x_input}")
    
    # 5. Setup Sampling Program
    # We clamp the visible block and sample the hidden block
    visible_block = Block(visible_nodes)
    hidden_block = Block(hidden_nodes)
    
    # In thrml, clamped_blocks are fixed. free_blocks are sampled.
    program = IsingSamplingProgram(model, free_blocks=[hidden_block], clamped_blocks=[visible_block])
    
    # 6. Run Sampling
    # We need to provide the initial state for the clamped block
    # The init_state list corresponds to [free_blocks..., clamped_blocks...] order in some contexts,
    # but let's check sample_states signature.
    # sample_states(key, program, schedule, init_state, clamped_state, observe_blocks)
    
    # init_state is for free blocks.
    # clamped_state is for clamped blocks.
    
    k_samp = jax.random.key(999)
    
    # Random init for hidden (free)
    # Shape: (batch, nodes)
    init_hidden = jax.random.bernoulli(k_samp, 0.5, (1000, n_out)) 
    
    # Fixed input for visible (clamped)
    # Shape: (batch, nodes)
    clamped_visible = jnp.tile(x_bool, (1000, 1))
    
    schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=1)
    
    print(f"Init Hidden Shape: {init_hidden.shape}")
    print(f"Clamped Visible Shape: {clamped_visible.shape}")
    
    print("Sampling...")
    
    # We need to vmap over the batch dimension because sample_states works on single chains
    # (or at least expects shapes that allow concatenation along axis 0, which (nodes,) does)
    
    # Define a single-chain sampling function
    def run_single_chain(key, init_h, clamped_v):
        # init_state expects a list of arrays (one per free block)
        # clamped_state expects a list of arrays (one per clamped block)
        return sample_states(
            key, 
            program, 
            schedule, 
            [init_h],     # Free blocks init
            [clamped_v],  # Clamped blocks values
            [hidden_block] # What to observe
        )

    # Split keys for vmap
    keys = jax.random.split(k_samp, 1000)
    
    # Run vmap
    # Output shape will be (batch, n_samples, nodes) because vmap adds batch dim at front
    # sample_states returns (samples_list, final_state_list)
    # We only care about samples_list[0]
    
    # Note: sample_states returns a list of PyTrees (one per observed block)
    # It does NOT return the final state in the return value (unlike sample_with_observation)
    
    # Let's wrap it to return just the hidden samples
    def run_single_chain_wrapper(key, init_h, clamped_v):
        samples_list = run_single_chain(key, init_h, clamped_v)
        return samples_list[0] # The first observed block (hidden)

    hidden_samples_batch = jax.vmap(run_single_chain_wrapper)(keys, init_hidden, clamped_visible)
    
    print(f"Samples Batch Shape: {hidden_samples_batch.shape}")
    # Shape: (batch, n_samples, nodes)
    
    # Flatten to (total_samples, nodes)
    hidden_samples_flat = hidden_samples_batch.reshape(-1, n_out)
    
    # 7. Compare with Analytical Expectation
    # P(h_j=1) = sigmoid(2 * (sum_i W_ij v_i + b_j)) 
    # Factor of 2 comes from spin difference (+1 vs -1 is distance 2) usually, 
    # but let's check thrml definition.
    # Ising Energy: E = - sum J s_i s_j - sum h s_i
    # Delta E = E(s_j=-1) - E(s_j=+1) = 2 * (sum J s_i + h)
    # P(s_j=1) = 1 / (1 + exp(-beta * Delta E)) = sigmoid(2 * beta * effective_field)
    
    effective_field = jnp.dot(x_input, W) + b
    expected_prob = sigmoid(2 * beta * effective_field)
    
    empirical_prob = jnp.mean(hidden_samples_flat, axis=0)
    
    print("\nResults:")
    print(f"Effective Field: {effective_field}")
    print(f"Expected P(h=1): {expected_prob}")
    print(f"Empirical P(h=1): {empirical_prob}")
    
    diff = jnp.abs(expected_prob - empirical_prob)
    print(f"Difference: {diff}")
    
    if jnp.all(diff < 0.05):
        print("\nSUCCESS: Sampling matches theoretical RBM prediction!")
    else:
        print("\nWARNING: Discrepancy detected.")

if __name__ == "__main__":
    main()
