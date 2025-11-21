import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block, SamplingSchedule, sample_states, FactorSamplingProgram, BlockGibbsSpec
from thrml.models import CategoricalEBMFactor, FactorizedEBM
from thrml.models.discrete_ebm import CategoricalGibbsConditional

def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()

def main():
    print("Experiment: Attention as Energy Sampling")
    
    # 1. Define Problem
    # Sequence length N=4
    # Query dimension d=8
    N = 4
    d = 8
    
    key = jax.random.key(42)
    k_q, k_k, k_samp = jax.random.split(key, 3)
    
    # Random Query and Keys
    Q = jax.random.normal(k_q, (d,))
    K = jax.random.normal(k_k, (N, d))
    
    # Calculate Attention Logits (Scaled Dot Product)
    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.dot(K, Q) * scale
    
    print(f"Logits: {logits}")
    expected_probs = softmax(logits)
    print(f"Expected Softmax Probs: {expected_probs}")
    
    # 2. Define Categorical Model
    node = CategoricalNode() 
    block = Block([node])
    
    # Factor:
    # Weights shape: (n_nodes_in_block, n_states) -> (1, N)
    factor = CategoricalEBMFactor([block], logits.reshape(1, N))
    
    ebm = FactorizedEBM([factor])
    
    # 3. Setup Sampling Program
    # Sampler needs to know number of categories
    sampler = CategoricalGibbsConditional(N)
    
    # Spec
    # We might need to specify node_shape_dtypes for CategoricalNode
    # Let's try to provide it explicitly to be safe
    # CategoricalGibbsConditional usually works with uint8 or int32
    node_shape_dtypes = {
        CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8)
    }
    
    spec = BlockGibbsSpec(
        [block],           # free_blocks
        [],                # clamped_blocks
        node_shape_dtypes  # node_shape_dtypes
    )
    
    program = FactorSamplingProgram(
        spec,          # gibbs_spec
        [sampler],     # samplers
        ebm.factors,   # factors
        []             # observers
    )
    
    # 4. Run Sampling
    n_batch = 1000
    
    # Init state: Random integers 0..N-1
    # Shape: (nodes, batch) -> (1, n_batch)
    init_state = jax.random.randint(k_samp, (1, n_batch), 0, N).astype(jnp.uint8)
    
    schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=1)
    
    print("Sampling...")
    
    def run_chain(key, init_s):
        # init_s: (1,) (single chain state)
        # sample_states returns list of samples for observed blocks
        samples_list = sample_states(key, program, schedule, [init_s], [], [block])
        return samples_list[0]

    keys = jax.random.split(k_samp, n_batch)
    
    # vmap over batch
    # init_state is (1, 1000). We need to transpose to (1000, 1) for vmap to iterate over batch
    init_state_T = init_state.T 
    
    samples_batch = jax.vmap(run_chain)(keys, init_state_T)
    
    # samples_batch shape: (batch, n_samples, nodes) -> (1000, 100, 1)
    
    # Flatten
    samples_flat = samples_batch.reshape(-1)
    
    # 5. Analyze
    print(f"Samples shape: {samples_flat.shape}")
    
    # Calculate histogram
    counts = jnp.bincount(samples_flat, minlength=N, length=N)
    probs = counts / counts.sum()
    
    print(f"Results:")
    print(f"Expected: {expected_probs}")
    print(f"Empirical: {probs}")
    
    diff = jnp.abs(expected_probs - probs)
    print(f"Difference: {diff}")
    
    if jnp.all(diff < 0.05):
        print("SUCCESS: Categorical sampling matches Softmax!")
    else:
        print("WARNING: Discrepancy detected.")

if __name__ == "__main__":
    main()
