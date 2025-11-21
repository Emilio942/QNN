import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, CategoricalNode, Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models import SpinEBMFactor, CategoricalEBMFactor, FactorizedEBM, DiscreteEBMFactor, IsingEBM
from thrml.models.discrete_ebm import SpinGibbsConditional, CategoricalGibbsConditional

class ThermalContext:
    """
    Manages the JAX PRNGKey state for deterministic sampling across the model.
    Acts as a central source of randomness and global parameters (Temperature).
    """
    def __init__(self, seed: int = 42, temperature: float = 1.0):
        self.key = jax.random.PRNGKey(seed)
        self.temperature = temperature
        
    def next_key(self):
        """Returns a new key and updates internal state."""
        self.key, subkey = jax.random.split(self.key)
        return subkey
    
    def set_temperature(self, temp: float):
        """Updates the global temperature."""
        self.temperature = temp

class TransformerToThermalAdapter:
    """
    Adapts PyTorch Transformer layers to Thermodynamic Energy-Based Models (EBMs)
    using the Extropic 'thrml' library.
    """
    
    def __init__(self, temperature=1.0, sparsity_threshold=0.0):
        self.temperature = temperature
        self.sparsity_threshold = sparsity_threshold
        self.factors = []
        self.all_nodes = []
        self.free_blocks = []
        self.clamped_blocks = []
        self.node_metadata = {} # Store metadata like n_categories
        
    def register_nodes(self, nodes: list, is_clamped=False, **kwargs):
        """Registers nodes to the adapter for tracking."""
        self.all_nodes.extend(nodes)
        block = Block(nodes)
        if is_clamped:
            self.clamped_blocks.append(block)
        else:
            self.free_blocks.append(block)
            
        # Store metadata
        for node in nodes:
            self.node_metadata[node] = kwargs

    def convert_linear_layer(self, layer: nn.Linear, input_nodes: list[SpinNode], output_nodes: list[SpinNode]) -> list[SpinEBMFactor]:
        """
        Converts a PyTorch Linear layer into a SpinEBMFactor (RBM-like coupling).
        
        Args:
            layer: The PyTorch nn.Linear layer.
            input_nodes: List of SpinNodes representing the input (visible units).
            output_nodes: List of SpinNodes representing the output (hidden units).
            
        Returns:
            list[SpinEBMFactor]: The thermodynamic factors defining the interaction.
        """
        # 1. Extract Weights and Biases
        W_torch = layer.weight.detach()
        b_torch = layer.bias.detach()
        
        W_numpy = W_torch.numpy() # (n_out, n_in)
        b_numpy = b_torch.numpy() # (n_out,)
        
        n_in = len(input_nodes)
        n_out = len(output_nodes)
        
        # 2. Construct Edges and Weights
        # Vectorized construction to avoid slow Python loops
        import itertools
        
        if self.sparsity_threshold > 0:
            # Sparse Construction
            # Find indices where |w| > threshold
            # W_numpy is (n_out, n_in)
            rows, cols = np.where(np.abs(W_numpy) > self.sparsity_threshold)
            
            # Extract weights
            edge_weights = W_numpy[rows, cols]
            
            # Construct edges
            # rows -> output_nodes indices
            # cols -> input_nodes indices
            edges = []
            # We still iterate, but only over non-zero elements.
            # For high sparsity, this is much faster than iterating all.
            # If sparsity is low (dense), this is slower than itertools.product.
            for r, c in zip(rows, cols):
                edges.append((output_nodes[r], input_nodes[c]))
                
        else:
            # Dense Construction (Vectorized)
            # W_numpy is (n_out, n_in)
            # flatten() yields row-major: (out_0, in_0), (out_0, in_1)...
            edge_weights = W_numpy.flatten()
            
            # Create edges in the same order: (out_i, in_j)
            # itertools.product(output_nodes, input_nodes) produces exactly this order.
            edges = list(itertools.product(output_nodes, input_nodes))
                
        # 3. Construct Biases
        # Nodes order: input_nodes + output_nodes
        all_layer_nodes = input_nodes + output_nodes
        
        # Biases: 0 for input, b for output
        biases = np.concatenate([np.zeros(n_in), b_numpy])
        
        # 4. Create IsingEBM
        # We use a temporary IsingEBM to generate the factors
        # Note: IsingEBM expects JAX arrays for weights/biases
        # Use adapter temperature
        beta_val = 1.0 / self.temperature
        ising_model = IsingEBM(
            nodes=all_layer_nodes,
            edges=edges,
            biases=jnp.array(biases), 
            weights=jnp.array(edge_weights), 
            beta=jnp.array(beta_val) 
        )
        
        # Extract factors
        new_factors = ising_model.factors
        self.factors.extend(new_factors)
        return new_factors

    def build_sampling_program(self) -> FactorSamplingProgram:
        """
        Constructs the THRML sampling program from registered factors and nodes.
        """
        # Create EBM
        ebm = FactorizedEBM(self.factors)
        
        # Samplers: One per free block.
        samplers = []
        for block in self.free_blocks:
            node = block.nodes[0]
            if isinstance(node, SpinNode):
                samplers.append(SpinGibbsConditional())
            elif isinstance(node, CategoricalNode):
                # Look up n_categories from metadata
                meta = self.node_metadata.get(node, {})
                n_categories = meta.get('n_categories')
                if n_categories is None:
                    raise ValueError(f"CategoricalNode {node} missing 'n_categories' metadata.")
                samplers.append(CategoricalGibbsConditional(n_categories=n_categories))
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
        
        spec = BlockGibbsSpec(
            free_super_blocks=self.free_blocks,
            clamped_blocks=self.clamped_blocks
        )
        
        program = FactorSamplingProgram(
            gibbs_spec=spec,
            samplers=samplers,
            factors=ebm.factors,
            other_interaction_groups=[]
        )
        
        return program

    def convert_attention_logits(self, query: jnp.ndarray, key_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the energy logits for attention sampling.
        E = - (Q @ K.T) / sqrt(d)
        """
        d = query.shape[-1]
        scale = 1.0 / jnp.sqrt(d)
        logits = jnp.dot(key_matrix, query) * scale
        return logits

    def create_attention_factor(self, logits: jnp.ndarray, target_node: CategoricalNode) -> CategoricalEBMFactor:
        """
        Creates a CategoricalEBMFactor from attention logits.
        This is a 'runtime' factor, as logits depend on the input.
        
        Args:
            logits: JAX array of shape (n_heads, seq_len) or similar.
                    For a single token generation, it's usually (vocab_size,) or (seq_len,).
            target_node: The CategoricalNode representing the attention choice.
            
        Returns:
            CategoricalEBMFactor
        """
        # Logits are directly the energy terms (negative log probs).
        # E(x) = -logits[x]
        # So we pass logits as the potential table.
        
        block = Block([target_node])
        # CategoricalEBMFactor expects weights of shape (n_states,) for a single node block?
        # Or (1, n_states)?
        # In attention_energy.py we used (1, N).
        N = logits.shape[0]
        factor = CategoricalEBMFactor([block], logits.reshape(1, N))
        self.factors.append(factor)
        return factor

# --- Optimized JAX Kernels for Linear (Spin Activation) ---

def _sample_spin_activation_kernel(key, biases, n_samples, temperature):
    """
    Samples spin states given local fields (biases).
    biases: (n_features,)
    """
    n_features = biases.shape[0]
    
    # 1. Define Nodes
    nodes = [SpinNode() for _ in range(n_features)]
    block = Block(nodes)
    
    # 2. Define Factor (Ising with only biases)
    beta = 1.0 / temperature
    
    # Workaround: IsingEBM might fail with empty edges.
    # We add dummy self-loops with 0 weight.
    # s_i * s_i = 1, so this adds a constant energy term which doesn't affect sampling.
    edges = [(nodes[i], nodes[i]) for i in range(n_features)]
    edge_weights = jnp.zeros(n_features)
    
    ising = IsingEBM(
        nodes=nodes,
        edges=edges,
        biases=biases,
        weights=edge_weights,
        beta=jnp.array(beta)
    )
    
    # 3. Program
    ebm = FactorizedEBM(ising.factors)
    sampler = SpinGibbsConditional()
    spec = BlockGibbsSpec(free_super_blocks=[block], clamped_blocks=[])
    
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=ebm.factors,
        other_interaction_groups=[]
    )
    
    # 4. Schedule
    schedule = SamplingSchedule(n_warmup=10, n_samples=n_samples, steps_per_sample=1)
    
    # 5. Init & Sample
    init_state = jax.random.bernoulli(key, 0.5, (n_features,))
    
    samples_list = sample_states(
        key,
        program,
        schedule,
        [init_state],
        [],
        [block]
    )
    
    return samples_list[0]

# Batched JIT for Spin Activation
_batched_spin_activation_sampler = jax.jit(
    jax.vmap(_sample_spin_activation_kernel, in_axes=(0, 0, None, None)),
    static_argnums=(2,) # n_samples is static
)

class ThermalActivationFunction(torch.autograd.Function):
    """
    Custom Autograd Function for Thermodynamic Activation.
    Forward: Samples spins from the TSU (via JAX).
    Backward: Straight-Through Estimator (STE).
    """
    @staticmethod
    def forward(ctx, h_eff, n_samples, temperature, context):
        # 1. Prepare JAX
        h_eff_jax = jnp.array(h_eff.detach().cpu().numpy())
        batch_size = h_eff.shape[0]
        
        # 2. Get Key
        rng_key = context.next_key()
        keys = jax.random.split(rng_key, batch_size)
        
        # 3. Sample
        samples = _batched_spin_activation_sampler(
            keys, 
            h_eff_jax, 
            n_samples, 
            temperature
        )
        
        # 4. Convert to Torch
        samples_torch = torch.tensor(np.array(samples), device=h_eff.device, dtype=torch.float32)
        output_mean = samples_torch.mean(dim=1) # (batch, out)
        
        # Map to [-1, 1]
        output_scaled = 2.0 * output_mean - 1.0
        
        # Save for backward (optional, if we want non-identity STE)
        # ctx.save_for_backward(h_eff, output_scaled)
        
        return output_scaled

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (Identity)
        # dL/dh_eff = dL/dy * 1
        # We could also use Tanh derivative here for better gradients.
        # For now, Identity is robust.
        return grad_output, None, None, None

class ThermalLinear(nn.Module):
    """
    A PyTorch module that wraps a Linear layer but executes it using 
    thermodynamic sampling via THRML.
    
    Refactored to use "Effective Fields" (Input Fidelity):
    Instead of clamping input nodes (which requires binarization),
    we compute the effective field h_eff = Wx + b in PyTorch,
    and use it as a bias for the output spins.
    """
    def __init__(self, original_layer: nn.Linear, adapter: TransformerToThermalAdapter, n_samples=1, context: ThermalContext = None):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter
        self.n_samples = n_samples
        self.out_features = original_layer.out_features
        # Initialize context with adapter's temperature if not provided
        self.context = context if context is not None else ThermalContext(temperature=adapter.temperature)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using thermodynamic sampling.
        x: (batch_size, in_features)
        Returns: (batch_size, out_features) - Mean of samples.
        """
        # 1. Compute Effective Fields (Linear Pass)
        # This preserves the magnitude of inputs!
        # h_eff: (batch, out)
        h_eff = self.original_layer(x)
        
        # 2. Apply Thermal Activation (with Autograd)
        output_scaled = ThermalActivationFunction.apply(
            h_eff, 
            self.n_samples, 
            self.context.temperature, 
            self.context
        )
        
        return output_scaled

def replace_linear_layers(model: nn.Module, adapter: TransformerToThermalAdapter, n_samples=1, context: ThermalContext = None):
    """
    Recursively replaces all nn.Linear layers in the model with ThermalLinear layers.
    """
    if context is None:
        context = ThermalContext()
        
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # Replace
            print(f"Replacing layer {name} with ThermalLinear...")
            thermal_layer = ThermalLinear(child, adapter, n_samples=n_samples, context=context)
            setattr(model, name, thermal_layer)
        else:
            # Recurse
            replace_linear_layers(child, adapter, n_samples, context)

# --- Optimized JAX Kernels for Attention ---

def _sample_attention_kernel(key, logits, n_samples):
    """
    Single-item JAX kernel for sampling attention indices.
    Constructs a temporary sampling program for the given logits.
    """
    seq_len = logits.shape[0]
    
    # 1. Define Graph (Nodes & Blocks)
    # We create a fresh node/block for this computation.
    # Inside JIT, these objects are created once during tracing.
    node = CategoricalNode()
    block = Block([node])
    
    # 2. Define Factor (Energy)
    # Logits are passed as weights. 
    # We assume logits are Energy terms (negative log probs).
    # Reshape to (1, seq_len) as expected by CategoricalEBMFactor
    factor = CategoricalEBMFactor([block], logits.reshape(1, seq_len))
    
    # 3. Define Program
    ebm = FactorizedEBM([factor])
    sampler = CategoricalGibbsConditional(n_categories=seq_len)
    spec = BlockGibbsSpec(free_super_blocks=[block], clamped_blocks=[])
    
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=ebm.factors,
        other_interaction_groups=[]
    )
    
    # 4. Schedule & Init
    # n_samples is static, so this is fine
    schedule = SamplingSchedule(n_warmup=10, n_samples=n_samples, steps_per_sample=1)
    init_val = jnp.array([0], dtype=jnp.uint8)
    
    # 5. Sample
    samples_list = sample_states(
        key,
        program,
        schedule,
        [init_val],
        [], # No clamped values
        [block] # Observe the block
    )
    
    return samples_list[0] # (n_samples,)

# JIT-compiled batched sampler
# static_argnums=2 corresponds to n_samples
_batched_attention_sampler = jax.jit(
    jax.vmap(_sample_attention_kernel, in_axes=(0, 0, None)), 
    static_argnums=(2,)
)

class ThermalAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention using Thermodynamic Sampling.
    Replaces the Softmax step with sampling from a Categorical distribution.
    """
    def __init__(self, temperature=1.0, n_samples=1, context: ThermalContext = None):
        super().__init__()
        self.n_samples = n_samples
        self.context = context if context is not None else ThermalContext(temperature=temperature)
        
    def forward(self, query, key, value):
        """
        query: (batch, 1, d) - Single token query
        key: (batch, seq_len, d)
        value: (batch, seq_len, d)
        
        Returns: (batch, 1, d) - Context vector
        """
        # 1. Compute Logits (Energy)
        # Q K^T / sqrt(d)
        d = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d)
        # scores shape: (batch, 1, seq_len)
        
        batch_size = query.shape[0]
        seq_len = key.shape[1]
        
        # 2. Prepare JAX Inputs
        # Flatten batch and head dims if necessary (here we have batch, 1, seq_len)
        logits_torch = scores.squeeze(1) # (batch, seq_len)
        logits_jax = jnp.array(logits_torch.detach().cpu().numpy())
        
        # Apply Temperature Scaling
        # P(x) ~ exp(logits/T)
        logits_jax = logits_jax / self.context.temperature
        
        # Generate Keys
        rng_key = self.context.next_key()
        keys = jax.random.split(rng_key, batch_size)
        
        # 3. Run Optimized Sampler
        # Returns: (batch, n_samples)
        indices_jax = _batched_attention_sampler(keys, logits_jax, self.n_samples)
        
        # 4. Aggregate Values
        # indices_jax: (batch, n_samples)
        indices_torch = torch.tensor(np.array(indices_jax), dtype=torch.long, device=value.device)
        
        # We need to gather values for each batch item
        # value: (batch, seq_len, d)
        # indices_torch: (batch, n_samples)
        
        # Expand indices to (batch, n_samples, d) for gathering?
        # Or use advanced indexing.
        
        # We want: for each b, select value[b][indices[b]] -> (n_samples, d)
        # Result: (batch, n_samples, d)
        
        # Torch gather requires same dim.
        # value is (B, L, D). We want to gather along L.
        # indices expanded: (B, S, D)
        
        # Easier way:
        batch_indices = torch.arange(batch_size, device=value.device).unsqueeze(1).expand(-1, self.n_samples) # (B, S)
        
        # Flatten for indexing
        flat_batch = batch_indices.flatten()
        flat_idx = indices_torch.flatten()
        
        selected_flat = value[flat_batch, flat_idx] # (B*S, d)
        selected = selected_flat.view(batch_size, self.n_samples, -1) # (B, S, d)
        
        # Mean over samples
        context = selected.mean(dim=1) # (B, d)
        
        return context.unsqueeze(1) # (B, 1, d)
