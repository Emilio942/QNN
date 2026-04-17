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
    Manages the JAX PRNGKey state and trainable thermodynamic parameters.
    """
    def __init__(self, seed: int = 42, temperature: float = 1.0):
        self.key = jax.random.PRNGKey(seed)
        # Temperature is now often managed as a torch.Parameter in the layers,
        # but we keep a default here for standalone use.
        self.temperature = temperature
        
    def next_key(self):
        """Returns a new key and updates internal state."""
        self.key, subkey = jax.random.split(self.key)
        return subkey

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

# --- Optimized JAX Kernels with Covariance Estimation ---

def _sample_and_correlate_kernel(key, h_eff, n_samples, temperature):
    """
    Samples spins and calculates the covariance with energy for the gradient.
    E(s) = -h_eff * s (Local Field Energy)
    """
    beta = 1.0 / temperature
    # Single-spin activation is simple: P(s=1) = sigmoid(2 * beta * h_eff)
    prob = jax.nn.sigmoid(2.0 * beta * h_eff)
    samples = jax.random.bernoulli(key, prob, (n_samples,)).astype(jnp.float32)
    # Map {0, 1} to {-1, 1}
    spins = 2.0 * samples - 1.0
    
    # Calculate Covariance for Gradient: d<s_i>/d_beta = -Cov(s_i, E_i)
    # Local energy E_i = -h_eff * s_i
    energies = -h_eff * spins
    
    mean_s = jnp.mean(spins)
    mean_e = jnp.mean(energies)
    mean_se = jnp.mean(spins * energies)
    
    cov_se = mean_se - mean_s * mean_e
    
    return mean_s, cov_se

_batched_thermal_sampler = jax.jit(
    # Vmap over batch (axis 0) and then over features (axis 1)
    jax.vmap(
        jax.vmap(_sample_and_correlate_kernel, in_axes=(0, 0, None, None)),
        in_axes=(0, 0, None, None)
    ),
    static_argnums=(2,)
)

class ThermalActivationFunction(torch.autograd.Function):
    """
    Thermodynamic Activation with Covariance-based Temperature Gradient.
    """
    @staticmethod
    def forward(ctx, h_eff, n_samples, temperature, context):
        # 1. Prepare JAX
        h_eff_jax = jnp.array(h_eff.detach().cpu().numpy())
        batch_size, out_features = h_eff.shape
        
        # 2. Get Keys for every single spin (batch * out_features)
        rng_key = context.next_key()
        keys = jax.random.split(rng_key, batch_size * out_features)
        keys_reshaped = keys.reshape(batch_size, out_features, -1)
        
        # 3. Sample and get Covariance
        # means: (batch, out), covs: (batch, out)
        means_jax, covs_jax = _batched_thermal_sampler(
            keys_reshaped, h_eff_jax, n_samples, temperature.item()
        )
        
        # 4. Convert to Torch
        output = torch.tensor(np.array(means_jax), device=h_eff.device, dtype=torch.float32)
        ctx.cov_se = torch.tensor(np.array(covs_jax), device=h_eff.device, dtype=torch.float32)
        ctx.save_for_backward(h_eff, temperature)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        h_eff, temperature = ctx.saved_tensors
        cov_se = ctx.cov_se
        
        # Gradient w.r.t. h_eff (Effective Field)
        grad_h = grad_output.clone() 
        
        # --- FIXED THERMODYNAMIC GRADIENT (Audit Results) ---
        # kappa = sqrt(12) * Delta_t / tau (Audit Question 50)
        # For our JAX sampler parameters, kappa = 6.29
        grad_T_elements = 6.29 * (grad_output * cov_se) / temperature
        grad_T = grad_T_elements.sum().view_as(temperature) 
        
        return grad_h, None, grad_T, None

class ThermalRG(nn.Module):
    """
    Layer-Wise Renormalization Group (RG) Step (Audit 4: Question 35).
    Cleans the signal between thermal layers by normalizing the noise floor.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Coarse-graining: Normalize and rescale to maintain information wedge
        return self.norm(x) * self.scale

class ThermalLinear(nn.Module):
    """
    Linear layer with Trainable Log-Temperature and Stability Patch.
    Uses Entropy Regularization to prevent the Heating Paradox.
    """
    def __init__(self, original_layer: nn.Linear, adapter: TransformerToThermalAdapter, n_samples=100, context: ThermalContext = None):
        super().__init__()
        self.original_layer = original_layer
        self.n_features = original_layer.out_features
        self.n_samples = n_samples
        self.context = context if context is not None else ThermalContext()
        
        # phi = log(T)
        init_phi = np.log(max(adapter.temperature, 1e-3))
        self.log_temperature = nn.Parameter(torch.tensor([init_phi], dtype=torch.float32))
        
        # Initial Damping factor (will be controlled via schedule)
        self.damping = 0.01

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_eff = self.original_layer(x)
        T = self.temperature
        
        # Thermodynamic Layer Norm (Normalized Field)
        # T is removed from scale because it is handled by the EBM (beta=1/T)
        std = h_eff.std()
        if std > 1e-6:
            scale = 2.0 / std
            h_eff = h_eff * scale

        # Apply Activation
        output = ThermalActivationFunction.apply(
            h_eff, 
            self.n_samples, 
            T, 
            self.context
        )

        # STABILITY PATCH: Entropy Regularization (Audit 3/42)
        if self.training:
            # Quadratic penalty to keep log(T) near 0 (T=1.0) unless forced
            # Fixed damping for stability as per Question 42 (Entropy Warm-up)
            entropy_penalty = self.damping * self.log_temperature.pow(2)
            output = output + (entropy_penalty - entropy_penalty.detach())

        return output

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
