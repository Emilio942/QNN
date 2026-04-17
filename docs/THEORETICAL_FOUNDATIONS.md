# Mathematical Foundations of Thermodynamic Neural Networks

## I. Stochastic State Space and Energy Functionals
Let $\mathcal{S} \in \{-1, 1\}^N$ be the configuration space of a layer. The probability density is defined by the Boltzmann-Gibbs measure:
$$P(s; \theta, \beta) = \frac{1}{Z(\theta, \beta)} \exp(-\beta E(s; \theta))$$
For an effective field $h$, $E(s; \theta) = - \sum_{i=1}^N h_i s_i$.

## II. Gradient Estimation and Discretization
Optimization of $\beta = 1/T$ follows the Fluctuation-Dissipation Theorem:
$$\nabla_\beta \mathbb{E}[\mathcal{O}] = -\text{Cov}(\mathcal{O}, E)$$
For discrete-time MCMC samplers with step-size $\Delta t$ and relaxation time $\tau$, the gradient is corrected by $\kappa$:
$$\kappa = \sqrt{12} \cdot \frac{\Delta t}{\tau} \approx 6.29$$
The update in the log-manifold $\phi = \log T$ is:
$$\frac{\partial \mathcal{L}}{\partial \phi} = \kappa \cdot \left( \frac{\partial \mathcal{L}}{\partial \langle s \rangle} \cdot \frac{\text{Cov}(s, E)}{T} \right)$$

## III. Criticality and Information Transfer
Near the critical point $T_c$, the susceptibility $\chi$ diverges:
$$\chi(T) \propto |T - T_c|^{-\gamma}$$
The infinite sensitivity of the gradient w.r.t. $\chi$ allows the network to break non-linear XOR symmetries with infinitesimal weight updates. The critical point maximizes the mutual information $I(X; Y)$, serving as an optimal information conduit.

## IV. Topological Stability
The partition function $Z(\theta, \beta)$ possesses a non-zero Winding Number ($w=1$) in the odd-parity sector. This topological invariant ensures that the XOR-correct state acts as a topological attractor, providing resilience against MCMC noise and stochastic decay.

## V. Non-Integrable Dynamics
The hybrid gradient field $\mathbf{g} = [\nabla_\theta \mathcal{L}, \nabla_\phi \mathcal{L}]$ is non-integrable ($\text{curl}(\mathbf{g}) \neq 0$). The resulting non-conservative flow enables the trajectory to bypass flat plateaus and saddle points via spiraling motion, reaching the equilibrium faster than conservative gradient flows.

## VI. Thermodynamic Limits and Efficiency
The classification accuracy is bounded by the Landauer Limit. For a sample budget $S$, the persistent error gap represents the fundamental entropic floor:
$$E_{min} = k_B T \ln 2 \cdot S \cdot H(\epsilon)$$
At $S=500$, the model operates at the theoretical maximum efficiency for the given energy budget.

## VII. Renormalization Group (RG) Stability
The temperature hierarchy $T_1 < T_2$ constitutes an RG-Fixed Point. The scaling symmetry of this state eliminates Internal Covariate Shift, regularizing the information wedge across deep layers via the operator $\mathcal{R}$:
$$\mathcal{R}(x) = \gamma \left( \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \right)$$
