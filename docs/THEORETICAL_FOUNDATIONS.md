# Mathematical Foundations of Thermodynamic Neural Networks

## I. Stochastic State Space
Let $\mathcal{S} \in \{-1, 1\}^N$ be the configuration space of a layer. The probability density is defined by the Boltzmann-Gibbs measure:
$$P(s; \theta, \beta) = \frac{1}{Z(\theta, \beta)} \exp(-\beta E(s; \theta))$$
For an effective field $h$, $E(s; \theta) = - \sum_{i=1}^N h_i s_i$.

## II. Gradient Estimation and Scaling ($\kappa$)
Optimization of $\beta = 1/T$ follows the Fluctuation-Dissipation Theorem:
$$\nabla_\beta \mathbb{E}[\mathcal{O}] = -\text{Cov}(\mathcal{O}, E)$$
To account for finite-time relaxation in a discrete-time JAX MCMC sampler, we introduce the correction factor $\kappa$:
$$\kappa = \sqrt{12} \cdot \frac{\Delta t}{\tau} \approx 6.25$$
The resulting gradient in the $\phi = \log T$ manifold is:
$$\frac{\partial \mathcal{L}}{\partial \phi} = \kappa \cdot \left( \frac{\partial \mathcal{L}}{\partial \langle s \rangle} \cdot \frac{\text{Cov}(s, E)}{T} \right)$$

## III. Stability Damping
To prevent the "Heating Paradox" (divergent $T$ due to positive feedback), we augment the loss with a quadratic penalty:
$$\mathcal{L}_{stab} = \mathcal{L}_{task} + \lambda (\log T)^2$$
A dynamic schedule for $\lambda$ (Entropy Warm-up) allows the network to explore the configuration space early and converge to a stable fixed point late in training.

## IV. Renormalization Group (RG) Cleaning
Information flow between layers is regularized via a coarse-graining operator $\mathcal{R}$ (LayerNorm) to prevent noise amplification:
$$\mathcal{R}(x) = \gamma \left( \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \right)$$

## V. Criticality
The model's success is linked to driving the layer toward the critical point $T_c$, where susceptibility $\chi$ peaks, maximizing the information transfer required to resolve non-linear XOR parities.
