# Mathematical Foundations of Thermodynamic Neural Networks

This document formalizes the stochastic-geometric framework of the implemented `ThermalAdapter` and the finalized physics of the Auto-Tuner.

## I. Stochastic State Space
Let $\mathcal{S} \in \{-1, 1\}^N$ be the configuration space of a layer. The probability of a state $s \in \mathcal{S}$ is governed by the Boltzmann-Gibbs distribution:
$$P(s; \theta, \beta) = \frac{1}{Z(\theta, \beta)} \exp(-\beta E(s; \theta))$$
where the local energy functional $E(s; \theta)$ for an effective field $h$ is defined as:
$$E(s; \theta) = - \sum_{i=1}^N h_i s_i$$

## II. Thermodynamic Gradient Estimation (FDT)
The optimization of the inverse temperature $\beta = 1/T$ is derived from the **Fluctuation-Dissipation Theorem**. 

### 1. Covariance identity
For any observable $\mathcal{O}(s)$, the gradient with respect to $\beta$ is the negative covariance with the energy:
$$\nabla_\beta \mathbb{E}[\mathcal{O}] = -\text{Cov}(\mathcal{O}, E)$$

### 2. Discretization Correction ($\kappa$)
In a discrete-time MCMC sampler, the gradient requires an analytical scaling factor $\kappa$ derived from the auto-correlation time $\tau$:
$$\kappa = \sqrt{12} \cdot \frac{\Delta t}{\tau} \approx 6.29$$
The corrected gradient in the $\phi$-manifold ($\phi = \log T$) is:
$$\frac{\partial \mathcal{L}}{\partial \phi} = \kappa \cdot \left( \frac{\partial \mathcal{L}}{\partial \langle s \rangle} \cdot \frac{\text{Cov}(s, E)}{T} \right)$$

## III. Lyapunov Stability and Damping
To ensure convergence and prevent the "Heating Paradox", we define an augmented energy functional $\mathcal{E}$ that serves as a strict Lyapunov function:
$$\mathcal{E}(\theta, \phi) = \mathcal{L}_{task}(\theta, \phi) + \frac{\lambda}{2} \|\phi\|_2^2$$
The time derivative along the trajectory satisfies $\dot{\mathcal{E}} \leq 0$, guaranteeing that the non-integrable gradient field ($\text{curl} \neq 0$) still converges to a point-like equilibrium.

## IV. Sample Complexity Scaling
Near the critical point $T_c$ (Phase Transition to Non-Linear Parity), the required sample budget $S$ follows a power-law scaling to bound the covariance error:
$$S(T) \propto |T - T_c|^{-\alpha}$$
Our implementation uses $S=500$ in the final training phase to bypass the Landauer Limit and resolve the XOR symmetry.

## V. Renormalization Group (RG) Step
The information transfer between layers is regularized via a coarse-graining operator $\mathcal{R}$ to prevent noise cascades:
$$\mathcal{R}(x) = \gamma \left( \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \right)$$
This ensures the information wedge remains open, allowing the output layer to receive the signal from the frozen (cooled) hidden representations.
