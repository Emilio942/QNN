# Mathematical Problem Statements and Solution Framework

This document formalizes the core challenges of our QNN pipeline and captures a practical solution framework for future work. See also `docs/REFERENCES.md`.

## A. Formal problem statements

1) Data and preprocessing
- Given x in R^d (or text t with an embedding E: T->R^d). Normalization N: R^d->R^d (z-score or L2).
- Spec constraint: d_spec in N. If d != d_spec, find a projection P: R^d->R^{d_spec} that minimizes distortion
  delta(P) = E[ || x - A^+ A x ||^2 ] with P(x) = A x.

2) QNN predictor (data re-uploading)
- Parametrized unitary: U_theta(x) = product_{l=1..L} [ V_l(theta_l) * E_l(x) ] on q qubits.
- State: |psi_theta(x)> = U_theta(x) |0^q>.
- Measurement M (e.g., Z \otimes I ...), score s_theta(x) = <psi_theta(x)| M |psi_theta(x)> in [-1, 1].
- Classification: y_hat = sign( s_theta(x) - tau ).

3) Training objective
- Data D = {(x_i, y_i)}, y_i in {-1, +1}.
- Minimize empirical risk: min_theta (1/n) sum_i loss(y_i, s_theta(x_i)), logistic or hinge.
- With regularization: min_theta R_emp(theta) + lambda * Omega(theta).

4) NN->QNN approximation (target; not a converter)
- Given f(x) = sigma(w^T x + b). Find A: (w, b) -> theta to minimize
  epsilon(A) = sup_{||x||<=B} | f(x) - g_theta(x) |, with g_theta(x) := (s_theta(x)+1)/2.
- Constraints: small (q, L), realizable U_theta, compilable on hardware.

5) Optimization under finite sampling
- Objective J(theta) = E_{(x,y)}[ loss(y, s_theta(x)) ]. SPSA est.: g_hat_k = (J(theta_k+c_k Delta_k) - J(theta_k-c_k Delta_k)) / (2 c_k) * Delta_k^{-1}.
- Convergence: sum a_k = inf, sum a_k^2 < inf, c_k -> 0. Measurement variance Var[\hat{s}] <= (1 - s^2)/S for S shots.

6) Barren plateaus (trainability)
- For expressive/random ansatz: Var[ dJ/d theta_i ] in O(2^{-q}). Need init/ansatz that preserves gradient variance.

7) Expressivity (re-uploading)
- F_L approximates trig polynomials in linear forms of x; effective frequency grows with L.
- Trade-off L (capacity) vs. noise/compile cost.

8) Noise and robustness
- Noise channel Lambda: s~_theta(x) = Tr[ M Lambda(U_theta rho_0 U_theta^dagger) ]. Depolarizing p: s~ = (1-p)s + p Tr[M]/2^q.
- Margin shrinks roughly by (1-p).

9) Compilation under hardware constraints
- Given coupling graph G=(V,E), gate set B. Choose layout and SWAPs to minimize C = alpha * depth + beta * cx_count.

10) Threshold calibration and drift
- tau* = argmax_tau F_beta(tau) with F_beta = (1+beta^2) PR / (beta^2 P + R). AUC alternative.

11) Generalization
- With hypothesis class H={ s_theta }, expected risk bound E[R(theta)] <= R_emp(theta) + O(R_n(H)) + O(sqrt(log(1/delta)/n)).

12) Embedding shift/stability
- Two embeddings E1, E2 with Delta = sup_t || E1(t) - A E2(t) ||. Error transfer bounded by Lipschitz(g_theta) * Delta.

13) Data scarcity vs. dimension
- n << d risks overfitting; need appropriate regularization and augmentation.

## B. Solution framework (engineering plan)

1) Data preprocessing & dimensionality reduction
- Projection for d != d_spec: PCA or random projections to minimize distortion delta(P). Optionally quantum PCA (research).

2) QNN architecture design
- Hardware-efficient ansatz with data re-uploading; alternate single-qubit rotations and entanglers. Balance depth L and qubits q.

3) NN->QNN translation (approximation)
- Approximate classical neuron f(x)=sigma(w^T x + b) via g_theta(x) = Tr(M U_theta(x) rho_0 U_theta^dagger(x)) with suitable encodings.
- Initialization: map w,b to angles (bucket averages + scaling) as a pragmatic start; refine by training.

4) Training & optimization
- SPSA with adaptive schedules (a_k, c_k), small-angle init, optional layer-wise training; consider gradient-free as fallback.

5) Noise & robustness
- Model depolarizing/thermal errors; apply zero-noise extrapolation (ZNE) or probabilistic error cancellation (PEC) where feasible; noise-aware training.

6) Transpilation & hardware constraints
- Use basis/coupling presets; minimize cost C = alpha*depth + beta*cx_count; select high-connectivity layouts.

7) Threshold calibration & drift
- Optimize tau for F_beta or AUC on validation; apply Platt scaling or isotonic regression; monitor drift and re-calibrate.

8) Generalization & regularization
- Penalize parameter norms/angles and circuit depth; explore capacity measures; cross-validate hyperparameters.

9) Embedding stability
- Prefer L2-normalized embeddings; track changes across model versions; optionally learn a linear adapter A.

10) Data scarcity mitigation
- Augment data, transfer-learn QNN, explore Bayesian priors over parameters.

## C. Mapping to repo status (Aug 2025)
- Implemented: data normalization/padding; QNN with re-uploading; training/predict CLI; PR/ROC plots; hardware transpile summary with presets; NN->QNN initializer (approximate); Ollama embeddings and offline pipeline.
- Planned: PCA/random projections utility; threshold sweep and calibration; explicit regularization/augmentation; error-mitigation hooks; expressivity/ansatz sweeps; theoretical bounds and stability tools.

## D. Immediate next steps
- Add PCA/random projections and spec-integration.
- Add tau-sweep/calibration report to eval script.
- Expose simple regularizers and an augmentation hook.
- Optional: Aer error-mitigation switch for demos.
