# References and Related Work

This project draws on established methods in quantum machine learning, variational circuits, compilation, and evaluation metrics. Below are key references and resources aligned with the implemented features.

## Quantum ML and Variational Circuits
- M. Cerezo et al., “Variational Quantum Algorithms,” Nature Reviews Physics 3, 625–644 (2021). doi:10.1038/s42254-021-00348-9
- V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” Nature 567, 209–212 (2019). doi:10.1038/s41586-019-0980-2
- M. Schuld and N. Killoran, “Quantum Machine Learning in Feature Hilbert Spaces,” Phys. Rev. Lett. 122, 040504 (2019). doi:10.1103/PhysRevLett.122.040504

## Data Re-uploading Ansatz (used in classifier demos)
- A. Pérez‑Salinas, A. Cervera‑Lierta, E. Gil‑Fuster, J. I. Latorre, “Data re-uploading for a universal quantum classifier,” Quantum 4, 240 (2020). doi:10.22331/q-2020-02-06-240

## Optimization (SPSA)
- J. C. Spall, “An Overview of the Simultaneous Perturbation Method for Efficient Optimization,” Johns Hopkins APL Technical Digest 19(4), 482–492 (1998).
- J. C. Spall, “Multivariate stochastic approximation using a simultaneous perturbation gradient approximation,” IEEE Trans. Automatic Control 37(3), 332–341 (1992). doi:10.1109/9.119632

## Framework and Compilation
- G. Aleksandrowicz et al., “Qiskit: An Open-source Framework for Quantum Computing” (2019). arXiv:1904.08986
- Qiskit Transpiler and Coupling Maps: https://qiskit.org/documentation/
- Heavy‑Hex topology (IBM Quantum devices): https://quantum-computing.ibm.com/docs/devices/

## State Preparation and Encodings
- Qiskit StatePreparation documentation: https://qiskit.org/documentation/apidoc/circuit_library.html#statepreparation
- Encodings overview (angle/phase/amplitude) in practice via Qiskit circuit library and custom circuits in this repo.

## Evaluation Metrics (PR/ROC)
- T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters 27(8), 861–874 (2006). doi:10.1016/j.patrec.2005.10.010
- J. Davis and M. Goadrich, “The Relationship Between Precision-Recall and ROC Curves,” ICML 2006. doi:10.1145/1143844.1143874

## Embeddings (for hybrid pipelines)
- Ollama (local LLM serving, embeddings API): https://github.com/ollama/ollama
- Nomic Embed Text model card: https://docs.nomic.ai/

## Note on NN→QNN
This repository does not contain a full neural‑network‑to‑quantum conversion method. It provides a pragmatic initializer (qnn-nn2qnn) mapping small linear/logistic weights to initial QNN parameters (approximate; not a 1:1 conversion). See README for usage.
