# Projekt: Transformer-to-TSU Adapter
*Hybrid Deterministic-Probabilistic Inference mit Thermodynamic Computing*

Dieses Projekt zielt darauf ab, vortrainierte Transformer-Modelle (z.B. BERT, GPT) auf thermodynamische Hardware (Extropic TSU) zu portieren, indem Gewichte in Energielandschaften übersetzt werden.

---

## 📦 Phase 1: THRML Setup & Verständnis
*Extropic hat bereits eine Open-Source-Bibliothek: `thrml` (JAX-basiert)*

- [x] **THRML installieren & Dokumentation studieren**
    - [x] `pip install thrml` ausführen.
    - [x] Beispiel-Notebooks durchgehen (`examples/02_spin_models.ipynb`).
    - [x] Verstehen: `SpinNode`, `Block`, `IsingEBM`, `IsingSamplingProgram`.
- [x] **Kern-Konzepte verinnerlichen**
    - [x] **Energy-Based Models (EBM):** Wie werden Energiefunktionen definiert?
    - [x] **Block Gibbs Sampling:** Wie funktioniert der `sample_states`-Aufruf?
    - [x] **Faktoren vs. Sampler:** `SpinEBMFactor` (definiert Interaktionen) vs. `SpinGibbsConditional` (sampled).
- [x] **Mini-Test: Einfaches Ising-Modell**
    - [x] 5-Node-Chain aus der README nachbauen.
    - [x] Samples ziehen und visualisieren (Spin-Korrelationen).

## 🧩 Phase 2: Mathematische Übersetzung (Transformer → EBM)
*Wie übersetzen wir Transformer-Gewichte in eine Energielandschaft?*

- [x] **Theorie: Linear Layer als RBM**
    - [x] Paper/Blog lesen: "Restricted Boltzmann Machines and Deep Learning".
    - [x] Verstehen: $E(v, h) = -v^T W h - b^T h$ (Visible-Hidden-Kopplung).
    - [x] Frage klären: Wie wird ein deterministischer Forward-Pass ($y = Wx + b$) in bedingtes Sampling übersetzt?
- [x] **Theorie: Attention als Energie**
    - [x] Softmax $\propto \exp(QK^T)$ ist eine Boltzmann-Verteilung.
    - [x] Wie kann man "Attention-Indizes" samplen statt exakt zu berechnen?
- [x] **Prototyp: Toy-Beispiel (ohne Transformer)**
    - [x] Ein 2-Layer MLP (Linear+ReLU+Linear) als `SpinEBMFactor` ausdrücken.
    - [x] Vergleich: Deterministischer Forward vs. Sampled Forward.

## ⚡ Phase 3: Adapter-Implementierung
*Transformer-Gewichte → THRML Faktoren*

- [x] **Klasse `TransformerToThermalAdapter`**
    - [x] Methode `convert_linear_to_factor(nn.Linear)` → gibt `SpinEBMFactor` zurück.
    - [x] Gewichts-Skalierung: Transformer-Weights normalisieren (z.B. auf [-1, 1]).
    - [x] Temperature-Parameter: Steuert "Kreativität" des Samplings.
- [x] **Integration mit THRML**
    - [x] Konstruiere `BlockGibbsSpec` aus Transformer-Architektur.
    - [x] Definiere `free_blocks` (die Layer, die gesampelt werden).
    - [x] Definiere `clamped_blocks` (Input-Embeddings als fixierte Bedingung).
- [x] **Forward-Pass Ersetzung**
    - [x] Ersetze `model.forward(x)` durch `sample_states(...)` (via `ThermalLinear`).
    - [x] Rückgabe: Ensemble von Outputs (mehrere Samples) statt einem deterministischen Wert (Mean returned).
    - [x] Utility `replace_linear_layers` implementiert.

## 🚀 Phase 4: Integration & Demo
*Zusammenfügen der Teile*

- [x] **End-to-End Demo**
    - [x] Lade ein kleines PyTorch Modell (z.B. MNIST MLP).
    - [x] Konvertiere es mit `replace_linear_layers`.
    - [x] Führe Inference durch und vergleiche Accuracy (Deterministic vs. Thermal).
- [x] **Attention-Integration**
    - [x] Implementiere `ThermalAttention` (analog zu `ThermalLinear`).
    - [x] Teste mit einem kleinen Transformer-Block (Core Attention Mechanism verified).

## 🛠️ Phase 5: Engineering & Optimierung (Refactoring)
*Vom Prototyp zur skalierbaren Lösung*

- [x] **Refactor `ThermalLinear`: Input Fidelity**
    - [x] Problem: Aktuell wird Input hart auf `x > 0` (binär) gesetzt. Informationsverlust.
    - [x] Lösung: "Effective Fields" nutzen. $B_{eff} = Wx + b$ in PyTorch berechnen.
    - [x] TSU als stochastische Aktivierungsfunktion nutzen (statt vollem Ising-Graph für MatMul).
- [x] **Refactor `ThermalAttention`: Performance**
    - [x] Problem: Graph wird bei jedem Forward-Pass neu gebaut (Python Loop Overhead).
    - [x] Lösung: JAX `jit` Kompilierung für statische Graphen nutzen.
- [x] **Scalability & Vectorization**
    - [x] Problem: Python-Loops in `convert_linear_layer` (O(N*M)).
    - [x] Lösung: `itertools.product` und NumPy Flattening für vektorisierte Edge-Erstellung.
    - [x] Test: 1M Edges in ~0.2s konstruiert.

## 🔌 Phase 6: Hardware Realism
*Vorbereitung auf physische Constraints*

- [x] **Sparsity Support**
    - [x] Problem: Fully Connected Graphen sind auf Hardware schwer abbildbar.
    - [x] Lösung: `sparsity_threshold` einführen. Nur Gewichte $|w| > \theta$ werden als Edges realisiert.
    - [x] Optimierung: Sparse-Construction (nur relevante Edges iterieren) statt Dense-Construction.

## 🏗️ Phase 7: Engineering Refactoring
*Vom Prototyp zur robusten Architektur*

- [x] **RNG State Management**
    - [x] Problem: `np.random` (Dirty Hack) bricht Reproduzierbarkeit.
    - [x] Lösung: `ThermalContext` Klasse eingeführt, die JAX PRNGKeys deterministisch verwaltet.
- [x] **Central Context & Annealing**
    - [x] Problem: Fragmentierte Adapter-Instanzen verhindern globale Steuerung.
    - [x] Lösung: `ThermalContext` hält globalen State (Temperatur).
    - [x] Feature: Globales Annealing (T_start -> T_end) über alle Layer hinweg möglich.
    - [x] Test: `scripts/test_global_annealing.py` verifiziert Steuerung.
- [x] **Backward Pass (Training)**
    - [x] Problem: Sampling ist nicht differenzierbar. Training unmöglich.
    - [x] Lösung: `ThermalActivationFunction` mit Straight-Through Estimator (STE) implementiert.
    - [x] Test: `scripts/test_training.py` zeigt erfolgreiches Lernen (Weight Update via SGD).

## 📝 Dokumentation & API
- [ ] Docstrings für alle Adapter-Methoden.
- [ ] Beispiel-Notebook `demo_thermal_transformer.ipynb` erstellen.

---

## 🔬 Phase 8: Advanced Mathematical Theory (Inquiry)
*Mathematische Fundierung und Optimierung*

- [ ] **Gradient Estimation & Variance Analysis**
    - [ ] Untersuchung des Fluctuation-Dissipation Theorems als Alternative zum STE.
    - [ ] Analyse der Varianz-Entwicklung bei steigender Layer-Tiefe.
- [ ] **Phase Transitions & Learning Stability**
    - [ ] Identifikation kritischer Punkte (Phasenübergänge) in den Gewichten.
    - [ ] Entwicklung einer "Thermodynamic Learning Rate" basierend auf magnetischer Suszeptibilität.
- [x] **Inquiry: "Carlin Questions"**
    - [x] Beantwortung der theoretischen Fragen in `docs/CARLIN_QUESTIONS.md` (Alle 31 Fragen beantwortet).
    - [x] Dokumentation der Ergebnisse in `docs/THEORETICAL_FOUNDATIONS.md`.

---

## 🏁 Phase 10: The Grand Integration & Research Paper
*Zusammenführung und finale Validierung*

- [ ] **Unified Hybrid Layer Prototype**
    - [ ] Konstruktion eines Layers, der QNN-Verschränkung und thermisches Auto-Tuning kombiniert.
- [ ] **Landauer Efficiency Benchmark**
    - [ ] Messung der Joul-Kosten pro Trainingsschritt und Vergleich mit Carlins Limit.
- [ ] **Final Research Summary**
    - [ ] Zusammenfassung der Ergebnisse für eine potenzielle Veröffentlichung oder Präsentation.
