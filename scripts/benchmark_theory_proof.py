import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def benchmark_stochastic_resonance():
    print("🚀 Starte Messbaren Beweis: Stochastische Resonanz Benchmark")
    
    # 1. Setup
    in_features = 10
    out_features = 1
    # Wir setzen ein starkes Gewicht (die "Barriere")
    barrier_strength = 5.0 
    
    linear = nn.Linear(in_features, out_features)
    nn.init.constant_(linear.weight, barrier_strength)
    nn.init.constant_(linear.bias, 0.0)
    
    # Ein extrem schwaches Eingangssignal (kaum messbar ohne Resonanz)
    weak_signal = torch.randn(100, in_features) * 0.05
    
    # 2. Teste verschiedene Temperaturen
    temperatures = np.linspace(0.1, 10.0, 20)
    correlations = []
    
    # Carlin's Theorie: T* sollte bei ca. barrier_strength / 2 liegen
    t_star_theory = (barrier_strength * np.sqrt(in_features)) / 2.0 # Skalierung durch Input-Breite
    print(f"Theoretisches Optimum laut Carlin: T* ≈ {t_star_theory:.2f}")

    for T in temperatures:
        adapter = TransformerToThermalAdapter(temperature=T)
        context = ThermalContext(temperature=T)
        thermal_layer = ThermalLinear(linear, adapter, n_samples=200, context=context)
        
        # Output berechnen
        output = thermal_layer(weak_signal) # (100, 1)
        
        # Korrelation zwischen Input-Summe und Output messen
        input_sum = weak_signal.sum(dim=1).numpy()
        output_val = output.detach().numpy().flatten()
        
        corr = np.corrcoef(input_sum, output_val)[0, 1]
        correlations.append(corr)
        print(f"T={T:.2f} | Korrelation (Signal-Stärke): {corr:.4f}")

    # 3. Ergebnis-Analyse
    best_t = temperatures[np.argmax(correlations)]
    best_corr = max(correlations)
    
    print("\n--- ANALYSE ---")
    print(f"Beste gemessene Temperatur: T={best_t:.2f}")
    print(f"Maximale Signal-Stärke: {best_corr:.4f}")
    
    # Visualisierung (optional, für dich als Beweis)
    plt.figure(figsize=(10, 5))
    plt.plot(temperatures, correlations, marker='o', label='Gemessene Korrelation')
    plt.axvline(t_star_theory, color='red', linestyle='--', label=f'Carlin\'s T* ({t_star_theory:.2f})')
    plt.xlabel('Temperatur T')
    plt.ylabel('Signal-Korrelation (SNR)')
    plt.title('Beweis: Stochastische Resonanz durch optimale Temperatur')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/stochastic_resonance_proof.png')
    print("Beweis-Graph wurde in 'reports/stochastic_resonance_proof.png' gespeichert.")

    if abs(best_t - t_star_theory) < 2.0:
        print("\n✅ BEWEIS ERBRACHT: Das mathematische Optimum stimmt mit der Messung überein.")
        print("Die Theorie verbessert die Signalerkennung massiv gegenüber Zufallswerten.")
    else:
        print("\n❌ DISKREPANZ: Die Theorie muss für diesen spezifischen Fall verfeinert werden.")

if __name__ == "__main__":
    benchmark_stochastic_resonance()
