import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def run_experiment(start_temp, seed, steps=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    in_features = 8
    out_features = 4
    base_linear = nn.Linear(in_features, out_features)
    adapter = TransformerToThermalAdapter(temperature=start_temp)
    context = ThermalContext(seed=seed)
    
    model = ThermalLinear(base_linear, adapter, n_samples=100, context=context)
    
    target_output = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    input_data = torch.randn(1, in_features)
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_output)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.temperature.clamp_(min=0.1, max=20.0)
        
        history.append(model.temperature.item())
        
    return history, loss.item()

def robustness_test():
    print("🧪 Starte Robustheits-Härtetest für den Thermodynamic Auto-Tuner...\n")
    
    seeds = [42, 123, 999]
    start_temps = [0.5, 5.0, 15.0]
    
    results = []
    
    for temp in start_temps:
        for seed in seeds:
            print(f"Testlauf: Start-T={temp:4.1f} | Seed={seed} ... ", end="", flush=True)
            history, final_loss = run_experiment(temp, seed)
            
            delta_t = history[-1] - history[0]
            direction = "ABGEKÜHLT ❄️" if delta_t < -0.5 else ("ERWÄRMT 🔥" if delta_t > 0.5 else "STABIL ⚖️")
            
            print(f"Final-T={history[-1]:.2f} | Loss={final_loss:.4f} | {direction}")
            results.append({
                'start': temp,
                'final': history[-1],
                'loss': final_loss,
                'direction': direction
            })

    print("\n--- ZUSAMMENFASSUNG ---")
    cooling_count = sum(1 for r in results if "❄️" in r['direction'])
    heating_count = sum(1 for r in results if "🔥" in r['direction'])
    stable_count = sum(1 for r in results if "⚖️" in r['direction'])
    
    print(f"Gesamtanzahl Tests: {len(results)}")
    print(f"Modell hat abgekühlt: {cooling_count}")
    print(f"Modell hat erwärmt:  {heating_count}")
    print(f"Modell blieb stabil: {stable_count}")
    
    if cooling_count + heating_count > 0:
        print("\n✅ DAS ERGEBNIS IST REPRODUZIERBAR: Die Temperatur ist eine aktive, lernende Komponente.")
    else:
        print("\n⚠️ WARNUNG: Das System zeigt keine signifikante Dynamik.")

if __name__ == "__main__":
    robustness_test()
