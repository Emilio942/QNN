import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext, ThermalRG

class ThermalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=100):
        super().__init__()
        self.context = ThermalContext()
        self.adapter = TransformerToThermalAdapter(temperature=1.0)
        self.layer1 = ThermalLinear(nn.Linear(input_dim, hidden_dim), self.adapter, n_samples=n_samples, context=self.context)
        # RG Step to wash the noise from Layer 1 before it hits Layer 2
        self.rg = ThermalRG(hidden_dim)
        self.layer2 = ThermalLinear(nn.Linear(hidden_dim, output_dim), self.adapter, n_samples=n_samples, context=self.context)

    def forward(self, x):
        h = self.layer1(x)
        h = self.rg(h)
        return self.layer2(h)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def generate_xor_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) > 0).astype(np.longlong)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def run_benchmark():
    print("🏆 FINALER PROFESSIONELLER BENCHMARK")
    X, y = generate_xor_data(1000)
    
    # --- Baseline ---
    print("\n[1/2] Training Baseline (ReLU MLP)...")
    base_model = BaselineMLP(2, 16, 2)
    base_opt = optim.Adam(base_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for _ in range(200):
        base_opt.zero_grad()
        loss = criterion(base_model(X), y)
        loss.backward()
        base_opt.step()
    base_acc = (base_model(X).argmax(dim=1) == y).float().mean()
    print(f"Baseline Fertig. Zeit: {time.time()-start:.2f}s | Acc: {base_acc:.4f}")

    # --- Thermal ---
    print("\n[2/2] Training Thermal Model (Auto-Tuner)...")
    thermal_model = ThermalMLP(2, 16, 2, n_samples=100)
    thermal_opt = optim.Adam(thermal_model.parameters(), lr=0.02)
    
    start = time.time()
    for epoch in range(200):
        # Adaptive Damping Schedule (Audit 42: Entropy Warm-up)
        # Low damping early (0.001) for exploration, higher late (0.05) for freezing the solution
        current_damping = 0.001 if epoch < 120 else 0.05
        thermal_model.layer1.damping = current_damping
        thermal_model.layer2.damping = current_damping

        # Dynamic Sample Budget (Audit 46/48/53: Bypassing the Landauer Limit)
        # Increase S late in training to reduce covariance bias near critical point
        current_samples = 100 if epoch < 150 else 500
        thermal_model.layer1.n_samples = current_samples
        thermal_model.layer2.n_samples = current_samples

        thermal_opt.zero_grad()
        outputs = thermal_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        thermal_opt.step()
        
        if (epoch + 1) % 50 == 0:
            acc = (outputs.argmax(dim=1) == y).float().mean()
            print(f"Epoch {epoch+1:03d} | Acc: {acc:.4f} | T1: {thermal_model.layer1.temperature.item():.2f} | T2: {thermal_model.layer2.temperature.item():.2f}")
            
    thermal_acc = (thermal_model(X).argmax(dim=1) == y).float().mean()
    print(f"Thermal Fertig. Zeit: {time.time()-start:.2f}s | Acc: {thermal_acc:.4f}")

    print("\n--- VERGLEICH ---")
    print(f"Vorsprung/Rückstand: {thermal_acc - base_acc:+.4%}")
    if thermal_acc >= base_acc:
        print("🚀 PHYSIK-VORTEIL: Das Thermal-System ist gleichwertig oder überlegen!")
    else:
        print("⚖️ INFO: Die Thermodynamik braucht noch mehr Samples für die gleiche Präzision.")

if __name__ == "__main__":
    run_benchmark()
