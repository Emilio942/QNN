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
    X = np.random.randn(n_samples, 2)
    y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) > 0).astype(np.longlong)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_one_model(model_type="thermal", n_epochs=200):
    X, y = generate_xor_data(1000)
    criterion = nn.CrossEntropyLoss()
    
    if model_type == "baseline":
        model = BaselineMLP(2, 16, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        model = ThermalMLP(2, 16, 2, n_samples=100)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    for epoch in range(n_epochs):
        if model_type == "thermal":
            # Best established schedule
            current_damping = 0.001 if epoch < 120 else 0.05
            model.layer1.damping = current_damping
            model.layer2.damping = current_damping
            current_samples = 100 if epoch < 150 else 500
            model.layer1.n_samples = current_samples
            model.layer2.n_samples = current_samples

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    acc = (model(X).argmax(dim=1) == y).float().mean().item()
    return acc

def run_multi_trial_benchmark(n_trials=5):
    print(f"🏆 MULTI-TRIAL BENCHMARK ({n_trials} Durchläufe)")
    
    base_accs = []
    print("\n[1/2] Training Baseline (ReLU MLP)...")
    for i in range(n_trials):
        acc = train_one_model("baseline")
        base_accs.append(acc)
        print(f"  Trial {i+1}: {acc:.4f}")
    
    thermal_accs = []
    print("\n[2/2] Training Thermal Model (FDT Physics)...")
    for i in range(n_trials):
        acc = train_one_model("thermal")
        thermal_accs.append(acc)
        print(f"  Trial {i+1}: {acc:.4f}")

    print("\n--- ZUSAMMENFASSUNG ---")
    print(f"Baseline: {np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")
    print(f"Thermal:  {np.mean(thermal_accs):.4f} ± {np.std(thermal_accs):.4f}")

if __name__ == "__main__":
    run_multi_trial_benchmark(5)
