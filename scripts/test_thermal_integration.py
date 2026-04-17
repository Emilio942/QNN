import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

class ThermalMLP(nn.Module):
    """
    A fully thermodynamic MLP using the Auto-Tuner for every layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=100):
        super().__init__()
        # Initialize context and adapter
        self.context = ThermalContext()
        self.adapter = TransformerToThermalAdapter(temperature=1.0)
        
        # Layer 1: Input to Hidden
        self.layer1 = ThermalLinear(
            nn.Linear(input_dim, hidden_dim), 
            self.adapter, 
            n_samples=n_samples, 
            context=self.context
        )
        
        # Layer 2: Hidden to Output
        self.layer2 = ThermalLinear(
            nn.Linear(hidden_dim, output_dim), 
            self.adapter, 
            n_samples=n_samples, 
            context=self.context
        )

    def forward(self, x):
        h = self.layer1(x)
        # ThermalLinear uses sigmoid-like activation internally, 
        # so we don't necessarily need an extra ReLU here.
        y = self.layer2(h)
        return y

def generate_xor_data(n_samples=400):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    # XOR logic: sign(x) * sign(y)
    y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) > 0).astype(np.longlong)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_integrated_thermal():
    print("🚀 INTEGRIERTES TRAINING: Thermal MLP mit Auto-Tuner")
    
    # 1. Setup Data
    X, y = generate_xor_data(600)
    
    # 2. Setup Model
    model = ThermalMLP(input_dim=2, hidden_dim=16, output_dim=2, n_samples=50)
    
    # Optimizer: Trains Weights AND all Log-Temperatures
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("\nInitial-Zustand:")
    print(f"Layer 1 Temp: {model.layer1.temperature.item():.4f}")
    print(f"Layer 2 Temp: {model.layer2.temperature.item():.4f}")
    
    # 3. Training
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass (uses Carlin's Covariance Formula for T)
        loss.backward()
        
        # Optimizer Step
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | T1: {model.layer1.temperature.item():.2f} | T2: {model.layer2.temperature.item():.2f}")

    print("\n--- FINALES ERGEBNIS ---")
    final_acc = (model(X).argmax(dim=1) == y).float().mean()
    print(f"Finale Accuracy: {final_acc:.4f}")
    
    if final_acc > 0.8:
        print("✅ ERFOLG: Das integrierte Thermal-System hat das Problem gelöst.")
    else:
        print("❌ FEHLER: Die Performance ist nicht ausreichend.")

if __name__ == "__main__":
    train_integrated_thermal()
