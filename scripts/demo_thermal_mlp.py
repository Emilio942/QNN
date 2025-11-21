import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from qnn.thermal_adapter import TransformerToThermalAdapter, replace_linear_layers

# 1. Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Generate Toy Data (XOR-like)
def generate_data(n_samples=200):
    np.random.seed(42)
    # Generate 4 clusters
    # Class 0: (1, 1), (-1, -1)
    # Class 1: (1, -1), (-1, 1)
    
    X = []
    y = []
    
    centers = [
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 1),
        (-1, 1, 1)
    ]
    
    for cx, cy, label in centers:
        n = n_samples // 4
        data = np.random.randn(n, 2) * 0.3 + np.array([cx, cy])
        X.append(data)
        y.append(np.full(n, label))
        
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long)

def train_model(model, X, y, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            acc = (outputs.argmax(dim=1) == y).float().mean()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Acc {acc:.4f}")

def evaluate(model, X, y, name="Model"):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        # If outputs are from ThermalLinear, they might be probabilities or logits?
        # ThermalLinear returns mean of samples (0..1).
        # If it's the last layer, we can treat them as probabilities.
        # Standard Linear returns logits (-inf..inf).
        
        # We need to handle both.
        # If max value is <= 1.0 and min >= 0.0, likely probabilities.
        
        if outputs.min() >= 0 and outputs.max() <= 1.0:
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs.argmax(dim=1)
            
        acc = (preds == y).float().mean()
        print(f"[{name}] Accuracy: {acc:.4f}")
        return acc

def main():
    # Setup
    X, y = generate_data(n_samples=400)
    
    # Train Baseline
    model = SimpleMLP(hidden_dim=8)
    train_model(model, X, y, epochs=200)
    evaluate(model, X, y, "Baseline (PyTorch)")
    
    # Convert to Thermal
    print("\nConverting to Thermal Model...")
    # Temperature controls the stochasticity. 
    # Lower T -> More deterministic (Argmax-like).
    # Higher T -> More random.
    # Since we trained with ReLU (deterministic), we want low T to mimic it?
    # Or T=1.0 if weights are scaled appropriately.
    # Let's try T=1.0 first.
    adapter = TransformerToThermalAdapter(temperature=0.5)
    
    # We need to be careful: replace_linear_layers modifies the model in-place.
    # But SimpleMLP uses nn.Sequential. 
    # replace_linear_layers recurses into children. nn.Sequential children are the layers.
    # It should work.
    
    replace_linear_layers(model, adapter, n_samples=20)
    
    print("\nEvaluating Thermal Model...")
    # Note: ThermalLinear thresholds input > 0.
    # Our input data is centered around +/- 1, so thresholding at 0 preserves the sign info.
    # This is good for the first layer.
    # The hidden layer (ReLU) outputs are >= 0.
    # Thresholding > 0 converts ReLU output to Binary (Active/Inactive).
    
    evaluate(model, X, y, "Thermal (Extropic)")

if __name__ == "__main__":
    main()
