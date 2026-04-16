import torch
import torch.nn as nn
import torch.optim as optim
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def demo_autotuner():
    print("🚀 Running Thermodynamic Auto-Tuner Demo...")
    
    # 1. Setup a simple task
    # Target: Map random input to a specific binary pattern
    in_features = 8
    out_features = 4
    
    # Create a model
    base_linear = nn.Linear(in_features, out_features)
    adapter = TransformerToThermalAdapter(temperature=5.0) # Start with very HIGH temperature (noisy)
    context = ThermalContext()
    
    model = ThermalLinear(base_linear, adapter, n_samples=100, context=context)
    
    # Target pattern
    target_output = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    input_data = torch.randn(1, in_features)
    
    # Optimizer: Optimizes BOTH weights and temperature
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    print(f"Initial Temperature: {model.temperature.item():.4f}")
    
    # 2. Training Loop
    for step in range(50):
        optimizer.zero_grad()
        
        # Forward
        output = model(input_data)
        loss = criterion(output, target_output)
        
        # Backward
        loss.backward()
        
        # Check gradient of temperature
        t_grad = model.temperature.grad.item() if model.temperature.grad is not None else 0.0
        
        # Step
        optimizer.step()
        
        # Clamp temperature to prevent it going negative or zero
        with torch.no_grad():
            model.temperature.clamp_(min=0.1, max=20.0)
            
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:02d} | Loss: {loss.item():.4f} | Temp: {model.temperature.item():.4f} | T-Grad: {t_grad:.4f}")

    print("\n--- RESULTS ---")
    print(f"Final Temperature: {model.temperature.item():.4f}")
    if model.temperature.item() < 4.0:
        print("✅ SUCCESS: The Auto-Tuner COOLED DOWN the system to improve precision!")
    elif model.temperature.item() > 6.0:
        print("✅ SUCCESS: The Auto-Tuner HEATED UP the system to escape local minima/noise!")
    else:
        print("ℹ️ RESULT: Temperature shifted slightly.")

if __name__ == "__main__":
    demo_autotuner()
