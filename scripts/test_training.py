import torch
import torch.optim as optim
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def test_training():
    print("Testing Backward Pass (Training)...")
    
    # Setup
    # Simple regression: y = -x
    # Model: y = ThermalLinear(x)
    # Target: -1.0 for x=1.0
    
    adapter = TransformerToThermalAdapter(temperature=0.1) # Low temp for cleaner gradients? Or high?
    # STE works best when forward is close to backward.
    # If T is low, forward is step function, backward is identity. Mismatch.
    # If T is high, forward is linear-ish, backward is identity. Match.
    # Let's try T=1.0.
    ctx = ThermalContext(temperature=1.0)
    
    linear = torch.nn.Linear(1, 1, bias=False)
    linear.weight.data = torch.tensor([[0.5]]) # Start wrong (positive)
    
    model = ThermalLinear(linear, adapter, n_samples=100, context=ctx)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    
    print(f"Initial Weight: {linear.weight.item():.4f}")
    
    # Training Loop
    for epoch in range(20):
        optimizer.zero_grad()
        
        # Input
        x = torch.tensor([[1.0]])
        target = torch.tensor([[-1.0]])
        
        # Forward
        output = model(x)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Weight={linear.weight.item():.4f}, Grad={linear.weight.grad.item():.4f}")
            
    print(f"Final Weight: {linear.weight.item():.4f}")
    
    # Validation
    # We want weight to move towards negative.
    # Since target is -1 and input is 1, weight should become negative.
    if linear.weight.item() < 0.0:
        print("SUCCESS: Weight learned to be negative.")
    else:
        print("FAILURE: Weight did not learn correctly.")

if __name__ == "__main__":
    test_training()
