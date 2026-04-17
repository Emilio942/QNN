import torch
import torch.nn as nn
import numpy as np
from qnn.thermal_adapter import ThermalLinear, TransformerToThermalAdapter, ThermalContext

def check_temperature_gradient():
    print("🕵️ PROFESSIONELLER GRADIENTEN-AUDIT")
    
    # Setup
    torch.manual_seed(42)
    in_features = 4
    out_features = 2
    linear = nn.Linear(in_features, out_features)
    # Gewichte fixieren für Reproduzierbarkeit
    nn.init.constant_(linear.weight, 0.5)
    nn.init.constant_(linear.bias, 0.1)
    
    adapter = TransformerToThermalAdapter(temperature=2.0)
    context = ThermalContext()
    
    # Wir nutzen 10.000 Samples, um das Rauschen fast auf Null zu drücken
    n_samples = 10000 
    model = ThermalLinear(linear, adapter, n_samples=n_samples, context=context)
    
    # Ein Test-Input
    x = torch.ones(1, in_features)
    
    # 1. Analytischer Gradient (Berechnung über unsere Formel in backward())
    output = model(x)
    loss = output.sum() # dL/ds ist hier einfach 1.0
    loss.backward()
    
    analytic_grad = model.log_temperature.grad.item()
    print(f"Analytischer Gradient (unser Code): {analytic_grad:.8f}")
    
    # 2. Numerischer Gradient (Finite Differenzen)
    eps = 1e-4
    with torch.no_grad():
        # Punkt + eps
        model.log_temperature.fill_(np.log(2.0) + eps)
        out_plus = model(x).sum().item()
        
        # Punkt - eps
        model.log_temperature.fill_(np.log(2.0) - eps)
        out_minus = model(x).sum().item()
        
    numeric_grad = (out_plus - out_minus) / (2 * eps)
    print(f"Numerischer Gradient (Realität):     {numeric_grad:.8f}")
    
    # 3. Urteil
    diff = analytic_grad - numeric_grad
    print(f"\nAbweichung: {diff:.8f}")
    
    if (analytic_grad * numeric_grad) > 0:
        print("✅ VORZEICHEN IST KORREKT.")
    else:
        print("❌ VORZEICHENFEHLER ENTDECKT! Der Gradient zieht in die falsche Richtung.")
        
    if abs(diff) < 0.01:
        print("✅ MAGNITUDE IST KORREKT.")
    else:
        print("⚠️ SKALIERUNGSFEHLER? Die Magnitude weicht ab.")

if __name__ == "__main__":
    check_temperature_gradient()
