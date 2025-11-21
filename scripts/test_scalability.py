import torch
import time
import numpy as np
from qnn.thermal_adapter import TransformerToThermalAdapter
from thrml import SpinNode

def test_scalability():
    print("Testing Graph Construction Scalability...")
    
    # Create a large layer
    # 1000x1000 = 1M edges.
    # Python loop would take ~1-2 seconds.
    # 4096 x 4096 = 16M edges. Python loop ~30s-1min.
    
    n_in = 1000
    n_out = 1000
    
    print(f"Creating layer {n_in}x{n_out} ({n_in*n_out/1e6:.1f}M edges)...")
    layer = torch.nn.Linear(n_in, n_out)
    
    adapter = TransformerToThermalAdapter()
    input_nodes = [SpinNode() for _ in range(n_in)]
    output_nodes = [SpinNode() for _ in range(n_out)]
    
    start_time = time.time()
    adapter.convert_linear_layer(layer, input_nodes, output_nodes)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Construction time: {duration:.4f}s")
    
    if duration < 1.0:
        print("SUCCESS: Construction is fast.")
    else:
        print("WARNING: Construction might be slow.")

if __name__ == "__main__":
    test_scalability()
