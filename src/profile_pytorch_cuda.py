import torch
import numpy as np
import time
from generate_clean_onnx import CausalANCModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Profiling on Hardware: {torch.cuda.get_device_name(0)}")

model = CausalANCModel(input_dim=257, hidden_dim=128, num_layers=2).to(device)
model.load_state_dict(torch.load("model_tactical_v2.pth", map_location=device, weights_only=True))
model.eval()

# Dummy frame on GPU: [1, 1, 257]
dummy_x = torch.randn(1, 1, 257, device=device)
dummy_h = torch.zeros(2, 1, 128, device=device)

# Warm-up runs
print("[*] Warming up CUDA kernels (200 runs)...")
with torch.no_grad():
    for _ in range(200):
        _, dummy_h = model(dummy_x, dummy_h)

# Synchronized CUDA timing over 2,000 frames
print("[*] Profiling 2,000 sequential frames on RTX 3050...")
timings = []

with torch.no_grad():
    for _ in range(2000):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        _, dummy_h = model(dummy_x, dummy_h)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms

avg_lat = np.mean(timings)
p95_lat = np.percentile(timings, 95)
p99_lat = np.percentile(timings, 99)
min_lat = np.min(timings)

print("\n" + "="*48)
print("     AQUANEX NATIVE CUDA GPU BENCHMARK       ")
print("="*48)
print(f" Execution Device    : {torch.cuda.get_device_name(0)}")
print(f" Frame Step Budget   : 16.000 ms")
print(f" Mean Latency        : {avg_lat:.3f} ms")
print(f" Minimum Latency     : {min_lat:.3f} ms")
print(f" 95th Percentile     : {p95_lat:.3f} ms")
print(f" 99th Percentile     : {p99_lat:.3f} ms")
print("="*48)