import onnxruntime as ort
import numpy as np
import time

# Target CUDA on your RTX 3050; fall back to CPU if needed
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

session = ort.InferenceSession("aquanex_causal_gru.onnx", providers=providers)
active_ep = session.get_providers()[0]
print(f"\n[+] Active Execution Provider: {active_ep}")

# 1 frame: 257 frequency bins, 2-layer GRU hidden state (128 units)
dummy_x = np.random.randn(1, 1, 257).astype(np.float32)
dummy_h = np.zeros((2, 1, 128), dtype=np.float32)

inputs = {
    'input_spectrum': dummy_x,
    'hidden_state_in': dummy_h
}

# Warmup runs (compiles and caches CUDA kernels)
print("[*] Warming up GPU cache (200 runs)...")
for _ in range(200):
    outputs = session.run(None, inputs)

# Timed runs over 2,000 sequential frames
print("[*] Profiling 2,000 sequential audio frames...")
timings = []
for _ in range(2000):
    t0 = time.perf_counter()
    outputs = session.run(None, inputs)
    t1 = time.perf_counter()
    timings.append((t1 - t0) * 1000.0)  # ms
    
    # Pass updated recurrent state back (simulates live streaming)
    inputs['hidden_state_in'] = outputs[1]

avg_lat = np.mean(timings)
p95_lat = np.percentile(timings, 95)
p99_lat = np.percentile(timings, 99)
min_lat = np.min(timings)

print("\n" + "="*48)
print("     AQUANEX EDGE ONNX RUNTIME BENCHMARK      ")
print("="*48)
print(f" Execution Target    : {active_ep}")
print(f" Frame Step Size     : 16.000 ms")
print(f" Mean Latency        : {avg_lat:.3f} ms")
print(f" Minimum Latency     : {min_lat:.3f} ms")
print(f" 95th Percentile     : {p95_lat:.3f} ms")
print(f" 99th Percentile     : {p99_lat:.3f} ms")
print("="*48)

if avg_lat < 5.0:
    print("[PASS] Verified: Frame latency is well below tactical limit (<16ms).")