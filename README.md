# AQUANEX: Real-Time Tactical Audio Enhancement Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Opset-14-green.svg)](https://onnxruntime.ai/)
[![Latency Target](https://img.shields.io/badge/Latency-%3C20ms_Cleared-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Target-RTX_3050_%7C_Edge_CPU-orange.svg)]()

> **AQUANEX** is a causal, edge-native dual-microphone speech enhancement pipeline engineered for extreme tactical noise fields (down to -8 dB SNR) under strict glass-to-glass latency budgets (< 20.0 ms). Designed for tactical radio headsets, vehicle intercoms, and dismounted combat operations.

---

## ⚡ Key Highlights & Empirical Benchmarks

* **Strictly Causal Architecture (L = 0):** Zero future lookahead frames. Recurrent state progression via a 2-layer Causal GRU core (1,035 KB footprint).
* **Hardware-Profiled Sub-Millisecond Execution:**
  * **Edge CPU (ONNX Runtime):** **0.200 ms** mean frame inference (uses only 1.25% of the 16 ms budget).
  * **NVIDIA RTX 3050 (Native CUDA):** **1.152 ms** mean frame inference.
* **Deterministic Glass-to-Glass Latency:** **17.90 ms (CPU) / 18.85 ms (GPU)** — strictly compliant with the tactical < 20.0 ms ceiling.
* **Standardized Metric Compliance:**
  * **STOI (Intelligibility):** **0.9303** (Target: > 0.8500)
  * **PESQ (Perceptual Quality):** **3.5867** (Target: > 2.5000)
  * **Hostile Recovery (@ -8 dB):** **+12.75 dB** net gain with STOI lift from 0.45 to 0.64.
* **Failsafe Combat Survivability:** Autonomous fallback to Single-Mic mode (beta = 0.20) preserving **0.8639 STOI** if the external reference microphone is severed or occluded.

---

## 🏗️ Pipeline Architecture

The system operates on 16 kHz discrete audio in 16.0 ms frame strides (R = 256 samples, 50% overlap, N = 512):

```text
[Primary & Reference Audio]
       │
       ▼
 1. Discrete Slew-Rate Limiter ─────────> Attenuates impulse shockwaves (>12 dB mitigation)
       │
       ▼
 2. Cross-Spectral Coherence Filter ────> Spatial ambient diffuse noise nulling
       │
       ▼
 3. Causal GRU Neural Core (L=0) ───────> Predicts sub-millisecond spectral gain mask
       │
       ▼
 4. Static Mask Floor Clamping ─────────> beta = 0.05 / 0.20 floor prevents musical noise
       │
       ▼
 5. Overlap-Add ISTFT Reconstruction ───> Reconstructed speech streamed to headset
⏱️ Glass-to-Glass Latency AllocationPipeline StageOperational MechanismMeasured LatencyInput Frame IngestionR = 256 samples @ 16 kHz (50% overlap, N = 512)16.00 msNeural Inferenceaquanex_causal_gru.onnx (1,035 KB, 2-layer Causal GRU)0.20 ms (CPU)1.15 ms (CUDA)Spectral SynthesisInverse STFT (ISTFT) & Overlap-Add reconstruction~0.50 msHardware Audio I/ODirect ALSA/DMA low-latency ring buffer margin~1.20 msTotal Glass-to-GlassMeasured End-to-End Tactical Audio Path17.90 ms (CPU)18.85 ms (GPU)Target Threshold: Strictly < 20.0 ms (CLEARED).📊 Performance Sweeps Across Noise TiersInput Baseline SNRSingle-Mic GainSingle-Mic STOISingle-Mic PESQDual-Mic GainDual-Mic STOIDual-Mic PESQTactical Viability-8 dB+5.51 dB0.40361.0847+12.75 dB0.63801.6612Voice Recovered (Prevents Call Drop)-4 dB+5.41 dB0.49871.1359+10.65 dB0.71801.9497Intelligibility Restored0 dB+4.99 dB0.60441.3063+8.37 dB0.79412.2852Monotonic Drone Suppression+4 dB+4.21 dB0.70781.5119+5.90 dB0.85682.6568All Mandatory Targets Cleared+8 dB+3.19 dB0.79361.7803+2.74 dB0.89833.0733High-Fidelity Radio Quality+12 dB+2.24 dB0.86392.3012-0.30 dB0.93033.5867Peak Perceptual Quality📂 Repository LayoutPlaintextaquanex/
├── models/
│   ├── model_tactical_v2.pth          # PyTorch checkpoint (~1.1 MB)
│   └── aquanex_causal_gru.onnx         # Production ONNX Opset 14 model (1,035 KB)
├── src/
│   ├── demo_live.py                   # Live mic enhancement with rolling latency telemetry
│   ├── profile_onnx.py                # Standalone ONNX CPU profiler
│   ├── profile_pytorch_cuda.py        # Standalone RTX 3050 CUDA profiler
│   └── generate_clean_onnx.py         # PyTorch -> ONNX Opset 14 export script
├── docs/
│   └── technical_blueprint.md         # Full mathematical derivation and system architecture
├── .gitignore                         # Excludes large cache and environment files
├── requirements.txt                   # Production dependencies
└── README.md                          # Technical overview and benchmarks
🚀 Quickstart & Reproduction1. Environment SetupBashgit clone [https://github.com/shashank-0912/aquanex.git](https://github.com/shashank-0912/aquanex.git)
cd aquanex
pip install -r requirements.txt
2. Run Latency Profilers (2,000 Frames)Bash# Verify Edge CPU ONNX runtime (0.20 ms mean)
python src/profile_onnx.py

# Verify Native CUDA GPU execution on RTX 3050 (1.15 ms mean)
python src/profile_pytorch_cuda.py
3. Launch Live Microphone Enhancement & TelemetryBashpython src/demo_live.py
Plaintext[TELEMETRY] NN Infer:  0.20 ms | Total Compute:  0.37 ms | Glass-to-Glass: 17.57 ms | Frame Budget Used:  2.3%
🛡️ Tactical Viability Profile100% Air-Gapped Security: Zero external network calls, zero RF emission signatures, functional in jammed and GPS-denied environments.SWaP-C Efficient: Fits within a low 7W to 15W thermal envelope on embedded edge hardware (NVIDIA Jetson Orin Nano / ARM Cortex SoCs).