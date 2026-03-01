# 🐾 SpikeGuard — Energy-Efficient SNN for Tamper-Resilient Wildlife Monitoring

**AMD Slingshot 2026 — Sustainable AI / AI for Good Track**
**Team RDx, Mumbai**

> An edge-AI training framework using Spiking Neural Networks to jointly optimize classification accuracy, energy efficiency (via spike-count regularization), and robustness to real-world camera-trap tampering — for solar-powered sensors in Indian forests.

---

## Architecture

```
SpikeGuard/
├── data/
│   ├── preprocess.py            # Wildlife dataset pipeline (sim + real)
│   ├── augmentations.py         # 5 physics-inspired tampering transforms (train)
│   └── unseen_attacks.py        # 5 distinct held-out transforms (eval only)
├── models/
│   └── snn_model.py             # SpikingJelly SNN (PLIF) + baseline ANN
├── energy_proxy/
│   └── energy.py                # Differentiable spike-count proxy
├── training/
│   ├── loss.py                  # Multi-objective: CE + spike + KL robustness
│   └── train.py                 # Full loop with 5 training modes
├── inference/
│   └── export.py                # ONNX export + quantization + benchmark
├── dashboard/
│   └── app.py                   # Streamlit demo (conservation-green theme)
├── docs/
│   ├── abstract.md              # 400-word submission abstract
│   ├── pitch_deck.md            # 10-slide outline
│   ├── video_script.md          # 2-minute demo narration
│   ├── architecture.md          # Mermaid architecture diagram
│   └── evaluation.md            # Evaluation + Pareto + limitations
├── utils.py                     # Seeding, visualization, spike raster
├── main.py                      # CLI entry point (6 commands)
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Loss Function

```
L_total = L_ce + λ1 × spike_energy_proxy + λ2 × KL(clean || tampered)
```

| Term | What It Does |
|---|---|
| `L_ce` | Cross-entropy on clean images |
| `spike_energy_proxy` | Normalized spike count (1–10 scale) + L1 rate penalty |
| `KL robustness` | Encourages stable output distributions under physical tampering |

**λ1=0.1, λ2=0.5** (configurable via CLI).

### Honest Claim

> Spike count is a **useful training proxy** for relative energy comparison between models trained with different objectives. **Actual hardware energy savings depend on deployment hardware** (neuromorphic or conventional). We do NOT claim absolute joule measurements.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate spikeguard
```

### 2. Train

```bash
# Full SpikeGuard (spike regularization + tamper robustness)
python main.py train --mode spikeguard --epochs 15 --timesteps 4

# Baseline ANN comparison
python main.py train --mode baseline_ann --epochs 15

# Run all 5 ablations
python main.py ablation --epochs 10
```

### 3. Export to ONNX

```bash
python main.py export --checkpoint checkpoints/spikeguard_best.pt
```

### 4. Benchmark

```bash
python main.py benchmark --model-path models/spikeguard.onnx
```

### 5. Launch Demo

```bash
python main.py demo
# → http://localhost:8501
```

### 6. TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Training Modes (Ablation)

| Mode | λ1 | λ2 | Description |
|---|---|---|---|
| `baseline_ann` | 0 | 0 | Standard ANN (ReLU, no spikes) |
| `vanilla_snn` | 0 | 0 | SNN without energy/robustness objectives |
| `energy_only` | 0.3 | 0 | SNN + spike-count regularization only |
| `robust_only` | 0 | 0.5 | SNN + tamper robustness only |
| `spikeguard` | 0.1 | 0.5 | Full system: accuracy + energy + robustness |

---

## Tampering Simulation

### Training Set (5 transforms)
Dust occlusion, IR glare, motion/rain blur, sensor noise, foliage coverage

### Unseen Evaluation Set (5 DIFFERENT transforms)
Textured dust overlay, angled tape patches, lens flare with rings, fog/condensation, camera-shake blur

Results are reported on the **unseen** set only for robustness claims.

---

## AMD Optimization

```bash
# ONNX export (opset 18 for AMD compatibility)
python main.py export --checkpoint checkpoints/spikeguard_best.pt

# Benchmark (auto-detects VitisAI / DirectML / ROCm / CPU)
python main.py benchmark --model-path models/spikeguard.onnx

# For Ryzen AI NPU readiness, install:
# pip install onnxruntime-vitisai  (2026 AMD provider)
# The benchmark will automatically attempt NPU acceleration.
```

**Providers detected**: VitisAIExecutionProvider > DmlExecutionProvider > ROCmExecutionProvider > CPUExecutionProvider

---

## Hardware

- **Training**: Any laptop with 8GB+ RAM (CPU or CUDA)
- **Inference**: AMD Ryzen AI / Intel Arc / CPU — all supported via ONNX Runtime
- **Model**: ~1.2M params (VGG-style SNN), <10MB ONNX
- **Training time**: ~2-4 hours for 15 epochs on CPU, ~30 min on GPU

---

## What This Is NOT

- ❌ Neuromorphic hardware project
- ❌ Real silicon energy measurement
- ❌ Military-grade deployment claim

## What This IS

- ✅ A hardware-aware SNN training framework
- ✅ Honest spike-count energy proxy (proxy, not measurement)
- ✅ Tamper-resilient inference for conservation edge sensors
- ✅ Reproducible, laptop-runnable, hackathon-ready

---

## Conservation Context

**Target deployment**: Solar-powered camera traps in Indian wildlife corridors (Western Ghats, Bandipur, Nilgiri Biosphere). Anti-poaching and biodiversity monitoring where:
- Cloud connectivity is unavailable
- Power budget is ≤500mW (solar cell)
- Cameras face physical tampering (dust, vegetation, IR sabotage)

**Impact**: Lower spike counts = longer battery life = more data from remote sensors.

---

## References

- SpikingJelly (ICLR 2026) — Memory-optimized SNN framework
- NeuEdge (arXiv Feb 2026) — Hardware-aware SNN training
- PyTorch-Wildlife (2025/26) — Conservation-focused computer vision
- AMD Quark / AI Analyzer (2026) — Unified quantization and profiling

---

**Team RDx, Mumbai | AMD Slingshot 2026**
