# SpikeGuard

**Energy-Efficient Spiking Neural Network Framework for Tamper-Resilient Wildlife Camera-Trap Monitoring**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-SNN-FF6F00?style=flat)](https://github.com/fangwei123456/spikingjelly)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> Wildlife camera traps face physical tampering — dust, IR glare, foliage, sensor degradation. SpikeGuard trains Spiking Neural Networks (SNNs) that stay accurate under attack while consuming fewer spikes (energy proxy) than conventional ANN baselines.

---

## Why This Matters

Camera traps deployed in the wild are routinely sabotaged or degraded:
- Poachers shine IR lights to blind sensors
- Dust, tape, and vegetation occlude the lens
- Rain and vibration cause motion blur
- Humidity degrades sensor quality

Standard CNNs fail catastrophically under these conditions. SpikeGuard uses **neuromorphic computing** (spiking neurons) with a **multi-objective loss** that jointly optimizes for accuracy, energy efficiency, and tamper robustness.

---

## Architecture

```
Input (64x64) -> Conv(3->32) -> PLIF -> Conv(32->64) -> PLIF -> Pool
              -> Conv(64->128) -> PLIF -> Conv(128->128) -> PLIF -> Pool
              -> Conv(128->256) -> PLIF -> Pool
              -> FC(256x8x8->512) -> PLIF -> FC(512->10 classes)

Output: Mean firing rate across T timesteps -> Wildlife class prediction
```

**Key design choices:**
- **PLIF neurons** (Parametric Leaky Integrate-and-Fire) with ATan surrogate gradients
- **Rate-coded input** — image repeated across configurable timesteps (T=4 default)
- **Spike tracking hooks** on every neuron layer for energy proxy computation
- **VGG-style conv blocks** with BatchNorm, no bias (SNN-compatible)

---

## Multi-Objective Training

```
L_total = L_ce + lambda1 * L_spike_energy + lambda2 * L_robustness
```

| Component | Purpose |
|---|---|
| `L_ce` | Cross-entropy on clean images |
| `L_spike_energy` | Penalizes excess spike count (energy proxy) |
| `L_robustness` | KL divergence between clean and tampered predictions |

The model sees **both clean and tampered images** every training step, learning to maintain stable predictions under physical perturbation.

---

## Physics-Inspired Tampering

SpikeGuard simulates real-world camera-trap attacks with differentiable PyTorch transforms that run on GPU:

| Attack | What It Simulates |
|---|---|
| **Dust Occlusion** | Brownish/greenish rectangular patches with soft edge blending |
| **IR Glare** | Gaussian brightness with reddish-white channel weighting (torch/headlight) |
| **Motion/Rain Blur** | Directional 1D convolution kernels at random angles |
| **Sensor Noise** | Gaussian + salt-and-pepper noise (extreme weather degradation) |
| **Foliage Coverage** | Irregular green-tinted patches with alpha blending |

**Unseen attacks** (held-out test set only) use distinct transforms — textured dust, angled tape, lens flare, fog, camera-shake blur — for honest robustness evaluation.

---

## Quick Start

### Install

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate spikeguard

# Or pip
pip install -r requirements.txt
```

### Train

```bash
# Full SpikeGuard (energy + robustness)
python main.py train --mode spikeguard --epochs 15

# Baselines for comparison
python main.py train --mode baseline_ann --epochs 15
python main.py train --mode vanilla_snn --epochs 15

# Run all 5 modes automatically
python main.py ablation --epochs 10
```

### Export & Benchmark

```bash
# Export to ONNX
python main.py export --checkpoint checkpoints/spikeguard_best.pt

# Quantize for deployment
python main.py quantize --checkpoint checkpoints/spikeguard_best.pt

# Benchmark latency + accuracy
python main.py benchmark --model-path models/spikeguard.onnx

# Launch Streamlit dashboard
python main.py demo
```

---

## Training Modes

| Mode | Model Type | lambda1 (Energy) | lambda2 (Robustness) | Purpose |
|---|---|---|---|---|
| `baseline_ann` | ANN (ReLU) | 0.0 | 0.0 | Standard CNN baseline |
| `vanilla_snn` | SNN (PLIF) | 0.0 | 0.0 | SNN without regularization |
| `energy_only` | SNN (PLIF) | 0.3 | 0.0 | SNN optimized for low spikes |
| `robust_only` | SNN (PLIF) | 0.0 | 0.5 | SNN optimized for tamper resistance |
| `spikeguard` | SNN (PLIF) | 0.1 | 0.5 | **Full method** — both objectives |

---

## Project Structure

```
SpikeGuard/
├── main.py                 # CLI entry point (6 subcommands)
├── utils.py                # Seeding, device, visualization (raster, energy, Pareto)
├── requirements.txt        # pip dependencies
├── environment.yml         # conda environment
│
├── models/
│   └── snn_model.py        # SpikeGuardSNN (PLIF) + BaselineANN (ReLU)
│
├── training/
│   ├── train.py            # Trainer class + ablation runner
│   └── loss.py             # SpikeGuardLoss (CE + energy + KL robustness)
│
├── data/
│   ├── preprocess.py       # CIFAR-10 simulation + real wildlife dataset loader
│   ├── augmentations.py    # Training tampering transforms (5 attacks)
│   └── unseen_attacks.py   # Held-out test transforms (5 different attacks)
│
├── energy_proxy/
│   └── energy.py           # Differentiable spike-count energy proxy
│
├── inference/
│   └── export.py           # ONNX export, quantization, benchmarking
│
├── dashboard/
│   └── app.py              # Streamlit demo interface
│
└── docs/                   # Research notes, abstract, architecture diagrams
```

---

## Wildlife Classes

10-class camera-trap classification:

`tiger` · `leopard` · `elephant` · `deer` · `wild_boar` · `bear` · `monkey` · `bird` · `human` · `blank`

**Data modes:**
- **Simulation** — CIFAR-10 remapped to wildlife classes (fast prototyping)
- **Real** — Folder-based wildlife image dataset (iWildCam or custom)

---

## Technical Highlights

- **Spike raster plots** — visualize neural firing patterns per layer and timestep
- **Pareto frontier** — accuracy vs. energy trade-off visualization across all modes
- **ONNX export** — SNN exported as single-step (T=1) for deployment compatibility
- **Dynamic quantization** — INT8 post-training quantization for edge deployment
- **AMD GPU support** — ONNX Runtime with VitisAI/DML/ROCm provider detection
- **Reproducible** — full seeding (torch, numpy, random, CUDA, PYTHONHASHSEED)

---

## Honest Limitations

- Spike count is a **relative energy proxy**, not absolute joule measurements. Actual hardware energy savings depend on deployment platform (neuromorphic chip vs. conventional GPU/CPU).
- CIFAR-10 simulation mode is for **rapid prototyping**. Real iWildCam data is needed for production evaluation.
- SNN->ONNX export uses single-step inference (T=1). Full temporal spiking dynamics are preserved in training but collapsed for deployment compatibility.

---

## Dependencies

| Category | Packages |
|---|---|
| Deep Learning | PyTorch 2.1+, torchvision, SpikingJelly |
| Inference | ONNX, ONNX Runtime |
| Dashboard | Streamlit, Plotly |
| Data | NumPy, Pandas, OpenCV, scikit-learn, Pillow |
| Logging | TensorBoard, Matplotlib, Seaborn, tqdm |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Neuromorphic computing for conservation — because wildlife deserves efficient AI.</em>
</p>
