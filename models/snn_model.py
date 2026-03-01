"""
SpikeGuard — SNN Model (SpikingJelly)

VGG-style Spiking Neural Network with PLIF neurons and surrogate gradient.
Designed for CIFAR-10-scale wildlife classification (64x64 input).

Key features:
- Parametric Leaky Integrate-and-Fire (PLIF) neurons
- Surrogate gradient training (ATan)
- Spike count tracking for energy proxy
- Rate coding input layer
- 4–8 timesteps configurable
"""
import torch
import torch.nn as nn

# SpikingJelly imports with fallback
try:
    from spikingjelly.activation_based import neuron, layer, functional, surrogate
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    print("[WARNING] SpikingJelly not installed. Using fallback surrogate LIF.")


# ─── Fallback LIF Neuron (if SpikingJelly not available) ──────
class FallbackLIFNeuron(nn.Module):
    """Simple LIF neuron with surrogate gradient for when SpikingJelly isn't available."""

    def __init__(self, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau))
        self.v_threshold = v_threshold
        self.v = None
        self.spike_count = 0

    def reset(self):
        self.v = None
        self.spike_count = 0

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)

        self.v = self.v * (1.0 - 1.0 / self.tau.abs().clamp(min=1.01)) + x

        # Surrogate gradient (ATan)
        spike = self._surrogate_fire(self.v - self.v_threshold)
        self.v = self.v * (1.0 - spike)
        self.spike_count += spike.sum().item()
        return spike

    def _surrogate_fire(self, x):
        """ATan surrogate gradient."""
        return (x > 0).float() + (torch.atan(x * 2.0) / math.pi + 0.5) - \
               (torch.atan(x * 2.0) / math.pi + 0.5).detach()


import math


# ─── SNN Model ────────────────────────────────────────────────
class SpikeGuardSNN(nn.Module):
    """
    VGG-style SNN for camera-trap wildlife classification.

    Architecture:
        Conv(3→32) → PLIF → Conv(32→64) → PLIF → Pool →
        Conv(64→128) → PLIF → Conv(128→128) → PLIF → Pool →
        Conv(128→256) → PLIF → Pool →
        FC(256*8*8→512) → PLIF → FC(512→num_classes)

    Input: Rate-coded image repeated for T timesteps.
    Output: Mean firing rate across timesteps as logits.
    """

    def __init__(self, num_classes=10, timesteps=4):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.spike_counts = {}

        if SPIKINGJELLY_AVAILABLE:
            NeuronClass = lambda: neuron.ParametricLIFNode(
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
            )
        else:
            NeuronClass = lambda: FallbackLIFNeuron(tau=2.0)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            NeuronClass(),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            NeuronClass(),
            nn.AvgPool2d(2),  # 64 → 32

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            NeuronClass(),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            NeuronClass(),
            nn.AvgPool2d(2),  # 32 → 16

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            NeuronClass(),
            nn.AvgPool2d(2),  # 16 → 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512, bias=False),
            NeuronClass(),
            nn.Linear(512, num_classes, bias=False),
        )

        # Register spike hooks
        self._register_spike_hooks()

    def _register_spike_hooks(self):
        """Register hooks to count spikes per layer."""
        self._spike_hooks = []
        self._spike_data = {}

        for name, module in self.named_modules():
            if SPIKINGJELLY_AVAILABLE:
                if isinstance(module, neuron.ParametricLIFNode):
                    hook = module.register_forward_hook(
                        self._make_spike_hook(name)
                    )
                    self._spike_hooks.append(hook)
            else:
                if isinstance(module, FallbackLIFNeuron):
                    hook = module.register_forward_hook(
                        self._make_spike_hook(name)
                    )
                    self._spike_hooks.append(hook)

    def _make_spike_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._spike_data[name] = output.detach()
        return hook_fn

    def forward(self, x):
        """
        Forward pass with temporal integration.

        Args:
            x: Input tensor (B, C, H, W) — will be rate-coded across timesteps.

        Returns:
            logits: (B, num_classes) — mean membrane potential over timesteps.
        """
        B = x.shape[0]

        # Reset spiking neuron states
        if SPIKINGJELLY_AVAILABLE:
            functional.reset_net(self)
        else:
            for m in self.modules():
                if isinstance(m, FallbackLIFNeuron):
                    m.reset()

        # Accumulate output over timesteps
        output_sum = torch.zeros(B, self.num_classes, device=x.device)
        self._all_spikes = {t: {} for t in range(self.timesteps)}

        for t in range(self.timesteps):
            # Rate coding: each timestep gets the same input
            # (with optional Poisson encoding — here we use direct)
            self._spike_data = {}
            feat = self.features(x)
            out = self.classifier(feat)
            output_sum += out

            # Store spikes for this timestep
            self._all_spikes[t] = {k: v.clone() for k, v in self._spike_data.items()}

        # Average over timesteps
        logits = output_sum / self.timesteps
        return logits

    def get_total_spike_count(self):
        """
        Get total spike count across all layers and timesteps.
        Returns normalized count (1-10 scale).
        """
        total_spikes = 0.0
        total_elements = 0

        for t, layer_spikes in self._all_spikes.items():
            for name, spikes in layer_spikes.items():
                total_spikes += spikes.sum().item()
                total_elements += spikes.numel()

        # Normalize spike rate to 1-10 range
        if total_elements > 0:
            avg_rate = total_spikes / total_elements  # [0, 1]
            normalized = 1.0 + avg_rate * 9.0  # [1, 10]
        else:
            normalized = 5.0

        return normalized, total_spikes, total_elements

    def get_spike_rates_per_layer(self):
        """Get spike rate per layer for visualization."""
        rates = {}
        for t, layer_spikes in self._all_spikes.items():
            for name, spikes in layer_spikes.items():
                if name not in rates:
                    rates[name] = []
                rate = spikes.mean().item()
                rates[name].append(rate)
        return {k: sum(v) / len(v) for k, v in rates.items()}

    def get_spike_raster_data(self, max_neurons=50):
        """Get spike data for raster plot visualization."""
        raster = {}
        for name in list(self._all_spikes[0].keys())[:5]:  # First 5 layers
            timestep_data = []
            for t in range(self.timesteps):
                if name in self._all_spikes[t]:
                    s = self._all_spikes[t][name]
                    # Take first sample, flatten, take first N neurons
                    flat = s[0].flatten()[:max_neurons]
                    timestep_data.append(flat.cpu())
            if timestep_data:
                raster[name] = torch.stack(timestep_data)  # (T, N)
        return raster


# ─── ANN Baseline ────────────────────────────────────────────
class BaselineANN(nn.Module):
    """
    Equivalent ANN architecture (same conv structure, ReLU instead of LIF).
    Used for fair comparison.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_snn(num_classes=10, timesteps=4):
    """Factory: create SpikeGuard SNN."""
    return SpikeGuardSNN(num_classes=num_classes, timesteps=timesteps)


def create_ann(num_classes=10):
    """Factory: create baseline ANN."""
    return BaselineANN(num_classes=num_classes)
