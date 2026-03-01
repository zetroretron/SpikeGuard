"""
SpikeGuard — Differentiable Spike-Count Energy Proxy

Primary: Total spike count across all timesteps/layers (normalized 1-10)
Secondary: L1 regularization on spike rates (attachable to loss)

HONEST CLAIM: "Spike count is a useful training proxy for relative energy
comparison. Actual hardware energy savings depend on deployment hardware
(neuromorphic or conventional). We do NOT claim absolute joule measurements."
"""
import torch


class SpikeEnergyProxy:
    """
    Differentiable spike-count energy proxy for SNN training.

    Computes:
        primary:   normalized total spike count (1-10 scale)
        secondary: L1 regularization on per-layer spike rates
    """

    def __init__(self, model, target_spike_rate=0.3):
        """
        Args:
            model: SpikeGuardSNN instance with spike tracking.
            target_spike_rate: Target average spike rate (0-1).
                              Lower = more energy efficient.
        """
        self.model = model
        self.target_spike_rate = target_spike_rate

    def compute(self):
        """
        Compute spike energy proxy AFTER a forward pass.

        Returns:
            energy_normalized: Spike count normalized to 1-10 scale.
            spike_rate_loss: L1 penalty on spike rates (differentiable-safe).
            components: Dict with detailed breakdown.
        """
        normalized, total_spikes, total_elements = self.model.get_total_spike_count()

        # Per-layer spike rates for L1 regularization
        layer_rates = self.model.get_spike_rates_per_layer()
        avg_rate = sum(layer_rates.values()) / max(len(layer_rates), 1)

        # L1 penalty: penalize spike rates above target
        spike_rate_loss = abs(avg_rate - self.target_spike_rate)

        components = {
            "energy_normalized": normalized,
            "total_spikes": total_spikes,
            "total_elements": total_elements,
            "avg_spike_rate": avg_rate,
            "spike_rate_loss": spike_rate_loss,
            "layer_rates": layer_rates,
        }

        return normalized, spike_rate_loss, components

    def compute_as_tensor(self, device="cpu"):
        """Returns energy as tensor for loss computation."""
        normalized, spike_rate_loss, components = self.compute()
        energy_tensor = torch.tensor(normalized / 10.0,  # Scale to ~[0.1, 1.0]
                                     dtype=torch.float32, device=device)
        spike_loss_tensor = torch.tensor(spike_rate_loss,
                                         dtype=torch.float32, device=device)
        return energy_tensor, spike_loss_tensor, components
