"""
SpikeGuard — Multi-Objective Loss

L_total = L_ce + λ1 × normalized_spike_count + λ2 × robustness_loss

robustness_loss = KL(clean_softmax || tampered_softmax)
Encourages stable predictions under physical perturbation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustnessLoss(nn.Module):
    """
    KL divergence between clean and tampered output distributions.
    Encourages prediction stability under physical tampering.
    """

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, clean_logits, tampered_logits):
        clean_log_prob = F.log_softmax(clean_logits / self.temperature, dim=1)
        tampered_prob = F.softmax(tampered_logits / self.temperature, dim=1)
        loss = F.kl_div(clean_log_prob, tampered_prob, reduction="batchmean")
        return loss * (self.temperature ** 2)


class SpikeGuardLoss(nn.Module):
    """
    Multi-objective loss for SpikeGuard.

    L_total = L_ce + λ1 × spike_energy + λ2 × robustness_loss

    Args:
        lambda1: Weight for spike energy regularization.
        lambda2: Weight for tamper robustness.
    """

    def __init__(self, lambda1=0.1, lambda2=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ce_loss = nn.CrossEntropyLoss()
        self.robust_loss = RobustnessLoss()

    def forward(self, clean_logits, tampered_logits, targets,
                spike_energy, spike_rate_loss):
        """
        Args:
            clean_logits: (B, C) clean predictions.
            tampered_logits: (B, C) tampered predictions.
            targets: (B,) ground truth labels.
            spike_energy: Scalar energy proxy (normalized).
            spike_rate_loss: Scalar L1 spike rate penalty.

        Returns:
            total_loss, components_dict
        """
        l_ce = self.ce_loss(clean_logits, targets)
        l_robust = self.robust_loss(clean_logits, tampered_logits)

        # Spike energy term (combines normalized energy + L1 rate penalty)
        if isinstance(spike_energy, torch.Tensor):
            l_spike = spike_energy + spike_rate_loss
        else:
            l_spike = torch.tensor(spike_energy + spike_rate_loss,
                                   dtype=torch.float32,
                                   device=clean_logits.device)

        total = l_ce + self.lambda1 * l_spike + self.lambda2 * l_robust

        components = {
            "loss_total": total.item(),
            "loss_ce": l_ce.item(),
            "loss_spike_energy": l_spike.item() if isinstance(l_spike, torch.Tensor) else l_spike,
            "loss_robust": l_robust.item(),
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
        }

        return total, components
