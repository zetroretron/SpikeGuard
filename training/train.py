"""
SpikeGuard — Full Training Pipeline

Training loop with:
- Dual forward pass (clean + tampered)
- Multi-objective backpropagation
- TensorBoard logging (clean_acc, tampered_acc, avg_spikes, total_loss)
- Checkpoint saving
- Support for baseline, vanilla SNN, energy-only, robust-only, and full SpikeGuard
"""
import os
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.preprocess import get_dataloaders
from data.augmentations import get_tamper_pipeline
from data.unseen_attacks import get_unseen_tamper_pipeline
from models.snn_model import create_snn, create_ann
from energy_proxy.energy import SpikeEnergyProxy
from training.loss import SpikeGuardLoss
from utils import set_seed, get_device, NUM_CLASSES


# ─── Training Modes ──────────────────────────────────────────
TRAINING_MODES = {
    "baseline_ann":    {"model": "ann", "lambda1": 0.0, "lambda2": 0.0},
    "vanilla_snn":     {"model": "snn", "lambda1": 0.0, "lambda2": 0.0},
    "energy_only":     {"model": "snn", "lambda1": 0.3, "lambda2": 0.0},
    "robust_only":     {"model": "snn", "lambda1": 0.0, "lambda2": 0.5},
    "spikeguard":      {"model": "snn", "lambda1": 0.1, "lambda2": 0.5},
}


class Trainer:
    """Multi-objective SNN trainer for SpikeGuard."""

    def __init__(self, config):
        self.config = config
        set_seed(config.get("seed", 42))
        self.device = get_device()

        # Resolve training mode
        mode_name = config.get("mode", "spikeguard")
        mode_config = TRAINING_MODES.get(mode_name, TRAINING_MODES["spikeguard"])

        # Override with explicit lambdas if provided
        lambda1 = config.get("lambda1", mode_config["lambda1"])
        lambda2 = config.get("lambda2", mode_config["lambda2"])

        # Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            mode=config.get("data_mode", "simulation"),
            batch_size=config.get("batch_size", 64),
            num_workers=config.get("num_workers", 2),
        )

        # Model
        timesteps = config.get("timesteps", 4)
        if mode_config["model"] == "snn":
            self.model = create_snn(
                num_classes=NUM_CLASSES, timesteps=timesteps,
            ).to(self.device)
            self.is_snn = True
        else:
            self.model = create_ann(num_classes=NUM_CLASSES).to(self.device)
            self.is_snn = False

        # Energy proxy (SNN only)
        self.energy_proxy = None
        if self.is_snn:
            self.energy_proxy = SpikeEnergyProxy(self.model)

        # Loss
        self.criterion = SpikeGuardLoss(lambda1=lambda1, lambda2=lambda2)

        # Tampering
        self.tamper_train = get_tamper_pipeline(max_transforms=2)
        self.tamper_unseen = get_unseen_tamper_pipeline(max_transforms=2)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=1e-4,
        )
        epochs = config.get("epochs", 15)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )

        # Logging
        self.log_dir = os.path.join(config.get("log_dir", "runs"), mode_name)
        self.writer = SummaryWriter(self.log_dir)

        # Checkpoints
        self.ckpt_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_val_acc = 0.0
        self.global_step = 0
        self.mode_name = mode_name

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        stats = {"loss": 0, "ce": 0, "spike": 0, "robust": 0,
                 "clean_ok": 0, "tamp_ok": 0, "total": 0, "spikes": 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 1. Clean forward
            clean_logits = self.model(images)

            # 2. Spike energy (SNN only)
            spike_energy = torch.tensor(0.0, device=self.device)
            spike_rate_loss = torch.tensor(0.0, device=self.device)
            if self.is_snn and self.energy_proxy:
                se, srl, energy_comp = self.energy_proxy.compute_as_tensor(self.device)
                spike_energy = se
                spike_rate_loss = srl
                stats["spikes"] += energy_comp["energy_normalized"]

            # 3. Tampered forward
            with torch.no_grad():
                tampered_images = self.tamper_train(images)
            tampered_logits = self.model(tampered_images)

            # 4. Loss
            loss, loss_comp = self.criterion(
                clean_logits, tampered_logits, labels,
                spike_energy, spike_rate_loss,
            )

            # 5. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 6. Track
            bs = labels.size(0)
            stats["loss"] += loss_comp["loss_total"] * bs
            stats["ce"] += loss_comp["loss_ce"] * bs
            stats["spike"] += loss_comp["loss_spike_energy"] * bs
            stats["robust"] += loss_comp["loss_robust"] * bs
            stats["clean_ok"] += (clean_logits.argmax(1) == labels).sum().item()
            stats["tamp_ok"] += (tampered_logits.argmax(1) == labels).sum().item()
            stats["total"] += bs

            self.global_step += 1
            if self.global_step % 30 == 0:
                self.writer.add_scalar("step/loss", loss_comp["loss_total"], self.global_step)
                self.writer.add_scalar("step/ce", loss_comp["loss_ce"], self.global_step)
                pbar.set_postfix(loss=f"{loss_comp['loss_total']:.3f}",
                                 acc=f"{stats['clean_ok']/stats['total']:.3f}")

        n = stats["total"]
        metrics = {
            "train/loss": stats["loss"] / n,
            "train/loss_ce": stats["ce"] / n,
            "train/loss_spike": stats["spike"] / n,
            "train/loss_robust": stats["robust"] / n,
            "train/clean_acc": stats["clean_ok"] / n,
            "train/tampered_acc": stats["tamp_ok"] / n,
            "train/avg_spikes": stats["spikes"] / max(len(self.train_loader), 1),
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)
        return metrics

    @torch.no_grad()
    def validate(self, epoch, use_unseen=False):
        """Validate on val set."""
        self.model.eval()
        tamper = self.tamper_unseen if use_unseen else self.tamper_train
        clean_ok, tamp_ok, total = 0, 0, 0
        total_spikes = 0.0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            clean_logits = self.model(images)
            if self.is_snn and self.energy_proxy:
                _, _, ec = self.energy_proxy.compute_as_tensor(self.device)
                total_spikes += ec["energy_normalized"]

            clean_ok += (clean_logits.argmax(1) == labels).sum().item()

            tampered = tamper(images)
            tamp_logits = self.model(tampered)
            tamp_ok += (tamp_logits.argmax(1) == labels).sum().item()

            total += labels.size(0)

        prefix = "val_unseen" if use_unseen else "val"
        metrics = {
            f"{prefix}/clean_acc": clean_ok / total,
            f"{prefix}/tampered_acc": tamp_ok / total,
            f"{prefix}/acc_drop": (clean_ok - tamp_ok) / total,
            f"{prefix}/avg_spikes": total_spikes / max(len(self.val_loader), 1),
        }
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)
        return metrics

    def save_checkpoint(self, epoch, metrics, filename=None):
        if filename is None:
            filename = f"{self.mode_name}_epoch{epoch}.pt"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "mode": self.mode_name,
            "is_snn": self.is_snn,
        }, path)
        return path

    def train(self):
        """Full training loop."""
        epochs = self.config.get("epochs", 15)
        print(f"\n{'='*60}")
        print(f"  🐾 SpikeGuard Training — [{self.mode_name.upper()}]")
        print(f"  Epochs: {epochs} | λ1={self.criterion.lambda1} | λ2={self.criterion.lambda2}")
        print(f"  Model: {'SNN' if self.is_snn else 'ANN'} | Device: {self.device}")
        print(f"  TensorBoard: {self.log_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch, use_unseen=False)
            unseen_m = self.validate(epoch, use_unseen=True)
            self.scheduler.step()
            elapsed = time.time() - t0

            print(f"E{epoch:2d}/{epochs} ({elapsed:.0f}s) | "
                  f"Loss:{train_m['train/loss']:.3f} | "
                  f"Clean:{val_m['val/clean_acc']:.3f} | "
                  f"Tamp:{val_m['val/tampered_acc']:.3f} | "
                  f"Unseen:{unseen_m['val_unseen/tampered_acc']:.3f} | "
                  f"Spikes:{val_m['val/avg_spikes']:.1f}")

            if val_m["val/clean_acc"] > self.best_val_acc:
                self.best_val_acc = val_m["val/clean_acc"]
                self.save_checkpoint(epoch, val_m, f"{self.mode_name}_best.pt")
                print(f"  ★ Best model saved ({self.best_val_acc:.4f})")

        self.save_checkpoint(epochs, val_m, f"{self.mode_name}_final.pt")
        self.writer.close()

        print(f"\n{'='*60}")
        print(f"  ✓ Training complete: {self.mode_name}")
        print(f"  Best accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        return self.model


def run_training(config):
    """Entry point."""
    trainer = Trainer(config)
    return trainer.train()


def run_all_ablations(base_config):
    """Run all 5 training modes for comparison."""
    results = {}
    for mode_name in TRAINING_MODES:
        config = {**base_config, "mode": mode_name}
        print(f"\n{'#'*60}")
        print(f"  ABLATION: {mode_name}")
        print(f"{'#'*60}")
        trainer = Trainer(config)
        trainer.train()

        # Final evaluation
        val_m = trainer.validate(0, use_unseen=True)
        results[mode_name] = val_m
        results[mode_name]["model"] = "ANN" if not trainer.is_snn else "SNN"

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Mode':<18} {'Type':<6} {'Clean Acc':<12} {'Unseen Tamp':<14} {'Acc Drop':<12} {'Spikes':<10}")
    print("-" * 80)
    for mode, m in results.items():
        print(f"{mode:<18} {m.get('model','?'):<6} "
              f"{m.get('val_unseen/clean_acc', 0):<12.4f} "
              f"{m.get('val_unseen/tampered_acc', 0):<14.4f} "
              f"{m.get('val_unseen/acc_drop', 0):<12.4f} "
              f"{m.get('val_unseen/avg_spikes', 0):<10.1f}")
    print(f"{'='*80}\n")
    return results
