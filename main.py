"""
SpikeGuard — CLI Entry Point

Commands:
  train      Train model (multiple modes: spikeguard, baseline_ann, vanilla_snn, etc.)
  ablation   Run all 5 modes for comparison table
  export     Export to ONNX
  quantize   Post-training quantization
  benchmark  Run benchmarks
  demo       Launch Streamlit dashboard
"""
import argparse
import os
import sys
import subprocess


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def cmd_train(args):
    from training.train import run_training
    config = {
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "timesteps": args.timesteps,
        "device": "cuda" if _has_cuda() else "cpu",
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.seed,
        "data_mode": args.data_mode,
        "num_workers": args.num_workers,
    }
    run_training(config)


def cmd_ablation(args):
    from training.train import run_all_ablations
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.seed,
        "data_mode": args.data_mode,
        "num_workers": args.num_workers,
    }
    run_all_ablations(config)


def cmd_export(args):
    from inference.export import export_to_onnx
    if not os.path.exists(args.checkpoint):
        print(f"Error: {args.checkpoint} not found"); sys.exit(1)
    export_to_onnx(args.checkpoint, args.output, timesteps=args.timesteps)


def cmd_quantize(args):
    from inference.export import quantize_model
    if not os.path.exists(args.checkpoint):
        print(f"Error: {args.checkpoint} not found"); sys.exit(1)
    quantize_model(args.checkpoint, args.output)


def cmd_benchmark(args):
    from inference.export import benchmark
    if not os.path.exists(args.model_path):
        print(f"Error: {args.model_path} not found"); sys.exit(1)
    benchmark(args.model_path, batch_size=args.batch_size)


def cmd_demo(args):
    app = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    print("\n🐾 Launching SpikeGuard demo dashboard...")
    print("   Open http://localhost:8501\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app,
                    "--server.headless", "true"])


def main():
    p = argparse.ArgumentParser(
        description="🐾 SpikeGuard — Energy-Efficient SNN for Wildlife Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --mode spikeguard --epochs 15
  python main.py train --mode baseline_ann --epochs 15
  python main.py ablation --epochs 10
  python main.py export --checkpoint checkpoints/spikeguard_best.pt
  python main.py benchmark --model-path models/spikeguard.onnx
  python main.py demo
        """)

    sub = p.add_subparsers(dest="command")

    # Train
    t = sub.add_parser("train", help="Train a model")
    t.add_argument("--mode", default="spikeguard",
                   choices=["spikeguard", "baseline_ann", "vanilla_snn",
                            "energy_only", "robust_only"])
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lambda1", type=float, default=0.1)
    t.add_argument("--lambda2", type=float, default=0.5)
    t.add_argument("--timesteps", type=int, default=4)
    t.add_argument("--log-dir", default="runs")
    t.add_argument("--checkpoint-dir", default="checkpoints")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--data-mode", default="simulation", choices=["simulation", "real"])
    t.add_argument("--num-workers", type=int, default=2)

    # Ablation
    a = sub.add_parser("ablation", help="Run all 5 modes for comparison")
    a.add_argument("--epochs", type=int, default=10)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--timesteps", type=int, default=4)
    a.add_argument("--log-dir", default="runs")
    a.add_argument("--checkpoint-dir", default="checkpoints")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--data-mode", default="simulation")
    a.add_argument("--num-workers", type=int, default=2)

    # Export
    e = sub.add_parser("export", help="Export to ONNX")
    e.add_argument("--checkpoint", default="checkpoints/spikeguard_best.pt")
    e.add_argument("--output", default="models/spikeguard.onnx")
    e.add_argument("--timesteps", type=int, default=4)

    # Quantize
    q = sub.add_parser("quantize", help="Quantize model")
    q.add_argument("--checkpoint", default="checkpoints/spikeguard_best.pt")
    q.add_argument("--output", default="models/spikeguard_quantized.pt")

    # Benchmark
    b = sub.add_parser("benchmark", help="Run benchmarks")
    b.add_argument("--model-path", default="models/spikeguard.onnx")
    b.add_argument("--batch-size", type=int, default=64)

    # Demo
    sub.add_parser("demo", help="Launch dashboard")

    args = p.parse_args()
    if args.command is None:
        p.print_help()
        sys.exit(0)

    {"train": cmd_train, "ablation": cmd_ablation, "export": cmd_export,
     "quantize": cmd_quantize, "benchmark": cmd_benchmark, "demo": cmd_demo
     }[args.command](args)


if __name__ == "__main__":
    main()
