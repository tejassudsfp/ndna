#!/usr/bin/env python3
"""
Developmental Genome experiments.

Usage:
    python3 run.py train       Train genome on MNIST
    python3 run.py transfer    Cross-task transfer test (Fashion -> MNIST)
    python3 run.py cifar10     Train genome on CIFAR-10 (MLP)
    python3 run.py cnn         Train genome CNN on CIFAR-10
    python3 run.py transfer100 Transfer CIFAR-10 genome to CIFAR-100
    python3 run.py transformer Genome transformer on IMDB (Rung 3)
    python3 run.py visualize   Generate dashboard from saved genome
    python3 run.py results     Print all saved results
"""

import sys
import os
import json
import glob

# Ensure we run from the worldmodel directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def cmd_train():
    from experiments.train_mnist import run
    run()


def cmd_transfer():
    from experiments.transfer import run
    run()


def cmd_cifar10():
    from experiments.train_cifar10 import run
    run()


def cmd_cnn():
    from experiments.train_cifar10_cnn import run
    run()


def cmd_transfer100():
    from experiments.transfer_cifar100 import run
    run()


def cmd_transformer():
    from experiments.rung3_transformer import run
    run()


def cmd_visualize():
    import torch
    from genome import Genome, GrownNetwork
    from genome.visualizer import show_dashboard

    # Find latest genome .pt file
    pts = sorted(glob.glob("results/genome_*.pt"), key=os.path.getmtime)
    if not pts:
        print("No saved genome found. Run 'python3 run.py train' first.")
        sys.exit(1)

    latest = pts[-1]
    print(f"Loading genome from {latest}")

    # Detect config from filename
    hidden_bands = [48, 48, 48, 48]
    input_dim = 784
    if 'cifar10' in latest:
        hidden_bands = [128, 128, 128, 128]
        input_dim = 3072

    dims = [input_dim] + hidden_bands + [10]
    genome = Genome(n_types=8, type_dim=8, n_bands=len(dims))
    genome.load_state_dict(torch.load(latest, weights_only=True))

    model = GrownNetwork(genome, input_dim, hidden_bands, 10)

    os.makedirs('results', exist_ok=True)
    show_dashboard(genome, model, save_path="results/genome_viz.png")
    print("Dashboard saved to results/genome_viz.png")


def cmd_results():
    jsons = sorted(glob.glob("results/*.json"))
    if not jsons:
        print("No saved results. Run an experiment first.")
        sys.exit(1)

    print("=" * 60)
    print("  SAVED EXPERIMENT RESULTS")
    print("=" * 60)

    for jpath in jsons:
        with open(jpath) as f:
            data = json.load(f)

        exp = data.get("experiment", "unknown")
        ts = data.get("timestamp", "")
        total_time = data.get("total_time", 0)
        config = data.get("config", {})
        results = data.get("results", {})

        print(f"\n  {exp} ({ts[:19]})")
        print(f"  Config: bands={config.get('hidden_bands')}, "
              f"epochs={config.get('n_epochs')}, "
              f"device={config.get('device')}")
        print(f"  Time: {total_time:.0f}s")

        print(f"  {'Model':<22} {'Accuracy':>10} {'Params':>10}")
        print(f"  {'-' * 46}")
        for name, r in results.items():
            acc = r.get('acc', 0)
            params = r.get('params', r.get('genome_params', ''))
            print(f"  {name:<22} {acc:>10.4f} {str(params):>10}")

        # Key comparison
        grown = results.get('grown', {})
        random = results.get('random_sparse', results.get('mnist_random', {}))
        if grown and random:
            gap = (grown.get('acc', 0) - random.get('acc', 0)) * 100
            label = "GENOME WINS" if gap > 0 else "RANDOM WINS"
            print(f"  Genome vs Random: {'+' if gap >= 0 else ''}{gap:.2f}% ({label})")

    print()


COMMANDS = {
    'train': cmd_train,
    'transfer': cmd_transfer,
    'cifar10': cmd_cifar10,
    'cnn': cmd_cnn,
    'transfer100': cmd_transfer100,
    'transformer': cmd_transformer,
    'visualize': cmd_visualize,
    'results': cmd_results,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"  Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
