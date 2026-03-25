"""
Genome Transfer Test.

Train genome on Fashion-MNIST, freeze it, use it to grow a network for MNIST.
Only train weights. Topology is fixed from Fashion.
Tests whether the genome learns general structural knowledge.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import time, json
from datetime import datetime

from genome import Genome, GrownNetwork, RandomSparseNetwork, NormalMLP

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data(name, bs=128):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DS = datasets.FashionMNIST if name == 'fashion' else datasets.MNIST
    tr = DS('./data', train=True, download=True, transform=tf)
    te = DS('./data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True),
            torch.utils.data.DataLoader(te, batch_size=bs, shuffle=False))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def evaluate(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            c += (model(bx).argmax(-1) == by).sum().item()
            t += bx.size(0)
    return c / t


def train_grown(model, tr, te, n_ep=25, lr=1e-3, sp_w=0.1, train_genome=True):
    if train_genome:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        w_params = list(model.weights.parameters()) + list(model.biases.parameters())
        opt = torch.optim.Adam(w_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
    best = 0
    for ep in range(n_ep):
        model.train()
        for bx, by in tr:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            ce = F.cross_entropy(model(bx), by)
            sp = model.genome.sparsity_loss(model.dims) if train_genome else 0
            loss = ce + sp_w * sp if train_genome else ce
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        if ep % 5 == 0 or ep == n_ep - 1:
            a, t, sd = model.count_effective()
            print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f} hard={a / t:.1%} soft={sd:.1%}")
    return best


def train_simple(model, tr, te, n_ep=25, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
    best = 0
    for ep in range(n_ep):
        model.train()
        for bx, by in tr:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        if ep % 5 == 0 or ep == n_ep - 1:
            print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f}")
    return best


def run():
    print("=" * 60)
    print("  GENOME TRANSFER TEST")
    print("  Train genome on Fashion, transfer to MNIST")
    print("=" * 60)
    print(f"  Device: {device}")

    hidden_bands = [48, 48, 48, 48]
    dims = [784] + hidden_bands + [10]
    n_ep = 25
    results = {}

    # PHASE 1: Train genome on Fashion-MNIST
    print("\n  PHASE 1: Train genome on Fashion-MNIST")
    print("  " + "-" * 50)

    fashion_tr, fashion_te = load_data('fashion')

    print(f"\n  [1] GROWN on Fashion (training genome)")
    genome_fashion = Genome(n_types=8, type_dim=8, n_bands=len(dims))
    m = GrownNetwork(genome_fashion, 784, hidden_bands, 10).to(device)
    gp = sum(x.numel() for x in genome_fashion.parameters())
    print(f"    Genome: {gp:,} params")
    t0 = time.time()
    acc = train_grown(m, fashion_tr, fashion_te, n_ep, sp_w=0.1)
    a, t, sd = m.count_effective()
    results['fashion_grown'] = {
        'acc': acc, 'time': time.time() - t0,
        'hard_density': a / t, 'soft_density': sd
    }
    print(f"    Fashion accuracy: {acc:.4f}, hard={a / t:.1%} soft={sd:.1%}")

    trained_genome_state = {k: v.clone() for k, v in genome_fashion.state_dict().items()}

    # PHASE 2: Transfer to MNIST
    print("\n\n  PHASE 2: Transfer genome to MNIST")
    print("  " + "-" * 50)

    mnist_tr, mnist_te = load_data('mnist')

    # 2a. Normal MLP
    print(f"\n  [2] NORMAL MLP on MNIST")
    m = NormalMLP(784, 128, 10, 2).to(device)
    print(f"    Params: {count_params(m):,}")
    t0 = time.time()
    acc = train_simple(m, mnist_tr, mnist_te, n_ep)
    results['mnist_normal'] = {'acc': acc, 'time': time.time() - t0}

    # 2b. Fresh genome
    print(f"\n  [3] FRESH GENOME on MNIST (genome trained)")
    genome_fresh = Genome(n_types=8, type_dim=8, n_bands=len(dims))
    m = GrownNetwork(genome_fresh, 784, hidden_bands, 10).to(device)
    t0 = time.time()
    acc = train_grown(m, mnist_tr, mnist_te, n_ep, sp_w=0.1)
    a, t, sd = m.count_effective()
    results['mnist_fresh'] = {'acc': acc, 'time': time.time() - t0, 'soft_density': sd}

    # 2c. Transferred genome (FROZEN)
    print(f"\n  [4] TRANSFERRED GENOME on MNIST (genome FROZEN)")
    genome_transfer = Genome(n_types=8, type_dim=8, n_bands=len(dims))
    genome_transfer.load_state_dict(trained_genome_state)
    m = GrownNetwork(genome_transfer, 784, hidden_bands, 10).to(device)
    t0 = time.time()
    acc = train_grown(m, mnist_tr, mnist_te, n_ep, train_genome=False)
    a, t, sd = m.count_effective()
    results['mnist_transfer'] = {'acc': acc, 'time': time.time() - t0, 'soft_density': sd}

    # 2d. Random sparse (same density as transferred)
    transfer_density = results['mnist_transfer']['soft_density']
    print(f"\n  [5] RANDOM SPARSE on MNIST (density={transfer_density:.1%})")
    m = RandomSparseNetwork(dims, transfer_density).to(device)
    t0 = time.time()
    acc = train_simple(m, mnist_tr, mnist_te, n_ep)
    results['mnist_random'] = {'acc': acc, 'time': time.time() - t0}

    # RESULTS
    print("\n" + "=" * 60)
    print("  RESULTS: GENOME TRANSFER")
    print("=" * 60)

    print(f"\n  Fashion-MNIST (source task):")
    r = results['fashion_grown']
    print(f"    Grown: {r['acc']:.4f} (soft density={r['soft_density']:.1%})")

    print(f"\n  MNIST (target task):")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Soft Density':>14}")
    print(f"  {'-' * 50}")
    for name in ['mnist_normal', 'mnist_fresh', 'mnist_transfer', 'mnist_random']:
        r = results[name]
        d = f"{r['soft_density']:.1%}" if 'soft_density' in r else "dense"
        print(f"  {name:<22} {r['acc']:>10.4f} {d:>14}")

    tr_acc = results['mnist_transfer']['acc']
    rn_acc = results['mnist_random']['acc']
    fr_acc = results['mnist_fresh']['acc']

    gap_vs_random = (tr_acc - rn_acc) * 100
    gap_vs_fresh = (tr_acc - fr_acc) * 100

    print(f"\n  TRANSFER vs RANDOM: {'+' if gap_vs_random >= 0 else ''}{gap_vs_random:.2f}%"
          f" -> {'TRANSFER WINS' if gap_vs_random > 0 else 'RANDOM WINS'}")
    print(f"  TRANSFER vs FRESH:  {'+' if gap_vs_fresh >= 0 else ''}{gap_vs_fresh:.2f}%"
          f" -> {'genome generalizes!' if abs(gap_vs_fresh) < 1 else 'task-specific'}")

    total_time = sum(r['time'] for r in results.values())
    print(f"\n  Time: {total_time:.0f}s")

    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Save fashion-trained genome
    genome_path = f"results/genome_fashion_{timestamp}.pt"
    torch.save(trained_genome_state, genome_path)
    print(f"\n  Genome saved to {genome_path}")

    result_data = {
        "experiment": "transfer",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_types": 8, "type_dim": 8,
            "hidden_bands": hidden_bands,
            "n_epochs": n_ep, "sparsity_weight": 0.1,
            "source_task": "fashion_mnist", "target_task": "mnist",
            "device": str(device)
        },
        "results": results,
        "genome_state": genome_path,
        "total_time": total_time
    }
    json_path = f"results/transfer_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    run()
