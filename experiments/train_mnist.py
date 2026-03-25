"""
Train a developmental genome on MNIST.

Compares: Genome-grown network vs Random sparse vs Dense skip vs Normal MLP.
Saves results to results/ as JSON + genome state as .pt.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import time, json
from datetime import datetime

from genome import Genome, GrownNetwork, RandomSparseNetwork, NormalMLP, DenseSkipNetwork

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_mnist(bs=128):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, transform=tf)
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


def train_grown(model, tr, te, n_epochs=25, lr=1e-3, sparsity_weight=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0
    for ep in range(n_epochs):
        model.train()
        for bx, by in tr:
            bx = bx.view(bx.size(0), -1).to(device)
            by = by.to(device)
            logits = model(bx)
            ce_loss = F.cross_entropy(logits, by)
            sp_loss = model.genome.sparsity_loss(model.dims)
            loss = ce_loss + sparsity_weight * sp_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        if ep % 5 == 0 or ep == n_epochs - 1:
            active, total, sd = model.count_effective()
            density = active / total
            print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f} "
                  f"hard={density:.1%} soft={sd:.1%} sp_loss={sp_loss.item():.3f}")
    return best


def train_model(model, tr, te, n_epochs=25, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0
    for ep in range(n_epochs):
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
        if ep % 5 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f}")
    return best


def run():
    print("=" * 60)
    print("  TRAIN MNIST: SPARSE DEVELOPMENTAL GROWTH")
    print("  Default=disconnected. Connections actively grown.")
    print("=" * 60)
    print(f"  Device: {device}")

    tr, te = load_mnist()
    n_epochs = 25
    hidden_bands = [48, 48, 48, 48]
    dims = [784] + hidden_bands + [10]
    results = {}

    # 1. Normal MLP
    print(f"\n  [1/4] NORMAL MLP (h=128)")
    m = NormalMLP(784, 128, 10, 2).to(device)
    p = count_params(m)
    print(f"    Params: {p:,}")
    t0 = time.time()
    acc = train_model(m, tr, te, n_epochs)
    results['normal_mlp'] = {'params': p, 'acc': acc, 'time': time.time() - t0}

    # 2. Grown with sparsity pressure
    print(f"\n  [2/4] GROWN (sparsity_weight=0.1)")
    genome = Genome(n_types=8, type_dim=8, n_bands=len(dims))
    m = GrownNetwork(genome, 784, hidden_bands, 10).to(device)
    gp = sum(x.numel() for x in genome.parameters())
    print(f"    Genome: {gp:,} params, Total: {count_params(m):,}")
    t0 = time.time()
    acc = train_grown(m, tr, te, n_epochs, sparsity_weight=0.1)
    active, total, sd = m.count_effective()
    density = active / total
    results['grown'] = {
        'params': count_params(m), 'genome_params': gp, 'acc': acc,
        'time': time.time() - t0, 'hard_density': density,
        'soft_density': sd, 'active': active, 'total': total
    }
    print(f"    Final: hard={density:.1%} soft={sd:.1%}")
    m.describe_topology()

    # 3. Random sparse at same soft density
    print(f"\n  [3/4] RANDOM SPARSE (soft_density={sd:.1%})")
    m_rs = RandomSparseNetwork(dims, sd).to(device)
    p = count_params(m_rs)
    print(f"    Params: {p:,}")
    t0 = time.time()
    acc = train_model(m_rs, tr, te, n_epochs)
    results['random_sparse'] = {'params': p, 'acc': acc, 'time': time.time() - t0}

    # 4. Dense skip
    print(f"\n  [4/4] DENSE SKIP (all connections)")
    m_ds = DenseSkipNetwork(dims).to(device)
    p = count_params(m_ds)
    print(f"    Params: {p:,}")
    t0 = time.time()
    acc = train_model(m_ds, tr, te, n_epochs)
    results['dense_skip'] = {'params': p, 'acc': acc, 'time': time.time() - t0}

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS: TRAIN MNIST")
    print("=" * 60)

    print(f"\n  {'Model':<18} {'Params':>10} {'Accuracy':>10}")
    print(f"  {'-' * 42}")
    for name, r in results.items():
        extra = f" genome:{r.get('genome_params', '')}" if 'genome_params' in r else ""
        print(f"  {name:<18} {r['params']:>10,} {r['acc']:>10.4f}{extra}")

    g = results.get('grown', {})
    rs = results.get('random_sparse', {})
    if g and rs:
        gap = (g['acc'] - rs['acc']) * 100
        print(f"\n  GROWN vs RANDOM: {'+' if gap >= 0 else ''}{gap:.2f}%"
              f" -> {'GENOME WINS' if gap > 0 else 'RANDOM WINS'}")

    if g:
        print(f"\n  GENOME: {g['genome_params']:,} params -> {g['total']:,} possible connections")
        print(f"  Hard density (>0.5): {g['active']:,} ({g['hard_density']:.1%})")
        print(f"  Soft density (mean mask): {g['soft_density']:.1%}")
        print(f"  Compression: {g['total'] // max(g['genome_params'], 1)}:1")

    total_time = sum(r['time'] for r in results.values())
    print(f"\n  Time: {total_time:.0f}s")

    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Save genome state
    genome_path = f"results/genome_mnist_{timestamp}.pt"
    torch.save(genome.state_dict(), genome_path)
    print(f"\n  Genome saved to {genome_path}")

    # Save JSON results
    result_data = {
        "experiment": "train_mnist",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_types": 8, "type_dim": 8,
            "hidden_bands": hidden_bands,
            "n_epochs": n_epochs, "sparsity_weight": 0.1,
            "device": str(device)
        },
        "results": results,
        "genome_state": genome_path,
        "total_time": total_time
    }
    json_path = f"results/train_mnist_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    run()
