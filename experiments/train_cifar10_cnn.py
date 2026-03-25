"""
Train genome CNN on CIFAR-10.

Maps the genome onto convolutional architectures. Each conv layer = one band,
each channel = one neuron. The genome's type compatibility generates masks
that determine which channels connect. Skip connections emerge naturally.

4-model comparison:
  1. Dense ResNet (ceiling)
  2. Genome CNN (the hypothesis)
  3. Random Sparse CNN (control, matched density)
  4. Dense Skip CNN (full skip architecture, no sparsity)

Uses hard binary masks via straight-through estimator.
Split optimizer: SGD+momentum for CNN weights, Adam for genome params.
200 epochs with cosine annealing (standard for CIFAR-10 ResNets).

Resume-safe: saves progress after each model. If interrupted, re-run and
it picks up from where it left off.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import time, json
from datetime import datetime

from genome import Genome, GrownConvNetwork, DenseResNet, RandomSparseResNet, DenseSkipResNet

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = 'results/_cnn_checkpoint.json'


def load_cifar10(bs=128):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf_train)
    te = datasets.CIFAR10('./data', train=False, transform=tf_test)
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(te, batch_size=bs, shuffle=False, num_workers=0))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def evaluate(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            c += (model(bx).argmax(-1) == by).sum().item()
            t += bx.size(0)
    return c / t


def save_checkpoint(results, genome_state_path=None):
    """Save partial results so we can resume after a crash."""
    os.makedirs('results', exist_ok=True)
    data = {'results': results}
    if genome_state_path:
        data['genome_state'] = genome_state_path
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint():
    """Load partial results from a previous interrupted run."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return data.get('results', {}), data.get('genome_state')
    return {}, None


def clear_checkpoint():
    """Remove checkpoint after successful completion."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def flush_cache():
    """Free cached GPU memory to prevent buildup."""
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def train_dense(model, tr, te, n_epochs=200, lr=0.1):
    """Train a standard CNN with SGD + momentum + cosine annealing."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0
    for ep in range(n_epochs):
        model.train()
        for bx, by in tr:
            bx, by = bx.to(device), by.to(device)
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        flush_cache()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f}")
    return best


def train_genome_cnn(model, tr, te, n_epochs=200, sparsity_weight=0.01):
    """Train genome CNN with split optimizer and temperature annealing.

    SGD+momentum for CNN weights (lr=0.1).
    Adam for genome params (lr=1e-3).

    Temperature annealing: sigmoid temperature goes from 1.0 to 10.0 over
    training. At temp=10, sigmoid is effectively binary (>0.99 or <0.01)
    but the transition is smooth so weights can adapt gradually.
    """
    genome_params = list(model.genome.parameters())
    genome_ids = set(id(p) for p in genome_params)
    weight_params = [p for p in model.parameters() if id(p) not in genome_ids]

    opt_weights = torch.optim.SGD(weight_params, lr=0.1, momentum=0.9, weight_decay=1e-4)
    opt_genome = torch.optim.Adam(genome_params, lr=1e-3)

    sched_weights = torch.optim.lr_scheduler.CosineAnnealingLR(opt_weights, n_epochs)
    sched_genome = torch.optim.lr_scheduler.CosineAnnealingLR(opt_genome, n_epochs)

    temp_start, temp_end = 1.0, 10.0
    print(f"    Temperature annealing: {temp_start} -> {temp_end} over {n_epochs} epochs")

    best = 0
    for ep in range(n_epochs):
        # Linearly anneal temperature
        model.temperature = temp_start + (temp_end - temp_start) * ep / (n_epochs - 1)

        model.train()
        for bx, by in tr:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            ce_loss = F.cross_entropy(logits, by)
            sp_loss = model.genome.sparsity_loss(model.band_channels)
            loss = ce_loss + sparsity_weight * sp_loss

            opt_weights.zero_grad()
            opt_genome.zero_grad()
            loss.backward()
            opt_weights.step()
            opt_genome.step()

        acc = evaluate(model, te)
        best = max(best, acc)
        sched_weights.step()
        sched_genome.step()

        flush_cache()
        if ep % 10 == 0 or ep == n_epochs - 1:
            active, total, sd = model.count_effective()
            density = active / total if total > 0 else 0
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f} "
                  f"hard={density:.1%} soft={sd:.1%} "
                  f"sp={sp_loss.item():.3f} temp={model.temperature:.1f}")
    return best


def train_sparse_cnn(model, tr, te, n_epochs=200, lr=0.1):
    """Train a sparse CNN (random or dense skip) with SGD + momentum."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0
    for ep in range(n_epochs):
        model.train()
        for bx, by in tr:
            bx, by = bx.to(device), by.to(device)
            loss = F.cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        flush_cache()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f}")
    return best


def run():
    print("=" * 60)
    print("  CNN GENOME: CONVOLUTIONAL ARCHITECTURE")
    print("  Genome controls channel-to-channel connectivity.")
    print("  Hard binary masks via straight-through estimator.")
    print("=" * 60)
    print(f"  Device: {device}")

    # Check for checkpoint from a previous interrupted run
    results, saved_genome_path = load_checkpoint()
    if results:
        done = list(results.keys())
        print(f"\n  RESUMING: found checkpoint with {done} complete.")
        print(f"  Skipping finished models.")
    else:
        results = {}

    tr, te = load_cifar10()
    n_epochs = 200
    genome = None

    # 1. Dense ResNet (ceiling)
    if 'dense_resnet' not in results:
        print(f"\n  [1/4] DENSE RESNET (ceiling baseline)")
        m = DenseResNet(num_classes=10).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_dense(m, tr, te, n_epochs)
        results['dense_resnet'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [1/4] DENSE RESNET: skipped (acc={results['dense_resnet']['acc']:.4f})")

    # 2. Genome CNN
    if 'genome_cnn' not in results:
        print(f"\n  [2/4] GENOME CNN (hard masks, split optimizer)")
        genome = Genome(n_types=8, type_dim=8, n_bands=8)
        m = GrownConvNetwork(genome, num_classes=10).to(device)
        gp = sum(x.numel() for x in genome.parameters())
        tp = count_params(m)
        print(f"    Genome: {gp:,} params, Total: {tp:,}")
        t0 = time.time()
        acc = train_genome_cnn(m, tr, te, n_epochs, sparsity_weight=0.01)
        active, total, sd = m.count_effective()
        density = active / total if total > 0 else 0
        results['genome_cnn'] = {
            'params': tp, 'genome_params': gp, 'acc': acc,
            'time': time.time() - t0, 'hard_density': density,
            'soft_density': sd, 'active': active, 'total': total
        }
        print(f"    Final: hard={density:.1%} soft={sd:.1%}")
        m.describe_topology()
        del m; flush_cache()
        # Save genome state
        os.makedirs('results', exist_ok=True)
        genome_path = f"results/genome_cnn_checkpoint.pt"
        torch.save(genome.state_dict(), genome_path)
        save_checkpoint(results, genome_path)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [2/4] GENOME CNN: skipped (acc={results['genome_cnn']['acc']:.4f})")
        sd = results['genome_cnn']['soft_density']

    # Get soft density for random sparse
    sd = results.get('genome_cnn', {}).get('soft_density', 0.1)

    # 3. Random Sparse CNN (control, matched density)
    if 'random_sparse_cnn' not in results:
        print(f"\n  [3/4] RANDOM SPARSE CNN (density={sd:.1%})")
        m_rs = RandomSparseResNet(density=sd, num_classes=10).to(device)
        p = count_params(m_rs)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_sparse_cnn(m_rs, tr, te, n_epochs)
        results['random_sparse_cnn'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m_rs; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [3/4] RANDOM SPARSE CNN: skipped (acc={results['random_sparse_cnn']['acc']:.4f})")

    # 4. Dense Skip CNN
    if 'dense_skip_cnn' not in results:
        print(f"\n  [4/4] DENSE SKIP CNN (all connections active)")
        m_ds = DenseSkipResNet(num_classes=10).to(device)
        p = count_params(m_ds)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_sparse_cnn(m_ds, tr, te, n_epochs)
        results['dense_skip_cnn'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m_ds; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [4/4] DENSE SKIP CNN: skipped (acc={results['dense_skip_cnn']['acc']:.4f})")

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS: CNN GENOME ON CIFAR-10")
    print("=" * 60)

    print(f"\n  {'Model':<22} {'Params':>10} {'Accuracy':>10}")
    print(f"  {'-' * 46}")
    for name, r in results.items():
        extra = f" genome:{r.get('genome_params', '')}" if 'genome_params' in r else ""
        print(f"  {name:<22} {r['params']:>10,} {r['acc']:>10.4f}{extra}")

    g = results.get('genome_cnn', {})
    rs = results.get('random_sparse_cnn', {})
    if g and rs:
        gap = (g['acc'] - rs['acc']) * 100
        print(f"\n  GENOME CNN vs RANDOM SPARSE: {'+' if gap >= 0 else ''}{gap:.2f}%"
              f" -> {'GENOME WINS' if gap > 0 else 'RANDOM WINS'}")

    dr = results.get('dense_resnet', {})
    if dr and g:
        gap = (dr['acc'] - g['acc']) * 100
        print(f"  DENSE RESNET vs GENOME CNN: {'+' if gap >= 0 else ''}{gap:.2f}% gap to ceiling")

    if g:
        print(f"\n  GENOME: {g['genome_params']:,} params -> {g['total']:,} possible connections")
        print(f"  Hard density (>0.5): {g['active']:,} ({g['hard_density']:.1%})")
        print(f"  Soft density (mean mask): {g['soft_density']:.1%}")
        print(f"  Compression: {g['total'] // max(g['genome_params'], 1)}:1")

    total_time = sum(r.get('time', 0) for r in results.values())
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Save genome (use checkpoint if we didn't train it this run)
    if genome is not None:
        genome_path = f"results/genome_cnn_{timestamp}.pt"
        torch.save(genome.state_dict(), genome_path)
        print(f"\n  Genome saved to {genome_path}")
    elif saved_genome_path and os.path.exists(saved_genome_path):
        genome_path = saved_genome_path
        print(f"\n  Genome from checkpoint: {genome_path}")
    else:
        genome_path = None

    result_data = {
        "experiment": "train_cifar10_cnn",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_types": 8, "type_dim": 8, "n_bands": 8,
            "n_epochs": n_epochs, "sparsity_weight": 0.01,
            "hard_masks": True, "device": str(device)
        },
        "results": results,
        "genome_state": genome_path,
        "total_time": total_time
    }
    json_path = f"results/train_cifar10_cnn_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")

    # Clean up checkpoint
    clear_checkpoint()
    print("  Checkpoint cleared. Run complete.")


if __name__ == "__main__":
    run()
