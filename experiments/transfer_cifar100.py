"""
Transfer test: CIFAR-10 genome -> CIFAR-100.

Does the genome learn general visual processing principles, or just
CIFAR-10-specific wiring? Freeze the CIFAR-10 trained genome, use its
topology on CIFAR-100 (100 classes, 500 samples/class), train only weights.

4-model comparison:
  1. Dense ResNet on CIFAR-100 (ceiling, ~65-70%)
  2. Frozen CIFAR-10 genome on CIFAR-100 (the transfer test)
  3. Fresh genome on CIFAR-100 (trained from scratch)
  4. Random sparse on CIFAR-100 (control, matched density)

Key comparisons:
  - Frozen vs random sparse: does the topology transfer?
  - Frozen vs fresh: how much task-specific adaptation does the genome need?
  - Frozen vs dense: practical cost of transfer

Resume-safe with checkpointing.
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import time, json
from datetime import datetime

from genome import Genome, GrownConvNetwork, DenseResNet, RandomSparseResNet

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = 'results/_cifar100_transfer_checkpoint.json'
NUM_CLASSES = 100
N_EPOCHS = 250


def load_cifar100(bs=128):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    tr = datasets.CIFAR100('./data', train=True, download=True, transform=tf_train)
    te = datasets.CIFAR100('./data', train=False, transform=tf_test)
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


def flush_mps():
    if device.type == 'mps':
        torch.mps.empty_cache()


def save_checkpoint(results, genome_state_path=None):
    os.makedirs('results', exist_ok=True)
    data = {'results': results}
    if genome_state_path:
        data['genome_state'] = genome_state_path
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return data.get('results', {}), data.get('genome_state')
    return {}, None


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def find_cifar10_genome():
    """Find the latest CIFAR-10 trained genome checkpoint."""
    pts = sorted(glob.glob("results/genome_cnn_*.pt"), key=os.path.getmtime)
    # Prefer the timestamped one over the checkpoint one
    pts = [p for p in pts if 'checkpoint' not in p] or pts
    if not pts:
        return None
    return pts[-1]


def train_dense(model, tr, te, n_epochs=N_EPOCHS, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
        flush_mps()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f}")
    return best


def train_frozen_genome(model, tr, te, n_epochs=N_EPOCHS):
    """Train CNN weights only. Genome is frozen (no grad)."""
    # Freeze genome
    for p in model.genome.parameters():
        p.requires_grad_(False)

    # High temperature so masks are near-binary
    model.temperature = 10.0

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.1, momentum=0.9, weight_decay=5e-4
    )
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
        flush_mps()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f}")
    return best


def train_fresh_genome(model, tr, te, n_epochs=N_EPOCHS, sparsity_weight=0.01):
    """Train genome + weights from scratch on CIFAR-100."""
    genome_params = list(model.genome.parameters())
    genome_ids = set(id(p) for p in genome_params)
    weight_params = [p for p in model.parameters() if id(p) not in genome_ids]

    opt_weights = torch.optim.SGD(weight_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    opt_genome = torch.optim.Adam(genome_params, lr=1e-3)

    sched_weights = torch.optim.lr_scheduler.CosineAnnealingLR(opt_weights, n_epochs)
    sched_genome = torch.optim.lr_scheduler.CosineAnnealingLR(opt_genome, n_epochs)

    temp_start, temp_end = 1.0, 10.0
    print(f"    Temperature annealing: {temp_start} -> {temp_end} over {n_epochs} epochs")

    best = 0
    for ep in range(n_epochs):
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

        flush_mps()
        if ep % 10 == 0 or ep == n_epochs - 1:
            active, total, sd = model.count_effective()
            density = active / total if total > 0 else 0
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f} "
                  f"hard={density:.1%} soft={sd:.1%} "
                  f"sp={sp_loss.item():.3f} temp={model.temperature:.1f}")
    return best


def train_sparse(model, tr, te, n_epochs=N_EPOCHS, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
        flush_mps()
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    Ep {ep:3d}: test={acc:.4f} best={best:.4f}")
    return best


def print_topology(model, label):
    """Print topology for comparison between genomes."""
    print(f"    {label} topology:")
    band_names = ['input', 'g1c1', 'g1c2', 'g2c1', 'g2c2', 'g3c1', 'g3c2', 'fc']
    with torch.no_grad():
        for tgt in range(1, 8):
            for src in range(tgt):
                src_ch = model.band_channels[src]
                tgt_ch = model.band_channels[tgt]
                m = model.genome.growth_mask(src, tgt, src_ch, tgt_ch, temperature=10.0)
                d = (m > 0.5).float().mean().item()
                if d > 0.01:
                    kind = "adj" if tgt - src == 1 else "skip"
                    print(f"      {band_names[src]}->{band_names[tgt]} "
                          f"({kind}): {d:.1%}")


def run():
    print("=" * 60)
    print("  CNN TRANSFER: CIFAR-10 GENOME -> CIFAR-100")
    print("  Does the genome learn general visual wiring principles?")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {N_EPOCHS}")

    # Find CIFAR-10 genome
    genome_path = find_cifar10_genome()
    if not genome_path:
        print("\n  ERROR: No CIFAR-10 genome found.")
        print("  Run 'python3 run.py cnn' first.")
        sys.exit(1)
    print(f"  CIFAR-10 genome: {genome_path}")

    # Check for checkpoint
    results, saved_genome_path = load_checkpoint()
    if results:
        done = list(results.keys())
        print(f"\n  RESUMING: found checkpoint with {done} complete.")
    else:
        results = {}

    tr, te = load_cifar100()

    # 1. Dense ResNet (ceiling)
    if 'dense_resnet' not in results:
        print(f"\n  [1/4] DENSE RESNET on CIFAR-100 (ceiling)")
        m = DenseResNet(num_classes=NUM_CLASSES).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_dense(m, tr, te)
        results['dense_resnet'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m; flush_mps()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [1/4] DENSE RESNET: skipped (acc={results['dense_resnet']['acc']:.4f})")

    # 2. Frozen CIFAR-10 genome on CIFAR-100
    if 'frozen_genome' not in results:
        print(f"\n  [2/4] FROZEN CIFAR-10 GENOME on CIFAR-100")
        genome_frozen = Genome(n_types=8, type_dim=8, n_bands=8)
        genome_frozen.load_state_dict(torch.load(genome_path, weights_only=True))
        m = GrownConvNetwork(genome_frozen, num_classes=NUM_CLASSES).to(device)
        tp = count_params(m)  # Before freezing
        print(f"    Total params: {tp:,} (genome frozen, only weights train)")

        # Show the transferred topology
        print_topology(m, "CIFAR-10 frozen")
        active, total, sd = m.count_effective()
        density = active / total if total > 0 else 0
        print(f"    Transferred density: hard={density:.1%} soft={sd:.1%}")

        t0 = time.time()
        acc = train_frozen_genome(m, tr, te)
        trainable = count_params(m)  # After freezing
        results['frozen_genome'] = {
            'params': tp, 'trainable_params': trainable, 'acc': acc,
            'time': time.time() - t0, 'hard_density': density,
            'soft_density': sd, 'source': genome_path
        }
        del m; flush_mps()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [2/4] FROZEN GENOME: skipped (acc={results['frozen_genome']['acc']:.4f})")
        sd = results['frozen_genome']['soft_density']

    # Get density for random sparse matching
    sd = results.get('frozen_genome', {}).get('soft_density', 0.45)

    # 3. Fresh genome on CIFAR-100
    fresh_genome = None
    if 'fresh_genome' not in results:
        print(f"\n  [3/4] FRESH GENOME on CIFAR-100 (trained from scratch)")
        fresh_genome = Genome(n_types=8, type_dim=8, n_bands=8)
        m = GrownConvNetwork(fresh_genome, num_classes=NUM_CLASSES).to(device)
        gp = sum(x.numel() for x in fresh_genome.parameters())
        tp = count_params(m)
        print(f"    Genome: {gp:,} params, Total: {tp:,}")
        t0 = time.time()
        acc = train_fresh_genome(m, tr, te, sparsity_weight=0.01)
        active, total, fresh_sd = m.count_effective()
        fresh_density = active / total if total > 0 else 0
        results['fresh_genome'] = {
            'params': tp, 'genome_params': gp, 'acc': acc,
            'time': time.time() - t0, 'hard_density': fresh_density,
            'soft_density': fresh_sd, 'active': active, 'total': total
        }
        print(f"    Final: hard={fresh_density:.1%} soft={fresh_sd:.1%}")

        # Show fresh topology for comparison
        print_topology(m, "CIFAR-100 fresh")

        # Save fresh genome
        os.makedirs('results', exist_ok=True)
        fresh_path = "results/genome_cifar100_fresh.pt"
        torch.save(fresh_genome.state_dict(), fresh_path)
        save_checkpoint(results, fresh_path)

        del m; flush_mps()
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [3/4] FRESH GENOME: skipped (acc={results['fresh_genome']['acc']:.4f})")

    # 4. Random sparse (matched density to frozen genome)
    if 'random_sparse' not in results:
        print(f"\n  [4/4] RANDOM SPARSE on CIFAR-100 (density={sd:.1%})")
        m_rs = RandomSparseResNet(density=sd, num_classes=NUM_CLASSES).to(device)
        p = count_params(m_rs)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_sparse(m_rs, tr, te)
        results['random_sparse'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m_rs; flush_mps()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [4/4] RANDOM SPARSE: skipped (acc={results['random_sparse']['acc']:.4f})")

    # --- Topology comparison ---
    print("\n" + "=" * 60)
    print("  TOPOLOGY COMPARISON: CIFAR-10 vs CIFAR-100 GENOME")
    print("=" * 60)

    # Load both genomes for comparison
    genome_c10 = Genome(n_types=8, type_dim=8, n_bands=8)
    genome_c10.load_state_dict(torch.load(genome_path, weights_only=True))
    m_c10 = GrownConvNetwork(genome_c10, num_classes=NUM_CLASSES)
    print_topology(m_c10, "CIFAR-10 genome")

    fresh_path = "results/genome_cifar100_fresh.pt"
    if os.path.exists(fresh_path):
        genome_c100 = Genome(n_types=8, type_dim=8, n_bands=8)
        genome_c100.load_state_dict(torch.load(fresh_path, weights_only=True))
        m_c100 = GrownConvNetwork(genome_c100, num_classes=NUM_CLASSES)
        print_topology(m_c100, "CIFAR-100 genome")

        # Compute topology similarity
        band_names = ['input', 'g1c1', 'g1c2', 'g2c1', 'g2c2', 'g3c1', 'g3c2', 'fc']
        match = total_pairs = 0
        with torch.no_grad():
            for tgt in range(1, 8):
                for src in range(tgt):
                    src_ch = m_c10.band_channels[src]
                    tgt_ch = m_c10.band_channels[tgt]
                    m10 = genome_c10.growth_mask(src, tgt, src_ch, tgt_ch, temperature=10.0)
                    m100 = genome_c100.growth_mask(src, tgt, src_ch, tgt_ch, temperature=10.0)
                    d10 = (m10 > 0.5).float().mean().item()
                    d100 = (m100 > 0.5).float().mean().item()
                    # Both agree on "active" (>10%) or "dead" (<10%)
                    if (d10 > 0.1) == (d100 > 0.1):
                        match += 1
                    total_pairs += 1
        print(f"\n    Topology agreement: {match}/{total_pairs} band pairs "
              f"({match/total_pairs:.0%} match)")

    # --- Results ---
    print("\n" + "=" * 60)
    print("  RESULTS: CIFAR-100 TRANSFER")
    print("=" * 60)

    print(f"\n  {'Model':<22} {'Params':>10} {'Accuracy':>10}")
    print(f"  {'-' * 46}")
    for name, r in results.items():
        extra = ""
        if 'genome_params' in r:
            extra = f" genome:{r['genome_params']}"
        elif 'trainable_params' in r:
            extra = f" trainable:{r['trainable_params']:,}"
        print(f"  {name:<22} {r['params']:>10,} {r['acc']:>10.4f}{extra}")

    frozen = results.get('frozen_genome', {})
    rs = results.get('random_sparse', {})
    fresh = results.get('fresh_genome', {})
    dr = results.get('dense_resnet', {})

    if frozen and rs:
        gap = (frozen['acc'] - rs['acc']) * 100
        label = "TOPOLOGY TRANSFERS" if gap > 0 else "NO TRANSFER"
        print(f"\n  FROZEN vs RANDOM SPARSE: {'+' if gap >= 0 else ''}{gap:.2f}% ({label})")

    if frozen and fresh:
        gap = (frozen['acc'] - fresh['acc']) * 100
        print(f"  FROZEN vs FRESH GENOME: {'+' if gap >= 0 else ''}{gap:.2f}%"
              f" ({'genome generalizes!' if gap >= -0.5 else 'task-specific adaptation helps'})")

    if frozen and dr:
        gap = (dr['acc'] - frozen['acc']) * 100
        print(f"  DENSE vs FROZEN: {'+' if gap >= 0 else ''}{gap:.2f}% gap to ceiling")

    total_time = sum(r.get('time', 0) for r in results.values())
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    result_data = {
        "experiment": "transfer_cifar100",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_types": 8, "type_dim": 8, "n_bands": 8,
            "n_epochs": N_EPOCHS, "num_classes": NUM_CLASSES,
            "cifar10_genome": genome_path, "device": str(device)
        },
        "results": results,
        "total_time": total_time
    }
    json_path = f"results/transfer_cifar100_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")

    clear_checkpoint()
    print("  Checkpoint cleared. Run complete.")


if __name__ == "__main__":
    run()
