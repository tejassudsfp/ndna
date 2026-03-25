"""
Rung 3: Genome-controlled Transformer on IMDB (v2: no free paths).

The first run failed: attention + residual flowed freely without genome masks.
The genome had no reason to grow connections, shrank to 0% density, and the
model ran entirely on unmasked attention.

Fix: replace nn.MultiheadAttention with manual multi-head attention so we can
genome-mask the output projection W_o. Every information path now goes through
a genome gate. No free highways. Same treatment that made the CNN work.

Small 6-layer transformer encoder for IMDB sentiment classification (binary).
Genome controls: attention W_o, FF first linear, skip projections, classifier.

4-model comparison:
  1. Dense Transformer (ceiling, PRESERVED from v1: 84.57%)
  2. Genome Transformer (hypothesis, re-trained with masked attention)
  3. Random Sparse Transformer (control, matched density)
  4. Dense Skip Transformer (full cross-layer, manual attention, no sparsity)

Memory-safe for 8GB M3: batch 16, gradient accumulation 2 steps,
sequence length 512 (fallback 256 on OOM), MPS cache flushing.

Resume-safe: checkpoint after each model completes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time, json
from datetime import datetime

from genome import (
    Genome, GrownTransformer,
    DenseTransformer, RandomSparseTransformer, DenseSkipTransformer,
)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = 'results/_transformer_checkpoint.json'

# Will be set after trying to load data
MAX_LEN = 512


def load_imdb(max_len=512, batch_size=16):
    """Load stanfordnlp/imdb from HuggingFace with bert-base-uncased tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("  Loading stanfordnlp/imdb dataset and tokenizer...")
    ds = load_dataset("stanfordnlp/imdb", cache_dir="./data/huggingface")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length",
                         max_length=max_len, return_tensors="pt")

    print("  Tokenizing...")
    ds = ds.map(tokenize, batched=True, batch_size=1000)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        ds["test"], batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train: {len(ds['train']):,}, Test: {len(ds['test']):,}")
    print(f"  Max length: {max_len}, Batch size: {batch_size}")
    return train_loader, test_loader


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def flush_cache():
    """Free cached GPU memory to prevent buildup."""
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, attention_mask=mask)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += ids.size(0)
    return correct / total


def save_checkpoint(results, genome_state_path=None):
    os.makedirs('results', exist_ok=True)
    data = {'results': results, 'max_len': MAX_LEN}
    if genome_state_path:
        data['genome_state'] = genome_state_path
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return data.get('results', {}), data.get('genome_state'), data.get('max_len', 512)
    return {}, None, 512


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def train_dense_transformer(model, tr, te, n_epochs=10, lr=2e-4, accum_steps=2):
    """Train standard transformer with AdamW + cosine annealing."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        step = 0
        for batch in tr:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, attention_mask=mask)
            loss = F.cross_entropy(logits, labels) / accum_steps
            loss.backward()

            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()

        # Handle leftover
        if step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        flush_cache()
        print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f}")

    return best


def train_genome_transformer(model, tr, te, n_epochs=15, lr=2e-4,
                              genome_lr=0.01, sparsity_weight=0.01, accum_steps=2):
    """Train genome transformer with split optimizer and temperature annealing."""
    genome_params = list(model.genome.parameters())
    genome_ids = set(id(p) for p in genome_params)
    weight_params = [p for p in model.parameters() if id(p) not in genome_ids]

    opt_weights = torch.optim.AdamW(weight_params, lr=lr, weight_decay=0.01)
    opt_genome = torch.optim.Adam(genome_params, lr=genome_lr)

    sched_weights = torch.optim.lr_scheduler.CosineAnnealingLR(opt_weights, n_epochs)
    sched_genome = torch.optim.lr_scheduler.CosineAnnealingLR(opt_genome, n_epochs)

    temp_start, temp_end = 1.0, 10.0
    print(f"    Temperature annealing: {temp_start} -> {temp_end} over {n_epochs} epochs")

    best = 0
    for ep in range(n_epochs):
        model.temperature = temp_start + (temp_end - temp_start) * ep / max(n_epochs - 1, 1)

        model.train()
        opt_weights.zero_grad()
        opt_genome.zero_grad()
        step = 0

        for batch in tr:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, attention_mask=mask)
            ce_loss = F.cross_entropy(logits, labels)
            sp_loss = model.genome.sparsity_loss(model.band_dims)
            loss = (ce_loss + sparsity_weight * sp_loss) / accum_steps
            loss.backward()

            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt_weights.step()
                opt_genome.step()
                opt_weights.zero_grad()
                opt_genome.zero_grad()

        if step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_weights.step()
            opt_genome.step()
            opt_weights.zero_grad()
            opt_genome.zero_grad()

        acc = evaluate(model, te)
        best = max(best, acc)
        sched_weights.step()
        sched_genome.step()
        flush_cache()

        active, total, sd = model.count_effective()
        density = active / total if total > 0 else 0
        print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f} "
              f"hard={density:.1%} soft={sd:.1%} "
              f"sp={sp_loss.item():.3f} temp={model.temperature:.1f}")

    return best


def train_sparse_transformer(model, tr, te, n_epochs=10, lr=2e-4, accum_steps=2):
    """Train sparse transformer (random or dense skip) with AdamW."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best = 0

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        step = 0
        for batch in tr:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, attention_mask=mask)
            loss = F.cross_entropy(logits, labels) / accum_steps
            loss.backward()

            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()

        if step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        acc = evaluate(model, te)
        best = max(best, acc)
        sched.step()
        flush_cache()
        print(f"    Ep {ep:2d}: test={acc:.4f} best={best:.4f}")

    return best


def run():
    global MAX_LEN

    print("=" * 60)
    print("  RUNG 3: GENOME-CONTROLLED TRANSFORMER ON IMDB")
    print("  Testing if genome generalizes from CNNs to Transformers.")
    print("  Cross-layer connectivity + FF sparsity via genome masks.")
    print("=" * 60)
    print(f"  Device: {device}")

    # Check for checkpoint
    results, saved_genome_path, saved_max_len = load_checkpoint()
    if results:
        done = list(results.keys())
        MAX_LEN = saved_max_len
        print(f"\n  CHECKPOINT: found {done} complete.")

        # v2 migration: manual attention replaces nn.MultiheadAttention so genome can mask W_o.
        # Dense Transformer: unchanged (no skips, no masks, standard architecture). PRESERVE.
        # Dense Skip Transformer: no masks, manual attention = same math. PRESERVE.
        # Genome/Random Sparse: architecture changed (W_o now masked). Must re-run.
        stale = ['genome_transformer', 'random_sparse_transformer']
        cleared = [k for k in stale if k in results]
        for k in cleared:
            del results[k]
        if cleared:
            print(f"  CLEARED stale results (v2 architecture change): {cleared}")
            save_checkpoint(results)

        for kept in ['dense_transformer', 'dense_skip_transformer']:
            if kept in results:
                print(f"  PRESERVED: {kept} (acc={results[kept]['acc']:.4f})")
        print(f"  Using max_len={MAX_LEN} from checkpoint.")
    else:
        results = {}
        MAX_LEN = 512

    # Load data (try 512, fall back to 256 on OOM during training)
    tr, te = load_imdb(max_len=MAX_LEN, batch_size=16)

    genome = None
    hidden = 256
    ff_dim = 512
    n_layers = 6
    n_heads = 4
    num_classes = 2

    # 1. Dense Transformer (ceiling)
    if 'dense_transformer' not in results:
        print(f"\n  [1/4] DENSE TRANSFORMER (ceiling baseline)")
        m = DenseTransformer(hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                             n_heads=n_heads, max_len=MAX_LEN, num_classes=num_classes).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        try:
            acc = train_dense_transformer(m, tr, te, n_epochs=10)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                print(f"    OOM at max_len={MAX_LEN}. Retrying with max_len=256...")
                del m; flush_cache()
                MAX_LEN = 256
                tr, te = load_imdb(max_len=MAX_LEN, batch_size=16)
                m = DenseTransformer(hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                                     n_heads=n_heads, max_len=MAX_LEN, num_classes=num_classes).to(device)
                p = count_params(m)
                t0 = time.time()
                acc = train_dense_transformer(m, tr, te, n_epochs=10)
            else:
                raise
        results['dense_transformer'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [1/4] DENSE TRANSFORMER: skipped (acc={results['dense_transformer']['acc']:.4f})")

    # 2. Genome Transformer
    if 'genome_transformer' not in results:
        print(f"\n  [2/4] GENOME TRANSFORMER (cross-layer + FF sparsity)")
        genome = Genome(n_types=8, type_dim=8, n_bands=8)
        m = GrownTransformer(genome, hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                             n_heads=n_heads, max_len=MAX_LEN, num_classes=num_classes).to(device)
        gp = sum(x.numel() for x in genome.parameters())
        tp = count_params(m)
        print(f"    Genome: {gp:,} params, Total: {tp:,}")
        t0 = time.time()
        try:
            acc = train_genome_transformer(m, tr, te, n_epochs=15, sparsity_weight=0.005)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                print(f"    OOM at max_len={MAX_LEN}. Retrying with max_len=256...")
                del m; flush_cache()
                MAX_LEN = 256
                tr, te = load_imdb(max_len=MAX_LEN, batch_size=16)
                if 'dense_transformer' in results:
                    print("    WARNING: Dense trained at different max_len. Results may not be comparable.")
                genome = Genome(n_types=8, type_dim=8, n_bands=8)
                m = GrownTransformer(genome, hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                                     n_heads=n_heads, max_len=MAX_LEN, num_classes=num_classes).to(device)
                gp = sum(x.numel() for x in genome.parameters())
                tp = count_params(m)
                t0 = time.time()
                acc = train_genome_transformer(m, tr, te, n_epochs=15, sparsity_weight=0.005)
            else:
                raise
        active, total, sd = m.count_effective()
        density = active / total if total > 0 else 0
        results['genome_transformer'] = {
            'params': tp, 'genome_params': gp, 'acc': acc,
            'time': time.time() - t0, 'hard_density': density,
            'soft_density': sd, 'active': active, 'total': total
        }
        print(f"    Final: hard={density:.1%} soft={sd:.1%}")
        m.describe_topology()
        del m; flush_cache()
        # Save genome
        os.makedirs('results', exist_ok=True)
        genome_path = "results/genome_transformer_checkpoint.pt"
        torch.save(genome.state_dict(), genome_path)
        save_checkpoint(results, genome_path)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [2/4] GENOME TRANSFORMER: skipped (acc={results['genome_transformer']['acc']:.4f})")
        sd = results['genome_transformer']['soft_density']

    # Get soft density for random sparse
    sd = results.get('genome_transformer', {}).get('soft_density', 0.1)

    # 3. Random Sparse Transformer (control, matched density)
    if 'random_sparse_transformer' not in results:
        print(f"\n  [3/4] RANDOM SPARSE TRANSFORMER (density={sd:.1%})")
        m = RandomSparseTransformer(density=sd, hidden=hidden, ff_dim=ff_dim,
                                     n_layers=n_layers, n_heads=n_heads,
                                     max_len=MAX_LEN, num_classes=num_classes).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_sparse_transformer(m, tr, te, n_epochs=10)
        results['random_sparse_transformer'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [3/4] RANDOM SPARSE TRANSFORMER: skipped (acc={results['random_sparse_transformer']['acc']:.4f})")

    # 4. Dense Skip Transformer
    if 'dense_skip_transformer' not in results:
        print(f"\n  [4/4] DENSE SKIP TRANSFORMER (full cross-layer connectivity)")
        m = DenseSkipTransformer(hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                                  n_heads=n_heads, max_len=MAX_LEN, num_classes=num_classes).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        acc = train_sparse_transformer(m, tr, te, n_epochs=10)
        results['dense_skip_transformer'] = {'params': p, 'acc': acc, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Checkpoint saved. Best: {acc:.4f}")
    else:
        print(f"\n  [4/4] DENSE SKIP TRANSFORMER: skipped (acc={results['dense_skip_transformer']['acc']:.4f})")

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS: GENOME TRANSFORMER ON IMDB")
    print("=" * 60)

    print(f"\n  {'Model':<28} {'Params':>10} {'Accuracy':>10}")
    print(f"  {'-' * 52}")
    for name, r in results.items():
        extra = f" genome:{r.get('genome_params', '')}" if 'genome_params' in r else ""
        print(f"  {name:<28} {r['params']:>10,} {r['acc']:>10.4f}{extra}")

    g = results.get('genome_transformer', {})
    rs = results.get('random_sparse_transformer', {})
    if g and rs:
        gap = (g['acc'] - rs['acc']) * 100
        print(f"\n  GENOME vs RANDOM SPARSE: {'+' if gap >= 0 else ''}{gap:.2f}%"
              f" -> {'GENOME WINS' if gap > 0 else 'RANDOM WINS'}")

    dt = results.get('dense_transformer', {})
    if dt and g:
        gap = (dt['acc'] - g['acc']) * 100
        print(f"  DENSE vs GENOME: {'+' if gap >= 0 else ''}{gap:.2f}% gap to ceiling")

    ds = results.get('dense_skip_transformer', {})
    if ds and g:
        gap = (g['acc'] - ds['acc']) * 100
        print(f"  GENOME vs DENSE SKIP: {'+' if gap >= 0 else ''}{gap:.2f}%")

    if g:
        print(f"\n  GENOME: {g['genome_params']:,} params -> {g['total']:,} masked connections")
        print(f"  Hard density (>0.5): {g['active']:,} ({g['hard_density']:.1%})")
        print(f"  Soft density (mean mask): {g['soft_density']:.1%}")
        print(f"  Compression: {g['total'] // max(g['genome_params'], 1)}:1")

    total_time = sum(r.get('time', 0) for r in results.values())
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if genome is not None:
        genome_path = f"results/genome_transformer_{timestamp}.pt"
        torch.save(genome.state_dict(), genome_path)
        print(f"\n  Genome saved to {genome_path}")
    elif saved_genome_path and os.path.exists(saved_genome_path):
        genome_path = saved_genome_path
        print(f"\n  Genome from checkpoint: {genome_path}")
    else:
        genome_path = None

    result_data = {
        "experiment": "rung3_transformer",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_types": 8, "type_dim": 8, "n_bands": 8,
            "hidden": hidden, "ff_dim": ff_dim, "n_layers": n_layers,
            "n_heads": n_heads, "max_len": MAX_LEN,
            "dense_epochs": 10, "genome_epochs": 15,
            "sparsity_weight": 0.005, "device": str(device)
        },
        "results": results,
        "genome_state": genome_path,
        "total_time": total_time
    }
    json_path = f"results/rung3_transformer_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")

    clear_checkpoint()
    print("  Checkpoint cleared. Run complete.")


if __name__ == "__main__":
    run()
