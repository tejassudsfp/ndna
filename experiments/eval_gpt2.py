"""
Evaluate GPT-2 models on standard benchmarks.

Benchmarks:
  1. HellaSwag (zero-shot) - commonsense reasoning
  2. WikiText-103 perplexity - standard LM benchmark
  3. Topology analysis - genome density patterns per layer

Usage:
    python3 experiments/eval_gpt2.py --checkpoint results/gpt2_full/genome_ckpt.pt --eval hellaswag
    python3 experiments/eval_gpt2.py --checkpoint results/gpt2_full/genome_ckpt.pt --eval wikitext
    python3 experiments/eval_gpt2.py --checkpoint results/gpt2_full/genome_ckpt.pt --eval topology
    python3 experiments/eval_gpt2.py --checkpoint results/gpt2_full/genome_ckpt.pt --eval all
"""

import sys
import os
import argparse
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from genome import Genome, GrownGPT2
from genome.baselines import DenseGPT2, RandomSparseGPT2


def load_model_from_checkpoint(ckpt_path, device):
    """Load a model from checkpoint, auto-detecting type."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt['config']
    model_name = ckpt['model_name']

    model_kwargs = dict(
        vocab_size=cfg['vocab_size'],
        hidden=cfg['hidden'],
        ff_dim=cfg['ff_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        max_len=cfg['block_size'],
    )

    genome = None
    if model_name == 'dense':
        model = DenseGPT2(**model_kwargs)
    elif model_name == 'genome':
        n_bands = cfg['n_layers'] + 2
        genome = Genome(n_types=cfg['n_types'], type_dim=cfg['type_dim'], n_bands=n_bands)
        if 'genome_state' in ckpt:
            genome.load_state_dict(ckpt['genome_state'])
        model = GrownGPT2(genome, **model_kwargs)
    elif model_name == 'sparse':
        density = 0.1  # Will be overridden by state dict
        model = RandomSparseGPT2(density, **model_kwargs)

    # Strip _orig_mod. prefix from torch.compile checkpoints
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, model_name, cfg, genome


# ---------------------------------------------------------------------------
# HellaSwag evaluation
# ---------------------------------------------------------------------------

def eval_hellaswag(model, device, cfg):
    """Zero-shot HellaSwag evaluation.

    For each example: context + 4 completions. Pick the one with lowest
    per-token loss (highest likelihood). Report accuracy.
    """
    import tiktoken
    from datasets import load_dataset

    print("\n  HellaSwag zero-shot evaluation")
    print("  Loading dataset...")

    ds = load_dataset("Rowan/hellaswag", split="validation",
                      cache_dir="./data/huggingface")
    enc = tiktoken.get_encoding("gpt2")

    correct = 0
    total = 0
    block_size = cfg['block_size']

    for idx, example in enumerate(ds):
        ctx_text = example['ctx']
        label = int(example['label'])
        endings = example['endings']

        ctx_tokens = enc.encode(ctx_text)

        # Score each ending
        scores = []
        for ending in endings:
            end_tokens = enc.encode(" " + ending)
            tokens = ctx_tokens + end_tokens
            # Truncate to block_size
            if len(tokens) > block_size:
                tokens = tokens[:block_size]

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _ = model(x)
                # Only score the ending tokens
                start = len(ctx_tokens) - 1
                if start >= logits.size(1):
                    scores.append(float('inf'))
                    continue
                ending_logits = logits[:, start:, :]
                ending_targets = y[:, start:]
                loss = F.cross_entropy(
                    ending_logits.reshape(-1, ending_logits.size(-1)),
                    ending_targets.reshape(-1),
                    reduction='mean'
                )
                scores.append(loss.item())

        pred = np.argmin(scores)
        if pred == label:
            correct += 1
        total += 1

        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(ds)}: accuracy={correct / total:.4f}")

    accuracy = correct / total
    print(f"\n  HellaSwag accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


# ---------------------------------------------------------------------------
# WikiText-103 perplexity
# ---------------------------------------------------------------------------

def eval_wikitext(model, device, cfg):
    """Compute perplexity on WikiText-103 test set."""
    import tiktoken
    from datasets import load_dataset

    print("\n  WikiText-103 perplexity evaluation")
    print("  Loading dataset...")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                      cache_dir="./data/huggingface")
    enc = tiktoken.get_encoding("gpt2")

    # Concatenate all text
    text = "\n\n".join([x['text'] for x in ds if x['text'].strip()])
    tokens = enc.encode(text)
    print(f"  Test set: {len(tokens):,} tokens")

    block_size = cfg['block_size']
    stride = block_size // 2  # Overlapping windows

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(tokens) - block_size, stride):
        chunk = tokens[i:i + block_size + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(x)
            # Only count the non-overlapping part for perplexity
            start = 0 if i == 0 else stride
            loss = F.cross_entropy(
                logits[:, start:, :].reshape(-1, logits.size(-1)),
                y[:, start:].reshape(-1),
                reduction='sum'
            )
            total_nll += loss.item()
            total_tokens += y[:, start:].numel()

        if (i // stride + 1) % 100 == 0:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"    chunks processed: {i // stride + 1}, perplexity so far: {ppl_so_far:.2f}")

    perplexity = math.exp(total_nll / total_tokens)
    print(f"\n  WikiText-103 perplexity: {perplexity:.2f}")
    return perplexity


# ---------------------------------------------------------------------------
# Topology analysis
# ---------------------------------------------------------------------------

def eval_topology(model, model_name, genome):
    """Analyze genome topology patterns."""
    print("\n  Topology analysis")

    if model_name != 'genome' or genome is None:
        print("  Skipping: not a genome model")
        return {}

    inner = model._orig_mod if hasattr(model, '_orig_mod') else model
    inner.describe_topology()

    # Detailed per-layer analysis
    n_layers = inner.n_layers
    hidden = inner.hidden
    ff_dim = inner.ff_dim

    layer_stats = []
    with torch.no_grad():
        for i in range(n_layers):
            wo_mask = inner._get_mask(i, i + 1, hidden, hidden)
            ff_mask = inner._get_mask(i, i + 1, hidden, ff_dim)

            wo_density = (wo_mask > 0.5).float().mean().item()
            ff_density = (ff_mask > 0.5).float().mean().item()
            wo_soft = wo_mask.mean().item()
            ff_soft = ff_mask.mean().item()

            layer_stats.append({
                'layer': i + 1,
                'wo_hard_density': wo_density,
                'wo_soft_density': wo_soft,
                'ff_hard_density': ff_density,
                'ff_soft_density': ff_soft,
            })

    print("\n  Per-layer density summary:")
    print(f"  {'Layer':>6} {'W_o hard':>10} {'W_o soft':>10} {'FF1 hard':>10} {'FF1 soft':>10}")
    print(f"  {'-' * 50}")
    for s in layer_stats:
        print(f"  {s['layer']:>6} {s['wo_hard_density']:>10.1%} {s['wo_soft_density']:>10.3f} "
              f"{s['ff_hard_density']:>10.1%} {s['ff_soft_density']:>10.3f}")

    # Overall stats
    active, total, sd = inner.count_effective()
    gp = sum(p.numel() for p in genome.parameters())
    print(f"\n  Genome params: {gp}")
    print(f"  Total masked connections: {total:,}")
    print(f"  Active connections (>0.5): {active:,} ({active / total:.1%})")
    print(f"  Soft density: {sd:.4f}")
    print(f"  Compression: {total // max(gp, 1):,}:1")

    return {
        'layer_stats': layer_stats,
        'genome_params': gp,
        'total_connections': total,
        'active_connections': active,
        'hard_density': active / total if total > 0 else 0,
        'soft_density': sd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eval', type=str, default='all',
                        choices=['hellaswag', 'wikitext', 'topology', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    print(f"  Device: {device}")
    print(f"  Loading checkpoint: {args.checkpoint}")

    model, model_name, cfg, genome = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Model: {model_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    results = {}

    if args.eval in ('hellaswag', 'all'):
        results['hellaswag_accuracy'] = eval_hellaswag(model, device, cfg)

    if args.eval in ('wikitext', 'all'):
        results['wikitext103_perplexity'] = eval_wikitext(model, device, cfg)

    if args.eval in ('topology', 'all'):
        topo = eval_topology(model, model_name, genome)
        results['topology'] = topo

    # Save results
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, f'eval_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Evaluation results saved to {out_path}")


if __name__ == "__main__":
    main()
