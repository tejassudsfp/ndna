"""
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
Rung 5: Train GPT-2-small from scratch with genome-controlled connectivity.

354 parameters grow GPT-2. Genome masks W_o and FF1 in every layer,
controlling 35.4M connections at ~100,000:1 compression.

Three models trained sequentially:
  1. DenseGPT2     - standard GPT-2-small, no masks (infrastructure validation)
  2. GrownGPT2     - genome masks on W_o + FF1 (the hypothesis)
  3. RandomSparseGPT2 - fixed random masks at matched density (the critical control)

Two configs:
  --config debug   Shakespeare on M3 MacBook (4 layers, hidden=128)
  --config full    OpenWebText on A100 (12 layers, hidden=768)

Usage:
    python3 run.py gpt2 --config debug
    python3 run.py gpt2 --config full
    python3 run.py gpt2 --config debug --model genome   (train only genome model)
    python3 run.py gpt2 --config debug --model dense    (train only dense model)
    python3 run.py gpt2 --config debug --model sparse   (train only sparse model)
"""

import sys
import os
import time
import json
import math
import argparse
from contextlib import nullcontext
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from genome import Genome, GrownGPT2
from genome.baselines import DenseGPT2, RandomSparseGPT2

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    'debug': dict(
        # Model
        n_layers=4,
        n_heads=4,
        hidden=128,
        ff_dim=512,
        vocab_size=50257,
        block_size=256,
        # Training
        max_iters=5000,
        eval_interval=250,
        eval_iters=50,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=6e-4,
        genome_lr=0.01,
        weight_decay=0.1,
        warmup_iters=200,
        lr_decay_iters=5000,
        min_lr=6e-5,
        grad_clip=1.0,
        sparsity_weight=0.005,
        temp_start=1.0,
        temp_end=10.0,
        temp_warmup_iters=2500,
        # Data
        dataset='shakespeare',
        data_dir='data/shakespeare',
        # Genome
        n_types=8,
        type_dim=8,
        # System
        dtype='float32',  # MPS doesn't support bf16 well
        compile_model=False,
        checkpoint_dir='results/gpt2_debug',
    ),
    'full': dict(
        # Model
        n_layers=12,
        n_heads=12,
        hidden=768,
        ff_dim=3072,
        vocab_size=50257,
        block_size=1024,
        # Training
        max_iters=100000,
        eval_interval=100,
        eval_iters=200,
        batch_size=12,
        gradient_accumulation_steps=40,
        learning_rate=6e-4,
        genome_lr=0.01,
        weight_decay=0.1,
        warmup_iters=2000,
        lr_decay_iters=100000,
        min_lr=6e-5,
        grad_clip=1.0,
        sparsity_weight=0.005,
        temp_start=1.0,
        temp_end=10.0,
        temp_warmup_iters=50000,
        # Data
        dataset='openwebtext',
        data_dir='data/openwebtext',
        # Genome
        n_types=8,
        type_dim=8,
        # System
        dtype='bfloat16',
        compile_model=True,
        checkpoint_dir='results/gpt2_full',
    ),
}


# ---------------------------------------------------------------------------
# Data loading (nanoGPT pattern)
# ---------------------------------------------------------------------------

def get_batch(split, cfg, device):
    """Load a random batch from memmap binary file."""
    data_dir = cfg['data_dir']
    filename = 'train.bin' if split == 'train' else 'val.bin'
    filepath = os.path.join(data_dir, filename)
    data = np.memmap(filepath, dtype=np.uint16, mode='r')

    block_size = cfg['block_size']
    batch_size = cfg['batch_size']

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(it, cfg):
    """Cosine decay with linear warmup."""
    warmup_iters = cfg['warmup_iters']
    lr_decay_iters = cfg['lr_decay_iters']
    learning_rate = cfg['learning_rate']
    min_lr = cfg['min_lr']

    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_temperature(it, cfg):
    """Temperature annealing for genome masks."""
    if it >= cfg['temp_warmup_iters']:
        return cfg['temp_end']
    ratio = it / cfg['temp_warmup_iters']
    return cfg['temp_start'] + (cfg['temp_end'] - cfg['temp_start']) * ratio


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, cfg, device, ctx):
    """Estimate loss on train and val splits."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(cfg['eval_iters']):
            x, y = get_batch(split, cfg, device)
            with ctx:
                _, loss = model(x, targets=y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Count params
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, iter_num, best_val, cfg, model_name,
                    genome=None, genome_optimizer=None):
    """Save training checkpoint for resume."""
    ckpt_dir = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val': best_val,
        'config': cfg,
        'model_name': model_name,
    }
    if genome is not None:
        ckpt['genome_state'] = genome.state_dict()
    if genome_optimizer is not None:
        ckpt['genome_optimizer_state'] = genome_optimizer.state_dict()
    path = os.path.join(ckpt_dir, f'{model_name}_ckpt.pt')
    torch.save(ckpt, path)
    return path


def load_checkpoint(cfg, model_name):
    """Load checkpoint if it exists."""
    path = os.path.join(cfg['checkpoint_dir'], f'{model_name}_ckpt.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_sample(model, device, max_new_tokens=100, top_k=40):
    """Generate a text sample from the model."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt = enc.encode("The meaning of life is")
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    model.eval()
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=top_k)
    model.train()
    return enc.decode(y[0].tolist())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model_name, cfg, device, completed_results):
    """Train a single model (dense, genome, or sparse)."""
    print(f"\n{'=' * 60}")
    print(f"  Training: {model_name}")
    print(f"{'=' * 60}")

    # Select dtype and autocast context
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[cfg['dtype']]
    # Only use autocast for mixed precision (bf16/fp16), not fp32
    if cfg['dtype'] == 'float32':
        ctx = nullcontext()
    else:
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    genome = None

    # Build model
    model_kwargs = dict(
        vocab_size=cfg['vocab_size'],
        hidden=cfg['hidden'],
        ff_dim=cfg['ff_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        max_len=cfg['block_size'],
    )

    if model_name == 'dense':
        model = DenseGPT2(**model_kwargs).to(device)
    elif model_name == 'genome':
        n_bands = cfg['n_layers'] + 2  # embedding + layers + LM head
        genome = Genome(n_types=cfg['n_types'], type_dim=cfg['type_dim'],
                        n_bands=n_bands).to(device)
        model = GrownGPT2(genome, **model_kwargs).to(device)
        gp = sum(p.numel() for p in genome.parameters())
        print(f"  Genome params: {gp}")
    elif model_name == 'sparse':
        # Get density from genome results
        density = completed_results.get('genome', {}).get('soft_density', 0.1)
        print(f"  Matched density: {density:.4f}")
        model = RandomSparseGPT2(density, **model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total_params = count_params(model)
    print(f"  Total params: {total_params:,}")

    # Compile if requested (PyTorch 2.0+)
    if cfg['compile_model'] and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer setup
    if model_name == 'genome':
        # Split optimizer: different LR for genome vs weights
        genome_params = list(genome.parameters())
        genome_ids = set(id(p) for p in genome_params)
        weight_params = [p for p in model.parameters() if id(p) not in genome_ids]

        optimizer = torch.optim.AdamW(
            weight_params, lr=cfg['learning_rate'],
            betas=(0.9, 0.95), weight_decay=cfg['weight_decay']
        )
        genome_optimizer = torch.optim.Adam(
            genome_params, lr=cfg['genome_lr']
        )
    else:
        # Standard optimizer
        # Separate weight decay: no WD on bias, layernorm, embeddings
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or 'ln' in name or 'bias' in name or 'emb' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': cfg['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=cfg['learning_rate'], betas=(0.9, 0.95))
        genome_optimizer = None

    # Check for resume
    ckpt = load_checkpoint(cfg, model_name)
    start_iter = 0
    best_val = float('inf')
    if ckpt is not None:
        print(f"  Resuming from iter {ckpt['iter_num']}...")
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_iter = ckpt['iter_num']
        best_val = ckpt['best_val']
        if genome is not None and 'genome_state' in ckpt:
            genome.load_state_dict(ckpt['genome_state'])
        if genome_optimizer is not None and 'genome_optimizer_state' in ckpt:
            genome_optimizer.load_state_dict(ckpt['genome_optimizer_state'])

    # GradScaler for float16 (not needed for bf16)
    scaler = torch.amp.GradScaler(enabled=(cfg['dtype'] == 'float16'))

    # Training
    model.train()
    t0 = time.time()
    accum_steps = cfg['gradient_accumulation_steps']
    running_loss = 0.0
    running_sp_loss = 0.0

    for it in range(start_iter, cfg['max_iters']):
        # Set learning rate
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Set temperature for genome model
        if model_name == 'genome' and hasattr(model, 'temperature'):
            temp = get_temperature(it, cfg)
            # Handle compiled model
            inner = model._orig_mod if hasattr(model, '_orig_mod') else model
            inner.temperature.fill_(temp)

        # Evaluation
        if it % cfg['eval_interval'] == 0 or it == cfg['max_iters'] - 1:
            losses = estimate_loss(model, cfg, device, ctx)
            dt = time.time() - t0
            tokens_per_sec = (it - start_iter + 1) * accum_steps * cfg['batch_size'] * cfg['block_size'] / max(dt, 1)

            msg = (f"  iter {it:6d} | train {losses['train']:.4f} | "
                   f"val {losses['val']:.4f} | lr {lr:.2e} | "
                   f"{tokens_per_sec:.0f} tok/s")

            if model_name == 'genome':
                inner = model._orig_mod if hasattr(model, '_orig_mod') else model
                active, total, sd = inner.count_effective()
                density = active / total if total > 0 else 0
                temp = inner.temperature
                msg += f" | hard={density:.1%} soft={sd:.1%} temp={temp:.1f}"
                if running_sp_loss > 0:
                    msg += f" sp={running_sp_loss / max(cfg['eval_interval'], 1):.4f}"
                    running_sp_loss = 0.0

            print(msg)

            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(model, optimizer, it, best_val, cfg, model_name,
                                genome=genome, genome_optimizer=genome_optimizer)

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        if genome_optimizer is not None:
            genome_optimizer.zero_grad(set_to_none=True)

        for micro_step in range(accum_steps):
            x, y = get_batch('train', cfg, device)
            with ctx:
                logits, loss = model(x, targets=y)
                loss = loss / accum_steps

                # Add sparsity loss for genome
                if model_name == 'genome':
                    inner = model._orig_mod if hasattr(model, '_orig_mod') else model
                    sp_loss = inner.genome.sparsity_loss_adjacent_only(
                        cfg['hidden'], cfg['ff_dim'], cfg['n_layers']
                    )
                    loss = loss + cfg['sparsity_weight'] * sp_loss / accum_steps
                    running_sp_loss += sp_loss.item()

            scaler.scale(loss).backward()

        # Gradient clipping
        if cfg['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

        scaler.step(optimizer)
        if genome_optimizer is not None:
            scaler.step(genome_optimizer)
        scaler.update()

        running_loss += loss.item() * accum_steps

    # Final evaluation
    losses = estimate_loss(model, cfg, device, ctx)
    total_time = time.time() - t0
    print(f"\n  {model_name} done. val_loss={losses['val']:.4f} best={best_val:.4f} "
          f"time={total_time:.0f}s")

    # Generate sample
    print("\n  Sample generation:")
    inner = model._orig_mod if hasattr(model, '_orig_mod') else model
    try:
        sample = generate_sample(inner, device, max_new_tokens=100)
        print(f"  {sample[:300]}...")
    except Exception as e:
        print(f"  Generation failed: {e}")

    # Collect results
    result = {
        'params': total_params,
        'val_loss': losses['val'],
        'best_val_loss': best_val,
        'train_loss': losses['train'],
        'time': total_time,
    }

    if model_name == 'genome':
        active, total, sd = inner.count_effective()
        density = active / total if total > 0 else 0
        gp = sum(p.numel() for p in genome.parameters())
        result.update({
            'genome_params': gp,
            'hard_density': density,
            'soft_density': sd,
            'active_connections': active,
            'total_connections': total,
            'compression': total // max(gp, 1),
        })
        inner.describe_topology()

        # Save genome separately
        genome_path = os.path.join(cfg['checkpoint_dir'], 'genome_final.pt')
        torch.save(genome.state_dict(), genome_path)
        result['genome_path'] = genome_path

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser(description='GPT-2 genome training')
    parser.add_argument('--config', type=str, default='debug', choices=['debug', 'full'])
    parser.add_argument('--model', type=str, default=None, choices=['dense', 'genome', 'sparse'],
                        help='Train only this model (default: all three)')

    # Parse from sys.argv, skipping the 'gpt2' command
    args = parser.parse_args(sys.argv[1:])

    cfg = CONFIGS[args.config].copy()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 60)
    print("  RUNG 5: GPT-2 WITH GENOME-CONTROLLED CONNECTIVITY")
    print("  354 parameters grow GPT-2.")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Dtype: {cfg['dtype']}")
    print(f"  Layers: {cfg['n_layers']}, Hidden: {cfg['hidden']}, FF: {cfg['ff_dim']}")
    print(f"  Block size: {cfg['block_size']}, Batch: {cfg['batch_size']}")
    print(f"  Grad accum: {cfg['gradient_accumulation_steps']}")
    tokens_per_iter = cfg['batch_size'] * cfg['gradient_accumulation_steps'] * cfg['block_size']
    print(f"  Tokens/iter: {tokens_per_iter:,}")
    print(f"  Max iters: {cfg['max_iters']:,}")
    print(f"  Dataset: {cfg['dataset']}")

    # Check data exists
    train_path = os.path.join(cfg['data_dir'], 'train.bin')
    if not os.path.exists(train_path):
        print(f"\n  ERROR: Data not found at {train_path}")
        print(f"  Run: python3 experiments/prepare_openwebtext.py {cfg['dataset']}")
        sys.exit(1)

    # Determine which models to train
    if args.model:
        models_to_train = [args.model]
    else:
        models_to_train = ['dense', 'genome', 'sparse']

    # Load any existing results
    results_path = os.path.join(cfg['checkpoint_dir'], 'results.json')
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Train models
    for model_name in models_to_train:
        if model_name == 'sparse' and 'genome' not in all_results:
            print(f"\n  WARNING: No genome results yet. Sparse will use default density 10%.")

        result = train_model(model_name, cfg, device, all_results)
        all_results[model_name] = result

        # Save incremental results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'Params':>12} {'Val Loss':>10} {'Best Val':>10}")
    print(f"  {'-' * 56}")
    for name, r in all_results.items():
        extra = ""
        if 'genome_params' in r:
            extra = f" (genome: {r['genome_params']})"
        print(f"  {name:<20} {r['params']:>12,} {r['val_loss']:>10.4f} {r['best_val_loss']:>10.4f}{extra}")

    # Key comparisons
    g = all_results.get('genome', {})
    d = all_results.get('dense', {})
    s = all_results.get('sparse', {})

    if g and s:
        gap = s['best_val_loss'] - g['best_val_loss']
        winner = 'GENOME WINS' if gap > 0 else 'RANDOM WINS'
        print(f"\n  Genome vs Random Sparse: {gap:+.4f} val loss ({winner})")

    if g and d:
        gap = g['best_val_loss'] - d['best_val_loss']
        print(f"  Genome vs Dense: {gap:+.4f} val loss gap to ceiling")

    if g:
        print(f"\n  GENOME: {g['genome_params']} params -> {g['total_connections']:,} masked connections")
        print(f"  Compression: {g['compression']:,}:1")
        print(f"  Hard density: {g['hard_density']:.1%}, Soft density: {g['soft_density']:.1%}")

    # Save final results with metadata
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    final_results = {
        "experiment": "rung5_gpt2",
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "results": all_results,
        "total_time": sum(r.get('time', 0) for r in all_results.values()),
    }
    final_path = os.path.join(cfg['checkpoint_dir'], f'rung5_gpt2_{timestamp}.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Final results saved to {final_path}")


if __name__ == "__main__":
    run()
