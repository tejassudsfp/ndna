"""
Download pretrained GPT-2-small and benchmark it.

This gives us the target numbers to compare against.
Downloads from HuggingFace, evaluates on HellaSwag and WikiText-103.

Usage:
    python3 experiments/benchmark_pretrained.py
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np


def eval_hellaswag(model, tokenizer, device, max_len=1024):
    """Zero-shot HellaSwag evaluation using HuggingFace model."""
    from datasets import load_dataset

    print("\n  HellaSwag zero-shot evaluation")
    ds = load_dataset("Rowan/hellaswag", split="validation",
                      cache_dir="./data/huggingface")

    correct = 0
    total = 0

    for idx, example in enumerate(ds):
        ctx_text = example['ctx']
        label = int(example['label'])
        endings = example['endings']

        ctx_tokens = tokenizer.encode(ctx_text)

        scores = []
        for ending in endings:
            end_tokens = tokenizer.encode(" " + ending)
            tokens = ctx_tokens + end_tokens
            if len(tokens) > max_len:
                tokens = tokens[:max_len]

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
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


def eval_wikitext(model, tokenizer, device, max_len=1024):
    """WikiText-103 perplexity using HuggingFace model."""
    from datasets import load_dataset

    print("\n  WikiText-103 perplexity evaluation")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                      cache_dir="./data/huggingface")

    text = "\n\n".join([x['text'] for x in ds if x['text'].strip()])
    tokens = tokenizer.encode(text)
    print(f"  Test set: {len(tokens):,} tokens")

    stride = max_len // 2
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(tokens) - max_len, stride):
        chunk = tokens[i:i + max_len + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            start = 0 if i == 0 else stride
            loss = F.cross_entropy(
                logits[:, start:, :].reshape(-1, logits.size(-1)),
                y[:, start:].reshape(-1),
                reduction='sum'
            )
            total_nll += loss.item()
            total_tokens += y[:, start:].numel()

        if (i // stride + 1) % 100 == 0:
            ppl = math.exp(total_nll / total_tokens)
            print(f"    chunks: {i // stride + 1}, perplexity: {ppl:.2f}")

    perplexity = math.exp(total_nll / total_tokens)
    print(f"\n  WikiText-103 perplexity: {perplexity:.2f}")
    return perplexity


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    print("=" * 60)
    print("  PRETRAINED GPT-2-SMALL BENCHMARK")
    print("=" * 60)
    print(f"  Device: {device}")

    print("\n  Downloading GPT-2-small from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='./data/huggingface')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./data/huggingface')
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    results = {'params': total_params}

    print("\n  Running benchmarks...")
    results['hellaswag_accuracy'] = eval_hellaswag(model, tokenizer, device)
    results['wikitext103_perplexity'] = eval_wikitext(model, tokenizer, device)

    # Save
    os.makedirs('results', exist_ok=True)
    out_path = 'results/pretrained_gpt2_benchmark.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print(f"\n  SUMMARY")
    print(f"  Params:              {results['params']:,}")
    print(f"  HellaSwag accuracy:  {results['hellaswag_accuracy']:.4f}")
    print(f"  WikiText-103 ppl:    {results['wikitext103_perplexity']:.2f}")


if __name__ == "__main__":
    main()
