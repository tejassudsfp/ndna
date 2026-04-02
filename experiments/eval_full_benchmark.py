"""
Full GPT-2 benchmark suite matching OpenAI's reported numbers.

lm-evaluation-harness tasks:
  - lambada_openai (PPL + ACC)
  - hellaswag (ACC)
  - wikitext (PPL for wikitext-2 and wikitext-103)

Custom perplexity evals:
  - PTB (Penn Treebank)
  - enwiki8 (bits per byte)
  - text8 (bits per character)
  - CBT-CN and CBT-NE (accuracy)
  - 1BW (One Billion Word benchmark PPL)
"""
import sys, os, argparse, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import tiktoken
from genome.model import GrownGPT2, Genome
from genome.baselines import DenseGPT2


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt['config']
    model_name = ckpt['model_name']

    model_kwargs = dict(
        vocab_size=cfg['vocab_size'], hidden=cfg['hidden'],
        ff_dim=cfg['ff_dim'], n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'], max_len=cfg['block_size'],
    )

    if model_name == 'genome':
        n_bands = cfg['n_layers'] + 2
        genome = Genome(n_types=cfg['n_types'], type_dim=cfg['type_dim'], n_bands=n_bands)
        if 'genome_state' in ckpt:
            genome.load_state_dict(ckpt['genome_state'])
        model = GrownGPT2(genome, **model_kwargs)
    elif model_name == 'dense':
        model = DenseGPT2(**model_kwargs)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    if hasattr(model, 'hard_masks'):
        model.hard_masks = True
    return model, cfg, ckpt.get('iter_num', 0)


def compute_perplexity(model, tokens, block_size, device, stride=None):
    """Sliding window perplexity."""
    if stride is None:
        stride = block_size // 2
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, max(1, len(tokens) - block_size), stride):
        chunk = tokens[i:i + block_size + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
            start = 0 if i == 0 else stride
            loss = F.cross_entropy(
                logits[:, start:, :].reshape(-1, logits.size(-1)),
                y[:, start:].reshape(-1),
                reduction='sum'
            )
            total_nll += loss.item()
            total_tokens += y[:, start:].numel()
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    return ppl, total_nll, total_tokens


def compute_bpb(model, raw_bytes, enc, block_size, device):
    """Bits per byte for byte-level benchmarks like enwiki8."""
    text = raw_bytes.decode('utf-8', errors='replace')
    tokens = enc.encode(text)
    _, total_nll, total_tokens = compute_perplexity(model, tokens, block_size, device)
    # Convert NLL (nats) to bits, then divide by number of bytes
    bits = total_nll / math.log(2)
    bpb = bits / len(raw_bytes)
    return bpb


def eval_cbt(model, enc, block_size, device, subset='CN'):
    """Children's Book Test - CN (common nouns) or NE (named entities)."""
    from datasets import load_dataset
    # HuggingFace CBT config names are uppercase: 'CN', 'NE' (not 'cn', 'ne')
    subset_upper = subset.upper()
    print(f'  Loading CBT-{subset_upper}...')
    try:
        ds = load_dataset('cbt', subset_upper, split='test', cache_dir='./data/huggingface', trust_remote_code=True)
    except Exception as e:
        print(f'  CBT-{subset_upper} failed to load: {e}')
        return None

    correct = 0
    total = 0
    for idx, ex in enumerate(ds):
        sentences = ex.get('sentences', [])
        question = ex.get('question', '')
        answer = ex.get('answer', '')
        options = ex.get('options', [])

        if not options or not answer:
            continue

        context = ' '.join(sentences) + ' '
        scores = []
        for opt in options:
            filled = question.replace('XXXXX', opt)
            full_text = context + filled
            tokens = enc.encode(full_text)
            ctx_tokens = enc.encode(context)
            if len(tokens) > block_size:
                tokens = tokens[-block_size:]
                ctx_len = max(0, len(tokens) - (len(enc.encode(filled))))
            else:
                ctx_len = len(ctx_tokens)

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(x)
                lp = F.log_softmax(logits, dim=-1)
                score = 0.0
                start = max(ctx_len - 1, 0)
                for j in range(start, len(tokens) - 1):
                    score += lp[0, j, tokens[j + 1]].item()
                scores.append(score)

        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        if options[pred_idx] == answer:
            correct += 1
        total += 1

        if (idx + 1) % 500 == 0:
            print(f'    {idx+1}/{len(ds)}: acc={correct/total:.4f}')

    acc = correct / total if total > 0 else 0
    print(f'  CBT-{subset_upper} accuracy: {acc:.4f} ({correct}/{total})')
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/gpt2_full/genome_ckpt.pt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, iter_num = load_model(args.checkpoint, device)
    block_size = cfg['block_size']
    enc = tiktoken.get_encoding('gpt2')

    n_params = sum(p.numel() for p in model.parameters())
    print('=' * 60)
    print('  NDNA GENOME GPT-2 — FULL BENCHMARK SUITE')
    print('=' * 60)
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Iteration: {iter_num}')
    print(f'  Parameters: {n_params:,}')
    print(f'  Block size: {block_size}')
    print()

    results = {'iter': iter_num, 'params': n_params}

    from datasets import load_dataset

    # 1. WikiText-2 PPL
    print('[1/9] WikiText-2 Perplexity')
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='./data/huggingface')
    text = '\n\n'.join([x['text'] for x in ds if x['text'].strip()])
    tokens = enc.encode(text)
    ppl, _, _ = compute_perplexity(model, tokens, block_size, device)
    results['wikitext2_ppl'] = ppl
    print(f'  WikiText-2 PPL: {ppl:.2f}')
    print()

    # 2. WikiText-103 PPL
    print('[2/9] WikiText-103 Perplexity')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test', cache_dir='./data/huggingface')
    text = '\n\n'.join([x['text'] for x in ds if x['text'].strip()])
    tokens = enc.encode(text)
    ppl, _, _ = compute_perplexity(model, tokens, block_size, device)
    results['wikitext103_ppl'] = ppl
    print(f'  WikiText-103 PPL: {ppl:.2f}')
    print()

    # 3. PTB PPL
    # Uses raw test file from Wojciech Zaremba's LSTM repo (the standard PTB source).
    # The HuggingFace ptb_text_only dataset was deprecated; this is more reliable.
    print('[3/9] Penn Treebank Perplexity')
    try:
        import urllib.request
        ptb_path = './data/ptb_test.txt'
        if not os.path.exists(ptb_path):
            print('  Downloading PTB test set...')
            os.makedirs('./data', exist_ok=True)
            url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
            urllib.request.urlretrieve(url, ptb_path)
        with open(ptb_path, 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        ppl, _, _ = compute_perplexity(model, tokens, block_size, device)
        results['ptb_ppl'] = ppl
        print(f'  PTB PPL: {ppl:.2f}')
    except Exception as e:
        print(f'  PTB failed: {e}')
        results['ptb_ppl'] = None
    print()

    # 4. LAMBADA (PPL + ACC)
    print('[4/9] LAMBADA')
    ds = load_dataset('EleutherAI/lambada_openai', 'default', split='test', cache_dir='./data/huggingface', trust_remote_code=True)
    correct = 0
    total_nll = 0.0
    total_toks = 0
    total = 0
    for idx, ex in enumerate(ds):
        text = ex['text']
        tokens = enc.encode(text)
        words = text.rsplit(' ', 1)
        if len(words) < 2:
            continue
        last_word_tokens = enc.encode(' ' + words[-1])
        ctx_len = len(tokens) - len(last_word_tokens)

        if len(tokens) > block_size:
            tokens = tokens[-block_size:]
            ctx_len = max(0, len(tokens) - len(last_word_tokens))

        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
            lp = F.log_softmax(logits, dim=-1)

            # ACC: greedy prediction of last word
            pred_tokens = []
            for j in range(ctx_len - 1, len(tokens) - 1):
                pred_tokens.append(logits[0, j].argmax().item())
            if pred_tokens == list(tokens[ctx_len:]):
                correct += 1

            # PPL: loss on last word only
            for j in range(ctx_len - 1, len(tokens) - 1):
                total_nll += -lp[0, j, tokens[j + 1]].item()
                total_toks += 1

        total += 1
        if (idx + 1) % 1000 == 0:
            print(f'    {idx+1}/{len(ds)}: acc={correct/total:.4f}')

    lambada_acc = correct / total if total > 0 else 0
    lambada_ppl = math.exp(total_nll / total_toks) if total_toks > 0 else float('inf')
    results['lambada_acc'] = lambada_acc
    results['lambada_ppl'] = lambada_ppl
    print(f'  LAMBADA ACC: {lambada_acc:.4f} ({correct}/{total})')
    print(f'  LAMBADA PPL: {lambada_ppl:.2f}')
    print()

    # 5. HellaSwag
    print('[5/9] HellaSwag')
    ds = load_dataset('Rowan/hellaswag', split='validation', cache_dir='./data/huggingface', trust_remote_code=True)
    correct = 0
    total = 0
    for idx, ex in enumerate(ds):
        ctx_tokens = enc.encode(ex['ctx'])
        label = int(ex['label'])
        scores = []
        for ending in ex['endings']:
            end_tokens = enc.encode(' ' + ending)
            tokens = ctx_tokens + end_tokens
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(x)
                start = len(ctx_tokens) - 1
                if start >= logits.size(1):
                    scores.append(float('inf'))
                    continue
                loss = F.cross_entropy(
                    logits[:, start:, :].reshape(-1, logits.size(-1)),
                    y[:, start:].reshape(-1), reduction='mean')
                scores.append(loss.item())
        if min(scores) < float('inf') and scores.index(min(scores)) == label:
            correct += 1
        total += 1
        if (idx + 1) % 2000 == 0:
            print(f'    {idx+1}/{len(ds)}: acc={correct/total:.4f}')
    hellaswag_acc = correct / total
    results['hellaswag_acc'] = hellaswag_acc
    print(f'  HellaSwag ACC: {hellaswag_acc:.4f} ({correct}/{total})')
    print()

    # 6. CBT-CN
    print('[6/9] CBT-CN')
    cbt_cn = eval_cbt(model, enc, block_size, device, 'CN')
    results['cbt_cn_acc'] = cbt_cn
    print()

    # 7. CBT-NE
    print('[7/9] CBT-NE')
    cbt_ne = eval_cbt(model, enc, block_size, device, 'NE')
    results['cbt_ne_acc'] = cbt_ne
    print()

    # 8. enwiki8 (bits per byte)
    print('[8/9] enwiki8 (BPB)')
    try:
        import urllib.request, zipfile, io
        enwiki8_path = './data/enwik8'
        if not os.path.exists(enwiki8_path):
            print('  Downloading enwik8...')
            os.makedirs('./data', exist_ok=True)
            url = 'http://mattmahoney.net/dc/enwik8.zip'
            urllib.request.urlretrieve(url, './data/enwik8.zip')
            with zipfile.ZipFile('./data/enwik8.zip', 'r') as z:
                z.extractall('./data/')
        with open(enwiki8_path, 'rb') as f:
            raw = f.read()
        # Standard split: last 5M bytes for test
        test_bytes = raw[-5_000_000:]
        bpb = compute_bpb(model, test_bytes, enc, block_size, device)
        results['enwiki8_bpb'] = bpb
        print(f'  enwiki8 BPB: {bpb:.3f}')
    except Exception as e:
        print(f'  enwiki8 failed: {e}')
        results['enwiki8_bpb'] = None
    print()

    # 9. text8 (bits per character)
    print('[9/9] text8 (BPC)')
    try:
        text8_path = './data/text8'
        if not os.path.exists(text8_path):
            print('  Downloading text8...')
            os.makedirs('./data', exist_ok=True)
            url = 'http://mattmahoney.net/dc/text8.zip'
            urllib.request.urlretrieve(url, './data/text8.zip')
            with zipfile.ZipFile('./data/text8.zip', 'r') as z:
                z.extractall('./data/')
        with open(text8_path, 'rb') as f:
            raw = f.read()
        # Standard split: last 5M chars for test
        test_bytes = raw[-5_000_000:]
        bpc = compute_bpb(model, test_bytes, enc, block_size, device)
        results['text8_bpc'] = bpc
        print(f'  text8 BPC: {bpc:.3f}')
    except Exception as e:
        print(f'  text8 failed: {e}')
        results['text8_bpc'] = None
    print()

    # Summary
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print(f'  {"Benchmark":<20} {"NDNA Genome":>15} {"GPT-2 (OpenAI)":>15}')
    print(f'  {"-"*50}')
    ref = {
        'lambada_ppl': 35.13, 'lambada_acc': 0.4599,
        'cbt_cn_acc': 0.8765, 'cbt_ne_acc': 0.834,
        'wikitext2_ppl': 29.41, 'ptb_ppl': 65.85,
        'enwiki8_bpb': 1.16, 'text8_bpc': 1.17,
        'wikitext103_ppl': 37.50, 'hellaswag_acc': 0.312,
    }
    for key, ref_val in ref.items():
        val = results.get(key)
        if val is None:
            val_str = 'N/A'
        elif isinstance(val, float):
            if 'acc' in key:
                val_str = f'{val*100:.2f}%'
                ref_str = f'{ref_val*100:.2f}%' if ref_val < 1 else f'{ref_val:.2f}%'
            elif 'bpb' in key or 'bpc' in key:
                val_str = f'{val:.3f}'
                ref_str = f'{ref_val:.2f}'
            else:
                val_str = f'{val:.2f}'
                ref_str = f'{ref_val:.2f}'
        else:
            val_str = str(val)
            ref_str = str(ref_val)
        print(f'  {key:<20} {val_str:>15} {ref_str:>15}')

    # Save
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_full_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved to {out_path}')


if __name__ == '__main__':
    main()
