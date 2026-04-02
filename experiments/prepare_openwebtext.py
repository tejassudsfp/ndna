"""
Prepare datasets for GPT-2 training.

Supports:
  - OpenWebText (full, ~9B tokens) for cloud training
  - Shakespeare (tiny, ~1M chars) for local debug

Tokenizes with tiktoken GPT-2 BPE and saves as numpy memmap files.
Pattern follows nanoGPT: train.bin + val.bin as uint16 arrays.

Usage:
    python3 experiments/prepare_openwebtext.py openwebtext
    python3 experiments/prepare_openwebtext.py shakespeare
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_shakespeare(out_dir='data/shakespeare'):
    """Download and tokenize Shakespeare for debug training."""
    import tiktoken
    import urllib.request

    os.makedirs(out_dir, exist_ok=True)

    # Download
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    filepath = os.path.join(out_dir, 'input.txt')
    if not os.path.exists(filepath):
        print(f"Downloading Shakespeare to {filepath}...")
        urllib.request.urlretrieve(url, filepath)

    with open(filepath, 'r') as f:
        text = f.read()
    print(f"Shakespeare: {len(text):,} characters")

    # Tokenize with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Tokenized: {len(tokens):,} tokens")

    # Split: 90% train, 10% val
    n = len(tokens)
    split = int(n * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Save as memmap
    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Train: {len(train_tokens):,} tokens -> {train_path}")
    print(f"Val:   {len(val_tokens):,} tokens -> {val_path}")
    print("Done.")


def prepare_openwebtext(out_dir='data/openwebtext'):
    """Download and tokenize OpenWebText for full training."""
    import tiktoken
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    print("Loading OpenWebText from HuggingFace (this may take a while)...")
    ds = load_dataset("Skylion007/openwebtext", trust_remote_code=True,
                      cache_dir="./data/huggingface")

    # Split: use 0.5% as val set
    split_ds = ds['train'].train_test_split(test_size=0.0005, seed=42, shuffle=True)
    split_ds['val'] = split_ds.pop('test')

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")

    def tokenize(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)  # end of text token
        return {'ids': ids, 'len': len(ids)}

    print("Tokenizing...")
    tokenized = split_ds.map(
        tokenize,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=os.cpu_count(),
    )

    # Concatenate and save as memmap
    for split, dset in tokenized.items():
        total_len = sum(dset['len'])
        print(f"{split}: {total_len:,} tokens across {len(dset):,} documents")

        filepath = os.path.join(out_dir, f'{split}.bin')
        arr = np.memmap(filepath, dtype=np.uint16, mode='w+', shape=(total_len,))

        idx = 0
        for example in dset:
            ids = np.array(example['ids'], dtype=np.uint16)
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
        print(f"  Saved to {filepath}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset = sys.argv[1].lower()
    if dataset == 'shakespeare':
        prepare_shakespeare()
    elif dataset == 'openwebtext':
        prepare_openwebtext()
    else:
        print(f"Unknown dataset: {dataset}")
        print("Use 'shakespeare' or 'openwebtext'")
        sys.exit(1)
