"""
Rung 4: Factored Spatiotemporal Genome on Moving MNIST.

A factored genome with separate temporal, spatial, and depth components
controls a video transformer that predicts the next frame from 10 input frames.

Factored Genome (~374 params):
  - Temporal genome (74 params): controls which frames attend to which (10x10)
  - Spatial genome (74 params): controls which patches attend to which (64x64)
  - Depth genome (226 params): controls FF and cross-layer connectivity

Architecture: Divided Space-Time Attention (like TimeSformer)
  - Each layer: spatial attention -> temporal attention -> FF
  - Spatial attention: within each frame, 64 patches attend to each other
  - Temporal attention: at each spatial position, 10 frames attend to each other
  - This gives each genome DIRECT gradient signal (no kron averaging)
  - Much more memory efficient: 64x64 + 10x10 vs 640x640

4-model comparison (all use divided attention for fair comparison):
  1. Dense Video Transformer (ceiling, 30 epochs)
  2. Genome Video Transformer (hypothesis, 50 epochs)
  3. Random Sparse Video Transformer (control, matched density, 30 epochs)
  4. Dense Skip Video Transformer (full cross-layer, 30 epochs)

Memory-safe for 8GB M3: batch 32, gradient accumulation 2, MPS cache flushing.
Resume-safe: checkpoint after each model completes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, json, math
from datetime import datetime

from genome.model import Genome

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = 'results/_video_checkpoint.json'


# --- Data: Moving MNIST ---

def generate_moving_mnist(n_sequences, n_frames=11, img_size=64, digit_size=28, n_digits=2):
    """Generate Moving MNIST sequences procedurally.

    Returns tensor of shape (n_sequences, n_frames, img_size, img_size) in [0, 1].
    """
    from torchvision import datasets
    mnist = datasets.MNIST('./data', train=True, download=True)
    mnist_images = mnist.data.float() / 255.0  # (60000, 28, 28)

    sequences = torch.zeros(n_sequences, n_frames, img_size, img_size)

    for i in range(n_sequences):
        for d in range(n_digits):
            idx = np.random.randint(len(mnist_images))
            digit = mnist_images[idx]

            x = np.random.randint(0, img_size - digit_size)
            y = np.random.randint(0, img_size - digit_size)
            vx = np.random.randint(-3, 4)
            vy = np.random.randint(-3, 4)
            while vx == 0 and vy == 0:
                vx = np.random.randint(-3, 4)
                vy = np.random.randint(-3, 4)

            for t in range(n_frames):
                x1, y1 = int(x), int(y)
                sequences[i, t, y1:y1+digit_size, x1:x1+digit_size] = torch.clamp(
                    sequences[i, t, y1:y1+digit_size, x1:x1+digit_size] + digit, 0, 1
                )
                x += vx
                y += vy
                if x < 0:
                    x = -x; vx = -vx
                if x > img_size - digit_size:
                    x = 2 * (img_size - digit_size) - x; vx = -vx
                if y < 0:
                    y = -y; vy = -vy
                if y > img_size - digit_size:
                    y = 2 * (img_size - digit_size) - y; vy = -vy
                x = max(0, min(img_size - digit_size, x))
                y = max(0, min(img_size - digit_size, y))

    return sequences


def make_dataloaders(batch_size=32):
    """Create train/test dataloaders for Moving MNIST.

    Memory-safe: generates sequences, splits immediately, frees originals.
    5000 train + 1000 test keeps peak RAM under 2GB.
    """
    import gc

    print("  Generating Moving MNIST training sequences...")
    train_seq = generate_moving_mnist(5000, n_frames=11)
    train_x = train_seq[:, :10].contiguous()  # (5000, 10, 64, 64)
    train_y = train_seq[:, 10].contiguous()    # (5000, 64, 64)
    del train_seq; gc.collect()

    print("  Generating Moving MNIST test sequences...")
    test_seq = generate_moving_mnist(1000, n_frames=11)
    test_x = test_seq[:, :10].contiguous()
    test_y = test_seq[:, 10].contiguous()
    del test_seq; gc.collect()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    test_ds = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train: {len(train_ds):,}, Test: {len(test_ds):,}")
    return train_loader, test_loader


# --- Factored Spatiotemporal Genome ---

class SpatiotemporalGenome(nn.Module):
    """Factored genome: temporal (74) + spatial (74) + depth (226) = 374 params."""

    def __init__(self):
        super().__init__()
        self.temporal = Genome(n_types=4, type_dim=4, n_bands=5)
        self.spatial = Genome(n_types=4, type_dim=4, n_bands=5)
        self.depth = Genome(n_types=8, type_dim=8, n_bands=6)

        # Warm-start temporal/spatial so attention begins around 50%
        # Also tame connection_scale and band_type variance to prevent
        # sigmoid saturation (the default 3.0 scale amplifies small random
        # affinities into extreme logits, killing gradient flow).
        with torch.no_grad():
            for g in (self.temporal, self.spatial):
                g.compatibility.data.fill_(0.0)
                g.depth_penalty.data.fill_(0.5)
                g.connection_scale.data.fill_(1.0)   # was 3.0
                g.band_type_base.data.mul_(0.2)       # reduce type variation
                g.band_type_grad.data.mul_(0.2)

    def temporal_mask(self, n_frames=10, temperature=1.0):
        """(n_frames, n_frames) mask for temporal attention."""
        mask = self.temporal.growth_mask(0, 1, n_frames, n_frames, temperature=temperature)
        return mask.clamp(0.01, 0.99)  # anti-saturation: keep sigmoid gradients alive

    def spatial_mask(self, n_patches=64, temperature=1.0):
        """(n_patches, n_patches) mask for spatial attention."""
        mask = self.spatial.growth_mask(0, 1, n_patches, n_patches, temperature=temperature)
        return mask.clamp(0.01, 0.99)  # anti-saturation: keep sigmoid gradients alive

    def depth_mask(self, src_band, tgt_band, src_n, tgt_n, temperature=1.0):
        """Depth mask for FF and cross-layer."""
        return self.depth.growth_mask(src_band, tgt_band, src_n, tgt_n, temperature=temperature)

    def sparsity_loss(self, depth_dims):
        """Sparsity on depth genome only. Temporal/spatial learn freely."""
        return self.depth.sparsity_loss(depth_dims)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- Patch Embedding ---

class PatchEmbed(nn.Module):
    """64x64 -> 8x8 grid of 8x8 patches -> hidden dim."""

    def __init__(self, img_size=64, patch_size=8, hidden=128):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 8
        self.n_patches = self.grid_size ** 2  # 64
        self.proj = nn.Linear(patch_size * patch_size, hidden)

    def forward(self, x):
        """x: (B, T, H, W) -> (B, T, P, hidden) where P=n_patches."""
        B, T, H, W = x.shape
        p = self.patch_size
        g = self.grid_size
        x = x.view(B, T, g, p, g, p)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, T, g, g, p, p)
        x = x.view(B, T, self.n_patches, p * p)
        return self.proj(x)  # (B, T, P, hidden)


# --- Divided Space-Time Attention Block ---

def spatial_attention(tokens, W_q, W_k, W_v, W_o, n_heads,
                      attn_mask=None, wo_mask=None, mask_bias_scale=20.0):
    """Spatial attention: within each frame, patches attend to each other.

    tokens: (B, T, P, hidden)
    attn_mask: (P, P) or None - genome attention pattern
    wo_mask: (hidden, hidden) or None - genome depth mask on W_o
    Returns: (B, T, P, hidden)
    """
    B, T, P, D = tokens.shape
    head_dim = D // n_heads
    scale = head_dim ** -0.5

    x = tokens.reshape(B * T, P, D)

    Q = W_q(x).view(B * T, P, n_heads, head_dim).transpose(1, 2)
    K = W_k(x).view(B * T, P, n_heads, head_dim).transpose(1, 2)
    V = W_v(x).view(B * T, P, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if attn_mask is not None:
        attn_bias = (attn_mask - 0.5) * mask_bias_scale
        scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    out = out.transpose(1, 2).contiguous().view(B * T, P, D)

    if wo_mask is not None:
        out = F.linear(out, W_o.weight * wo_mask)
    else:
        out = W_o(out)

    return out.view(B, T, P, D)


def temporal_attention(tokens, W_q, W_k, W_v, W_o, n_heads,
                       attn_mask=None, wo_mask=None, mask_bias_scale=20.0):
    """Temporal attention: at each spatial position, frames attend to each other.

    tokens: (B, T, P, hidden)
    attn_mask: (T, T) or None
    wo_mask: (hidden, hidden) or None
    Returns: (B, T, P, hidden)
    """
    B, T, P, D = tokens.shape
    head_dim = D // n_heads
    scale = head_dim ** -0.5

    x = tokens.permute(0, 2, 1, 3).reshape(B * P, T, D)

    Q = W_q(x).view(B * P, T, n_heads, head_dim).transpose(1, 2)
    K = W_k(x).view(B * P, T, n_heads, head_dim).transpose(1, 2)
    V = W_v(x).view(B * P, T, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if attn_mask is not None:
        attn_bias = (attn_mask - 0.5) * mask_bias_scale
        scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    out = out.transpose(1, 2).contiguous().view(B * P, T, D)

    if wo_mask is not None:
        out = F.linear(out, W_o.weight * wo_mask)
    else:
        out = W_o(out)

    return out.reshape(B, P, T, D).permute(0, 2, 1, 3)


# --- Genome Video Transformer ---

class GrownVideoTransformer(nn.Module):
    """Video transformer with divided space-time attention, genome-controlled.

    Each layer: spatial attn -> temporal attn -> FF
    Genome controls: spatial attention pattern, temporal attention pattern,
    FF gating, cross-layer skip connections.
    """

    def __init__(self, genome, hidden=128, ff_dim=256, n_layers=4, n_heads=4,
                 n_frames=10, img_size=64, patch_size=8):
        super().__init__()
        self.genome = genome
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 64
        self.img_size = img_size
        self.temperature = 1.0

        self.depth_dims = [hidden] * (n_layers + 1) + [patch_size * patch_size]

        self.patch_embed = PatchEmbed(img_size, patch_size, hidden)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, hidden) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, hidden) * 0.02)

        # Per-layer: spatial Q/K/V/O + temporal Q/K/V/O + FF + LayerNorms
        self.s_Wq = nn.ModuleList()
        self.s_Wk = nn.ModuleList()
        self.s_Wv = nn.ModuleList()
        self.s_Wo = nn.ModuleList()
        self.t_Wq = nn.ModuleList()
        self.t_Wk = nn.ModuleList()
        self.t_Wv = nn.ModuleList()
        self.t_Wo = nn.ModuleList()
        self.ln_s = nn.ModuleList()  # before spatial attn
        self.ln_t = nn.ModuleList()  # before temporal attn
        self.ln_f = nn.ModuleList()  # before FF
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.s_Wq.append(nn.Linear(hidden, hidden))
            self.s_Wk.append(nn.Linear(hidden, hidden))
            self.s_Wv.append(nn.Linear(hidden, hidden))
            self.s_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.t_Wq.append(nn.Linear(hidden, hidden))
            self.t_Wk.append(nn.Linear(hidden, hidden))
            self.t_Wv.append(nn.Linear(hidden, hidden))
            self.t_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.ln_s.append(nn.LayerNorm(hidden))
            self.ln_t.append(nn.LayerNorm(hidden))
            self.ln_f.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        # Skip projections for non-adjacent layers
        self.skip_projs = nn.ModuleDict()
        self.skip_pairs = []
        for t in range(1, n_layers + 1):
            for s in range(t):
                if t - s == 1:
                    continue
                key = f'{t}_{s}'
                self.skip_projs[key] = nn.Linear(hidden, hidden, bias=False)
                self.skip_pairs.append((s, t))

        self.ln_final = nn.LayerNorm(hidden)
        self.pred_head = nn.Linear(hidden, patch_size * patch_size)

    def forward(self, x):
        """x: (B, 10, 64, 64) -> (B, 64, 64) predicted next frame."""
        B = x.shape[0]

        tokens = self.patch_embed(x)  # (B, T, P, hidden)
        tokens = tokens + self.temporal_pos + self.spatial_pos

        # Genome masks (computed once, shared across layers)
        s_mask = self.genome.spatial_mask(self.n_patches, temperature=self.temperature)
        t_mask = self.genome.temporal_mask(self.n_frames, temperature=self.temperature)

        acts = [tokens]

        for i in range(self.n_layers):
            tgt_band = i + 1
            h = acts[tgt_band - 1]

            # 1. Spatial attention with genome masks
            ln_h = self.ln_s[i](h)
            wo_mask = self.genome.depth_mask(tgt_band - 1, tgt_band, self.hidden, self.hidden,
                                              temperature=self.temperature)
            s_out = spatial_attention(ln_h, self.s_Wq[i], self.s_Wk[i], self.s_Wv[i],
                                      self.s_Wo[i], self.n_heads,
                                      attn_mask=s_mask, wo_mask=wo_mask)
            h = h + s_out

            # 2. Temporal attention with genome masks
            ln_h = self.ln_t[i](h)
            t_out = temporal_attention(ln_h, self.t_Wq[i], self.t_Wk[i], self.t_Wv[i],
                                       self.t_Wo[i], self.n_heads,
                                       attn_mask=t_mask, wo_mask=wo_mask)
            h = h + t_out

            # 3. Feed-forward with depth genome mask
            ln_h = self.ln_f[i](h)
            ff_mask = self.genome.depth_mask(tgt_band - 1, tgt_band, self.hidden, self.ff_dim,
                                              temperature=self.temperature)
            ff_out = F.linear(ln_h, self.ff1s[i].weight * ff_mask, self.ff1s[i].bias)
            ff_out = F.relu(ff_out)
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            # 4. Skip connections
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                skip_mask = self.genome.depth_mask(skip_src, tgt_band, self.hidden, self.hidden,
                                                    temperature=self.temperature)
                proj = self.skip_projs[key]
                h = h + F.linear(acts[skip_src], proj.weight * skip_mask)

            acts.append(h)

        # Predict next frame from last input frame's tokens
        final = self.ln_final(acts[-1])
        last_frame = final[:, -1, :, :]  # (B, P, hidden) - last frame tokens

        # Depth mask on prediction head
        head_mask = self.genome.depth_mask(self.n_layers, self.n_layers + 1,
                                            self.hidden, self.patch_size ** 2,
                                            temperature=self.temperature)
        patches = F.linear(last_frame, self.pred_head.weight * head_mask, self.pred_head.bias)
        patches = torch.sigmoid(patches)  # (B, P, patch_size^2)

        # Reconstruct frame
        g = int(math.sqrt(self.n_patches))
        pred = patches.view(B, g, g, self.patch_size, self.patch_size)
        pred = pred.permute(0, 1, 3, 2, 4).contiguous()
        pred = pred.view(B, self.img_size, self.img_size)
        return pred

    def count_effective(self, threshold=0.5):
        """Count active connections across all genome masks."""
        total = active = 0
        mask_sum = 0.0
        with torch.no_grad():
            # Spatial attention mask
            s = self.genome.spatial_mask(self.n_patches)
            total += s.numel(); active += (s > threshold).sum().item(); mask_sum += s.sum().item()
            # Temporal attention mask
            t = self.genome.temporal_mask(self.n_frames)
            total += t.numel(); active += (t > threshold).sum().item(); mask_sum += t.sum().item()
            # Depth masks (FF + skips + head)
            for tgt in range(1, self.n_layers + 2):
                for src in range(tgt):
                    if tgt <= self.n_layers and tgt - src == 1:
                        m = self.genome.depth_mask(src, tgt, self.hidden, self.hidden)
                        total += m.numel(); active += (m > threshold).sum().item(); mask_sum += m.sum().item()
                        m = self.genome.depth_mask(src, tgt, self.hidden, self.ff_dim)
                        total += m.numel(); active += (m > threshold).sum().item(); mask_sum += m.sum().item()
                    elif tgt == self.n_layers + 1 and src == self.n_layers:
                        m = self.genome.depth_mask(src, tgt, self.hidden, self.patch_size**2)
                        total += m.numel(); active += (m > threshold).sum().item(); mask_sum += m.sum().item()
                    elif tgt <= self.n_layers and tgt - src > 1:
                        m = self.genome.depth_mask(src, tgt, self.hidden, self.hidden)
                        total += m.numel(); active += (m > threshold).sum().item(); mask_sum += m.sum().item()
        return active, total, mask_sum / total if total > 0 else 0.0


# --- Baselines ---

class DenseVideoTransformer(nn.Module):
    """Divided space-time attention, full attention, no masks. Ceiling baseline."""

    def __init__(self, hidden=128, ff_dim=256, n_layers=4, n_heads=4,
                 n_frames=10, img_size=64, patch_size=8):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size, patch_size, hidden)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, hidden) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, hidden) * 0.02)

        self.s_Wq = nn.ModuleList()
        self.s_Wk = nn.ModuleList()
        self.s_Wv = nn.ModuleList()
        self.s_Wo = nn.ModuleList()
        self.t_Wq = nn.ModuleList()
        self.t_Wk = nn.ModuleList()
        self.t_Wv = nn.ModuleList()
        self.t_Wo = nn.ModuleList()
        self.ln_s = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        self.ln_f = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.s_Wq.append(nn.Linear(hidden, hidden))
            self.s_Wk.append(nn.Linear(hidden, hidden))
            self.s_Wv.append(nn.Linear(hidden, hidden))
            self.s_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.t_Wq.append(nn.Linear(hidden, hidden))
            self.t_Wk.append(nn.Linear(hidden, hidden))
            self.t_Wv.append(nn.Linear(hidden, hidden))
            self.t_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.ln_s.append(nn.LayerNorm(hidden))
            self.ln_t.append(nn.LayerNorm(hidden))
            self.ln_f.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        self.ln_final = nn.LayerNorm(hidden)
        self.pred_head = nn.Linear(hidden, patch_size * patch_size)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = tokens + self.temporal_pos + self.spatial_pos

        h = tokens
        for i in range(self.n_layers):
            ln_h = self.ln_s[i](h)
            s_out = spatial_attention(ln_h, self.s_Wq[i], self.s_Wk[i], self.s_Wv[i],
                                      self.s_Wo[i], self.n_heads)
            h = h + s_out
            ln_h = self.ln_t[i](h)
            t_out = temporal_attention(ln_h, self.t_Wq[i], self.t_Wk[i], self.t_Wv[i],
                                       self.t_Wo[i], self.n_heads)
            h = h + t_out
            ln_h = self.ln_f[i](h)
            ff_out = F.relu(self.ff1s[i](ln_h))
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

        final = self.ln_final(h)
        last_frame = final[:, -1, :, :]
        patches = torch.sigmoid(self.pred_head(last_frame))

        g = int(math.sqrt(self.n_patches))
        pred = patches.view(B, g, g, self.patch_size, self.patch_size)
        pred = pred.permute(0, 1, 3, 2, 4).contiguous()
        pred = pred.view(B, self.img_size, self.img_size)
        return pred


class RandomSparseVideoTransformer(nn.Module):
    """Divided space-time attention with fixed random masks. Control baseline."""

    def __init__(self, spatial_density, temporal_density, depth_density,
                 hidden=128, ff_dim=256, n_layers=4, n_heads=4,
                 n_frames=10, img_size=64, patch_size=8):
        super().__init__()
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size, patch_size, hidden)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, hidden) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, hidden) * 0.02)

        # Fixed random masks
        s_mask = (torch.rand(self.n_patches, self.n_patches) < spatial_density).float()
        self.register_buffer('s_mask', s_mask)
        t_mask = (torch.rand(n_frames, n_frames) < temporal_density).float()
        self.register_buffer('t_mask', t_mask)

        self.s_Wq = nn.ModuleList()
        self.s_Wk = nn.ModuleList()
        self.s_Wv = nn.ModuleList()
        self.s_Wo = nn.ModuleList()
        self.t_Wq = nn.ModuleList()
        self.t_Wk = nn.ModuleList()
        self.t_Wv = nn.ModuleList()
        self.t_Wo = nn.ModuleList()
        self.ln_s = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        self.ln_f = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for i in range(n_layers):
            self.s_Wq.append(nn.Linear(hidden, hidden))
            self.s_Wk.append(nn.Linear(hidden, hidden))
            self.s_Wv.append(nn.Linear(hidden, hidden))
            self.s_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.t_Wq.append(nn.Linear(hidden, hidden))
            self.t_Wk.append(nn.Linear(hidden, hidden))
            self.t_Wv.append(nn.Linear(hidden, hidden))
            self.t_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.ln_s.append(nn.LayerNorm(hidden))
            self.ln_t.append(nn.LayerNorm(hidden))
            self.ln_f.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))
            # Depth masks
            wo_mask = (torch.rand(hidden, hidden) < depth_density).float()
            self.register_buffer(f'wo_mask_{i}', wo_mask)
            ff_mask = (torch.rand(ff_dim, hidden) < depth_density).float()
            self.register_buffer(f'ff_mask_{i}', ff_mask)

        # Skip projections
        self.skip_projs = nn.ModuleDict()
        self.skip_pairs = []
        for t in range(1, n_layers + 1):
            for s in range(t):
                if t - s == 1:
                    continue
                key = f'{t}_{s}'
                self.skip_projs[key] = nn.Linear(hidden, hidden, bias=False)
                sm = (torch.rand(hidden, hidden) < depth_density).float()
                self.register_buffer(f'sm_{key}', sm)
                self.skip_pairs.append((s, t))

        self.ln_final = nn.LayerNorm(hidden)
        self.pred_head = nn.Linear(hidden, patch_size * patch_size)
        hm = (torch.rand(patch_size * patch_size, hidden) < depth_density).float()
        self.register_buffer('head_mask', hm)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = tokens + self.temporal_pos + self.spatial_pos

        acts = [tokens]
        for i in range(self.n_layers):
            tgt_band = i + 1
            h = acts[tgt_band - 1]

            # Spatial attention with random masks
            ln_h = self.ln_s[i](h)
            wo_mask = getattr(self, f'wo_mask_{i}')
            s_out = spatial_attention(ln_h, self.s_Wq[i], self.s_Wk[i], self.s_Wv[i],
                                      self.s_Wo[i], self.n_heads,
                                      attn_mask=self.s_mask, wo_mask=wo_mask)
            h = h + s_out

            # Temporal attention with random masks
            ln_h = self.ln_t[i](h)
            t_out = temporal_attention(ln_h, self.t_Wq[i], self.t_Wk[i], self.t_Wv[i],
                                       self.t_Wo[i], self.n_heads,
                                       attn_mask=self.t_mask, wo_mask=wo_mask)
            h = h + t_out

            # FF with random mask
            ln_h = self.ln_f[i](h)
            ff_mask = getattr(self, f'ff_mask_{i}')
            ff_out = F.linear(ln_h, self.ff1s[i].weight * ff_mask, self.ff1s[i].bias)
            ff_out = F.relu(ff_out)
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            # Skips
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                sm = getattr(self, f'sm_{key}')
                h = h + F.linear(acts[skip_src], self.skip_projs[key].weight * sm)

            acts.append(h)

        final = self.ln_final(acts[-1])
        last_frame = final[:, -1, :, :]
        patches = F.linear(last_frame, self.pred_head.weight * self.head_mask, self.pred_head.bias)
        patches = torch.sigmoid(patches)

        g = int(math.sqrt(self.n_patches))
        pred = patches.view(B, g, g, self.patch_size, self.patch_size)
        pred = pred.permute(0, 1, 3, 2, 4).contiguous()
        pred = pred.view(B, self.img_size, self.img_size)
        return pred


class DenseSkipVideoTransformer(nn.Module):
    """Divided space-time attention, full cross-layer connectivity, no masks."""

    def __init__(self, hidden=128, ff_dim=256, n_layers=4, n_heads=4,
                 n_frames=10, img_size=64, patch_size=8):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size, patch_size, hidden)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, hidden) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, hidden) * 0.02)

        self.s_Wq = nn.ModuleList()
        self.s_Wk = nn.ModuleList()
        self.s_Wv = nn.ModuleList()
        self.s_Wo = nn.ModuleList()
        self.t_Wq = nn.ModuleList()
        self.t_Wk = nn.ModuleList()
        self.t_Wv = nn.ModuleList()
        self.t_Wo = nn.ModuleList()
        self.ln_s = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        self.ln_f = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.s_Wq.append(nn.Linear(hidden, hidden))
            self.s_Wk.append(nn.Linear(hidden, hidden))
            self.s_Wv.append(nn.Linear(hidden, hidden))
            self.s_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.t_Wq.append(nn.Linear(hidden, hidden))
            self.t_Wk.append(nn.Linear(hidden, hidden))
            self.t_Wv.append(nn.Linear(hidden, hidden))
            self.t_Wo.append(nn.Linear(hidden, hidden, bias=False))
            self.ln_s.append(nn.LayerNorm(hidden))
            self.ln_t.append(nn.LayerNorm(hidden))
            self.ln_f.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        self.skip_projs = nn.ModuleDict()
        self.skip_pairs = []
        for t in range(1, n_layers + 1):
            for s in range(t):
                if t - s == 1:
                    continue
                key = f'{t}_{s}'
                self.skip_projs[key] = nn.Linear(hidden, hidden, bias=False)
                self.skip_pairs.append((s, t))

        self.ln_final = nn.LayerNorm(hidden)
        self.pred_head = nn.Linear(hidden, patch_size * patch_size)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = tokens + self.temporal_pos + self.spatial_pos

        acts = [tokens]
        for i in range(self.n_layers):
            tgt_band = i + 1
            h = acts[tgt_band - 1]

            ln_h = self.ln_s[i](h)
            s_out = spatial_attention(ln_h, self.s_Wq[i], self.s_Wk[i], self.s_Wv[i],
                                      self.s_Wo[i], self.n_heads)
            h = h + s_out
            ln_h = self.ln_t[i](h)
            t_out = temporal_attention(ln_h, self.t_Wq[i], self.t_Wk[i], self.t_Wv[i],
                                       self.t_Wo[i], self.n_heads)
            h = h + t_out
            ln_h = self.ln_f[i](h)
            ff_out = F.relu(self.ff1s[i](ln_h))
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                h = h + self.skip_projs[key](acts[skip_src])

            acts.append(h)

        final = self.ln_final(acts[-1])
        last_frame = final[:, -1, :, :]
        patches = torch.sigmoid(self.pred_head(last_frame))

        g = int(math.sqrt(self.n_patches))
        pred = patches.view(B, g, g, self.patch_size, self.patch_size)
        pred = pred.permute(0, 1, 3, 2, 4).contiguous()
        pred = pred.view(B, self.img_size, self.img_size)
        return pred


# --- Training ---

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def flush_cache():
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def evaluate(model, loader):
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_mse += F.mse_loss(pred, y, reduction='sum').item()
            n += y.shape[0]
    return total_mse / n


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


def train_dense(model, tr, te, n_epochs=30, lr=3e-4, accum_steps=2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best_mse = float('inf')

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        step = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y) / accum_steps
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

        mse = evaluate(model, te)
        best_mse = min(best_mse, mse)
        sched.step()
        flush_cache()
        print(f"    Ep {ep:2d}: MSE={mse:.6f} best={best_mse:.6f}")

    return best_mse


def train_genome(model, tr, te, n_epochs=50, lr=3e-4, genome_lr=0.01,
                 attn_genome_lr=0.01, sparsity_weight=0.005, accum_steps=2):
    """Train genome video transformer. Three LR groups."""
    depth_params = list(model.genome.depth.parameters())
    attn_params = list(model.genome.temporal.parameters()) + list(model.genome.spatial.parameters())
    genome_ids = set(id(p) for p in depth_params + attn_params)
    weight_params = [p for p in model.parameters() if id(p) not in genome_ids]

    opt_weights = torch.optim.AdamW(weight_params, lr=lr, weight_decay=0.01)
    opt_depth = torch.optim.Adam(depth_params, lr=genome_lr)
    opt_attn = torch.optim.Adam(attn_params, lr=attn_genome_lr)
    sched_weights = torch.optim.lr_scheduler.CosineAnnealingLR(opt_weights, n_epochs)
    sched_depth = torch.optim.lr_scheduler.CosineAnnealingLR(opt_depth, n_epochs)
    sched_attn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_attn, n_epochs)

    temp_start, temp_end = 1.0, 10.0
    print(f"    Temperature: {temp_start} -> {temp_end}")
    print(f"    LR: weights={lr}, depth={genome_lr}, attn={attn_genome_lr}")
    best_mse = float('inf')

    for ep in range(n_epochs):
        model.temperature = temp_start + (temp_end - temp_start) * ep / max(n_epochs - 1, 1)
        model.train()
        opt_weights.zero_grad()
        opt_depth.zero_grad()
        opt_attn.zero_grad()
        step = 0

        for x, y in tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = F.mse_loss(pred, y)
            sp_loss = model.genome.sparsity_loss(model.depth_dims)
            loss = (mse_loss + sparsity_weight * sp_loss) / accum_steps
            loss.backward()
            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt_weights.step()
                opt_depth.step()
                opt_attn.step()
                opt_weights.zero_grad()
                opt_depth.zero_grad()
                opt_attn.zero_grad()

        if step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_weights.step()
            opt_depth.step()
            opt_attn.step()
            opt_weights.zero_grad()
            opt_depth.zero_grad()
            opt_attn.zero_grad()

        mse = evaluate(model, te)
        best_mse = min(best_mse, mse)
        sched_weights.step()
        sched_depth.step()
        sched_attn.step()
        flush_cache()

        active, total, sd = model.count_effective()
        density = active / total if total > 0 else 0

        # Report temporal/spatial mask state
        with torch.no_grad():
            tm = model.genome.temporal_mask(model.n_frames)
            sm = model.genome.spatial_mask(model.n_patches)
        print(f"    Ep {ep:2d}: MSE={mse:.6f} best={best_mse:.6f} "
              f"hard={density:.1%} soft={sd:.1%} "
              f"temp=[{tm.min():.2f},{tm.max():.2f}] "
              f"spat=[{sm.min():.2f},{sm.max():.2f}] "
              f"T={model.temperature:.1f}")

    return best_mse


def train_sparse(model, tr, te, n_epochs=30, lr=3e-4, accum_steps=2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best_mse = float('inf')

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        step = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y) / accum_steps
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

        mse = evaluate(model, te)
        best_mse = min(best_mse, mse)
        sched.step()
        flush_cache()
        print(f"    Ep {ep:2d}: MSE={mse:.6f} best={best_mse:.6f}")

    return best_mse


# --- Visualization ---

def save_visualizations(model, test_loader, genome, save_dir='results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # 1. Temporal mask heatmap (10x10)
        t_mask = genome.temporal_mask(10).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(t_mask, cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Source Frame')
        ax.set_ylabel('Target Frame')
        ax.set_title('Temporal Attention (Genome-Learned)')
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rung4_temporal_mask.png', dpi=150)
        plt.close()
        print(f"  Saved {save_dir}/rung4_temporal_mask.png")

        # 2. Spatial mask: distance vs attention strength
        s_mask = genome.spatial_mask(64).cpu().numpy()
        distances = []
        strengths = []
        for i in range(64):
            for j in range(64):
                r1, c1 = i // 8, i % 8
                r2, c2 = j // 8, j % 8
                dist = abs(r1 - r2) + abs(c1 - c2)
                distances.append(dist)
                strengths.append(s_mask[i, j])

        from collections import defaultdict
        bins = defaultdict(list)
        for d, s in zip(distances, strengths):
            bins[d].append(s)
        bin_x = sorted(bins.keys())
        bin_y = [np.mean(bins[d]) for d in bin_x]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(distances, strengths, alpha=0.05, s=1, color='blue')
        ax.plot(bin_x, bin_y, 'r-o', linewidth=2, label='Mean')
        ax.set_xlabel('Manhattan Distance Between Patches')
        ax.set_ylabel('Attention Strength')
        ax.set_title('Spatial Attention vs Distance (Genome-Learned)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rung4_spatial_decay.png', dpi=150)
        plt.close()
        print(f"  Saved {save_dir}/rung4_spatial_decay.png")

        # 3. Prediction samples
        model.eval()
        x, y = next(iter(test_loader))
        x, y = x[:4].to(device), y[:4]
        pred = model(x).cpu()

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(4):
            axes[i, 0].imshow(x[i, 8].cpu(), cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Frame 9' if i == 0 else '')
            axes[i, 1].imshow(x[i, 9].cpu(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Frame 10' if i == 0 else '')
            axes[i, 2].imshow(y[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Target (11)' if i == 0 else '')
            axes[i, 3].imshow(pred[i].detach(), cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Predicted (11)' if i == 0 else '')
            for j in range(4):
                axes[i, j].axis('off')

        plt.suptitle('Genome Video Transformer: Next Frame Prediction')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rung4_predictions.png', dpi=150)
        plt.close()
        print(f"  Saved {save_dir}/rung4_predictions.png")


# --- Main ---

def run():
    print("=" * 60)
    print("  RUNG 4: FACTORED SPATIOTEMPORAL GENOME ON MOVING MNIST")
    print("  Divided Space-Time Attention with genome-learned masks.")
    print("  Temporal (74) + Spatial (74) + Depth (226) = 374 params.")
    print("=" * 60)
    print(f"  Device: {device}")

    results, saved_genome_path = load_checkpoint()
    if results:
        print(f"\n  CHECKPOINT: found {list(results.keys())} complete.")
        for name, r in results.items():
            print(f"  PRESERVED: {name} (MSE={r['mse']:.6f})")
    else:
        results = {}

    hidden = 128
    ff_dim = 256
    n_layers = 4
    n_heads = 4
    batch_size = 32

    tr, te = make_dataloaders(batch_size=batch_size)
    genome = None

    # 1. Dense
    if 'dense' not in results:
        print(f"\n  [1/4] DENSE VIDEO TRANSFORMER (ceiling)")
        m = DenseVideoTransformer(hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                                   n_heads=n_heads).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        try:
            mse = train_dense(m, tr, te, n_epochs=30)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                print(f"    OOM! Retrying batch_size=16...")
                del m; flush_cache()
                batch_size = 16
                tr, te = make_dataloaders(batch_size=batch_size)
                m = DenseVideoTransformer(hidden=hidden, ff_dim=ff_dim, n_layers=n_layers,
                                           n_heads=n_heads).to(device)
                p = count_params(m)
                t0 = time.time()
                mse = train_dense(m, tr, te, n_epochs=30)
            else:
                raise
        results['dense'] = {'params': p, 'mse': mse, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Best MSE: {mse:.6f}")
    else:
        print(f"\n  [1/4] DENSE: skipped (MSE={results['dense']['mse']:.6f})")

    # 2. Genome
    if 'genome' not in results:
        print(f"\n  [2/4] GENOME VIDEO TRANSFORMER (factored spatiotemporal)")
        genome = SpatiotemporalGenome()
        m = GrownVideoTransformer(genome, hidden=hidden, ff_dim=ff_dim,
                                   n_layers=n_layers, n_heads=n_heads).to(device)
        gp = genome.count_params()
        tp = count_params(m)
        print(f"    Genome: {gp:,} params, Total: {tp:,}")
        t0 = time.time()
        try:
            mse = train_genome(m, tr, te, n_epochs=50, sparsity_weight=0.005)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                print(f"    OOM! Retrying batch_size=16...")
                del m; flush_cache()
                batch_size = 16
                tr, te = make_dataloaders(batch_size=batch_size)
                genome = SpatiotemporalGenome()
                m = GrownVideoTransformer(genome, hidden=hidden, ff_dim=ff_dim,
                                           n_layers=n_layers, n_heads=n_heads).to(device)
                gp = genome.count_params()
                tp = count_params(m)
                t0 = time.time()
                mse = train_genome(m, tr, te, n_epochs=50, sparsity_weight=0.005)
            else:
                raise
        active, total, sd = m.count_effective()
        density = active / total if total > 0 else 0
        results['genome'] = {
            'params': tp, 'genome_params': gp, 'mse': mse,
            'time': time.time() - t0, 'hard_density': density,
            'soft_density': sd, 'active': active, 'total': total
        }
        print(f"    Final: hard={density:.1%} soft={sd:.1%}")

        try:
            save_visualizations(m, te, genome)
        except Exception as e:
            print(f"    Visualization failed: {e}")

        del m; flush_cache()
        os.makedirs('results', exist_ok=True)
        genome_path = "results/genome_video_checkpoint.pt"
        torch.save(genome.state_dict(), genome_path)
        save_checkpoint(results, genome_path)
        print(f"    >> Best MSE: {mse:.6f}")
    else:
        print(f"\n  [2/4] GENOME: skipped (MSE={results['genome']['mse']:.6f})")

    # Get densities for random sparse
    genome_r = results.get('genome', {})
    spatial_d = genome_r.get('soft_density', 0.3)
    temporal_d = spatial_d
    depth_d = spatial_d

    # 3. Random Sparse
    if 'random_sparse' not in results:
        print(f"\n  [3/4] RANDOM SPARSE (density~{spatial_d:.1%})")
        m = RandomSparseVideoTransformer(
            spatial_density=spatial_d, temporal_density=temporal_d, depth_density=depth_d,
            hidden=hidden, ff_dim=ff_dim, n_layers=n_layers, n_heads=n_heads
        ).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        mse = train_sparse(m, tr, te, n_epochs=30)
        results['random_sparse'] = {'params': p, 'mse': mse, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Best MSE: {mse:.6f}")
    else:
        print(f"\n  [3/4] RANDOM SPARSE: skipped (MSE={results['random_sparse']['mse']:.6f})")

    # 4. Dense Skip
    if 'dense_skip' not in results:
        print(f"\n  [4/4] DENSE SKIP (full cross-layer)")
        m = DenseSkipVideoTransformer(hidden=hidden, ff_dim=ff_dim,
                                       n_layers=n_layers, n_heads=n_heads).to(device)
        p = count_params(m)
        print(f"    Params: {p:,}")
        t0 = time.time()
        mse = train_sparse(m, tr, te, n_epochs=30)
        results['dense_skip'] = {'params': p, 'mse': mse, 'time': time.time() - t0}
        del m; flush_cache()
        save_checkpoint(results)
        print(f"    >> Best MSE: {mse:.6f}")
    else:
        print(f"\n  [4/4] DENSE SKIP: skipped (MSE={results['dense_skip']['mse']:.6f})")

    # --- Results ---
    print("\n" + "=" * 60)
    print("  RESULTS: FACTORED SPATIOTEMPORAL GENOME ON MOVING MNIST")
    print("=" * 60)
    print(f"\n  {'Model':<24} {'Params':>10} {'MSE':>12}")
    print(f"  {'-' * 50}")
    for name, r in results.items():
        extra = f" genome:{r.get('genome_params', '')}" if 'genome_params' in r else ""
        print(f"  {name:<24} {r['params']:>10,} {r['mse']:>12.6f}{extra}")

    g = results.get('genome', {})
    rs = results.get('random_sparse', {})
    if g and rs:
        gap = ((rs['mse'] - g['mse']) / rs['mse']) * 100
        print(f"\n  GENOME vs RANDOM: {gap:+.1f}% MSE ({'GENOME WINS' if g['mse'] < rs['mse'] else 'RANDOM WINS'})")

    dt = results.get('dense', {})
    if dt and g:
        gap = ((g['mse'] - dt['mse']) / dt['mse']) * 100
        print(f"  GENOME vs DENSE: {gap:+.1f}% gap to ceiling")

    if g:
        print(f"\n  GENOME: {g['genome_params']:,} params -> {g['total']:,} masked connections")
        print(f"  Compression: {g['total'] // max(g['genome_params'], 1)}:1")

    total_time = sum(r.get('time', 0) for r in results.values())
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    # Save final results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if genome is not None:
        genome_path = f"results/genome_video_{timestamp}.pt"
        torch.save(genome.state_dict(), genome_path)
        print(f"\n  Genome saved to {genome_path}")
    elif saved_genome_path and os.path.exists(saved_genome_path):
        genome_path = saved_genome_path
    else:
        genome_path = None

    result_data = {
        "experiment": "rung4_video",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "architecture": "divided_spacetime_attention",
            "genome_type": "factored_spatiotemporal",
            "temporal": {"n_types": 4, "type_dim": 4, "n_bands": 5},
            "spatial": {"n_types": 4, "type_dim": 4, "n_bands": 5},
            "depth": {"n_types": 8, "type_dim": 8, "n_bands": 6},
            "hidden": hidden, "ff_dim": ff_dim, "n_layers": n_layers,
            "n_heads": n_heads, "n_frames": 10, "img_size": 64,
            "patch_size": 8, "batch_size": batch_size,
            "dense_epochs": 30, "genome_epochs": 50,
            "sparsity_weight": 0.005, "device": str(device)
        },
        "results": results,
        "genome_state": genome_path,
        "total_time": total_time
    }
    json_path = f"results/rung4_video_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to {json_path}")

    clear_checkpoint()
    print("  Checkpoint cleared. Run complete.")


if __name__ == "__main__":
    run()
