"""
Baseline models for comparison against genome-grown networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RandomSparseNetwork(nn.Module):
    """Same band structure as GrownNetwork, but with random fixed masks."""

    def __init__(self, dims, density):
        super().__init__()
        self.dims = dims
        self.n_bands = len(dims)
        self.weights = nn.ParameterDict()
        for t in range(1, self.n_bands):
            for s in range(t):
                key = f'{t}_{s}'
                fan = dims[s] + dims[t]
                self.weights[key] = nn.Parameter(
                    torch.randn(dims[t], dims[s]) * (2.0 / math.sqrt(fan))
                )
                mask = (torch.rand(dims[t], dims[s]) < density).float()
                self.register_buffer(f'm_{key}', mask)
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for d in dims[1:]
        ])

    def forward(self, x):
        acts = [x]
        for t in range(1, self.n_bands):
            h = self.biases[t - 1]
            for s in range(t):
                key = f'{t}_{s}'
                h = h + acts[s] @ (getattr(self, f'm_{key}') * self.weights[key]).T
            if t < self.n_bands - 1:
                h = F.relu(h)
            acts.append(h)
        return acts[-1]


class DenseSkipNetwork(nn.Module):
    """Same band structure, ALL connections active (no mask)."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.n_bands = len(dims)
        self.weights = nn.ParameterDict()
        for t in range(1, self.n_bands):
            for s in range(t):
                fan = dims[s] + dims[t]
                self.weights[f'{t}_{s}'] = nn.Parameter(
                    torch.randn(dims[t], dims[s]) * (2.0 / math.sqrt(fan))
                )
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for d in dims[1:]
        ])

    def forward(self, x):
        acts = [x]
        for t in range(1, self.n_bands):
            h = self.biases[t - 1]
            for s in range(t):
                h = h + acts[s] @ self.weights[f'{t}_{s}'].T
            if t < self.n_bands - 1:
                h = F.relu(h)
            acts.append(h)
        return acts[-1]


class NormalMLP(nn.Module):
    """Standard dense MLP baseline."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- CNN baselines ---


class BasicBlock(nn.Module):
    """Standard ResNet basic block (two 3x3 convs + shortcut)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class DenseResNet(nn.Module):
    """Standard ResNet-20 style for CIFAR-10. The ceiling baseline.

    3 groups of [16, 32, 64] channels, 2 BasicBlocks each.
    Target: ~91% on CIFAR-10 with proper training.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.group1 = nn.Sequential(
            BasicBlock(16, 16), BasicBlock(16, 16)
        )
        self.group2 = nn.Sequential(
            BasicBlock(16, 32, stride=2), BasicBlock(32, 32)
        )
        self.group3 = nn.Sequential(
            BasicBlock(32, 64, stride=2), BasicBlock(64, 64)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)


class RandomSparseResNet(nn.Module):
    """Same architecture as GrownConvNetwork but with fixed random binary masks.

    Matched density. The control for genome CNN experiments.
    """

    def __init__(self, density, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.band_channels = [3, 16, 16, 32, 32, 64, 64, num_classes]
        self.band_spatial = [32, 32, 32, 16, 16, 8, 8]

        # Primary convs (adjacent bands)
        self.primary_convs = nn.ModuleList()
        self.primary_bns = nn.ModuleList()
        self.adjacent_pairs = [
            (0, 1, 1), (1, 2, 1), (2, 3, 2),
            (3, 4, 1), (4, 5, 2), (5, 6, 1)
        ]
        for src, tgt, stride in self.adjacent_pairs:
            in_ch, out_ch = self.band_channels[src], self.band_channels[tgt]
            self.primary_convs.append(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            )
            self.primary_bns.append(nn.BatchNorm2d(out_ch))
            mask = (torch.rand(out_ch, in_ch) < density).float()
            self.register_buffer(f'pm_{tgt}_{src}', mask)

        # Skip convs (non-adjacent)
        self.skip_convs = nn.ModuleDict()
        self.skip_bns = nn.ModuleDict()
        self.skip_pairs = []
        for tgt in range(1, 7):
            for src in range(tgt):
                if tgt - src == 1:
                    continue
                in_ch, out_ch = self.band_channels[src], self.band_channels[tgt]
                key = f'{tgt}_{src}'
                self.skip_convs[key] = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.skip_bns[key] = nn.BatchNorm2d(out_ch)
                mask = (torch.rand(out_ch, in_ch) < density).float()
                self.register_buffer(f'sm_{key}', mask)
                self.skip_pairs.append((src, tgt))

        # FC
        self.fc = nn.Linear(64, num_classes, bias=True)
        fc_mask = (torch.rand(num_classes, 64) < density).float()
        self.register_buffer('fc_mask', fc_mask)

    def forward(self, x):
        acts = [x]
        adj_idx = 0
        for tgt in range(1, 7):
            src = tgt - 1
            conv = self.primary_convs[adj_idx]
            bn = self.primary_bns[adj_idx]
            mask = getattr(self, f'pm_{tgt}_{src}')
            stride = self.adjacent_pairs[adj_idx][2]
            w = conv.weight * mask[:, :, None, None]
            h = F.conv2d(acts[src], w, stride=stride, padding=1)
            adj_idx += 1

            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt:
                    continue
                key = f'{tgt}_{skip_src}'
                skip_mask = getattr(self, f'sm_{key}')
                skip_w = self.skip_convs[key].weight * skip_mask[:, :, None, None]
                src_act = acts[skip_src]
                tgt_spatial = self.band_spatial[tgt]
                if src_act.shape[2] != tgt_spatial:
                    src_act = F.adaptive_avg_pool2d(src_act, tgt_spatial)
                h = h + self.skip_bns[key](F.conv2d(src_act, skip_w))

            h = F.relu(bn(h))
            acts.append(h)

        out = F.adaptive_avg_pool2d(acts[6], 1).view(x.size(0), -1)
        return F.linear(out, self.fc.weight * self.fc_mask, self.fc.bias)


class DenseSkipResNet(nn.Module):
    """Same architecture as GrownConvNetwork but all masks = 1.0.

    Shows what the full skip architecture achieves without sparsity.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.band_channels = [3, 16, 16, 32, 32, 64, 64, num_classes]
        self.band_spatial = [32, 32, 32, 16, 16, 8, 8]

        # Primary convs
        self.primary_convs = nn.ModuleList()
        self.primary_bns = nn.ModuleList()
        self.adjacent_pairs = [
            (0, 1, 1), (1, 2, 1), (2, 3, 2),
            (3, 4, 1), (4, 5, 2), (5, 6, 1)
        ]
        for src, tgt, stride in self.adjacent_pairs:
            in_ch, out_ch = self.band_channels[src], self.band_channels[tgt]
            self.primary_convs.append(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            )
            self.primary_bns.append(nn.BatchNorm2d(out_ch))

        # Skip convs
        self.skip_convs = nn.ModuleDict()
        self.skip_bns = nn.ModuleDict()
        self.skip_pairs = []
        for tgt in range(1, 7):
            for src in range(tgt):
                if tgt - src == 1:
                    continue
                in_ch, out_ch = self.band_channels[src], self.band_channels[tgt]
                key = f'{tgt}_{src}'
                self.skip_convs[key] = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.skip_bns[key] = nn.BatchNorm2d(out_ch)
                self.skip_pairs.append((src, tgt))

        self.fc = nn.Linear(64, num_classes, bias=True)

    def forward(self, x):
        acts = [x]
        adj_idx = 0
        for tgt in range(1, 7):
            src = tgt - 1
            conv = self.primary_convs[adj_idx]
            bn = self.primary_bns[adj_idx]
            stride = self.adjacent_pairs[adj_idx][2]
            h = F.conv2d(acts[src], conv.weight, stride=stride, padding=1)
            adj_idx += 1

            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt:
                    continue
                key = f'{tgt}_{skip_src}'
                src_act = acts[skip_src]
                tgt_spatial = self.band_spatial[tgt]
                if src_act.shape[2] != tgt_spatial:
                    src_act = F.adaptive_avg_pool2d(src_act, tgt_spatial)
                h = h + self.skip_bns[key](F.conv2d(src_act, self.skip_convs[key].weight))

            h = F.relu(bn(h))
            acts.append(h)

        out = F.adaptive_avg_pool2d(acts[6], 1).view(x.size(0), -1)
        return self.fc(out)


# --- Transformer baselines ---


class DenseTransformer(nn.Module):
    """Standard 6-layer transformer encoder. No skip connections beyond normal residual.

    The ceiling baseline for IMDB sentiment classification.
    """

    def __init__(self, vocab_size=30522, hidden=256, ff_dim=512,
                 n_layers=6, n_heads=4, max_len=512, num_classes=2):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        self.attentions = nn.ModuleList()
        self.ln1s = nn.ModuleList()
        self.ln2s = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.attentions.append(nn.MultiheadAttention(hidden, n_heads, batch_first=True))
            self.ln1s.append(nn.LayerNorm(hidden))
            self.ln2s.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        self.ln_final = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        for i in range(self.n_layers):
            # Self-attention
            ln_x = self.ln1s[i](x)
            attn_out, _ = self.attentions[i](ln_x, ln_x, ln_x,
                                              key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
            x = x + attn_out

            # Feed-forward
            ln_x = self.ln2s[i](x)
            ff_out = F.relu(self.ff1s[i](ln_x))
            ff_out = self.ff2s[i](ff_out)
            x = x + ff_out

        x = self.ln_final(x)

        # Mean pool
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


class RandomSparseTransformer(nn.Module):
    """Same architecture as GrownTransformer but with fixed random binary masks.

    Manual multi-head attention with random mask on W_o, matching GrownTransformer.
    Every path masked, no free highways. Matched density control.
    """

    def __init__(self, density, vocab_size=30522, hidden=256, ff_dim=512,
                 n_layers=6, n_heads=4, max_len=512, num_classes=2):
        super().__init__()
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.num_classes = num_classes
        self.scale = self.head_dim ** -0.5

        assert hidden % n_heads == 0

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        self.W_qs = nn.ModuleList()
        self.W_ks = nn.ModuleList()
        self.W_vs = nn.ModuleList()
        self.W_os = nn.ModuleList()
        self.ln1s = nn.ModuleList()
        self.ln2s = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for i in range(n_layers):
            self.W_qs.append(nn.Linear(hidden, hidden))
            self.W_ks.append(nn.Linear(hidden, hidden))
            self.W_vs.append(nn.Linear(hidden, hidden))
            self.W_os.append(nn.Linear(hidden, hidden, bias=False))
            self.ln1s.append(nn.LayerNorm(hidden))
            self.ln2s.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))
            # Attention output mask
            attn_mask = (torch.rand(hidden, hidden) < density).float()
            self.register_buffer(f'attn_mask_{i}', attn_mask)
            # FF mask
            ff_mask = (torch.rand(ff_dim, hidden) < density).float()
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
                skip_mask = (torch.rand(hidden, hidden) < density).float()
                self.register_buffer(f'sm_{key}', skip_mask)
                self.skip_pairs.append((s, t))

        self.classifier = nn.Linear(hidden, num_classes)
        cls_mask = (torch.rand(num_classes, hidden) < density).float()
        self.register_buffer('cls_mask', cls_mask)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        acts = [x]

        kpm = ~attention_mask.bool() if attention_mask is not None else None

        for i in range(self.n_layers):
            tgt_band = i + 1
            residual = acts[tgt_band - 1]

            # Manual multi-head attention with random mask on W_o
            ln_x = self.ln1s[i](residual)
            Q = self.W_qs[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.W_ks[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.W_vs[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if kpm is not None:
                scores = scores.masked_fill(kpm.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_out = torch.matmul(F.softmax(scores, dim=-1), V)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden)
            attn_mask = getattr(self, f'attn_mask_{i}')
            attn_out = F.linear(attn_out, self.W_os[i].weight * attn_mask)
            h = residual + attn_out

            # FF with fixed mask
            ln_h = self.ln2s[i](h)
            ff_mask = getattr(self, f'ff_mask_{i}')
            ff_out = F.linear(ln_h, self.ff1s[i].weight * ff_mask, self.ff1s[i].bias)
            ff_out = F.relu(ff_out)
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            # Skip connections
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                skip_mask = getattr(self, f'sm_{key}')
                proj = self.skip_projs[key]
                h = h + F.linear(acts[skip_src], proj.weight * skip_mask)

            acts.append(h)

        # Mean pool
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (acts[-1] * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = acts[-1].mean(dim=1)

        return F.linear(pooled, self.classifier.weight * self.cls_mask, self.classifier.bias)


class DenseSkipTransformer(nn.Module):
    """Same architecture as GrownTransformer but all masks = 1.0.

    Manual multi-head attention (matching GrownTransformer architecture),
    W_o unmasked. Full cross-layer connectivity, no sparsity.
    Shows the skip architecture ceiling.
    """

    def __init__(self, vocab_size=30522, hidden=256, ff_dim=512,
                 n_layers=6, n_heads=4, max_len=512, num_classes=2):
        super().__init__()
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.num_classes = num_classes
        self.scale = self.head_dim ** -0.5

        assert hidden % n_heads == 0

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        self.W_qs = nn.ModuleList()
        self.W_ks = nn.ModuleList()
        self.W_vs = nn.ModuleList()
        self.W_os = nn.ModuleList()
        self.ln1s = nn.ModuleList()
        self.ln2s = nn.ModuleList()
        self.ff1s = nn.ModuleList()
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.W_qs.append(nn.Linear(hidden, hidden))
            self.W_ks.append(nn.Linear(hidden, hidden))
            self.W_vs.append(nn.Linear(hidden, hidden))
            self.W_os.append(nn.Linear(hidden, hidden, bias=False))
            self.ln1s.append(nn.LayerNorm(hidden))
            self.ln2s.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        # Skip projections (all active, no masks)
        self.skip_projs = nn.ModuleDict()
        self.skip_pairs = []
        for t in range(1, n_layers + 1):
            for s in range(t):
                if t - s == 1:
                    continue
                key = f'{t}_{s}'
                self.skip_projs[key] = nn.Linear(hidden, hidden, bias=False)
                self.skip_pairs.append((s, t))

        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        acts = [x]

        kpm = ~attention_mask.bool() if attention_mask is not None else None

        for i in range(self.n_layers):
            tgt_band = i + 1
            residual = acts[tgt_band - 1]

            # Manual multi-head attention, W_o unmasked
            ln_x = self.ln1s[i](residual)
            Q = self.W_qs[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.W_ks[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.W_vs[i](ln_x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if kpm is not None:
                scores = scores.masked_fill(kpm.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_out = torch.matmul(F.softmax(scores, dim=-1), V)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden)
            attn_out = self.W_os[i](attn_out)  # No mask
            h = residual + attn_out

            # Feed-forward (no mask)
            ln_h = self.ln2s[i](h)
            ff_out = F.relu(self.ff1s[i](ln_h))
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            # Skip connections (all active, no mask)
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                h = h + self.skip_projs[key](acts[skip_src])

            acts.append(h)

        # Mean pool
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (acts[-1] * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = acts[-1].mean(dim=1)

        return self.classifier(pooled)
