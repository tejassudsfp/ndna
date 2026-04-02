"""
Developmental Genome: the core model.

A tiny genome (~226 params for MLP, ~258 for CNN) specifies cell types
and growth rules that determine which neurons connect. Default is
disconnected. Connections must be actively grown. Sparsity pressure
acts as metabolic cost.

The genome learns general wiring principles, not task-specific structure.
A genome trained on Fashion-MNIST grows networks for MNIST just as well
as one trained from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StraightThroughHardSigmoid(torch.autograd.Function):
    """Binary mask with straight-through estimator.

    Forward: sigmoid(logits) > 0.5 -> binary {0, 1}
    Backward: gradient of sigmoid (allows genome to learn)
    """

    @staticmethod
    def forward(ctx, logits):
        sig = torch.sigmoid(logits)
        ctx.save_for_backward(sig)
        return (sig > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        sig, = ctx.saved_tensors
        # Gradient of sigmoid: sig * (1 - sig)
        return grad_output * sig * (1 - sig)


class Genome(nn.Module):
    """Developmental genome that specifies neural growth rules.

    ~226 params (with default settings) that control 174K+ potential
    connections through cell type affinities and compatibility rules.

    Args:
        n_types: Number of cell types (default: 8)
        type_dim: Dimensionality of type affinity vectors (default: 8)
        n_bands: Number of neuron bands including input and output (default: 6)
    """

    def __init__(self, n_types=8, type_dim=8, n_bands=6):
        super().__init__()
        self.n_types = n_types
        self.type_dim = type_dim
        self.n_bands = n_bands

        self.affinity = nn.Parameter(torch.randn(n_types, type_dim) * 0.1)

        # Start NEGATIVE: default is disconnected
        self.compatibility = nn.Parameter(torch.randn(n_types, n_types) * 0.3 - 1.0)

        self.connection_scale = nn.Parameter(torch.tensor(3.0))

        # Depth distance penalty (learned)
        self.depth_penalty = nn.Parameter(torch.tensor(2.0))

        self.band_type_base = nn.Parameter(torch.randn(n_bands, n_types) * 0.5)
        self.band_type_grad = nn.Parameter(torch.randn(n_bands, n_types) * 0.3)

    def type_distribution(self, band_idx, n_neurons):
        """Get the cell type distribution for neurons in a band."""
        pos = torch.linspace(0, 1, n_neurons, device=self.affinity.device)
        base = self.band_type_base[band_idx]
        grad = self.band_type_grad[band_idx]
        logits = base + pos.unsqueeze(1) * grad
        return F.softmax(logits * 3, dim=-1)

    def connection_rule(self):
        """Compute type-to-type connection logits."""
        aff = self.affinity @ self.affinity.T / math.sqrt(self.type_dim)
        scale = F.softplus(self.connection_scale)
        return (aff + self.compatibility) * scale

    def growth_mask(self, src_band, tgt_band, src_n, tgt_n, hard=False, temperature=1.0):
        """Compute the growth mask between two bands.

        Returns a (tgt_n, src_n) matrix of connection probabilities [0, 1].
        When hard=True, uses straight-through estimator for binary {0,1} masks.
        Temperature > 1.0 sharpens sigmoid toward binary smoothly.
        """
        src_t = self.type_distribution(src_band, src_n)
        tgt_t = self.type_distribution(tgt_band, tgt_n)
        rule = self.connection_rule()

        # Base connection logits from type compatibility
        logits = tgt_t @ rule @ src_t.T

        # Depth distance penalty: farther bands = harder to connect
        depth_dist = abs(tgt_band - src_band) / self.n_bands
        penalty = F.softplus(self.depth_penalty) * depth_dist
        logits = logits - penalty

        if hard:
            return StraightThroughHardSigmoid.apply(logits)
        return torch.sigmoid(logits * temperature)

    def sparsity_loss(self, dims):
        """Metabolic cost: penalize total connection strength."""
        total = 0
        n_bands = len(dims)
        for t in range(1, n_bands):
            for s in range(t):
                total = total + self.growth_mask(s, t, dims[s], dims[t]).sum()
        n_possible = sum(
            dims[t] * dims[s]
            for t in range(1, n_bands) for s in range(t)
        )
        return total / n_possible

    def sparsity_loss_adjacent_only(self, hidden_dim, ff_dim, n_layers):
        """Metabolic cost for adjacent-band masks only (no quadratic skip iteration).

        Used by GrownGPT2 where skip connections are provided by the residual
        stream and the genome only controls W_o and FF1 per layer.
        """
        total = 0
        n_possible = 0
        for i in range(n_layers):
            # W_o mask: (hidden, hidden)
            wo_mask = self.growth_mask(i, i + 1, hidden_dim, hidden_dim)
            total = total + wo_mask.sum()
            n_possible += wo_mask.numel()
            # FF1 mask: (ff_dim, hidden)
            ff_mask = self.growth_mask(i, i + 1, hidden_dim, ff_dim)
            total = total + ff_mask.sum()
            n_possible += ff_mask.numel()
        return total / n_possible


class GrownNetwork(nn.Module):
    """Network with genome-grown topology. Skip connections emerge.

    Args:
        genome: A Genome instance that provides growth masks
        input_dim: Input dimensionality
        hidden_bands: List of hidden band sizes, e.g. [48, 48, 48, 48]
        output_dim: Output dimensionality (number of classes)
        hard_masks: If True, use straight-through estimator for binary masks
    """

    def __init__(self, genome, input_dim, hidden_bands, output_dim, hard_masks=False):
        super().__init__()
        self.genome = genome
        self.hard_masks = hard_masks
        self.dims = [input_dim] + hidden_bands + [output_dim]
        self.n_bands = len(self.dims)

        self.weights = nn.ParameterDict()
        for t in range(1, self.n_bands):
            for s in range(t):
                fan = self.dims[s] + self.dims[t]
                self.weights[f'{t}_{s}'] = nn.Parameter(
                    torch.randn(self.dims[t], self.dims[s]) * (2.0 / math.sqrt(fan))
                )
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for d in self.dims[1:]
        ])

    def forward(self, x):
        acts = [x]
        for t in range(1, self.n_bands):
            h = self.biases[t - 1]
            for s in range(t):
                W = self.weights[f'{t}_{s}']
                mask = self.genome.growth_mask(
                    s, t, self.dims[s], self.dims[t], hard=self.hard_masks
                )
                h = h + acts[s] @ (mask * W).T
            if t < self.n_bands - 1:
                h = F.relu(h)
            acts.append(h)
        return acts[-1]

    def count_effective(self, threshold=0.5):
        """Count active connections and compute density metrics."""
        total = active = 0
        mask_sum = 0.0
        with torch.no_grad():
            for t in range(1, self.n_bands):
                for s in range(t):
                    m = self.genome.growth_mask(s, t, self.dims[s], self.dims[t])
                    total += m.numel()
                    active += (m > threshold).sum().item()
                    mask_sum += m.sum().item()
        soft_density = mask_sum / total if total > 0 else 0.0
        return active, total, soft_density

    def describe_topology(self):
        """Print which band pairs have connections."""
        print("    Connection density per band pair:")
        with torch.no_grad():
            for t in range(1, self.n_bands):
                for s in range(t):
                    m = self.genome.growth_mask(s, t, self.dims[s], self.dims[t])
                    d = (m > 0.5).float().mean().item()
                    avg = m.mean().item()
                    if d > 0.01 or avg > 0.01:
                        label_s = "input" if s == 0 else f"band{s}"
                        label_t = f"band{t}" if t < self.n_bands - 1 else "output"
                        print(f"      {label_s}->{label_t}: {d:.1%} density, avg={avg:.3f}")


class GrownConvNetwork(nn.Module):
    """CNN with genome-grown channel connectivity. Skip connections emerge.

    Band mapping (n_bands=8):
      0: Input         3ch   32x32
      1: Group1 Conv1  16ch  32x32
      2: Group1 Conv2  16ch  32x32
      3: Group2 Conv1  32ch  16x16
      4: Group2 Conv2  32ch  16x16
      5: Group3 Conv1  64ch  8x8
      6: Group3 Conv2  64ch  8x8
      7: FC output     10ch  1x1

    Genome mask shape (out_ch, in_ch) applied as weight * mask[:,:,None,None]
    to zero entire 3x3 kernels for masked channel pairs.
    """

    def __init__(self, genome, num_classes=10, hard_masks=False):
        super().__init__()
        self.genome = genome
        self.hard_masks = hard_masks
        self.temperature = 1.0  # Anneal from 1.0 to 10.0 during training
        self.num_classes = num_classes

        # Band definitions: (channels, spatial_size)
        self.band_channels = [3, 16, 16, 32, 32, 64, 64, num_classes]
        # Spatial sizes for reference: [32, 32, 32, 16, 16, 8, 8, 1]

        # Primary convs: adjacent band pairs (3x3)
        # Band 0->1: 3->16, stride 1
        # Band 1->2: 16->16, stride 1
        # Band 2->3: 16->32, stride 2 (downsample)
        # Band 3->4: 32->32, stride 1
        # Band 4->5: 32->64, stride 2 (downsample)
        # Band 5->6: 64->64, stride 1
        self.primary_convs = nn.ModuleList()
        self.primary_bns = nn.ModuleList()
        adjacent_pairs = [
            (0, 1, 1), (1, 2, 1), (2, 3, 2),
            (3, 4, 1), (4, 5, 2), (5, 6, 1)
        ]
        for src, tgt, stride in adjacent_pairs:
            in_ch = self.band_channels[src]
            out_ch = self.band_channels[tgt]
            self.primary_convs.append(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            )
            self.primary_bns.append(nn.BatchNorm2d(out_ch))
        self.adjacent_pairs = adjacent_pairs

        # Skip convs: non-adjacent band pairs (1x1)
        self.skip_convs = nn.ModuleDict()
        self.skip_bns = nn.ModuleDict()
        self.skip_pairs = []
        for tgt in range(1, 7):  # bands 1-6 (conv bands)
            for src in range(tgt):
                if tgt - src == 1:
                    continue  # adjacent, handled above
                in_ch = self.band_channels[src]
                out_ch = self.band_channels[tgt]
                key = f'{tgt}_{src}'
                self.skip_convs[key] = nn.Conv2d(
                    in_ch, out_ch, 1, bias=False
                )
                self.skip_bns[key] = nn.BatchNorm2d(out_ch)
                self.skip_pairs.append((src, tgt))

        # FC layer: band 6 -> band 7 (after global avg pool)
        self.fc = nn.Linear(64, num_classes, bias=True)

        # Spatial sizes for each band (for adaptive pooling in skips)
        self.band_spatial = [32, 32, 32, 16, 16, 8, 8]

    def _get_mask(self, src_band, tgt_band):
        """Get genome mask for a band pair."""
        src_ch = self.band_channels[src_band]
        tgt_ch = self.band_channels[tgt_band]
        return self.genome.growth_mask(
            src_band, tgt_band, src_ch, tgt_ch,
            hard=self.hard_masks, temperature=self.temperature
        )

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        acts = [x]  # band 0

        # Build activations for bands 1-6
        adj_idx = 0
        for tgt in range(1, 7):
            # Primary conv from adjacent band
            src = tgt - 1
            mask = self._get_mask(src, tgt)
            conv = self.primary_convs[adj_idx]
            bn = self.primary_bns[adj_idx]
            # Apply mask to conv weights: (out_ch, in_ch, 3, 3)
            w = conv.weight * mask[:, :, None, None]
            stride = self.adjacent_pairs[adj_idx][2]
            h = F.conv2d(acts[src], w, stride=stride, padding=1)
            adj_idx += 1

            # Add skip connections from non-adjacent bands
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt:
                    continue
                key = f'{tgt}_{skip_src}'
                skip_mask = self._get_mask(skip_src, tgt)
                skip_conv = self.skip_convs[key]
                skip_w = skip_conv.weight * skip_mask[:, :, None, None]
                # Match spatial dimensions
                src_act = acts[skip_src]
                tgt_spatial = self.band_spatial[tgt]
                if src_act.shape[2] != tgt_spatial:
                    src_act = F.adaptive_avg_pool2d(src_act, tgt_spatial)
                skip_h = F.conv2d(src_act, skip_w)
                skip_h = self.skip_bns[key](skip_h)
                h = h + skip_h

            h = bn(h)
            h = F.relu(h)
            acts.append(h)

        # Global average pool
        out = F.adaptive_avg_pool2d(acts[6], 1).view(x.size(0), -1)

        # FC with genome mask (band 6 -> band 7)
        fc_mask = self._get_mask(6, 7)
        out = F.linear(out, self.fc.weight * fc_mask, self.fc.bias)

        return out

    def count_effective(self, threshold=0.5):
        """Count active channel connections and density metrics."""
        total = active = 0
        mask_sum = 0.0
        with torch.no_grad():
            for tgt in range(1, 8):
                for src in range(tgt):
                    m = self._get_mask(src, tgt)
                    total += m.numel()
                    active += (m > threshold).sum().item()
                    mask_sum += m.sum().item()
        soft_density = mask_sum / total if total > 0 else 0.0
        return active, total, soft_density

    def describe_topology(self):
        """Print channel connectivity per band pair."""
        print("    Channel connectivity per band pair:")
        band_names = ['input', 'g1c1', 'g1c2', 'g2c1', 'g2c2', 'g3c1', 'g3c2', 'fc']
        with torch.no_grad():
            for tgt in range(1, 8):
                for src in range(tgt):
                    m = self._get_mask(src, tgt)
                    d = (m > 0.5).float().mean().item()
                    avg = m.mean().item()
                    if d > 0.01 or avg > 0.01:
                        kind = "adj" if tgt - src == 1 else "skip"
                        print(f"      {band_names[src]}->{band_names[tgt]} "
                              f"({kind}): {d:.1%} hard, avg={avg:.3f}")


class GrownTransformer(nn.Module):
    """Transformer encoder where EVERY information path goes through a genome mask.

    Band mapping (n_bands=8):
      0: Embedding output   256 dims
      1: Transformer layer 1  256 dims
      2: Transformer layer 2  256 dims
      3: Transformer layer 3  256 dims
      4: Transformer layer 4  256 dims
      5: Transformer layer 5  256 dims
      6: Transformer layer 6  256 dims
      7: Classifier           num_classes dims

    Manual multi-head attention replaces nn.MultiheadAttention so we can
    genome-mask the output projection W_o. Q, K, V projections and the
    attention computation itself are standard (no genome). The genome masks
    W_o, which combines head outputs back to hidden dim. Analogous to CNN:
    spatial convolution is standard, but channel connectivity is genome-controlled.

    Genome controls:
      1. Attention output projection: genome mask on W_o (hidden x hidden)
      2. FF first linear: genome mask (ff_dim x hidden)
      3. Cross-layer skip projections: genome mask (hidden x hidden)
      4. Classifier: genome mask (num_classes x hidden)

    No free paths. Every information flow goes through a genome gate.
    """

    def __init__(self, genome, vocab_size=30522, hidden=256, ff_dim=512,
                 n_layers=6, n_heads=4, max_len=512, num_classes=2):
        super().__init__()
        self.genome = genome
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.num_classes = num_classes
        self.hard_masks = False
        self.temperature = 1.0

        assert hidden % n_heads == 0, f"hidden={hidden} not divisible by n_heads={n_heads}"

        # Band dims for sparsity_loss: [256, 256, 256, 256, 256, 256, 256, num_classes]
        self.band_dims = [hidden] * (n_layers + 1) + [num_classes]

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        # Transformer layers: manual attention + FF
        self.W_qs = nn.ModuleList()
        self.W_ks = nn.ModuleList()
        self.W_vs = nn.ModuleList()
        self.W_os = nn.ModuleList()  # output projection, genome-masked
        self.ln1s = nn.ModuleList()
        self.ln2s = nn.ModuleList()
        self.ff1s = nn.ModuleList()  # hidden -> ff_dim
        self.ff2s = nn.ModuleList()  # ff_dim -> hidden

        for _ in range(n_layers):
            self.W_qs.append(nn.Linear(hidden, hidden))
            self.W_ks.append(nn.Linear(hidden, hidden))
            self.W_vs.append(nn.Linear(hidden, hidden))
            self.W_os.append(nn.Linear(hidden, hidden, bias=False))
            self.ln1s.append(nn.LayerNorm(hidden))
            self.ln2s.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        # Skip projections for non-adjacent band pairs
        self.skip_projs = nn.ModuleDict()
        self.skip_pairs = []
        for t in range(1, n_layers + 1):  # bands 1-6
            for s in range(t):
                if t - s == 1:
                    continue
                key = f'{t}_{s}'
                self.skip_projs[key] = nn.Linear(hidden, hidden, bias=False)
                self.skip_pairs.append((s, t))

        # Classifier: band 6 -> band 7
        self.classifier = nn.Linear(hidden, num_classes)

        # Scale for attention
        self.scale = self.head_dim ** -0.5

    def _get_mask(self, src_band, tgt_band, src_n, tgt_n):
        """Get genome mask for a band pair."""
        return self.genome.growth_mask(
            src_band, tgt_band, src_n, tgt_n,
            hard=self.hard_masks, temperature=self.temperature
        )

    def _manual_attention(self, x, layer_idx, key_padding_mask=None):
        """Manual multi-head attention. Q/K/V are standard, W_o is genome-masked."""
        B, L, _ = x.shape

        Q = self.W_qs[layer_idx](x)  # (B, L, hidden)
        K = self.W_ks[layer_idx](x)
        V = self.W_vs[layer_idx](x)

        # Reshape to (B, n_heads, L, head_dim)
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (B, L), True = ignore
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: (B, n_heads, L, head_dim)
        attn_out = torch.matmul(attn_weights, V)

        # Concatenate heads: (B, L, hidden)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden)

        # Output projection with GENOME MASK
        tgt_band = layer_idx + 1
        attn_mask = self._get_mask(tgt_band - 1, tgt_band, self.hidden, self.hidden)
        # attn_mask: (hidden, hidden) - masks W_o weights
        out = F.linear(attn_out, self.W_os[layer_idx].weight * attn_mask)

        return out

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        # Band 0: embedding
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        acts = [x]  # (B, L, hidden)

        # Key padding mask for attention (True = ignore position)
        kpm = ~attention_mask.bool() if attention_mask is not None else None

        # Bands 1-6: transformer layers
        for i in range(self.n_layers):
            tgt_band = i + 1
            residual = acts[tgt_band - 1]

            # Self-attention with genome-masked output projection
            ln_x = self.ln1s[i](residual)
            attn_out = self._manual_attention(ln_x, i, key_padding_mask=kpm)
            h = residual + attn_out

            # Feed-forward with genome mask on first linear
            ln_h = self.ln2s[i](h)
            ff_w1 = self.ff1s[i].weight  # (ff_dim, hidden)
            ff_b1 = self.ff1s[i].bias
            ff_mask = self._get_mask(tgt_band - 1, tgt_band, self.hidden, self.ff_dim)
            ff_out = F.linear(ln_h, ff_w1 * ff_mask, ff_b1)
            ff_out = F.relu(ff_out)
            ff_out = self.ff2s[i](ff_out)
            h = h + ff_out

            # Skip connections from non-adjacent earlier layers
            for skip_src, skip_tgt in self.skip_pairs:
                if skip_tgt != tgt_band:
                    continue
                key = f'{tgt_band}_{skip_src}'
                skip_mask = self._get_mask(skip_src, tgt_band, self.hidden, self.hidden)
                proj = self.skip_projs[key]
                skip_out = F.linear(acts[skip_src], proj.weight * skip_mask)
                h = h + skip_out

            acts.append(h)

        # Mean pool over sequence length
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (acts[-1] * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = acts[-1].mean(dim=1)

        # Classifier with genome mask (band 6 -> band 7)
        cls_mask = self._get_mask(self.n_layers, self.n_layers + 1,
                                   self.hidden, self.num_classes)
        out = F.linear(pooled, self.classifier.weight * cls_mask, self.classifier.bias)
        return out

    def count_effective(self, threshold=0.5):
        """Count active connections and density metrics."""
        total = active = 0
        mask_sum = 0.0
        with torch.no_grad():
            for t in range(1, self.n_layers + 2):  # bands 1-7
                for s in range(t):
                    if t <= self.n_layers and t - s == 1:
                        # Adjacent band pair: attention W_o mask + FF mask
                        # Attention output mask: (hidden, hidden)
                        m_attn = self._get_mask(s, t, self.hidden, self.hidden)
                        total += m_attn.numel()
                        active += (m_attn > threshold).sum().item()
                        mask_sum += m_attn.sum().item()
                        # FF mask: (ff_dim, hidden)
                        m_ff = self._get_mask(s, t, self.hidden, self.ff_dim)
                        total += m_ff.numel()
                        active += (m_ff > threshold).sum().item()
                        mask_sum += m_ff.sum().item()
                    elif t == self.n_layers + 1 and s == self.n_layers:
                        # Classifier mask (only from last layer)
                        m = self._get_mask(s, t, self.hidden, self.num_classes)
                        total += m.numel()
                        active += (m > threshold).sum().item()
                        mask_sum += m.sum().item()
                    elif t <= self.n_layers and t - s > 1:
                        # Skip projection mask
                        m = self._get_mask(s, t, self.hidden, self.hidden)
                        total += m.numel()
                        active += (m > threshold).sum().item()
                        mask_sum += m.sum().item()
        soft_density = mask_sum / total if total > 0 else 0.0
        return active, total, soft_density

    def describe_topology(self):
        """Print connection density per band pair."""
        print("    Transformer connectivity per band pair:")
        band_names = ['emb', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'cls']
        with torch.no_grad():
            for t in range(1, self.n_layers + 2):
                for s in range(t):
                    if t <= self.n_layers and t - s == 1:
                        # Show attention and FF masks separately
                        m_attn = self._get_mask(s, t, self.hidden, self.hidden)
                        m_ff = self._get_mask(s, t, self.hidden, self.ff_dim)
                        d_attn = (m_attn > 0.5).float().mean().item()
                        d_ff = (m_ff > 0.5).float().mean().item()
                        avg_attn = m_attn.mean().item()
                        avg_ff = m_ff.mean().item()
                        if d_attn > 0.01 or avg_attn > 0.01:
                            print(f"      {band_names[s]}->{band_names[t]} "
                                  f"(attn W_o): {d_attn:.1%} hard, avg={avg_attn:.3f}")
                        if d_ff > 0.01 or avg_ff > 0.01:
                            print(f"      {band_names[s]}->{band_names[t]} "
                                  f"(FF): {d_ff:.1%} hard, avg={avg_ff:.3f}")
                    elif t == self.n_layers + 1 and s == self.n_layers:
                        m = self._get_mask(s, t, self.hidden, self.num_classes)
                        d = (m > 0.5).float().mean().item()
                        avg = m.mean().item()
                        if d > 0.01 or avg > 0.01:
                            print(f"      {band_names[s]}->{band_names[t]} "
                                  f"(cls): {d:.1%} hard, avg={avg:.3f}")
                    elif t <= self.n_layers and t - s > 1:
                        m = self._get_mask(s, t, self.hidden, self.hidden)
                        d = (m > 0.5).float().mean().item()
                        avg = m.mean().item()
                        if d > 0.01 or avg > 0.01:
                            print(f"      {band_names[s]}->{band_names[t]} "
                                  f"(skip): {d:.1%} hard, avg={avg:.3f}")


class GrownGPT2(nn.Module):
    """GPT-2 with genome-controlled connectivity on W_o and FF1.

    Causal (decoder-only) transformer. The genome masks the attention output
    projection (W_o) and feed-forward expansion (FF1) in each layer. No
    genome-controlled skip connections because GPT-2's residual stream already
    provides free skip paths.

    Band mapping (n_bands=14):
      0:     Embedding output    (hidden dims)
      1-12:  Transformer layers  (hidden dims each)
      13:    LM head output      (vocab dims, weight-tied, unmasked)

    Masked connections per layer: W_o (768x768) + FF1 (3072x768) = 2.95M
    Total masked: 12 layers x 2.95M = 35.4M
    Genome params: 354 (K=8, D=8, L=14)
    Compression: ~100,000:1

    Key differences from GrownTransformer:
    - Causal attention (lower-triangular mask)
    - GELU activation (not ReLU)
    - Weight-tied LM head (output = embedding.T)
    - No genome skip connections (residual stream is free)
    - Adjacent-only sparsity loss (linear, not quadratic)
    - FlashAttention compatible (genome masks W_o only, not attention)
    """

    def __init__(self, genome, vocab_size=50257, hidden=768, ff_dim=3072,
                 n_layers=12, n_heads=12, max_len=1024):
        super().__init__()
        self.genome = genome
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.max_len = max_len
        self.hard_masks = False
        self.register_buffer('temperature', torch.tensor(1.0))

        assert hidden % n_heads == 0, f"hidden={hidden} not divisible by n_heads={n_heads}"

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(0.0)  # configurable

        # Transformer layers
        self.ln1s = nn.ModuleList()
        self.W_qs = nn.ModuleList()
        self.W_ks = nn.ModuleList()
        self.W_vs = nn.ModuleList()
        self.W_os = nn.ModuleList()  # genome-masked
        self.ln2s = nn.ModuleList()
        self.ff1s = nn.ModuleList()  # genome-masked
        self.ff2s = nn.ModuleList()

        for _ in range(n_layers):
            self.ln1s.append(nn.LayerNorm(hidden))
            self.W_qs.append(nn.Linear(hidden, hidden))
            self.W_ks.append(nn.Linear(hidden, hidden))
            self.W_vs.append(nn.Linear(hidden, hidden))
            self.W_os.append(nn.Linear(hidden, hidden, bias=False))
            self.ln2s.append(nn.LayerNorm(hidden))
            self.ff1s.append(nn.Linear(hidden, ff_dim))
            self.ff2s.append(nn.Linear(ff_dim, hidden))

        self.ln_f = nn.LayerNorm(hidden)

        # Weight-tied LM head (no separate linear, uses token_emb.weight)
        # No bias for LM head

        # Attention scale
        self.scale = self.head_dim ** -0.5

        # Mask cache for efficiency (recomputed each forward when training)
        self._mask_cache = {}

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Scale residual projections (W_o and ff2) by 1/sqrt(2*n_layers)
        # following GPT-2 convention
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for i in range(self.n_layers):
            self.W_os[i].weight.data *= scale
            self.ff2s[i].weight.data *= scale

    def _get_mask(self, src_band, tgt_band, src_n, tgt_n):
        """Get genome mask, using cache during eval."""
        key = (src_band, tgt_band, src_n, tgt_n)
        if not self.training and key in self._mask_cache:
            return self._mask_cache[key]
        mask = self.genome.growth_mask(
            src_band, tgt_band, src_n, tgt_n,
            hard=self.hard_masks, temperature=self.temperature
        )
        if not self.training:
            self._mask_cache[key] = mask
        return mask

    def _causal_attention(self, x, layer_idx):
        """Causal multi-head attention. Q/K/V standard, W_o genome-masked.

        Uses F.scaled_dot_product_attention for FlashAttention when available.
        """
        B, L, _ = x.shape

        Q = self.W_qs[layer_idx](x)
        K = self.W_ks[layer_idx](x)
        V = self.W_vs[layer_idx](x)

        # Reshape to (B, n_heads, L, head_dim)
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's SDPA with causal mask (enables FlashAttention)
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Concatenate heads: (B, L, hidden)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden)

        # Output projection with GENOME MASK
        # Band i -> band i+1 for layer i
        wo_mask = self._get_mask(layer_idx, layer_idx + 1, self.hidden, self.hidden)
        out = F.linear(attn_out, self.W_os[layer_idx].weight * wo_mask)

        return out

    def forward(self, input_ids, targets=None):
        """Forward pass. Returns logits (and loss if targets provided).

        Args:
            input_ids: (B, L) token indices
            targets: (B, L) target token indices for computing loss
        Returns:
            logits: (B, L, vocab_size)
            loss: scalar if targets provided, else None
        """
        B, L = input_ids.shape
        assert L <= self.max_len, f"Sequence length {L} > max_len {self.max_len}"

        # Clear mask cache at start of forward
        self._mask_cache = {}

        positions = torch.arange(L, device=input_ids.device)

        # Embedding (band 0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Transformer layers (bands 1-12)
        for i in range(self.n_layers):
            # Pre-norm attention
            ln_x = self.ln1s[i](x)
            attn_out = self._causal_attention(ln_x, i)
            x = x + attn_out

            # Pre-norm feed-forward with genome mask on FF1
            ln_x = self.ln2s[i](x)
            ff1_mask = self._get_mask(i, i + 1, self.hidden, self.ff_dim)
            ff_out = F.linear(ln_x, self.ff1s[i].weight * ff1_mask, self.ff1s[i].bias)
            ff_out = F.gelu(ff_out)
            ff_out = self.ff2s[i](ff_out)
            x = x + ff_out

        x = self.ln_f(x)

        # Weight-tied LM head: logits = x @ embedding.T
        logits = F.linear(x, self.token_emb.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def count_effective(self, threshold=0.5):
        """Count active connections and density metrics across all masked layers."""
        total = active = 0
        mask_sum = 0.0
        with torch.no_grad():
            for i in range(self.n_layers):
                # W_o mask
                m_wo = self._get_mask(i, i + 1, self.hidden, self.hidden)
                total += m_wo.numel()
                active += (m_wo > threshold).sum().item()
                mask_sum += m_wo.sum().item()
                # FF1 mask
                m_ff = self._get_mask(i, i + 1, self.hidden, self.ff_dim)
                total += m_ff.numel()
                active += (m_ff > threshold).sum().item()
                mask_sum += m_ff.sum().item()
        soft_density = mask_sum / total if total > 0 else 0.0
        return active, total, soft_density

    def describe_topology(self):
        """Print per-layer density for W_o and FF1."""
        print("    GPT-2 connectivity per layer:")
        with torch.no_grad():
            for i in range(self.n_layers):
                m_wo = self._get_mask(i, i + 1, self.hidden, self.hidden)
                m_ff = self._get_mask(i, i + 1, self.hidden, self.ff_dim)
                d_wo = (m_wo > 0.5).float().mean().item()
                d_ff = (m_ff > 0.5).float().mean().item()
                avg_wo = m_wo.mean().item()
                avg_ff = m_ff.mean().item()
                print(f"      L{i+1:2d} W_o: {d_wo:5.1%} hard, avg={avg_wo:.3f}  |  "
                      f"FF1: {d_ff:5.1%} hard, avg={avg_ff:.3f}")

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            # Crop to max_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_len else input_ids[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
